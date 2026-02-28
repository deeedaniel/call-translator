from __future__ import annotations

import asyncio
import fractions
import logging
import time
from typing import Optional

import av
import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

from services.audio_bus import AudioBus

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_DURATION_S = 0.020  # 20 ms frames to match Twilio cadence
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION_S)  # 320
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)


class TelephonyAudioTrack(MediaStreamTrack):
    """Custom audio track that reads PCM16 16kHz frames from an AudioBus queue
    and yields them as av.AudioFrame objects for WebRTC delivery to the browser.
    """

    kind = "audio"

    def __init__(self, queue: asyncio.Queue[bytes | None]) -> None:
        super().__init__()
        self._queue = queue
        self._pts = 0

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        pcm_bytes = await self._queue.get()
        if pcm_bytes is None:
            self.stop()
            raise MediaStreamError

        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)

        frame = AudioFrame.from_ndarray(
            pcm.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = AUDIO_TIME_BASE
        self._pts += len(pcm)
        return frame


class WebRTCPeerManager:
    """Manages a single aiortc RTCPeerConnection for one operator session.

    Responsibilities:
    - Add a TelephonyAudioTrack so the operator hears the caller.
    - Accept the operator's audio track and make its frames available for
      reverse-transcoding back into the Twilio stream.
    """

    def __init__(
        self,
        call_id: str,
        audio_bus: AudioBus,
        ice_servers: Optional[list[dict]] = None,
    ) -> None:
        self.call_id = call_id
        self._bus = audio_bus
        self._pc = RTCPeerConnection()
        self._telephony_track: TelephonyAudioTrack | None = None
        self._bus_queue: asyncio.Queue | None = None
        self._operator_audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._closed = False

        self._setup_event_handlers()

    @property
    def operator_audio_queue(self) -> asyncio.Queue[bytes]:
        """Queue of PCM16 16kHz frames captured from the operator's microphone."""
        return self._operator_audio_queue

    def _setup_event_handlers(self) -> None:
        @self._pc.on("track")
        async def on_track(track: MediaStreamTrack) -> None:
            if track.kind != "audio":
                return
            logger.info("Operator audio track received for call %s", self.call_id)
            asyncio.ensure_future(self._consume_operator_track(track))

        @self._pc.on("connectionstatechange")
        async def on_state_change() -> None:
            state = self._pc.connectionState
            logger.info(
                "WebRTC connection state for %s: %s", self.call_id, state
            )
            if state in ("failed", "closed"):
                await self.close()

    async def _consume_operator_track(self, track: MediaStreamTrack) -> None:
        """Read frames from the operator's browser audio track."""
        try:
            while True:
                frame: AudioFrame = await track.recv()
                pcm = frame.to_ndarray().flatten().astype(np.int16)
                try:
                    self._operator_audio_queue.put_nowait(pcm.tobytes())
                except asyncio.QueueFull:
                    # Drop oldest to prevent backpressure
                    try:
                        self._operator_audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self._operator_audio_queue.put_nowait(pcm.tobytes())
        except MediaStreamError:
            logger.info("Operator audio track ended for call %s", self.call_id)

    async def create_answer(self, offer_sdp: str, offer_type: str = "offer") -> str:
        """Process an SDP offer from the browser and return an SDP answer.

        Before answering, subscribes to AudioBus and adds the telephony track.
        """
        self._bus_queue = await self._bus.subscribe(self.call_id)
        self._telephony_track = TelephonyAudioTrack(self._bus_queue)
        self._pc.addTrack(self._telephony_track)

        offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        await self._pc.setRemoteDescription(offer)
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        return self._pc.localDescription.sdp

    async def add_ice_candidate(self, candidate_dict: dict) -> None:
        """Add a remote ICE candidate received from the browser."""
        from aiortc import RTCIceCandidate
        # aiortc accepts candidates via addIceCandidate
        # but the simplest path is to include them in the offer
        # For trickle ICE we'd parse and add; for now log it.
        logger.debug("ICE candidate for %s: %s", self.call_id, candidate_dict)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._telephony_track:
            self._telephony_track.stop()
        if self._bus_queue:
            await self._bus.unsubscribe(self.call_id, self._bus_queue)

        await self._pc.close()
        logger.info("WebRTC peer closed for call %s", self.call_id)
