from __future__ import annotations

import re
from dataclasses import dataclass

from models.asr_events import TranscriptEvent

CLAUSE_BOUNDARY = re.compile(r"(?<=[.!?;:,])\s")


@dataclass
class BufferChunk:
    """A clause- or sentence-level text segment ready for translation."""

    text: str
    source_language: str
    call_id: str


class PunctuatedBufferStreamer:
    """Yields sentence-boundary chunks from streaming absolute ASR transcripts.

    Streaming ASR emits *absolute* (accumulated) transcripts, not deltas.
    This class tracks a ``_yielded_len`` cursor per call so that each
    ``feed()`` only processes the new suffix beyond what has already been
    sent downstream, avoiding text duplication.
    """

    def __init__(self) -> None:
        self._yielded_len: dict[str, int] = {}

    def feed(self, event: TranscriptEvent) -> list[BufferChunk]:
        yielded = self._yielded_len.get(event.call_id, 0)
        absolute_text = event.original_transcript
        pending = absolute_text[yielded:]
        chunks: list[BufferChunk] = []

        while m := CLAUSE_BOUNDARY.search(pending):
            segment = pending[: m.start()].strip()
            if segment:
                chunks.append(
                    BufferChunk(
                        text=segment,
                        source_language=event.detected_language,
                        call_id=event.call_id,
                    )
                )
            yielded += m.end()
            pending = pending[m.end() :]

        self._yielded_len[event.call_id] = yielded

        if event.is_final:
            residual = pending.strip()
            if residual:
                chunks.append(
                    BufferChunk(
                        text=residual,
                        source_language=event.detected_language,
                        call_id=event.call_id,
                    )
                )
            self._yielded_len.pop(event.call_id, None)

        return chunks

    def flush(self, call_id: str) -> None:
        """Discard any buffered state for a call."""
        self._yielded_len.pop(call_id, None)
