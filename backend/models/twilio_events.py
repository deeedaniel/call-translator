from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class MediaFormat(BaseModel):
    encoding: str = "audio/x-mulaw"
    sample_rate: int = Field(alias="sampleRate", default=8000)
    channels: int = 1


class ConnectedEvent(BaseModel):
    event: Literal["connected"]
    protocol: str
    version: str


class StartMeta(BaseModel):
    stream_sid: str = Field(alias="streamSid")
    account_sid: str = Field(alias="accountSid")
    call_sid: str = Field(alias="callSid")
    tracks: list[str] = ["inbound"]
    custom_parameters: dict[str, str] = Field(
        alias="customParameters", default_factory=dict
    )
    media_format: MediaFormat = Field(alias="mediaFormat", default_factory=MediaFormat)


class StartEvent(BaseModel):
    event: Literal["start"]
    sequence_number: str = Field(alias="sequenceNumber")
    start: StartMeta
    stream_sid: str = Field(alias="streamSid")


class MediaPayload(BaseModel):
    track: str = "inbound"
    chunk: str
    timestamp: str
    payload: str


class MediaEvent(BaseModel):
    event: Literal["media"]
    sequence_number: str = Field(alias="sequenceNumber")
    media: MediaPayload
    stream_sid: str = Field(alias="streamSid")


class StopEvent(BaseModel):
    event: Literal["stop"]
    sequence_number: str = Field(alias="sequenceNumber")
    stream_sid: str = Field(alias="streamSid")


class MarkEvent(BaseModel):
    event: Literal["mark"]
    sequence_number: str = Field(alias="sequenceNumber")
    stream_sid: str = Field(alias="streamSid")
    mark: dict


TwilioStreamEvent = Union[ConnectedEvent, StartEvent, MediaEvent, StopEvent, MarkEvent]


def parse_twilio_event(data: dict) -> TwilioStreamEvent:
    """Parse a raw Twilio Media Stream JSON message into the appropriate event model."""
    event_type = data.get("event")
    dispatch = {
        "connected": ConnectedEvent,
        "start": StartEvent,
        "media": MediaEvent,
        "stop": StopEvent,
        "mark": MarkEvent,
    }
    model = dispatch.get(event_type)
    if model is None:
        raise ValueError(f"Unknown Twilio stream event type: {event_type!r}")
    return model.model_validate(data)
