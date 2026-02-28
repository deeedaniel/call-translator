from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SDPMessage(BaseModel):
    type: Literal["offer", "answer"]
    sdp: str


class ICECandidate(BaseModel):
    candidate: str
    sdp_mid: Optional[str] = Field(default=None, alias="sdpMid")
    sdp_m_line_index: Optional[int] = Field(default=None, alias="sdpMLineIndex")


class SignalMessage(BaseModel):
    """Discriminated union for WebRTC signaling messages."""

    kind: Literal["sdp", "ice", "bye"]
    sdp: Optional[SDPMessage] = None
    ice: Optional[ICECandidate] = None
