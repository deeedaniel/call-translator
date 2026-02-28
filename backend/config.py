from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Publicly reachable hostname used in TwiML <Stream> URL
    public_host: str = "localhost:8000"

    # Twilio credentials (optional; used for request signature validation)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""

    # STUN/TURN server for WebRTC ICE negotiation
    webrtc_stun_url: str = "stun:stun.l.google.com:19302"

    # CORS origin allowed for the frontend dev server
    cors_allowed_origin: str = "http://localhost:5173"

    # ---- VAD settings ----
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 700

    # ---- NVIDIA NIM ASR settings ----
    nvidia_api_key: str = ""
    nvidia_asr_function_id: str = "d3fe9151-442b-4204-a70d-5fcc597fd610"
    nvidia_grpc_endpoint: str = "grpc.nvcf.nvidia.com:443"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
