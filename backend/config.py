from __future__ import annotations
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Publicly reachable hostname used in TwiML <Stream> URL
    public_host: str = "localhost:8000"

    # Twilio credentials (optional; used for request signature validation)
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN")

    # STUN/TURN server for WebRTC ICE negotiation
    webrtc_stun_url: str = "stun:stun.l.google.com:19302"

    # CORS origin allowed for the frontend dev server
    cors_allowed_origin: str = "http://localhost:5173"

    # ---- VAD settings ----
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 700

    # ---- NVIDIA NIM ASR settings ----
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY")
    nvidia_asr_function_id: str = "d3fe9151-442b-4204-a70d-5fcc597fd610"
    nvidia_grpc_endpoint: str = "grpc.nvcf.nvidia.com:443"

    # ---- NVIDIA NIM Translation settings ----
    nvidia_translate_function_id: str = "0778f2eb-b64d-45e7-acae-7dd9b9b35b4d"
    nvidia_translate_grpc_endpoint: str = "grpc.nvcf.nvidia.com:443"

    # ---- NVIDIA NIM TTS settings ----
    nvidia_tts_function_id: str = ""

    # ---- Analytics pipeline settings ----
    analytics_chunk_ms: int = 500
    analytics_emit_interval_ms: int = 500
    analytics_keywords_path: str = ""
    analytics_stress_classifier: str = "rule_based"
    analytics_ml_model_path: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
