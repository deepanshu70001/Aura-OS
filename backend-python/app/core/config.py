# app/core/config.py
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_origins(raw_origins: str) -> tuple[str, ...]:
    values = tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip())
    if values:
        return values
    return ("http://localhost:5173", "http://127.0.0.1:5173")


@dataclass(frozen=True)
class Settings:
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")
    FRONTEND_ORIGINS: tuple[str, ...] = _parse_origins(
        os.getenv("FRONTEND_ORIGINS", "")
    )
    NODE_BACKEND_URL: str = os.getenv("NODE_BACKEND_URL", "http://localhost:5001")

    TWILIO_ACCOUNT_SID: str | None = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: str | None = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_WHATSAPP_NUMBER: str | None = os.getenv("TWILIO_WHATSAPP_NUMBER")

    SMTP_SERVER: str | None = os.getenv("SMTP_SERVER")
    SMTP_PORT: int = _env_int("SMTP_PORT", 587)
    SMTP_USERNAME: str | None = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD: str | None = os.getenv("SMTP_PASSWORD")

    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

    VOICE_SAMPLE_RATE: int = _env_int("VOICE_SAMPLE_RATE", 16000)
    VOICE_VAD_THRESHOLD: float = _env_float("VOICE_VAD_THRESHOLD", 0.005)


settings = Settings()
