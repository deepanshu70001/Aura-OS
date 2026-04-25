# app/core/database.py
from __future__ import annotations

from typing import Any

from app.core.config import settings

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except Exception as exc:  # pragma: no cover
    AsyncIOMotorClient = None
    _MOTOR_IMPORT_ERROR = exc
else:
    _MOTOR_IMPORT_ERROR = None


class Database:
    client: Any = None
    db: Any = None


db_config = Database()


async def connect_to_mongo() -> None:
    """Connects to MongoDB if the driver and URI are available."""
    if AsyncIOMotorClient is None:
        if _MOTOR_IMPORT_ERROR is None:
            print("[Mongo] motor is not installed; DB-backed triage features are disabled.")
        else:
            print(
                "[Mongo] motor import failed; DB-backed features are disabled. "
                f"Import error: {_MOTOR_IMPORT_ERROR!r}"
            )
        return

    if not settings.MONGODB_URI:
        print("[Mongo] MONGODB_URI is empty; DB-backed triage features are disabled.")
        return

    try:
        db_config.client = AsyncIOMotorClient(settings.MONGODB_URI)
        db_config.db = db_config.client.auraos
        print("[Mongo] Connected.")
    except Exception as exc:
        db_config.client = None
        db_config.db = None
        print(f"[Mongo] Connection failed: {exc}")


async def close_mongo_connection() -> None:
    """Closes MongoDB connection if open."""
    if db_config.client is not None:
        db_config.client.close()
        db_config.client = None
        db_config.db = None
        print("[Mongo] Closed.")
