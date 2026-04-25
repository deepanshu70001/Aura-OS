from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.voice_service import VoiceSession

logger = logging.getLogger(__name__)
router = APIRouter()


async def _safe_send(websocket: WebSocket, data: dict) -> bool:
    """Send JSON to the WebSocket, returning False if the connection is gone."""
    try:
        await websocket.send_json(data)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception as exc:
        logger.debug(f"[ws/audio] send failed: {exc!r}")
        return False


async def _safe_task(coro):
    """Run a coroutine and suppress errors from closed WebSockets."""
    try:
        await coro
    except (WebSocketDisconnect, RuntimeError, Exception) as exc:
        logger.debug(f"[ws/audio] background task ended: {exc!r}")


@router.get("/api/v1/audio/health")
async def health():
    return {"ok": True, "service": "audio-stream"}


@router.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()

    user_id = websocket.query_params.get("userId")
    task_context = websocket.query_params.get("taskContext")
    session = VoiceSession(user_id=user_id, task_context=task_context)

    await _safe_send(
        websocket,
        {
            "type": "emotion_update",
            "emotion": "calm",
            "pitch_score": 0.0,
            "cadence_score": 0.0,
        },
    )

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            raw_chunk = message.get("bytes")
            if raw_chunk is not None:
                processed = await session.process_chunk(raw_chunk)
                for outbound in processed:
                    if outbound.get("type") == "_async_utterance":
                        merged_audio = outbound["_merged_audio"]

                        # Send immediate "thinking" indicator (error-safe)
                        asyncio.create_task(_safe_task(_safe_send(websocket, {
                            "type": "transcript",
                            "text": "Listening... 🎙️",
                        })))

                        # Fire and forget the LLM/transcript call so acoustic logic doesn't block
                        asyncio.create_task(_safe_task(
                            session.finish_utterance_async(merged_audio, websocket)
                        ))
                    else:
                        if not await _safe_send(websocket, outbound):
                            # Connection is dead, stop processing
                            return
                continue

            text_message = message.get("text")
            if text_message and text_message.strip().lower() == "ping":
                await _safe_send(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        session.flush_pending_event()
    except Exception:
        session.flush_pending_event()
        raise
