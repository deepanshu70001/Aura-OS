from __future__ import annotations

import logging
import time

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Rate-limit repeated timeout/error logs to avoid flooding the console.
_last_log_time: dict[str, float] = {}
_LOG_INTERVAL_SEC = 30.0


def _should_log(key: str) -> bool:
    """Return True at most once per _LOG_INTERVAL_SEC for a given key."""
    now = time.monotonic()
    last = _last_log_time.get(key, 0.0)
    if now - last >= _LOG_INTERVAL_SEC:
        _last_log_time[key] = now
        return True
    return False


async def post_vocal_stress_event(
    *,
    user_id: str | None,
    emotion: str,
    arousal_score: float,
    task_context: str | None = None,
    clinical_alert: bool = False,
    transcript: str | None = None,
    semantic_summary: str | None = None,
    semantic_intent: str | None = None,
    risk_level: str | None = None,
    companion_message: str | None = None,
) -> None:
    if not user_id:
        return

    url = f"{settings.NODE_BACKEND_URL.rstrip('/')}/api/clinical/vocal-stress"
    payload = {
        "userId": user_id,
        "emotion": emotion,
        "arousalScore": arousal_score,
        "taskContext": task_context,
        "clinicalAlert": bool(clinical_alert),
    }
    if transcript:
        payload["transcript"] = transcript
    if semantic_summary:
        payload["semanticSummary"] = semantic_summary
    if semantic_intent:
        payload["semanticIntent"] = semantic_intent
    if risk_level:
        payload["riskLevel"] = risk_level
    if companion_message:
        payload["companionMessage"] = companion_message

    timeout = httpx.Timeout(connect=2.0, read=4.0, write=4.0, pool=4.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            if response.status_code >= 400:
                if _should_log("node_http_error"):
                    logger.warning(
                        f"[NodeBridge] sync failed ({response.status_code}): {response.text[:100]}"
                    )
    except httpx.ConnectError:
        if _should_log("node_connect_refused"):
            logger.warning(f"[NodeBridge] connection refused at {url}. Is the Node backend running?")
    except httpx.ConnectTimeout:
        if _should_log("node_connect_timeout"):
            logger.warning(f"[NodeBridge] connection timeout at {url}. Network is slow or server is overloaded.")
    except Exception as exc:
        if _should_log("node_other"):
            logger.warning(f"[NodeBridge] sync error: {exc!r}")
