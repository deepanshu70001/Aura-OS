# app/api/routes_behavioral.py
# ---------------------------------------------------------------
# REST endpoint for the Behavioral Health Scorer
# ---------------------------------------------------------------
from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.services.behavioral_scorer import compute_behavioral_profile

router = APIRouter()


class SessionTelemetry(BaseModel):
    """Inbound telemetry payload from the frontend."""
    tasks_created: int = 0
    tasks_completed: int = 0
    tasks_abandoned: int = 0
    worry_weights: list[float] = Field(default_factory=list)
    cognitive_forge_score: float = 50.0
    task_shatterer_steps: int = 0
    task_shatterer_completed: int = 0
    emotion_timeline: list[str] = Field(default_factory=list)
    arousal_timeline: list[float] = Field(default_factory=list)
    session_duration_minutes: float = 1.0
    voice_segments_count: int = 0


@router.post("/behavioral-profile")
async def get_behavioral_profile(data: SessionTelemetry):
    """
    Compute a behavioral health profile from session telemetry.
    Returns risk tier, composite score, and clinical recommendations.
    """
    profile = compute_behavioral_profile(data.model_dump())

    return {
        "success": True,
        "profile": {
            "cognitive_load_index": profile.cognitive_load_index,
            "emotional_volatility": profile.emotional_volatility,
            "task_avoidance_ratio": profile.task_avoidance_ratio,
            "engagement_decay_rate": profile.engagement_decay_rate,
            "worry_amplification": profile.worry_amplification,
            "cognitive_forge_score": profile.cognitive_forge_score,
            "task_shatterer_completion": profile.task_shatterer_completion,
            "vocal_stress_trend": profile.vocal_stress_trend,
            "mean_arousal": profile.mean_arousal,
            "peak_arousal": profile.peak_arousal,
            "risk_tier": profile.risk_tier,
            "risk_score": profile.risk_score,
            "primary_concern": profile.primary_concern,
            "recommended_intervention": profile.recommended_intervention,
        }
    }
