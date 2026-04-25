# backend-python/app/services/behavioral_scorer.py
# ---------------------------------------------------------------
# AuraOS Behavioral Health Scorer
#
# Deterministic feature engineering layer that computes a
# behavioral health profile from in-session interaction data.
# No ML training required -- uses clinical heuristics.
# ---------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BehavioralProfile:
    """Clinical behavioral feature vector derived from session interactions."""

    # Composite scores (0-1 range, higher = more concerning)
    cognitive_load_index: float       # Executive function strain
    emotional_volatility: float       # How rapidly emotions shift
    task_avoidance_ratio: float       # Abandoned / total tasks
    engagement_decay_rate: float      # How quickly engagement drops over session
    worry_amplification: float        # Anxiety amplification factor from Worry Weights

    # Game-derived metrics
    cognitive_forge_score: float      # 0-100 from the naming game
    task_shatterer_completion: float  # 0-1, proportion completed

    # Voice-derived (from accumulated session voice data)
    vocal_stress_trend: float         # +1 = escalating, -1 = de-escalating
    mean_arousal: float               # Average arousal across session
    peak_arousal: float               # Maximum arousal spike

    # Overall risk tier
    risk_tier: str                    # "low" | "moderate" | "high" | "critical"
    risk_score: float                 # 0-100 composite

    # Recommendations
    primary_concern: str              # Top clinical flag
    recommended_intervention: str     # Suggested next action


def compute_behavioral_profile(session_data: dict) -> BehavioralProfile:
    """
    Compute a behavioral health profile from raw session telemetry.

    Expected session_data keys:
      - tasks_created: int
      - tasks_completed: int
      - tasks_abandoned: int
      - worry_weights: list[float]         (0-10 ratings from Worry Weights game)
      - cognitive_forge_score: float       (0-100 from naming game)
      - task_shatterer_steps: int          (total sub-tasks generated)
      - task_shatterer_completed: int      (sub-tasks marked done)
      - emotion_timeline: list[str]        (sequence of detected emotions)
      - arousal_timeline: list[float]      (sequence of arousal scores 1-10)
      - session_duration_minutes: float
      - voice_segments_count: int
    """

    # -- Extract raw metrics with safe defaults --
    tasks_created = max(session_data.get("tasks_created", 0), 0)
    tasks_completed = max(session_data.get("tasks_completed", 0), 0)
    tasks_abandoned = max(session_data.get("tasks_abandoned", 0), 0)

    worry_weights = session_data.get("worry_weights", [])
    forge_score = _clamp(session_data.get("cognitive_forge_score", 50.0), 0, 100)

    shatterer_steps = max(session_data.get("task_shatterer_steps", 0), 0)
    shatterer_done = max(session_data.get("task_shatterer_completed", 0), 0)

    emotions = session_data.get("emotion_timeline", [])
    arousals = session_data.get("arousal_timeline", [])
    duration = max(session_data.get("session_duration_minutes", 1.0), 0.1)
    voice_count = max(session_data.get("voice_segments_count", 0), 0)

    # ---------------------------------------------------------------
    # 1. Cognitive Load Index (0-1)
    #    High = many tasks created but few completed, low forge score
    # ---------------------------------------------------------------
    if tasks_created > 0:
        completion_ratio = tasks_completed / tasks_created
    else:
        completion_ratio = 1.0  # No tasks = no overload

    forge_norm = forge_score / 100.0
    cognitive_load_index = _clamp(
        0.4 * (1.0 - completion_ratio) +
        0.3 * (1.0 - forge_norm) +
        0.3 * min(tasks_created / 10.0, 1.0)  # Creating too many tasks = overload
    )

    # ---------------------------------------------------------------
    # 2. Emotional Volatility (0-1)
    #    Measure how often emotion changes in the timeline
    # ---------------------------------------------------------------
    if len(emotions) > 1:
        transitions = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        emotional_volatility = _clamp(transitions / (len(emotions) - 1))
    else:
        emotional_volatility = 0.0

    # ---------------------------------------------------------------
    # 3. Task Avoidance Ratio (0-1)
    # ---------------------------------------------------------------
    total_tasks = tasks_created + tasks_abandoned
    if total_tasks > 0:
        task_avoidance_ratio = _clamp(tasks_abandoned / total_tasks)
    else:
        task_avoidance_ratio = 0.0

    # ---------------------------------------------------------------
    # 4. Engagement Decay Rate (0-1)
    #    Compare first-half vs second-half voice activity
    # ---------------------------------------------------------------
    if len(arousals) > 4:
        mid = len(arousals) // 2
        first_half = sum(arousals[:mid]) / max(mid, 1)
        second_half = sum(arousals[mid:]) / max(len(arousals) - mid, 1)
        # If second half is less engaged (lower arousal), decay is positive
        engagement_decay_rate = _clamp((first_half - second_half) / 10.0 + 0.5)
    else:
        engagement_decay_rate = 0.5  # Neutral

    # ---------------------------------------------------------------
    # 5. Worry Amplification (0-1)
    #    How extreme are the worry ratings?
    # ---------------------------------------------------------------
    if worry_weights:
        mean_worry = sum(worry_weights) / len(worry_weights)
        max_worry = max(worry_weights)
        worry_amplification = _clamp(
            0.5 * (mean_worry / 10.0) + 0.5 * (max_worry / 10.0)
        )
    else:
        worry_amplification = 0.0

    # ---------------------------------------------------------------
    # 6. Task Shatterer Completion (0-1)
    # ---------------------------------------------------------------
    if shatterer_steps > 0:
        task_shatterer_completion = _clamp(shatterer_done / shatterer_steps)
    else:
        task_shatterer_completion = 0.0

    # ---------------------------------------------------------------
    # 7. Vocal Stress Trend (-1 to +1)
    #    Linear regression slope of arousal over time
    # ---------------------------------------------------------------
    if len(arousals) >= 3:
        n = len(arousals)
        x_mean = (n - 1) / 2.0
        y_mean = sum(arousals) / n
        numerator = sum((i - x_mean) * (a - y_mean) for i, a in enumerate(arousals))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / max(denominator, 1e-9)
        vocal_stress_trend = _clamp(slope, -1.0, 1.0)
    else:
        vocal_stress_trend = 0.0

    # ---------------------------------------------------------------
    # 8. Mean / Peak Arousal
    # ---------------------------------------------------------------
    mean_arousal = sum(arousals) / max(len(arousals), 1) if arousals else 5.0
    peak_arousal = max(arousals) if arousals else 5.0

    # ---------------------------------------------------------------
    # 9. Risk Score (0-100) and Risk Tier
    # ---------------------------------------------------------------
    risk_score = _clamp(
        15 * cognitive_load_index +
        10 * emotional_volatility +
        15 * task_avoidance_ratio +
        10 * engagement_decay_rate +
        20 * worry_amplification +
        10 * (1.0 - task_shatterer_completion) +
        10 * _clamp(vocal_stress_trend, 0, 1) +
        10 * _clamp((mean_arousal - 1.0) / 9.0),
        lo=0.0, hi=1.0
    ) * 100

    if risk_score >= 75:
        risk_tier = "critical"
    elif risk_score >= 50:
        risk_tier = "high"
    elif risk_score >= 25:
        risk_tier = "moderate"
    else:
        risk_tier = "low"

    # ---------------------------------------------------------------
    # 10. Primary Concern + Recommendation
    # ---------------------------------------------------------------
    concerns = [
        (cognitive_load_index, "Executive Function Overload",
         "Break tasks into smaller steps; use Task Shatterer"),
        (emotional_volatility, "Emotional Dysregulation",
         "Practice grounding exercises; 4-7-8 breathing"),
        (task_avoidance_ratio, "Task Avoidance Pattern",
         "Identify the blocked task and apply the 2-minute rule"),
        (worry_amplification, "Anxiety Amplification",
         "Cognitive restructuring; challenge catastrophic thinking"),
        (_clamp(vocal_stress_trend, 0, 1), "Escalating Vocal Stress",
         "Take a sensory break; cold water on wrists"),
    ]
    top = max(concerns, key=lambda x: x[0])
    primary_concern = top[1]
    recommended_intervention = top[2]

    return BehavioralProfile(
        cognitive_load_index=round(cognitive_load_index, 3),
        emotional_volatility=round(emotional_volatility, 3),
        task_avoidance_ratio=round(task_avoidance_ratio, 3),
        engagement_decay_rate=round(engagement_decay_rate, 3),
        worry_amplification=round(worry_amplification, 3),
        cognitive_forge_score=round(forge_score, 1),
        task_shatterer_completion=round(task_shatterer_completion, 3),
        vocal_stress_trend=round(vocal_stress_trend, 3),
        mean_arousal=round(mean_arousal, 2),
        peak_arousal=round(peak_arousal, 2),
        risk_tier=risk_tier,
        risk_score=round(risk_score, 1),
        primary_concern=primary_concern,
        recommended_intervention=recommended_intervention,
    )


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))
