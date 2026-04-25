# backend-python/app/services/audio_engine.py
# ---------------------------------------------------------------
# AuraOS Emotion Engine — 4-Level ML Pipeline
#
# Level 1: Whisper Foundation Encoder (frozen feature extractor)
# Level 2: PyTorch MLP Classifier (3-class + arousal regression)
# Level 3: Acoustic Physics Gate (Praat jitter/shimmer/HNR)
# Level 4: Temporal Bi-LSTM (30-sec escalation detection)
# Final:   Decision Gate (multi-stage clinical alert logic)
#
# Accuracy Improvements (v2):
# - Physics-informed confidence boosting (Level 3 corroborates Level 2)
# - Soft ensemble fallback instead of hard heuristic override
# - Adaptive confidence threshold based on signal quality
# ---------------------------------------------------------------
from __future__ import annotations

import os
import sys
import time
import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import parselmouth
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False

# Add training directory to path so we can import the level modules
_TRAINING_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "training"))
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)


# -- Data class returned to callers ------------------------------------

@dataclass(frozen=True)
class AudioFeatures:
    rms: float
    zero_crossing_rate: float
    pitch_score: float
    cadence_score: float
    stress_score: float
    emotion: str
    arousal_score: float = 5.0
    confidence: float = 0.0
    jitter: float = 0.0
    shimmer: float = 0.0
    hnr: float = 0.0
    f0_mean: float = 0.0
    escalation_prob: float = 0.0
    clinical_alert: bool = False
    model_source: str = "heuristic"


_EMPTY = AudioFeatures(
    rms=0.0, zero_crossing_rate=0.0, pitch_score=0.0, cadence_score=0.0,
    stress_score=0.0, emotion="calm", arousal_score=2.5,
)


# -- Locate model directory --------------------------------------------

def _find_model_dir() -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "models"),
        os.path.join(os.getcwd(), "models"),
        os.path.join(os.getcwd(), "backend-python", "models"),
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.isdir(p):
            return p
    return os.path.normpath(candidates[0])


# -- Emotion Engine (New 4-Level Pipeline) -----------------------------

class EmotionEngine:
    """
    Production inference engine using the 4-Level architecture:
      Level 1: Whisper encoder (frozen) -> 512-dim embeddings
      Level 2: MLP classifier -> 3-class emotion + arousal 1-10
      Level 3: Praat acoustic physics -> jitter/shimmer/HNR gate
      Level 4: Bi-LSTM temporal context -> escalation detection
    """

    LABEL_MAP = {0: "calm", 1: "mild_anxiety", 2: "high_anxiety"}

    def __init__(self):
        self.level1 = None
        self.level2 = None
        self.level3 = None
        self.level4 = None
        self.final_decision_gate_fn = None
        self.is_ready = False
        self.device = None

        # Session-scoped rolling arousal buffers (prevent cross-user leakage).
        self.arousal_buffers: dict[str, collections.deque] = {}
        self.buffer_last_seen: dict[str, float] = {}
        self.max_temporal_sessions = 512
        self.temporal_session_ttl_sec = 10 * 60

        self._load()

    def _get_temporal_buffer(self, session_key: Optional[str]) -> collections.deque:
        now = time.monotonic()
        key = session_key or "__global__"

        if key not in self.arousal_buffers:
            self.arousal_buffers[key] = collections.deque(maxlen=6)
        self.buffer_last_seen[key] = now

        if len(self.arousal_buffers) > self.max_temporal_sessions:
            # Drop oldest inactive sessions first.
            stale = sorted(self.buffer_last_seen.items(), key=lambda kv: kv[1])[:32]
            for stale_key, _ in stale:
                if stale_key == key:
                    continue
                self.arousal_buffers.pop(stale_key, None)
                self.buffer_last_seen.pop(stale_key, None)
                if len(self.arousal_buffers) <= self.max_temporal_sessions:
                    break

        # Opportunistic TTL cleanup.
        expired = [
            sid for sid, ts in self.buffer_last_seen.items()
            if (now - ts) > self.temporal_session_ttl_sec and sid != key
        ]
        for sid in expired[:32]:
            self.arousal_buffers.pop(sid, None)
            self.buffer_last_seen.pop(sid, None)

        return self.arousal_buffers[key]

    def _load(self):
        model_dir = _find_model_dir()

        # -- Level 1: Whisper Foundation --
        if HAS_TORCH:
            try:
                from level1_foundation import Level1FeatureExtractor
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.level1 = Level1FeatureExtractor(model_type="whisper")
                self.level1.eval()
                print(f"[OK] [EmotionEngine] Level 1 (Whisper) loaded on {self.device}")
            except Exception as e:
                print(f"[WARN] [EmotionEngine] Level 1 load failed: {e}")

        # -- Level 2: MLP Classifier --
        if HAS_TORCH and self.level1 is not None:
            l2_path = os.path.join(model_dir, "level2_mlp.pth")
            if os.path.exists(l2_path):
                try:
                    from level2_classification import Level2Classifier
                    self.level2 = Level2Classifier(input_dim=512)
                    self.level2.load_state_dict(
                        torch.load(l2_path, map_location=self.device, weights_only=True)
                    )
                    self.level2.to(self.device).eval()
                    print(f"[OK] [EmotionEngine] Level 2 (MLP) loaded from {l2_path}")
                except Exception as e:
                    print(f"[WARN] [EmotionEngine] Level 2 load failed: {e}")

        # -- Level 3: Acoustic Physics --
        if HAS_PARSELMOUTH:
            try:
                from level3_acoustic_physics import AcousticPhysicsExtractor
                self.level3 = AcousticPhysicsExtractor()
                print("[OK] [EmotionEngine] Level 3 (Praat Physics) loaded")
            except Exception as e:
                print(f"[WARN] [EmotionEngine] Level 3 load failed: {e}")

        # -- Level 4: Temporal Bi-LSTM --
        if HAS_TORCH:
            l4_path = os.path.join(model_dir, "level4_bilstm.pth")
            if os.path.exists(l4_path):
                try:
                    from level4_temporal_gate import TemporalContextLSTM, final_decision_gate
                    self.level4 = TemporalContextLSTM()
                    self.level4.load_state_dict(
                        torch.load(l4_path, map_location=self.device, weights_only=True)
                    )
                    self.level4.to(self.device).eval()
                    self.final_decision_gate_fn = final_decision_gate
                    print(f"[OK] [EmotionEngine] Level 4 (Bi-LSTM) loaded from {l4_path}")
                except Exception as e:
                    print(f"[WARN] [EmotionEngine] Level 4 load failed: {e}")

        self.is_ready = (self.level1 is not None and self.level2 is not None)
        if self.is_ready:
            print("[OK] [EmotionEngine] 4-Level pipeline READY")
        else:
            print("[WARN] [EmotionEngine] Falling back to heuristic mode")

    # -- Main analysis entrypoint --------------------------------------

    def analyze_chunk(
        self,
        samples: np.ndarray,
        sr: int = 16000,
        session_key: Optional[str] = None,
    ) -> AudioFeatures:
        if samples.size == 0:
            return _EMPTY

        wave = np.asarray(samples, dtype=np.float32).flatten()
        rms = float(np.sqrt(np.mean(np.square(wave))))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(wave)))))

        # Heuristic baseline (always available as fallback)
        pitch_h = _clamp((rms - 0.008) / 0.06)
        cadence_h = _clamp((zcr - 0.03) / 0.25)
        stress_h = _clamp(pitch_h * 0.62 + cadence_h * 0.38)

        if not self.is_ready:
            return AudioFeatures(
                rms=rms, zero_crossing_rate=zcr,
                pitch_score=round(pitch_h, 4), cadence_score=round(cadence_h, 4),
                stress_score=round(stress_h, 4),
                emotion=_emotion_from_stress(stress_h),
                arousal_score=round(1.0 + stress_h * 9.0, 2),
            )

        # ---- Level 3: Acoustic Physics (Stage 1) ----
        jitter, shimmer, hnr, f0_mean = 0.0, 0.0, 0.0, 0.0
        physiology_stressed = False
        if self.level3 is not None and wave.size >= sr * 0.5:
            try:
                biomarkers = self.level3.extract_biomarkers(wave, sr)
                jitter = biomarkers['jitter']
                shimmer = biomarkers['shimmer']
                hnr = biomarkers['hnr']
                f0_mean = biomarkers['f0_mean']
                physiology_stressed = biomarkers['physiologically_stressed']
            except Exception:
                pass

        # ---- Level 1 + Level 2: Neural Classification (Stage 2) ----
        emotion = "calm"
        arousal = round(1.0 + stress_h * 9.0, 2)
        confidence = 0.0
        source = "heuristic"

        if wave.size >= sr * 0.5:
            try:
                import torch
                # Pad/truncate to 5 seconds for stable Whisper dimensions
                target_len = int(sr * 5.0)
                if len(wave) < target_len:
                    audio_input = np.pad(wave, (0, target_len - len(wave)))
                else:
                    audio_input = wave[:target_len]

                with torch.no_grad():
                    embeddings = self.level1(audio_input, sr)  # [1, 512]
                    logits, arousal_pred = self.level2(embeddings)

                    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    pred_idx = int(probs.argmax())
                    emotion = self.LABEL_MAP[pred_idx]
                    confidence = float(probs.max())
                    arousal = float(arousal_pred.squeeze().cpu().item())
                    arousal = max(1.0, min(10.0, arousal))
                    source = "level2_mlp"
            except Exception as e:
                print(f"[WARN] Neural inference failed: {e}")

        # ---- Physics-Informed Confidence Boosting (NEW) ----
        # When Level 3 (physics) corroborates Level 2 (neural), boost confidence.
        # When they disagree, reduce confidence. This prevents the model from
        # confidently predicting "calm" when vocal cords show stress tremors.
        if source == "level2_mlp" and self.level3 is not None:
            neural_says_stressed = (emotion != "calm")
            physics_says_stressed = physiology_stressed

            if neural_says_stressed and physics_says_stressed:
                # Both agree on stress → boost confidence
                confidence = min(1.0, confidence * 1.15)
            elif not neural_says_stressed and physics_says_stressed:
                # Physics detects stress but neural doesn't → reduce confidence
                # This forces the soft fallback below to blend in heuristic stress
                confidence *= 0.70
            elif neural_says_stressed and not physics_says_stressed:
                # Neural detects stress but physics doesn't → mild reduction
                confidence *= 0.85

        # ---- Level 4: Temporal Context (Stage 3) ----
        escalation_prob = 0.0
        temporal_buffer = self._get_temporal_buffer(session_key)
        temporal_buffer.append(arousal)

        if self.level4 is not None and len(temporal_buffer) == 6:
            try:
                import torch
                seq = torch.tensor(
                    list(temporal_buffer), dtype=torch.float32
                ).unsqueeze(0).unsqueeze(-1).to(self.device)  # [1, 6, 1]

                with torch.no_grad():
                    escalation_prob = float(self.level4(seq).squeeze().cpu().item())
                source = "full_pipeline"
            except Exception:
                pass

        # ---- Final Decision Gate ----
        clinical_alert = False
        if self.final_decision_gate_fn is not None:
            try:
                gate = self.final_decision_gate_fn(
                    stage1_physiology=physiology_stressed,
                    stage2_arousal_score=arousal,
                    stage3_escalation_prob=escalation_prob,
                )
                clinical_alert = bool(gate.get("alert_triggered", False))
            except Exception:
                clinical_alert = False

        # ---- Soft Ensemble Fallback (replaces hard heuristic override) ----
        # Instead of completely overriding the neural prediction when confidence
        # is low, blend the neural and heuristic predictions proportionally.
        # This preserves partial signal from the neural model even at low confidence.
        if confidence < 0.35:
            # Very low confidence: use heuristic entirely
            emotion = _emotion_from_stress(stress_h)
            arousal = round(1.0 + stress_h * 9.0, 2)
            source += " (low_conf_fallback)"
        elif confidence < 0.55:
            # Medium confidence: blend neural arousal with heuristic arousal
            heuristic_arousal = 1.0 + stress_h * 9.0
            blend_weight = (confidence - 0.35) / 0.20  # 0.0 at conf=0.35, 1.0 at conf=0.55
            arousal = round(blend_weight * arousal + (1 - blend_weight) * heuristic_arousal, 2)
            arousal = max(1.0, min(10.0, arousal))
            # Use neural emotion if its confidence is reasonable
            source += " (blended)"

        stress_score = _clamp((arousal - 1.0) / 9.0)

        return AudioFeatures(
            rms=rms,
            zero_crossing_rate=zcr,
            pitch_score=round(pitch_h, 4),
            cadence_score=round(cadence_h, 4),
            stress_score=round(stress_score, 4),
            emotion=emotion,
            arousal_score=round(arousal, 2),
            confidence=round(confidence, 3),
            jitter=jitter,
            shimmer=shimmer,
            hnr=hnr,
            f0_mean=f0_mean,
            escalation_prob=round(escalation_prob, 4),
            clinical_alert=clinical_alert,
            model_source=source,
        )

    def get_engine_status(self) -> dict:
        return {
            "heuristic_fallback": not self.is_ready,
            "level1_loaded": self.level1 is not None,
            "level2_loaded": self.level2 is not None,
            "level3_loaded": self.level3 is not None,
            "level4_loaded": self.level4 is not None,
            "pipeline_ready": self.is_ready,
            "temporal_sessions_active": len(self.arousal_buffers),
        }


# -- Helpers -----------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _emotion_from_stress(s: float) -> str:
    if s >= 0.72:
        return "high_anxiety"
    if s >= 0.42:
        return "mild_anxiety"
    return "calm"


# -- Lazy singleton + backward-compatible API --------------------------

_engine: EmotionEngine | None = None


def _get_engine() -> EmotionEngine:
    global _engine
    if _engine is None:
        _engine = EmotionEngine()
    return _engine


def get_audio_engine_status() -> dict:
    """Check the load status of ML models in the engine singleton."""
    return _get_engine().get_engine_status()


def analyze_audio_chunk(samples: np.ndarray, session_key: Optional[str] = None) -> AudioFeatures:
    """Drop-in replacement -- voice_service.py calls this unchanged."""
    return _get_engine().analyze_chunk(samples, session_key=session_key)
