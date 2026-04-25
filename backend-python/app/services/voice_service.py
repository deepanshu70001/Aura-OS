import asyncio
import io
import json
import logging
import time
import uuid
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
from fastapi import WebSocketDisconnect

logger = logging.getLogger(__name__)

from app.core.config import settings
from app.core.database import db_config
from app.services.audio_engine import analyze_audio_chunk
from app.services.node_bridge import post_vocal_stress_event

SILENCE_CHUNKS_TO_CLOSE_UTTERANCE = 6
EMOTION_UPDATE_INTERVAL_SECONDS = 0.75
MAX_BUFFER_SECONDS = 12
MAX_STORED_VOICE_INSIGHTS = 12

EMOTION_RESPONSES = {
    "calm": "I hear you. Keep speaking at your pace, one thought at a time.",
    "mild_anxiety": "You are carrying a lot. Inhale for four counts, then exhale slowly.",
    "high_anxiety": "You're safe right now. Feel your feet on the ground and name three things you can see.",
}

EMOTION_TRANSCRIPTS = {
    "calm": "I am talking through what I feel.",
    "mild_anxiety": "I feel anxious and need a small reset.",
    "high_anxiety": "I feel overwhelmed right now.",
}


@dataclass
class VoiceSession:
    user_id: str | None = None
    task_context: str | None = None
    sample_rate: int = settings.VOICE_SAMPLE_RATE
    vad_threshold: float = settings.VOICE_VAD_THRESHOLD
    speech_chunks: list[np.ndarray] = field(default_factory=list)
    is_speaking: bool = False
    silence_chunks: int = 0
    last_emotion: str = "calm"
    last_emotion_emit: float = 0.0
    pending_vocal_stress_event: dict[str, Any] | None = None
    session_key: str = field(default_factory=lambda: str(uuid.uuid4()))

    async def process_chunk(self, payload: bytes) -> list[dict[str, Any]]:
        if not payload:
            return []

        try:
            samples = np.frombuffer(payload, dtype=np.float32)
        except ValueError:
            return []

        if samples.size == 0:
            return []

        features = analyze_audio_chunk(samples, session_key=self.session_key)
        messages: list[dict[str, Any]] = []
        now = time.monotonic()

        should_emit_emotion = (
            features.emotion != self.last_emotion
            or (now - self.last_emotion_emit) >= EMOTION_UPDATE_INTERVAL_SECONDS
        )
        if should_emit_emotion:
            messages.append(
                {
                    "type": "emotion_update",
                    "emotion": features.emotion,
                    "pitch_score": features.pitch_score,
                    "cadence_score": features.cadence_score,
                    "arousal_score": getattr(features, "arousal_score", 5.0),
                    "confidence": getattr(features, "confidence", 0.0),
                    "jitter": getattr(features, "jitter", 0.0),
                    "shimmer": getattr(features, "shimmer", 0.0),
                    "hnr": getattr(features, "hnr", 0.0),
                    "f0_mean": getattr(features, "f0_mean", 0.0),
                    "escalation_prob": getattr(features, "escalation_prob", 0.0),
                    "clinical_alert": getattr(features, "clinical_alert", False),
                }
            )
            self.last_emotion = features.emotion
            self.last_emotion_emit = now

        if features.rms >= self.vad_threshold:
            self.is_speaking = True
            self.silence_chunks = 0
            self.speech_chunks.append(np.array(samples, copy=True))
            self._trim_buffer()
            return messages

        if not self.is_speaking:
            return messages

        self.silence_chunks += 1
        if self.silence_chunks >= SILENCE_CHUNKS_TO_CLOSE_UTTERANCE:
            if self.speech_chunks:
                merged = np.concatenate(self.speech_chunks)
                messages.append(
                    {
                        "type": "_async_utterance",
                        "_merged_audio": merged,
                        "_vocal_emotion": self.last_emotion,
                    }
                )
            self.is_speaking = False
            self.silence_chunks = 0
            self.speech_chunks.clear()

        return messages

    def flush_pending_event(self) -> None:
        if not self.pending_vocal_stress_event:
            return

        payload = self.pending_vocal_stress_event
        self.pending_vocal_stress_event = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(post_vocal_stress_event(**payload))

    def _trim_buffer(self) -> None:
        max_samples = self.sample_rate * MAX_BUFFER_SECONDS
        total_samples = int(sum(chunk.size for chunk in self.speech_chunks))
        if total_samples <= max_samples:
            return

        merged = np.concatenate(self.speech_chunks)
        keep_tail = merged[-max_samples:]
        self.speech_chunks = [keep_tail]

    async def finish_utterance_async(self, merged: np.ndarray, websocket: Any) -> None:
        features = analyze_audio_chunk(merged, session_key=self.session_key)
        arousal_score = getattr(features, "arousal_score", 5.0)
        utterance_seconds = round(float(merged.size / self.sample_rate), 2)
        clinical_alert = getattr(features, "clinical_alert", False)

        transcript = await self._transcribe_audio(merged)
        if not transcript:
            transcript = self._build_transcript(features.emotion, utterance_seconds)
        else:
            transcript = f"{transcript} ({utterance_seconds}s)"

        semantic = await self._analyze_semantics(transcript, features.emotion)
        response = await self._generate_ai_response(transcript, features.emotion, semantic)

        self.pending_vocal_stress_event = {
            "user_id": self.user_id,
            "emotion": features.emotion,
            "arousal_score": arousal_score,
            "task_context": self.task_context,
            "clinical_alert": clinical_alert,
            "transcript": transcript,
            "semantic_summary": semantic["summary"],
            "semantic_intent": semantic["intent"],
            "risk_level": semantic["risk_level"],
            "companion_message": response,
        }
        self.flush_pending_event()

        await self._persist_voice_insight(
            transcript=transcript,
            emotion=features.emotion,
            arousal_score=arousal_score,
            semantic=semantic,
            companion_message=response,
            utterance_seconds=utterance_seconds,
            clinical_alert=clinical_alert,
        )

        # Guard all WebSocket sends — connection may have closed while we
        # were doing transcription/LLM work in the background.
        try:
            await websocket.send_json({"type": "transcript", "text": transcript})
            await websocket.send_json(
                {
                    "type": "semantic_analysis",
                    "summary": semantic["summary"],
                    "intent": semantic["intent"],
                    "sentiment": semantic["sentiment"],
                    "risk_level": semantic["risk_level"],
                    "actionable_signal": semantic["actionable_signal"],
                    "emotion": features.emotion,
                    "arousal_score": arousal_score,
                }
            )
            await websocket.send_json(
                {
                    "type": "response",
                    "text": response,
                    "emotion": features.emotion,
                    "arousal_score": arousal_score,
                    "confidence": getattr(features, "confidence", 0.0),
                    "clinical_alert": clinical_alert,
                    "semantic_summary": semantic["summary"],
                    "semantic_risk_level": semantic["risk_level"],
                }
            )
            await websocket.send_json(
                {
                    "type": "emotion_update",
                    "emotion": features.emotion,
                    "pitch_score": features.pitch_score,
                    "cadence_score": features.cadence_score,
                    "arousal_score": arousal_score,
                    "confidence": getattr(features, "confidence", 0.0),
                    "jitter": getattr(features, "jitter", 0.0),
                    "shimmer": getattr(features, "shimmer", 0.0),
                    "hnr": getattr(features, "hnr", 0.0),
                    "f0_mean": getattr(features, "f0_mean", 0.0),
                    "escalation_prob": getattr(features, "escalation_prob", 0.0),
                    "clinical_alert": clinical_alert,
                }
            )
        except (WebSocketDisconnect, RuntimeError):
            logger.debug("[VoiceSession] WebSocket closed before utterance results could be sent")
        except Exception as exc:
            logger.debug(f"[VoiceSession] send error in finish_utterance_async: {exc!r}")

    async def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        if not settings.GROQ_API_KEY:
            return ""

        try:
            audio_int16 = (audio_data * 32767).astype(np.int16)

            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())

            buffer.seek(0)

            timeout = httpx.Timeout(connect=10.0, read=25.0, write=25.0, pool=25.0)
            max_retries = 2
            last_err = None
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.post(
                            "https://api.groq.com/openai/v1/audio/transcriptions",
                            headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                            files={"file": ("speech.wav", buffer, "audio/wav")},
                            data={
                                "model": "whisper-large-v3",
                                "response_format": "json",
                                "language": "hi"
                            },
                        )
                        if response.status_code == 200:
                            return response.json().get("text", "")

                        logger.warning(
                            f"[GroqWhisper] API error {response.status_code}: "
                            f"{response.text[:240]}"
                        )
                        break  # Don't retry on API errors (4xx/5xx)
                except httpx.ConnectTimeout:
                    last_err = "ConnectTimeout"
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5)  # Brief pause before retry
                        buffer.seek(0)  # Reset buffer for retry
                        continue
                except httpx.TimeoutException as e:
                    last_err = repr(e)
                    break

            if last_err:
                logger.warning(f"[GroqWhisper] Transcription failed after {max_retries} attempts: {last_err}")
        except Exception as e:
            logger.warning(f"[GroqWhisper] Error during transcription: {e!r}")
        return ""

    async def _analyze_semantics(self, transcript: str, emotion: str) -> dict[str, str]:
        fallback = self._semantic_fallback(transcript, emotion)
        if not transcript.strip():
            return fallback

        from app.services.llm_langchain import _get_llm

        llm = _get_llm()
        if not llm:
            return fallback

        prompt = (
            "You are a semantic analyzer for short voice transcripts in an ADHD support app. "
            "Return STRICT JSON with keys: summary, intent, sentiment, risk_level, actionable_signal. "
            "Use concise language. summary <= 20 words. risk_level must be low, medium, or high.\n\n"
            f"Transcript: {transcript}\n"
            f"Detected vocal emotion: {emotion}"
        )
        try:
            res = await llm.ainvoke(prompt)
            content = getattr(res, "content", res)
            if isinstance(content, list):
                normalized_chunks: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        normalized_chunks.append(str(item.get("text", "")))
                    else:
                        normalized_chunks.append(str(item))
                content = "\n".join(chunk for chunk in normalized_chunks if chunk).strip()
            else:
                content = str(content).strip()
            parsed = self._parse_json_object(content)
            if not isinstance(parsed, dict):
                return fallback
            return self._normalize_semantic(parsed, fallback)
        except Exception as e:
            print(f"[GroqSemantic] Error: {e}")
            return fallback

    async def _generate_ai_response(
        self,
        transcript: str,
        emotion: str,
        semantic: dict[str, str],
    ) -> str:
        from app.services.llm_langchain import _get_llm

        llm = _get_llm()
        if not llm or not transcript:
            return EMOTION_RESPONSES.get(emotion, EMOTION_RESPONSES["calm"])

        prompt = (
            f"The user just said: '{transcript}'. They sound {emotion.replace('_', ' ')}. "
            f"Semantic summary: {semantic.get('summary', '')}. "
            f"Semantic risk: {semantic.get('risk_level', 'low')}. "
            "Provide a short, empathetic, supportive response (max 20 words) as an ADHD companion. "
            "Do not use generic affirmations; be specific to their tone if possible."
        )
        try:
            res = await llm.ainvoke(prompt)
            return str(res.content).strip()
        except Exception as e:
            print(f"[GroqLLM] Error: {e}")
            return EMOTION_RESPONSES.get(emotion, EMOTION_RESPONSES["calm"])

    async def _persist_voice_insight(
        self,
        *,
        transcript: str,
        emotion: str,
        arousal_score: float,
        semantic: dict[str, str],
        companion_message: str,
        utterance_seconds: float,
        clinical_alert: bool,
    ) -> None:
        if not self.user_id or db_config.db is None:
            return

        now = datetime.now(timezone.utc)
        insight_event = {
            "timestamp": now,
            "transcript": transcript,
            "emotion": emotion,
            "arousal_score": round(float(arousal_score), 2),
            "semantic_summary": semantic["summary"],
            "semantic_intent": semantic["intent"],
            "semantic_sentiment": semantic["sentiment"],
            "semantic_risk_level": semantic["risk_level"],
            "actionable_signal": semantic["actionable_signal"],
            "companion_message": companion_message,
            "utterance_seconds": round(float(utterance_seconds), 2),
            "clinical_alert": bool(clinical_alert),
        }

        try:
            await db_config.db["active_sessions"].update_one(
                {"user_id": self.user_id},
                {
                    "$setOnInsert": {
                        "user_id": self.user_id,
                        "created_at": now,
                    },
                    "$set": {
                        "updated_at": now,
                        "latest_voice_transcript": transcript,
                        "latest_voice_message": companion_message,
                        "latest_voice_semantic_summary": semantic["summary"],
                    },
                    "$push": {
                        "voice_insights": {
                            "$each": [insight_event],
                            "$slice": -MAX_STORED_VOICE_INSIGHTS,
                        }
                    },
                },
                upsert=True,
            )
        except Exception as exc:
            print(f"[VoicePersistence] Unable to store voice insight: {exc!r}")

    @staticmethod
    def _semantic_fallback(transcript: str, emotion: str) -> dict[str, str]:
        normalized_transcript = transcript.strip()
        if len(normalized_transcript) > 180:
            normalized_transcript = f"{normalized_transcript[:177].rstrip()}..."

        sentiment = {
            "high_anxiety": "distressed",
            "mild_anxiety": "uneasy",
            "calm": "stable",
        }.get(emotion, "neutral")

        risk_level = (
            "high"
            if emotion == "high_anxiety"
            else "medium" if emotion == "mild_anxiety" else "low"
        )
        summary = normalized_transcript or EMOTION_TRANSCRIPTS.get(
            emotion, EMOTION_TRANSCRIPTS["calm"]
        )

        return {
            "summary": summary,
            "intent": "emotional_check_in",
            "sentiment": sentiment,
            "risk_level": risk_level,
            "actionable_signal": EMOTION_RESPONSES.get(
                emotion, EMOTION_RESPONSES["calm"]
            ),
        }

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any] | None:
        candidate = content.strip()
        if candidate.startswith("```"):
            candidate = candidate.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return None

        try:
            parsed = json.loads(candidate[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None

        return None

    @staticmethod
    def _normalize_semantic(
        raw: dict[str, Any], fallback: dict[str, str]
    ) -> dict[str, str]:
        risk = str(raw.get("risk_level") or fallback["risk_level"]).strip().lower()
        if risk not in {"low", "medium", "high"}:
            risk = fallback["risk_level"]

        def normalize_field(name: str) -> str:
            value = str(raw.get(name) or fallback[name]).strip()
            return value[:240] if value else fallback[name]

        return {
            "summary": normalize_field("summary"),
            "intent": normalize_field("intent"),
            "sentiment": normalize_field("sentiment"),
            "risk_level": risk,
            "actionable_signal": normalize_field("actionable_signal"),
        }

    @staticmethod
    def _build_transcript(emotion: str, utterance_seconds: float) -> str:
        template = EMOTION_TRANSCRIPTS.get(emotion, EMOTION_TRANSCRIPTS["calm"])
        return f"{template} (voice segment: {utterance_seconds}s)"
