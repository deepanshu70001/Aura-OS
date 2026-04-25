# app/services/pdf_generator.py
from html import escape
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (value or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or "AuraUser"


def _safe_text(value: object, default: str = "N/A") -> str:
    text = str(value).strip() if value is not None else ""
    return escape(text or default)


def create_clinical_pdf(user_name: str, session_data: dict, ai_summary: str) -> str:
    """Generates a PDF report and returns the file path."""

    report_dir = Path(__file__).resolve().parent.parent.parent / "temp_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _safe_name(user_name)
    file_path = report_dir / f"AuraOS_Report_{safe_name}_{timestamp}.pdf"

    doc = SimpleDocTemplate(str(file_path), pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []

    # Title
    Story.append(Paragraph(f"AuraOS Clinical Triage Report", styles['Title']))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    Story.append(Paragraph(f"User: {_safe_text(user_name)}", styles['Normal']))
    Story.append(Spacer(1, 12))

    # AI Clinical Summary
    Story.append(Paragraph("AI Clinical Summary:", styles['Heading2']))
    Story.append(Paragraph(_safe_text(ai_summary), styles['Normal']))
    Story.append(Spacer(1, 12))

    # Session Details
    Story.append(Paragraph("Session Details:", styles['Heading2']))
    Story.append(Paragraph(
        f"<b>Initial Trigger/Query:</b> {_safe_text(session_data.get('initial_query'))}",
        styles['Normal'],
    ))
    Story.append(Spacer(1, 12))

    # Shattered Tasks (If any)
    if 'tasks' in session_data and session_data['tasks']:
        Story.append(Paragraph("Shattered Tasks (In User's Order):", styles['Heading3']))
        for idx, task in enumerate(session_data['tasks'], 1):
            title = _safe_text(task.get("title", "Untitled"))
            action = _safe_text(task.get("action", ""))
            Story.append(Paragraph(f"{idx}. {title} - {action}", styles['Normal']))
        Story.append(Spacer(1, 12))

    # Worry Blocks Shattered (If any)
    if 'worries' in session_data and session_data['worries']:
        Story.append(Paragraph("Worry Blocks Shattered:", styles['Heading3']))
        for worry in session_data['worries']:
            Story.append(Paragraph(f"- {_safe_text(worry)}", styles['Normal']))
        Story.append(Spacer(1, 12))

    # Voice semantic insights (if any)
    voice_insights = session_data.get("voice_insights") or []
    if voice_insights:
        Story.append(Paragraph("Voice Semantic Analysis:", styles['Heading3']))
        for idx, event in enumerate(voice_insights[-5:], 1):
            transcript = _safe_text(event.get("transcript", ""))
            emotion = _safe_text(event.get("emotion", "unknown"))
            arousal = _safe_text(event.get("arousal_score", "N/A"))
            summary = _safe_text(event.get("semantic_summary", "N/A"))
            intent = _safe_text(event.get("semantic_intent", "N/A"))
            risk = _safe_text(event.get("semantic_risk_level", "N/A"))
            Story.append(
                Paragraph(
                    f"{idx}. <b>Transcript:</b> {transcript} "
                    f"| <b>Emotion:</b> {emotion} | <b>Arousal:</b> {arousal}/10",
                    styles['Normal'],
                )
            )
            Story.append(
                Paragraph(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<b>Semantic:</b> {summary} "
                    f"| <b>Intent:</b> {intent} | <b>Risk:</b> {risk}",
                    styles['Normal'],
                )
            )
        Story.append(Spacer(1, 12))

    # Build PDF
    doc.build(Story)
    return str(file_path.resolve())
