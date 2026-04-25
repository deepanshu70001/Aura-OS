# app/services/notification.py
import smtplib
from pathlib import Path
from email.message import EmailMessage

try:
    from twilio.rest import Client
except ImportError:
    Client = None

from app.core.config import settings

def send_whatsapp_alert(guardian_number: str, message_body: str):
    """Sends a WhatsApp alert to the guardian."""
    if not guardian_number:
        return

    if Client is None:
        print("[Twilio] twilio package not installed; skipping WhatsApp dispatch.")
        return

    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        print("Twilio credentials not configured; skipping WhatsApp dispatch.")
        return

    try:
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            from_=settings.TWILIO_WHATSAPP_NUMBER,
            body=message_body,
            to=f"whatsapp:{guardian_number}"
        )
        print(f"WhatsApp sent successfully: {message.sid}")
    except Exception as e:
        print(f"Failed to send WhatsApp: {e}")

def send_email_with_pdf(guardian_email: str, subject: str, body: str, pdf_path: str):
    """Sends an email with the PDF report attached."""
    if not guardian_email:
        return

    if not settings.SMTP_SERVER or not settings.SMTP_USERNAME or not settings.SMTP_PASSWORD:
        print("SMTP credentials not configured; skipping email dispatch.")
        return

    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = settings.SMTP_USERNAME
        msg['To'] = guardian_email
        msg.set_content(body)

        # Attach PDF
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
            msg.add_attachment(
                pdf_data,
                maintype='application',
                subtype='pdf',
                filename=Path(pdf_path).name,
            )

        # Send Email
        with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.send_message(msg)
            
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send Email: {e}")
