# app/api/routes_triage.py
import os
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from app.core.database import db_config
from app.services.llm_langchain import generate_clinical_summary
from app.services.pdf_generator import create_clinical_pdf
from app.services.notification import send_whatsapp_alert, send_email_with_pdf

router = APIRouter()

# Schema for incoming request
class TriageTrigger(BaseModel):
    user_id: str

async def process_triage_background(user_id: str):
    """
    Pulls live data from MongoDB, generates AI summary and PDF, 
    and dispatches alerts to parents/guardians.
    """
    try:
        if db_config.db is None:
            print("[Triage] Database unavailable; skipping triage pipeline.")
            return

        # 1. Fetch the user's active session and profile from DB
        session = await db_config.db["active_sessions"].find_one({"user_id": user_id})
        user_profile = await db_config.db["users"].find_one({"user_id": user_id})
        
        # Guard clauses: Ensure data exists before proceeding
        if not session:
            print(f"Triage Error: No active session found for user {user_id}.")
            return
        
        if not user_profile:
            print(f"Triage Error: No user profile found for user {user_id}. Cannot send alerts.")
            return

        # Clean up MongoDB's internal '_id' object before passing to LangChain
        session.pop('_id', None)
        
        # 2. Generate AI Summary using Groq
        ai_summary = generate_clinical_summary(str(session))
        
        # 3. Generate PDF
        user_name = user_profile.get("name", "AuraOS User")
        pdf_path = create_clinical_pdf(user_name, session, ai_summary)
        
        # 4. Dispatch Alerts
        guardian_number = user_profile.get("guardian_number")
        guardian_email = user_profile.get("guardian_email")
        
        wa_message = (f"🚨 AuraOS Alert: {user_name} just utilized the anxiety toolkit.\n\n"
                      f"AI Summary: {ai_summary}\n\n"
                      f"Please check your email for the full clinical report.")
        
        if guardian_number:
            send_whatsapp_alert(guardian_number, wa_message)
            
        if guardian_email:
            email_subject = f"AuraOS Clinical Triage Report for {user_name}"
            email_body = f"Please find attached the clinical session report for {user_name}.\n\nSummary:\n{ai_summary}"
            send_email_with_pdf(guardian_email, email_subject, email_body, pdf_path)
        
        # 5. Clean up the temporary PDF to save server space
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        # 6. Clear the active session so they start fresh next time
        await db_config.db["active_sessions"].delete_one({"user_id": user_id})
        print(f"✅ Triage completed successfully for {user_name}.")

    except Exception as e:
        print(f"❌ Background Triage Error: {str(e)}")


@router.post("/trigger-triage")
async def api_trigger_triage(data: TriageTrigger, background_tasks: BackgroundTasks):
    """
    Triggers the emergency protocol. 
    Returns instantly to the frontend while processing heavily in the background.
    """
    try:
        if db_config.db is None:
            raise HTTPException(
                status_code=503,
                detail="Triage database is not connected. Check MongoDB configuration.",
            )

        # Add the triage function to FastAPI's background task queue
        background_tasks.add_task(process_triage_background, data.user_id)
        
        # Immediately respond so the frontend UI doesn't lag
        return {"success": True, "message": "Triage protocol initiated in the background."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
