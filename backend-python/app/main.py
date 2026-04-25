# app/main.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api.routes_audio import router as audio_router
from app.api.routes_tasks import router as tasks_router
from app.api.routes_triage import router as triage_router
from app.api.routes_rag import router as rag_router
from app.api.routes_behavioral import router as behavioral_router
from app.core.database import connect_to_mongo, close_mongo_connection
from app.core.config import settings
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    yield
    await close_mongo_connection()


app = FastAPI(
    title="AuraOS API",
    description="Backend for AuraOS Mental Health Engine",
    lifespan=lifespan,
)

cors_origins = list(dict.fromkeys([*settings.FRONTEND_ORIGINS, settings.FRONTEND_URL]))
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audio_router, tags=["Audio"])
app.include_router(tasks_router, prefix="/api/v1/ai", tags=["LangChain Logic"])
app.include_router(triage_router, prefix="/api/v1/triage", tags=["Triage System"])
app.include_router(rag_router, prefix="/api/v1/rag", tags=["Clinical RAG"])
app.include_router(behavioral_router, prefix="/api/v1/behavioral", tags=["Behavioral Health"])


from app.services.audio_engine import get_audio_engine_status

@app.get("/")
async def root():
    return {"message": "AuraOS Engine is online and ready."}


@app.get("/health")
async def health():
    ml_status = get_audio_engine_status()
    # Let's also include whether it's operating purely on heuristic fallback
    status_msg = "online" if not ml_status["heuristic_fallback"] else "online (heuristic fallback mode)"
    
    return {
        "ok": True, 
        "service": "aura-python-backend",
        "status": status_msg,
        "ml_engine": ml_status
    }
