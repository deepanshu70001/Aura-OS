# app/api/routes_rag.py
# Clinical RAG Knowledge Base API endpoints.
#
# POST /api/v1/rag/recovery-protocol
#   → Generates a personalized recovery protocol (diet + exercise)
#     grounded in clinical research via RAG retrieval.
#
# GET /api/v1/rag/status
#   → Returns the status of the knowledge base (document count, etc.)
#
# POST /api/v1/rag/ingest
#   → Force re-ingestion of the clinical knowledge base.

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_service import (
    generate_recovery_protocol,
    ingest_knowledge_base,
    query_clinical_rag,
    _get_chroma,
)

router = APIRouter()


class ProtocolRequest(BaseModel):
    condition: str  # anxiety | adhd | depression | burnout
    severity: str = "moderate"  # mild | moderate | high
    arousal_score: int = 5  # 1-10
    user_id: str | None = None


class QueryRequest(BaseModel):
    query: str
    n_results: int = 4


@router.post("/recovery-protocol")
async def api_recovery_protocol(request: ProtocolRequest):
    """
    Generates a personalized, evidence-based recovery protocol
    using RAG retrieval from the clinical knowledge base.
    """
    try:
        valid_conditions = {"anxiety", "adhd", "depression", "burnout"}
        condition = request.condition.lower().strip()
        if condition not in valid_conditions:
            raise HTTPException(
                status_code=400,
                detail=f"condition must be one of: {', '.join(valid_conditions)}",
            )

        protocol = generate_recovery_protocol(
            condition=condition,
            severity=request.severity,
            arousal_score=max(1, min(10, request.arousal_score)),
        )

        return {"success": True, "protocol": protocol}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def api_rag_status():
    """Returns the current status of the clinical knowledge base."""
    collection = _get_chroma()
    if collection is None:
        return {
            "success": True,
            "status": "disabled",
            "reason": "chromadb not installed or not initialized",
            "document_count": 0,
        }

    return {
        "success": True,
        "status": "ready",
        "document_count": collection.count(),
        "collection_name": collection.name,
    }


@router.post("/ingest")
async def api_force_ingest():
    """Force re-ingestion of the clinical knowledge base."""
    try:
        result = ingest_knowledge_base(force=True)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def api_query_rag(request: QueryRequest):
    """
    Direct RAG query — useful for debugging/demo.
    Returns raw retrieved passages without LLM processing.
    """
    try:
        passages = query_clinical_rag(request.query, n_results=request.n_results)
        return {"success": True, "passages": passages, "count": len(passages)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
