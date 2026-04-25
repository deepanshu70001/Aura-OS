# app/services/rag_service.py
# Clinical RAG (Retrieval-Augmented Generation) Knowledge Base service.
#
# Uses ChromaDB as a local vector store (no API key / no external service).
# Embeds curated clinical knowledge from Harvard Nutritional Psychiatry
# and Dr. Ratey's exercise neurochemistry research.
#
# Flow:
#   1. On startup → ingest_knowledge_base() loads .txt files, chunks them,
#      and stores embeddings in a local ChromaDB collection.
#   2. On request → query_clinical_rag() retrieves top-k relevant passages.
#   3. generate_recovery_protocol() combines RAG context + user profile
#      and sends to Groq LLM for a structured, grounded response.

from __future__ import annotations

import os
import glob
from pathlib import Path
from functools import lru_cache

from app.core.config import settings

# ── Graceful imports (system works without RAG deps installed) ───────────
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None
    ChromaSettings = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

try:
    from langchain.prompts import PromptTemplate
    from langchain_groq import ChatGroq
except ImportError:
    PromptTemplate = None
    ChatGroq = None

from pydantic import BaseModel, Field


# ── Paths ───────────────────────────────────────────────────────────────
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "clinical_knowledge"
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent / ".chroma_db"

# ── Collection name ─────────────────────────────────────────────────────
COLLECTION_NAME = "auraos_clinical_kb"

# ── Singleton ChromaDB client ──────────────────────────────────────────
_chroma_client = None
_collection = None


def _get_chroma():
    """Lazy-init ChromaDB client and collection."""
    global _chroma_client, _collection
    if _chroma_client is not None:
        return _collection

    if chromadb is None:
        print("[RAG] chromadb is not installed; RAG features are disabled.")
        return None

    try:
        _chroma_client = chromadb.Client(ChromaSettings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=str(CHROMA_PERSIST_DIR),
        ))
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[RAG] ChromaDB ready. Collection '{COLLECTION_NAME}' "
              f"has {_collection.count()} documents.")
        return _collection
    except Exception as exc:
        print(f"[RAG] ChromaDB init failed: {exc}")
        _chroma_client = None
        _collection = None
        return None


# ═══════════════════════════════════════════════════════════════════════
#  INGESTION — Load clinical knowledge into vector store
# ═══════════════════════════════════════════════════════════════════════

def ingest_knowledge_base(force: bool = False) -> dict:
    """
    Reads all .txt files from clinical_knowledge/, chunks them using
    RecursiveCharacterTextSplitter, and upserts into ChromaDB.

    Only re-ingests if the collection is empty (or force=True).
    Returns {"status": ..., "documents": N, "chunks": N}.
    """
    collection = _get_chroma()
    if collection is None:
        return {"status": "disabled", "reason": "chromadb not available"}

    # Skip if already populated (unless forced)
    if collection.count() > 0 and not force:
        return {
            "status": "already_populated",
            "documents": collection.count(),
        }

    if RecursiveCharacterTextSplitter is None:
        return {"status": "disabled", "reason": "langchain-text-splitters not installed"}

    # Find all .txt files in knowledge directory
    txt_files = sorted(glob.glob(str(KNOWLEDGE_DIR / "*.txt")))
    if not txt_files:
        return {"status": "no_files", "path": str(KNOWLEDGE_DIR)}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n---\n", "\n\n", "\n", ". ", " "],
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for filepath in txt_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}::chunk_{i:04d}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source_file": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

    # Upsert into ChromaDB (uses default embedding function: all-MiniLM-L6-v2)
    # Batch in groups of 100 to avoid memory issues
    batch_size = 100
    for start in range(0, len(all_chunks), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=all_ids[start:end],
            documents=all_chunks[start:end],
            metadatas=all_metadatas[start:end],
        )

    print(f"[RAG] Ingested {len(all_chunks)} chunks from {len(txt_files)} files.")
    return {
        "status": "ingested",
        "files": len(txt_files),
        "chunks": len(all_chunks),
    }


# ═══════════════════════════════════════════════════════════════════════
#  RETRIEVAL — Query the vector store for relevant clinical passages
# ═══════════════════════════════════════════════════════════════════════

def query_clinical_rag(query: str, n_results: int = 4) -> list[dict]:
    """
    Searches the clinical knowledge base for passages relevant to the query.
    Returns a list of {text, source, score} dicts.
    """
    collection = _get_chroma()
    if collection is None or collection.count() == 0:
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
        )
    except Exception as exc:
        print(f"[RAG] Query failed: {exc}")
        return []

    passages = []
    if results and results.get("documents"):
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        for doc, meta, dist in zip(docs, metas, dists):
            passages.append({
                "text": doc,
                "source": meta.get("source_file", "unknown"),
                "relevance_score": round(1.0 - dist, 3),  # cosine: lower distance = more relevant
            })

    return passages


# ═══════════════════════════════════════════════════════════════════════
#  GENERATION — Combine RAG context + user profile → LLM → protocol
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _get_rag_llm():
    """Returns a Groq LLM instance configured for RAG generation."""
    if ChatGroq is None or not settings.GROQ_API_KEY:
        return None
    try:
        return ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name="llama3-8b-8192",
            temperature=0.4,  # Lower temp for clinical accuracy
            max_tokens=1200,
        )
    except Exception:
        return None


class RecoveryProtocolModel(BaseModel):
    diagnosis_baseline: str = Field(..., description="Baseline diagnosis summary (e.g. 'High Cortisol / Acute Anxiety')")
    neuro_diet_plan: list[str] = Field(..., description="List of specific, strict neurochemical diet interventions")
    somatic_exercise_plan: str = Field(..., description="Specific, somatic exercise plan to lower or raise appropriate biomarkers")
    confidence_anchor: str = Field(..., description="A brief reminder to anchor the user's confidence based on any small wins")
    disclaimer: str = Field(
        default="AuraOS provides neuro-supportive lifestyle suggestions, not medical prescriptions. Consult a doctor for severe symptoms.",
        description="Standard medical liability disclaimer"
    )

_RAG_PROMPT_TEMPLATE = """You are a clinical recovery protocol generator for AuraOS.
Synthesize evidence-based recommendations from real medical research provided below.

USER PROFILE:
- Condition: {condition}
- Severity: {severity}
- Baseline Arousal Score: {arousal_score}/10

RETRIEVED CLINICAL EVIDENCE:
{rag_context}

RULES YOU MUST FOLLOW STRICTLY:
1. If Condition implies ADHD/Executive Freeze: Prescribe Dopamine-boosting protocols (High protein breakfast, Tyrosine-heavy snacks, HIIT/complex motor exercises).
2. If Condition implies High Arousal/Severe Anxiety: Prescribe GABA-boosting and Cortisol-lowering protocols (Magnesium-rich foods, Omega-3s, strict zero-caffeine, Zone 2 steady-state cardio or Yoga).
3. If Condition implies Depression/Low Confidence: Prescribe Serotonin-boosting protocols (Tryptophan-rich foods, gut-health probiotics, outdoor sunlight exposure).

Generate a personalized recovery protocol applying the clinical evidence subject strictly to the aforementioned neuro-chemical rules."""

def _fallback_protocol(condition: str, severity: str) -> dict:
    """Returns a minimal protocol when RAG or LLM is unavailable."""
    return {
        "diagnosis_baseline": f"{severity.capitalize()} {condition.capitalize()}",
        "neuro_diet_plan": [
            "Magnesium heavy dinner (spinach/seeds)",
            "Zero caffeine after 12 PM",
            "High protein breakfast"
        ],
        "somatic_exercise_plan": "20-minute Zone 2 cardio (brisk walk) to lower resting heart rate.",
        "confidence_anchor": "Remind yourself you successfully initiated steps toward mental clarity today.",
        "disclaimer": "AuraOS provides neuro-supportive lifestyle suggestions, not medical prescriptions. Consult a doctor for severe symptoms.",
        "_fallback": True,
    }


def generate_recovery_protocol(
    condition: str,
    severity: str = "moderate",
    arousal_score: int = 5,
) -> dict:
    """
    Full RAG pipeline:
    1. Query vector store with condition-specific prompt
    2. Format retrieved passages as context
    3. Send to Groq LLM with structured prompt
    4. Parse and return JSON protocol

    Falls back to a static protocol if RAG or LLM is unavailable.
    """
    # Step 1: Retrieve relevant clinical passages
    query = f"{condition} diet exercise protocol neurochemistry treatment"
    passages = query_clinical_rag(query, n_results=6)

    if not passages:
        print("[RAG] No passages retrieved; using fallback protocol.")
        return _fallback_protocol(condition, severity)

    # Step 2: Format context from retrieved passages
    rag_context = "\n\n".join(
        f"[Source: {p['source']}] (relevance: {p['relevance_score']})\n{p['text']}"
        for p in passages
    )

    # Step 3: Generate with LLM
    llm = _get_rag_llm()
    if llm is None or PromptTemplate is None:
        print("[RAG] LLM unavailable; using fallback protocol.")
        result = _fallback_protocol(condition, severity)
        # Still attach the RAG passages so frontend can display sources
        result["rag_passages"] = passages
        return result

    prompt = PromptTemplate(
        template=_RAG_PROMPT_TEMPLATE,
        input_variables=["condition", "severity", "arousal_score", "rag_context"],
    )

    try:
        if hasattr(llm, 'with_structured_output') and BaseModel is not None:
            llm_with_structure = llm.with_structured_output(RecoveryProtocolModel)
            chain = prompt | llm_with_structure
            structured_response = chain.invoke({
                "condition": condition,
                "severity": severity,
                "arousal_score": str(arousal_score),
                "rag_context": rag_context,
            })
            # Convert BaseModel to dict
            protocol = structured_response.model_dump()
        else:
            print("[RAG] with_structured_output unavailable. Using fallback.")
            protocol = _fallback_protocol(condition, severity)
        
        protocol["rag_passages"] = passages
        protocol["_source"] = "rag_llm"
        return protocol

    except Exception as exc:
        print(f"[RAG] LLM generation failed: {exc}")
        result = _fallback_protocol(condition, severity)
        result["rag_passages"] = passages
        return result
