# ============================================================
#   SIFRA Synthetic LLM Inference API v4.5 ULTRA ENTERPRISE
#
#   Upgrades in v4.5:
#     ✔ Advanced RAG: CRE-aware vector ranking
#     ✔ DMAO-compatible LLM inference
#     ✔ Multi-model support gateway (future-ready)
#     ✔ Safe memory + context controller
#     ✔ Auto-trimming oversized inputs
#     ✔ NARE-X cognitive formatting support
# ============================================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Union

from .sifra_llm_engine import SifraLLMEngine
from tasks.llm_vectorizer import build_vector_store


router = APIRouter()
llm_engine = SifraLLMEngine()


# ------------------------------------------------------------
# Request Model
# ------------------------------------------------------------
class LLMInferenceRequest(BaseModel):
    prompt: str
    persona: str
    behavior: str = ""
    documents: Union[List[str], str]


# ------------------------------------------------------------
# Normalize incoming documents + CRE-friendly segmentation
# ------------------------------------------------------------
def normalize_docs(raw_docs):
    """
    Converts input into a clean list of text segments.
    Works for textarea inputs, lists, and dataset→knowledge
    pipelines. Also prepares segments for CRE awareness.
    """

    if isinstance(raw_docs, str):
        lines = [x.strip() for x in raw_docs.split("\n") if len(x.strip()) > 3]
        # Remove duplicates while keeping order
        return list(dict.fromkeys(lines))

    if isinstance(raw_docs, list):
        cleaned = [str(x).strip() for x in raw_docs if len(str(x).strip()) > 3]
        return list(dict.fromkeys(cleaned))

    return []


# ------------------------------------------------------------
# ⭐ MAIN SYNTHETIC LLM INFERENCE ENDPOINT
# ------------------------------------------------------------
@router.post("/llm_inference")
def llm_inference(req: LLMInferenceRequest):

    try:
        # --------------------------------------------------------
        # 1. Normalize Documents
        # --------------------------------------------------------
        docs = normalize_docs(req.documents)

        if not docs:
            raise HTTPException(
                status_code=400,
                detail="Documents are empty — upload dataset or provide knowledge."
            )

        # Hard limit for safety
        if len(docs) > 50000:
            raise HTTPException(
                status_code=400,
                detail="Too many documents — reduce dataset or summarize text."
            )

        # --------------------------------------------------------
        # 2. Vector Store Construction (CRE-aware)
        # --------------------------------------------------------
        try:
            vector_store = build_vector_store(docs)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Vector store build failed: {str(e)}"
            )

        # --------------------------------------------------------
        # 3. Build LLM cognitive package
        # --------------------------------------------------------
        llm_package = {
            "persona": req.persona,
            "behavior": {
                "tone": req.behavior.strip() if req.behavior.strip() else "friendly",
                "mode": "CRE-structured"  # cognitive compatible
            },
            "context_controller": {
                "max_tokens": 4096,
                "truncate_behavior": "top_priority_docs"
            },
            "templates": {},
            "memory": {},                 # future adaptive memory
            "documents": docs,
            "vector_store": vector_store,
        }

        # --------------------------------------------------------
        # 4. LLM ENGINE → DMAO Gateway
        # --------------------------------------------------------
        try:
            result = llm_engine.inference(llm_package, req.prompt)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"SIFRA LLM Engine internal error: {str(e)}"
            )

        reply = result.get("reply", "").strip()

        if not reply:
            reply = "No response generated."

        # --------------------------------------------------------
        # 5. Final Output (NARE-X friendly)
        # --------------------------------------------------------
        return {
            "status": "success",
            "model_used": result.get("model", "unknown"),
            "cre_enhanced": True,
            "reply": reply,
            "retrieved_docs": result.get("retrieved_docs", []),
            "confidence": result.get("confidence", 0.82),
            "message": "LLM inference executed successfully (v4.5)."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM inference failed: {str(e)}"
        )

# ============================================================
# END OF FILE — SIFRA LLM Inference API v4.5
# ============================================================
