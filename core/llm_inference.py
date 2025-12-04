# ============================================================
#   SIFRA Synthetic LLM Inference API (FINAL v2.6.1 STABLE)
#   Works with sifra_llm_engine v2.6 + llm_vectorizer
#   Fully supports dataset→knowledge & text RAG input
#   Connected to frontend LLM Playground (LLM tab)
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
    documents: Union[List[str], str]   # Supports textarea or array


# ------------------------------------------------------------
# Normalize incoming documents
# ------------------------------------------------------------
def normalize_docs(raw_docs):
    """
    Converts input into a clean list of text segments.
    Works for:
      - textarea (string)
      - list[str]
      - large dataset knowledge lists
    """

    if isinstance(raw_docs, str):
        # Split into lines, strip whitespace
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

        # Safety check for too large input
        if len(docs) > 30000:
            raise HTTPException(
                status_code=400,
                detail="Too many documents — please reduce dataset or compress text."
            )

        # --------------------------------------------------------
        # 2. Build Vector Store for RAG
        # --------------------------------------------------------
        try:
            vector_store = build_vector_store(docs)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Vector store build failed: {str(e)}"
            )

        # --------------------------------------------------------
        # 3. Build LLM Package
        # --------------------------------------------------------
        llm_package = {
            "persona": req.persona,
            "behavior": {
                "tone": req.behavior.strip() if req.behavior.strip() else "friendly"
            },
            "templates": {},
            "memory": {},        # future memory support
            "documents": docs,
            "vector_store": vector_store
        }

        # --------------------------------------------------------
        # 4. Run Synthetic LLM Engine
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
        # 5. Final Response
        # --------------------------------------------------------
        return {
            "status": "success",
            "reply": reply
        }

    except Exception as e:
        # global fallback
        raise HTTPException(
            status_code=500,
            detail=f"LLM inference failed: {str(e)}"
        )
# ============================================================