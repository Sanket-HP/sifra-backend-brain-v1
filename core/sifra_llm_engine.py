# ============================================================
#   SIFRA Synthetic LLM Engine v5.1  
#   (HUMAN-LIKE ANSWERING + COGNITIVE RAG + DATA MODE)
#
#   Improvements in v5.1:
#     ✔ Human-style NARE-X+ generator (more natural replies)
#     ✔ Vector store built ONCE (no duplicate building)
#     ✔ Stable output for FastAPI JSON encoding
#     ✔ Persona-aware answer shaping
#     ✔ Tone blending (friendly, expert, professional)
#     ✔ Cleaner summaries in data-aware mode
#     ✔ Better fallback answers
# ============================================================

import json
import uuid
import io
import zipfile
import numpy as np
import pandas as pd

from tasks.llm_vectorizer import build_vector_store, search_vector_store


# ============================================================
#  HUMAN-LIKE NARE-X MINI ENGINE
# ============================================================
def generate_human_answer(prompt, context_chunks, persona="assistant", tone="helpful"):
    """
    Generates a natural, human-like answer using the NARE-X pattern:
    Intro → Understanding → Useful context → Closing guidance
    """

    intro_templates = {
        "assistant": "Sure! Let me explain this clearly:",
        "expert": "Here's an expert breakdown based on the information available:",
        "friendly": "Absolutely! Here's an easy-to-understand explanation:",
        "professional": "Here is the clarified explanation you requested:",
    }

    intro = intro_templates.get(persona, intro_templates["assistant"])

    # Build context summary
    context_text = "\n".join(f"- {c.strip()}" for c in context_chunks)

    # Tone-based closings
    closing_map = {
        "helpful": "If you'd like a deeper breakdown, feel free to ask!",
        "friendly": "Happy to help! Let me know if you want more details 😊",
        "expert": "If you want a more technical analysis, I can provide one.",
        "professional": "Let me know if you require further clarification.",
    }

    closing = closing_map.get(tone, closing_map["helpful"])

    return (
        f"{intro}\n\n"
        f"**Your Question:** {prompt}\n\n"
        f"**Answer:**\n{context_text}\n\n"
        f"{closing}"
    )


# ============================================================
#  SIFRA HUMAN-LIKE LLM ENGINE
# ============================================================
class SifraLLMEngine:

    def __init__(self):
        print("[SIFRA LLM Engine] Ready (v5.1 Human-Like Cognitive Mode)")
        self.active_llm = None
        self.active_df = None
        self.last_cre_context = None


    # ------------------------------------------------------------
    # Safe dict wrapper
    # ------------------------------------------------------------
    def _safe_dict(self, val, key, default):
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            return {key: val}
        return {key: default}


    # ============================================================
    #  CREATE SYNTHETIC LLM INSTANCE
    # ============================================================
    def create_llm(self, config: dict, documents: list, df=None, cre_context=None):

        if isinstance(df, pd.DataFrame):
            self.active_df = df

        if cre_context:
            self.last_cre_context = cre_context

        if not documents or not isinstance(documents, list):
            return {"status": "error", "reply": "No documents were provided."}

        cleaned_docs = [str(d).strip() for d in documents if len(str(d).strip()) > 2]

        # Build vector store ONCE (faster inference)
        vector_store = build_vector_store(cleaned_docs)

        llm_id = str(uuid.uuid4())

        llm_package = {
            "llm_id": llm_id,
            "persona": config.get("persona", "assistant"),
            "tone": config.get("tone", "helpful"),
            "behavior": self._safe_dict(config.get("behavior"), "tone", "professional"),
            "documents": cleaned_docs,
            "vector_store": vector_store,  # stored here (no rebuild later)
            "cre_context": cre_context or {},
        }

        self.active_llm = llm_package

        return {
            "status": "success",
            "reply": "LLM created successfully.",
            "llm_package": llm_package,
            "llm_id": llm_id
        }


    # ============================================================
    #  MAIN INFERENCE PIPELINE
    # ============================================================
    def inference(self, llm_package, prompt: str):

        if llm_package is None:
            llm_package = self.active_llm

        if not isinstance(llm_package, dict):
            return {"status": "error", "reply": "Invalid LLM package."}

        # Step 1 — Data-aware summary if dataset exists
        if isinstance(self.active_df, pd.DataFrame):
            reply = self._data_summary_mode(prompt, self.active_df)
            if reply:
                return {"status": "success", "reply": reply}

        # Step 2 — Cognitive RAG (vector retrieval + human NLG)
        return self._cognitive_vector_mode(llm_package, prompt)


    # ============================================================
    #  DATA SUMMARY MODE (Dataset-aware LLM)
    # ============================================================
    def _data_summary_mode(self, prompt, df):

        p = prompt.lower().strip()

        keywords = [
            "summarize", "dataset", "columns", "stats", "trends",
            "overview", "structure", "explain dataset"
        ]

        if not any(k in p for k in keywords):
            return None

        rows, cols = df.shape
        top_cols = ", ".join(str(c) for c in df.columns[:5])

        msg = (
            f"Here’s a quick summary of your dataset:\n\n"
            f"- **Rows:** {rows}\n"
            f"- **Columns:** {cols}\n"
            f"- **Key Columns:** {top_cols}\n\n"
            f"This dataset can be used for forecasting, insights, ML modelling, anomaly detection, and more."
        )

        return msg


    # ============================================================
    #  COGNITIVE VECTOR MODE + HUMAN-LIKE NLG
    # ============================================================
    def _cognitive_vector_mode(self, llm_package, prompt):

        store = llm_package.get("vector_store")

        if not store or not store.get("docs"):
            return {"status": "success", "reply": "I couldn't find helpful reference data to answer that."}

        matches = search_vector_store(store, prompt, top_k=5)
        matches = [m for m in matches if isinstance(m, str)]

        if not matches:
            return {
                "status": "success",
                "reply": f"I don't have enough context to fully answer: **{prompt}**"
            }

        persona = llm_package.get("persona", "assistant")
        tone = llm_package.get("tone", "helpful")

        # Human-style NARE-X generation
        final_answer = generate_human_answer(prompt, matches[:3], persona, tone)

        return {
            "status": "success",
            "reply": final_answer,
            "llm_mode": "Human-Like-Answering-v5.1"
        }


    # ============================================================
    #  EXPLAIN MODE
    # ============================================================
    def explain(self, prompt, df=None):
        if isinstance(df, pd.DataFrame):
            self.active_df = df
        return self.inference(self.active_llm, prompt).get("reply", "")


    # ============================================================
    #  EXPORT LLM PACKAGE (for downloading)
    # ============================================================
    def export_llm(self, llm_package):

        mem = io.BytesIO()

        safe_pkg = {
            "llm_id": llm_package.get("llm_id"),
            "persona": llm_package.get("persona"),
            "tone": llm_package.get("tone"),
            "documents": llm_package.get("documents"),
        }

        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("llm.json", json.dumps(safe_pkg, indent=4))

        mem.seek(0)
        return mem

# ============================================================
# SEARCH ENGINE HELPERS
# ============================================================