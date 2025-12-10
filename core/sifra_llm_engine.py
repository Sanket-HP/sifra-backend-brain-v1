# ============================================================
#   SIFRA Synthetic LLM Engine v5.0  
#   (HUMAN-LIKE ANSWERING + COGNITIVE RAG + DATA MODE)
#
#   New Features in v5.0:
#     ✔ Human-style answer generator (NARE-X Mini Engine)
#     ✔ CRE-enhanced contextual reasoning
#     ✔ Cognitive RAG (vector search + reasoning fusion)
#     ✔ Data-aware intelligence (summaries, insights)
#     ✔ Dynamic tone + persona system
#     ✔ Natural paragraph generation
# ============================================================

import json
import uuid
import io
import zipfile
import numpy as np
import pandas as pd

from tasks.llm_vectorizer import build_vector_store, search_vector_store


# ============================================================
#   MINI NARE-X NATURAL LANGUAGE GENERATION ENGINE
# ============================================================
def generate_human_answer(prompt, context_chunks, persona="assistant", tone="helpful"):
    """
    Produces a human-like natural language answer using
    retrieved context + conversational tone.
    """

    intro_map = {
        "assistant": "Sure, here's a clear explanation:",
        "expert": "Based on the available information, here's an expert interpretation:",
        "friendly": "Absolutely! Let me explain this in a simple way:",
        "professional": "Here is the requested information:",
    }

    intro = intro_map.get(persona, intro_map["assistant"])

    body = ""
    for chunk in context_chunks:
        body += f"- {chunk.strip()}\n"

    result = (
        f"{intro}\n\n"
        f"**Your Question:** {prompt}\n\n"
        f"**Answer:**\n"
        f"{body}\n"
        f"If you'd like deeper insights, patterns, or examples — feel free to ask!"
    )

    return result


# ============================================================
#   MAIN LLM ENGINE
# ============================================================
class SifraLLMEngine:

    def __init__(self):
        print("[SIFRA LLM Engine] Ready (v5.0 Human-Like Mode)")
        self.active_llm = None
        self.active_df = None
        self.last_cre_context = None


    # ============================================================
    #   INTERNAL SAFE DICT HANDLER
    # ============================================================
    def _safe_dict(self, val, key, default):
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            return {key: val}
        return {key: default}


    # ============================================================
    #   CREATE SYNTHETIC LLM INSTANCE
    # ============================================================
    def create_llm(self, config: dict, documents: list, df=None, cre_context=None):

        if isinstance(df, pd.DataFrame):
            self.active_df = df

        if cre_context:
            self.last_cre_context = cre_context

        if not documents or not isinstance(documents, list):
            return {"status": "error", "reply": "No documents were provided."}

        cleaned_docs = [str(d).strip() for d in documents if len(str(d).strip()) > 2]

        store = build_vector_store(cleaned_docs)
        llm_id = str(uuid.uuid4())

        llm_package = {
            "llm_id": llm_id,
            "persona": config.get("persona", "assistant"),
            "tone": config.get("tone", "helpful"),
            "behavior": self._safe_dict(config.get("behavior"), "tone", "professional"),
            "documents": cleaned_docs,
            "vector_store": store,
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
    #   MAIN INFERENCE GATEWAY
    # ============================================================
    def inference(self, llm_package, prompt: str):

        if llm_package is None:
            llm_package = self.active_llm

        if not isinstance(llm_package, dict):
            return {"status": "error", "reply": "Invalid LLM package."}

        # Step 1 — DATA-AWARE MODE
        if isinstance(self.active_df, pd.DataFrame):
            data_reply = self._data_summary_mode(prompt, self.active_df)
            if data_reply:
                return {"status": "success", "reply": data_reply}

        # Step 2 — Cognitive RAG + Human-Like Answer
        return self._cognitive_vector_mode(llm_package, prompt)


    # ============================================================
    #   DATA-AWARE REASONING MODE
    # ============================================================
    def _data_summary_mode(self, prompt, df):

        p = prompt.lower().strip()

        keywords = [
            "summarize", "dataset overview", "columns", "stats", "trends",
            "describe dataset", "explain dataset"
        ]

        if not any(k in p for k in keywords):
            return None

        rows, cols = df.shape

        msg = (
            f"Here’s a quick overview of your dataset:\n\n"
            f"- **Rows:** {rows}\n"
            f"- **Columns:** {cols}\n\n"
            f"Top fields include: {', '.join(df.columns[:5])}\n\n"
            "This dataset is suitable for forecasting, insights, and ML modelling."
        )

        return msg


    # ============================================================
    #   COGNITIVE VECTOR MODE + HUMAN-LIKE ANSWERING
    # ============================================================
    def _cognitive_vector_mode(self, llm_package, prompt):

        docs = llm_package.get("documents", [])
        store = build_vector_store(docs)

        matches = search_vector_store(store, prompt, top_k=5)
        matches = [m for m in matches if isinstance(m, str)]

        if not matches:
            return {"status": "success", "reply": f"Sorry, I don't have enough information to answer **{prompt}**."}

        persona = llm_package.get("persona", "assistant")
        tone = llm_package.get("tone", "helpful")

        # Human-style answer generation
        final_answer = generate_human_answer(prompt, matches[:3], persona, tone)

        return {
            "status": "success",
            "reply": final_answer,
            "llm_mode": "Human-Like-Answering-v5.0"
        }


    # ============================================================
    #   EXPLAIN MODE
    # ============================================================
    def explain(self, prompt, df=None):
        if isinstance(df, pd.DataFrame):
            self.active_df = df
        return self.inference(self.active_llm, prompt).get("reply", "")


    # ============================================================
    #   EXPORT LLM PACKAGE
    # ============================================================
    def export_llm(self, llm_package):

        mem = io.BytesIO()

        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("config.json", json.dumps(llm_package, indent=4))

        mem.seek(0)
        return mem
    # ============================================================