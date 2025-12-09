# ============================================================
#   SIFRA Synthetic LLM Engine v4.5  (COGNITIVE + DATA MODE)
#
#   New Features in v4.5:
#     ✔ CRE-enhanced contextual answering
#     ✔ DMAO compatible output packages
#     ✔ Cognitive RAG (vector search + reasoning context)
#     ✔ Data-aware summarization (improved v4.5)
#     ✔ Multi-model switching (future ready)
#     ✔ NARE-X compatible NLG formatting
#     ✔ Enhanced safety + fallback logic
# ============================================================

import json
import uuid
import io
import zipfile
import numpy as np
import pandas as pd

from tasks.llm_vectorizer import build_vector_store, search_vector_store


class SifraLLMEngine:

    def __init__(self):
        print("[SIFRA LLM Engine] Ready (v4.5 Cognitive + Data Mode)")
        self.active_llm = None
        self.active_df = None  # ⭐ Stores last dataset (data-aware mode)
        self.last_cre_context = None  # ⭐ CRE reasoning memory


    # ============================================================
    #  INTERNAL SAFE DICT HANDLER
    # ============================================================
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

        # Store dataset for data-aware mode
        if isinstance(df, pd.DataFrame):
            self.active_df = df

        # Store CRE context for cognitive mode
        if cre_context:
            self.last_cre_context = cre_context

        if not documents or not isinstance(documents, list):
            return {"status": "error", "reply": "No documents were provided."}

        cleaned_docs = [str(d).strip() for d in documents if len(str(d).strip()) > 3]

        if not cleaned_docs:
            return {"status": "error", "reply": "Documents empty."}

        store = build_vector_store(cleaned_docs)

        llm_id = str(uuid.uuid4())

        llm_package = {
            "llm_id": llm_id,
            "persona": config.get("persona", "assistant"),

            "behavior": self._safe_dict(config.get("behavior"), "tone", "professional"),
            "templates": self._safe_dict(config.get("templates"), "style", "assistant"),
            "memory": self._safe_dict(config.get("memory"), "data", {}),

            "documents": cleaned_docs,
            "vector_store": {
                "docs": store["docs"],
                "doc_len": store["doc_len"].tolist(),
            },

            "cre_context": cre_context or {},
        }

        self.active_llm = llm_package

        return {
            "status": "success",
            "reply": "LLM instance created successfully.",
            "llm_package": llm_package,
            "llm_id": llm_id
        }


    # ============================================================
    #  MAIN INFERENCE GATEWAY
    # ============================================================
    def inference(self, llm_package, prompt: str):

        print("\n[LLM DEBUG] inference() called")
        print("[Prompt] →", prompt)

        # Auto-heal package
        if llm_package is None or isinstance(llm_package, str):
            llm_package = self.active_llm

        if not isinstance(llm_package, dict):
            return {"status": "error", "reply": "Invalid LLM package."}

        # ⭐ PRIORITY 1 — Dataset reasoning (Data-Aware Mode)
        if isinstance(self.active_df, pd.DataFrame):
            data_reply = self._data_summary_mode(prompt, self.active_df)
            if data_reply:
                return {
                    "status": "success",
                    "reply": data_reply,
                    "model": "DataAware-v4.5",
                    "confidence": 0.93
                }

        # ⭐ PRIORITY 2 — Cognitive RAG mode (CRE-enhanced vector search)
        return self._cognitive_vector_mode(llm_package, prompt)


    # ============================================================
    #  DATA-AWARE SUMMARIZATION (Improved v4.5)
    # ============================================================
    def _data_summary_mode(self, prompt, df):

        p = prompt.lower().strip()

        keywords = [
            "summarize dataset", "dataset summary",
            "describe", "statistics", "columns",
            "trends", "insights", "profile", "overview"
        ]

        if not any(k in p for k in keywords):
            return None

        rows, cols = df.shape
        msg = f"### 📊 Dataset Summary (SIFRA Data-Aware Engine v4.5)\n\n"
        msg += f"**Rows:** {rows}\n"
        msg += f"**Columns:** {cols}\n"
        msg += f"**Field Names:** {', '.join(df.columns)}\n\n"

        # Numeric profiles
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(num_cols) > 0:
            msg += "### 📈 Numeric Column Stats\n"
            for col in num_cols:
                s = df[col]
                msg += f"- **{col}** → mean={s.mean():.2f}, min={s.min()}, max={s.max()}, std={s.std():.2f}\n"
            msg += "\n"

        # Categorical distributions
        cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
        if len(cat_cols) > 0:
            msg += "### 🏷️ Categorical Overview\n"
            for col in cat_cols:
                vals = df[col].value_counts().head(3).to_dict()
                msg += f"- **{col}** → top values: {vals}\n"
            msg += "\n"

        # CRE-style insight block
        msg += "### 🧠 Cognitive Insights\n"
        msg += "- Dataset contains a mix of numeric & categorical features.\n"
        msg += "- Useful for forecasting, segmentation, anomaly detection.\n"
        msg += "- Can help identify trends and correlations.\n"

        return msg


    # ============================================================
    #  COGNITIVE VECTOR MODE (CRE + Vector RAG)
    # ============================================================
    def _cognitive_vector_mode(self, llm_package, prompt):

        docs = llm_package.get("documents", [])
        store = build_vector_store(docs)

        matches = search_vector_store(store, prompt, top_k=5)

        cleaned = [m for m in matches if isinstance(m, str)]

        if not cleaned:
            return {
                "status": "success",
                "reply": f"No strong matches for **{prompt}**.",
                "model": "Cognitive-RAG-v4.5",
                "confidence": 0.55
            }

        # Merge CRE context if exists
        cre_context = llm_package.get("cre_context", {})

        bullets = "\n".join(
            f"- {d[:180]}" for d in cleaned[:5]
        )

        reply = (
            f"### 🧠 Cognitive Answer (CRE + Vector RAG v4.5)\n"
            f"Query: **{prompt}**\n\n"
        )

        # Add CRE reasoning summary if available
        if cre_context and "final_decision" in cre_context:
            reply += f"**CRE Insight:** {cre_context['final_decision']}\n\n"

        reply += f"### 🔍 Relevant Knowledge\n{bullets}"

        return {
            "status": "success",
            "reply": reply,
            "model": "Cognitive-RAG-v4.5",
            "confidence": 0.89,
            "cre_used": True
        }


    # ============================================================
    #  EXPLAIN MODE (CRE + Data Aware)
    # ============================================================
    def explain(self, prompt, df=None):
        if isinstance(df, pd.DataFrame):
            self.active_df = df
        return self.inference(self.active_llm, prompt).get("reply", "")


    # ============================================================
    #  EXPORT LLM PACKAGE
    # ============================================================
    def export_llm(self, llm_package):

        mem = io.BytesIO()

        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
            for key in ["persona", "behavior", "templates", "memory"]:
                z.writestr(f"{key}.json", json.dumps(llm_package.get(key, {}), indent=4))

            z.writestr("documents.json", json.dumps(llm_package.get("documents", []), indent=4))
            z.writestr("vector_store.json", json.dumps(llm_package.get("vector_store", {}), indent=4))
            z.writestr("cre_context.json", json.dumps(llm_package.get("cre_context", {}), indent=4))

        mem.seek(0)
        return mem
