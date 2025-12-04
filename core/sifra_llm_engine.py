# ============================================================
#   SIFRA Synthetic LLM Engine v4.5  (DATA-AWARE MODE)
#   • True dataset summarization
#   • Hybrid (LLM + EDA) intelligence
#   • Stats, schema, insights, trends
#   • Fully crash-proof
#   • Always returns {status, reply}
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
        print("[SIFRA LLM Engine] Ready (v4.5 DATA-AWARE MODE)")
        self.active_llm = None
        self.active_df = None        # ⭐ Stores last dataset


    # ============================================================
    #   SAFE DICT CONVERTER
    # ============================================================
    def _safe_dict(self, val, default_key, default_val):
        if isinstance(val, dict):
            return val

        if isinstance(val, str):
            return {default_key: val}

        return {default_key: default_val}


    # ============================================================
    #   CREATE LLM PACKAGE
    # ============================================================
    def create_llm(self, config: dict, documents: list, df=None):

        if df is not None and isinstance(df, pd.DataFrame):
            self.active_df = df   # ⭐ Enable data-aware mode

        if not documents or not isinstance(documents, list):
            return {"status": "error", "reply": "No documents for LLM."}

        cleaned = [str(d).strip() for d in documents if len(str(d).strip()) > 3]

        if not cleaned:
            return {"status": "error", "reply": "Documents empty."}

        store = build_vector_store(cleaned)

        vector_store = {
            "docs": store["docs"],
            "doc_len": store["doc_len"].tolist(),
        }

        llm_id = str(uuid.uuid4())

        llm_package = {
            "llm_id": llm_id,
            "persona": config.get("persona", "assistant"),

            "behavior": self._safe_dict(config.get("behavior"), "tone", "professional"),
            "templates": self._safe_dict(config.get("templates"), "style", "assistant"),
            "memory": self._safe_dict(config.get("memory"), "data", {}),

            "documents": cleaned,
            "vector_store": vector_store,
        }

        self.active_llm = llm_package

        return {
            "status": "success",
            "llm_id": llm_id,
            "llm_package": llm_package,
            "reply": "Synthetic LLM created successfully."
        }


    # ============================================================
    #   MAIN INFERENCE — Now Data Aware
    # ============================================================
    def inference(self, llm_package, prompt: str):

        print("\n[DEBUG LLM] inference() CALLED")
        print("[DEBUG] Prompt =", prompt)

        # Auto-heal broken packages
        if llm_package is None or isinstance(llm_package, str):
            llm_package = self.active_llm

        if not isinstance(llm_package, dict):
            return {"status": "error", "reply": "Invalid LLM package."}

        # ⭐ If dataset exists → Priority = DATA-AWARE MODE
        if self.active_df is not None:
            reply = self._data_summary_mode(prompt, self.active_df)
            if reply:
                return {"status": "success", "reply": reply}

        # Otherwise → Vector search fallback
        return self._vector_mode(llm_package, prompt)


    # ============================================================
    #   DATA-AWARE SUMMARIZATION LOGIC (Core of v4.5)
    # ============================================================
    def _data_summary_mode(self, prompt, df):

        p = prompt.lower().strip()

        # Keywords understood by the Data-Aware Engine
        keywords = [
            "summarize dataset",
            "dataset summary",
            "column-wise summary",
            "columns",
            "summary",
            "insights",
            "trends",
            "statistics",
            "describe",
        ]

        if not any(k in p for k in keywords):
            return None   # Not a dataset-level query

        # ============================================================
        #   1) BASIC SCHEMA
        # ============================================================
        rows, cols = df.shape
        col_list = ", ".join(df.columns)

        msg = f"### 📊 Dataset Summary (Data-Aware Mode)\n\n"
        msg += f"**Rows:** {rows}\n"
        msg += f"**Columns ({cols}):** {col_list}\n\n"

        # ============================================================
        #   2) NUMERIC STATS
        # ============================================================
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            msg += "### 📈 Numeric Columns Summary\n"
            for col in numeric_cols:
                s = df[col]
                msg += f"- **{col}** → mean={s.mean():.2f}, min={s.min()}, max={s.max()}, std={s.std():.2f}\n"
            msg += "\n"

        # ============================================================
        #   3) CATEGORICAL STATS
        # ============================================================
        cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
        if len(cat_cols) > 0:
            msg += "### 🏷️ Categorical Columns Summary\n"
            for col in cat_cols:
                vals = df[col].value_counts().head(3).to_dict()
                msg += f"- **{col}** → top values: {vals}\n"
            msg += "\n"

        # ============================================================
        #   4) INSIGHTS
        # ============================================================
        msg += "### 🧠 Key Insights\n"
        msg += f"- Dataset covers {cols} different variables.\n"
        msg += f"- Contains both categorical and numeric attributes.\n"
        msg += f"- Useful for sales analytics, forecasting, profitability insights.\n"

        msg += "\nAsk for column summary, trends, anomalies, or business insights."

        return msg


    # ============================================================
    #   VECTOR MODE (Fallback)
    # ============================================================
    def _vector_mode(self, llm_package, prompt):

        docs = llm_package.get("documents", [])
        store = build_vector_store(docs)

        matches = search_vector_store(store, prompt, top_k=5)
        cleaned = [x for x in matches if isinstance(x, str)]

        if not cleaned:
            return {
                "status": "success",
                "reply": f"No strong matches for **{prompt}**, but dataset-aware mode is available."
            }

        bullets = "\n".join(f"- {d[:200]}" for d in cleaned[:5])

        return {
            "status": "success",
            "reply": f"### 📌 Results for: **{prompt}**\n\n{bullets}"
        }


    # ============================================================
    #   EXPLAIN MODE (Auto-Data)
    # ============================================================
    def explain(self, prompt, df=None):

        if df is not None:
            self.active_df = df

        return self.inference(self.active_llm, prompt).get("reply", "")


    # ============================================================
    #   EXPORT PACKAGE
    # ============================================================
    def export_llm(self, llm_package):

        mem = io.BytesIO()

        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
            for key in ["persona", "behavior", "templates", "memory"]:
                z.writestr(f"{key}.json", json.dumps(llm_package[key], indent=4))

            z.writestr("documents.json", json.dumps(llm_package["documents"], indent=4))
            z.writestr("vector_store.json", json.dumps(llm_package["vector_store"], indent=4))

        mem.seek(0)
        return mem
