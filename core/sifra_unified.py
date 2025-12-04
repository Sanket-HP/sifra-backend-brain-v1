# ============================================================
#   SIFRA Unified Intelligence Engine v9.4 (ENTERPRISE HYBRID MODE)
#   FINAL VERSION – Smart Dataset Brain + LLM Hybrid + Safe Numeric Engine
#
#   Major Features:
#       ✓ Smart Intent Detection for Data Questions
#       ✓ Professional Dataset Summary System
#       ✓ Automatic Numeric Conversion (Crash-Proof)
#       ✓ Fully Safe Business Insights
#       ✓ Hybrid LLM: explain() + vector fallback
#       ✓ Auto-dataset detection (docs → CSV)
#       ✓ Debug Mode
#       ✓ FE-Compatible AutoML Response Wrapper   ★ NEW
# ============================================================

import pandas as pd
from io import StringIO
import traceback

from core.sifra_core import SifraCore
from utils.logger import SifraLogger
from tasks.auto_modeler import AutoModeler
from core.sifra_llm_engine import SifraLLMEngine
from tasks.dataset_to_knowledge import df_to_sentences


class SIFRAUnifiedEngine:

    # ------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------
    def __init__(self):
        self.debug = True
        self.log = SifraLogger("SIFRA_UNIFIED_9_4")
        self.core = SifraCore()
        self.modeler = AutoModeler()
        self.llm_engine = SifraLLMEngine()

        self.active_df = None
        self._dbg("Unified Engine Loaded (v9.4 ENTERPRISE MODE)")

    def _dbg(self, *msg):
        if self.debug:
            print("[DEBUG]", *msg)

    # ------------------------------------------------------------
    # UNIVERSAL DATASET LOADER
    # ------------------------------------------------------------
    def load_dataset(self, src):

        self._dbg("load_dataset() src type:", type(src))

        try:
            if src is None:
                return pd.DataFrame()

            if isinstance(src, pd.DataFrame):
                return src

            # JSON table format
            if isinstance(src, dict) and "columns" in src and "data" in src:
                return pd.DataFrame(src["data"], columns=src["columns"])

            # list-of-dicts
            if isinstance(src, list) and len(src) > 0 and isinstance(src[0], dict):
                return pd.DataFrame(src)

            # list-of-lists
            if isinstance(src, list) and len(src) > 0 and isinstance(src[0], list):
                cols = [f"col_{i+1}" for i in range(len(src[0]))]
                return pd.DataFrame(src, columns=cols)

            # CSV raw text
            if isinstance(src, str) and "," in src and "\n" in src:
                return pd.read_csv(StringIO(src))

        except Exception as e:
            self._dbg("load_dataset ERROR:", e)

        return pd.DataFrame()

    # ------------------------------------------------------------
    # MAIN ROUTER
    # ------------------------------------------------------------
    def run(self, goal, ctx):
        self._dbg("RUN →", goal, ctx)

        try:
            match goal:

                case "create_llm":
                    return self._handle_create_llm(ctx)

                case "test_llm":
                    return self._handle_test_llm(ctx)

                case "automl_train":
                    return self._handle_automl(ctx)

                case "brain_pipeline":
                    return self._handle_brain(ctx)

                case _:
                    return {"status": "error", "detail": f"Unknown goal '{goal}'"}

        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "detail": str(e)}

    # ------------------------------------------------------------
    # CREATE LLM – HYBRID DATA MODE
    # ------------------------------------------------------------
    def _handle_create_llm(self, ctx):

        docs = ctx.get("documents")
        dataset = ctx.get("dataset")
        config = ctx.get("config", {})

        df = None

        # 1. Direct dataset
        if dataset is not None:
            df = self.load_dataset(dataset)

        # 2. Auto-detect CSV-like docs
        if df is None and isinstance(docs, list) and len(docs) > 3:
            try:
                if "," in docs[0]:
                    df = pd.DataFrame([r.split(",") for r in docs])
                    self._dbg("Auto-detected dataset from documents")
            except:
                pass

        # 3. Store dataset globally
        if df is not None and not df.empty:
            self.active_df = df
            self._dbg("Dataset stored globally")

        # 4. Convert docs → sentences if needed
        if not isinstance(docs, list):
            try:
                df2 = self.load_dataset(docs)
                docs = df_to_sentences(df2)
            except:
                docs = []

        return self.llm_engine.create_llm(config, docs)

    # ------------------------------------------------------------
    # ENTERPRISE DATA SUMMARY ENGINE (NUMERIC SAFE)
    # ------------------------------------------------------------
    def _data_summary(self, df: pd.DataFrame):

        # ---- CLEAN NUMERIC COLUMNS SAFELY ----
        df_clean = df.copy()

        for col in df_clean.columns:

            # Clean text
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )

            # Convert numeric only where possible
            df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")

        out = []

        # BASIC SUMMARY
        out.append("### 📊 Dataset Summary")
        out.append(f"- Rows: {df_clean.shape[0]}")
        out.append(f"- Columns: {df_clean.shape[1]}")

        # COLUMNS
        col_names = list(map(str, df_clean.columns))
        out.append("\n### 📌 Columns")
        out.append(", ".join(col_names))

        # COLUMN ANALYSIS
        out.append("\n### 🧱 Column Analysis")

        for col in df_clean.columns:
            s = df_clean[col]
            col_str = str(col)

            if pd.api.types.is_numeric_dtype(s):
                out.append(
                    f"**{col_str}** → mean={s.mean():.2f}, min={s.min()}, "
                    f"max={s.max()}, std={s.std():.2f}"
                )
            else:
                vc = s.value_counts().head(3).to_dict()
                out.append(f"**{col_str}** → top values: {vc}")

        # NUMERIC INSIGHTS
        num_cols = [
            c for c in df_clean.columns
            if pd.api.types.is_numeric_dtype(df_clean[c])
        ]

        if num_cols:
            out.append("\n### 📈 Key Numeric Insights")
            for c in num_cols[:5]:
                s = df_clean[c]
                try:
                    trend = "increasing" if s.iloc[-1] > s.iloc[0] else "decreasing"
                except:
                    trend = "stable"
                out.append(f"- {str(c)}: trending **{trend}**")

        # BUSINESS INSIGHT (ONLY if numeric)
        if "Total Profit" in df_clean.columns:
            s = df_clean["Total Profit"]
            if pd.api.types.is_numeric_dtype(s):
                profit = s.sum()
                out.append(
                    f"\n### 💰 Business Insight\nTotal Profit: **{profit:,.2f}**"
                )

        return "\n".join(out)

    # ------------------------------------------------------------
    # HYBRID LLM + DATA INFERENCE
    # ------------------------------------------------------------
    def _handle_test_llm(self, ctx):

        llm_pkg = ctx.get("llm_package")
        prompt = ctx.get("prompt", "").lower()
        df = self.active_df

        self._dbg("Dataset available?", df is not None)

        DATA_KEYWORDS = [
            "summarize", "summary", "column", "columns", "insights",
            "analysis", "analyze", "describe", "statistics", "stats",
            "trend", "trends", "profit", "revenue", "dataset"
        ]

        is_data_query = any(k in prompt for k in DATA_KEYWORDS)

        # Dataset question → summary engine
        if df is not None and not df.empty and is_data_query:
            self._dbg("DATA QUERY → Summary Engine")
            return {"status": "success", "reply": self._data_summary(df)}

        # Dataset exists → use LLM explain
        if df is not None and not df.empty:
            self._dbg("General Query → LLM explain()")
            return {
                "status": "success",
                "reply": self.llm_engine.explain(prompt, df)
            }

        # No dataset → vector LLM fallback
        self._dbg("No dataset → Vector LLM")
        raw = self.llm_engine.inference(llm_pkg, prompt)

        if isinstance(raw, dict):
            return raw
        return {"status": "success", "reply": str(raw)}

    # ------------------------------------------------------------
    # FIXED AUTOML (FE-Compatible Wrapper)
    # ------------------------------------------------------------
    def _handle_automl(self, ctx):

        df = self.load_dataset(ctx.get("dataset"))
        self.active_df = df

        meta = self.modeler.run(df)

        # the AutoModeler sometimes returns { "result": {...} }
        result = meta.get("result", meta)

        return {
            "status": "success",
            "mode": "train",
            "result": {
                "task": result.get("task"),
                "best_model": result.get("best_model", "undefined"),
                "best_score": result.get("best_score"),
                "accuracy": result.get("accuracy"),
                "model_summary": result.get("model_summary"),
                "model_hex": result.get("model_hex"),
                "preprocessor_hex": result.get("preprocessor_hex"),
                "runtime": result.get("runtime", 0)
            }
        }

    # ------------------------------------------------------------
    def _handle_brain(self, ctx):
        df = self.load_dataset(ctx.get("dataset"))
        self.active_df = df
        summary = self._data_summary(df)
        return {"status": "success", "response": {"reply": summary}}
