# ============================================================
#   SIFRA Unified Intelligence Engine v10.0 (ENTERPRISE COGNITIVE MODE)
#
#   Integrates:
#       ✓ SifraCore (HDP + HDS + NAREX + CRE + DMAO + ALL)
#       ✓ AutoML v8.1.2
#       ✓ Synthetic LLM Engine v4.5
#       ✓ Dataset → Knowledge Engine v9.0
#       ✓ Enterprise Data Summary System
#       ✓ FE-Compatible Responses
#
#   Supports:
#       • create_llm
#       • test_llm
#       • automl_train
#       • brain_pipeline
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
        self.log = SifraLogger("SIFRA_UNIFIED_10_0")

        # Core Cognitive Engine
        self.core = SifraCore()

        # AutoML
        self.modeler = AutoModeler()

        # LLM Engine
        self.llm_engine = SifraLLMEngine()

        # Dataset Context Memory
        self.active_df = None

        self._dbg("Unified Engine Loaded (v10.0 Cognitive Mode)")

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

            # DataFrame directly
            if isinstance(src, pd.DataFrame):
                return src.copy()

            # JSON style: {columns: [], data: []}
            if isinstance(src, dict) and "columns" in src and "data" in src:
                return pd.DataFrame(src["data"], columns=src["columns"])

            # List of dicts
            if isinstance(src, list) and len(src) > 0 and isinstance(src[0], dict):
                return pd.DataFrame(src)

            # List of lists
            if isinstance(src, list) and len(src) > 0 and isinstance(src[0], list):
                cols = [f"col_{i+1}" for i in range(len(src[0]))]
                return pd.DataFrame(src, columns=cols)

            # CSV string
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
    # CREATE LLM – HYBRID DATA & KNOWLEDGE MODE
    # ------------------------------------------------------------
    def _handle_create_llm(self, ctx):

        docs = ctx.get("documents")
        dataset = ctx.get("dataset")
        config = ctx.get("config", {})

        df = None

        # ---- If dataset exists
        if dataset is not None:
            df = self.load_dataset(dataset)

        # ---- Auto-detect CSV-like docs as dataset
        if df is None and isinstance(docs, list) and len(docs) > 3:
            try:
                if "," in docs[0]:
                    df = pd.DataFrame([r.split(",") for r in docs])
                    self._dbg("Auto-detected dataset from documents")
            except:
                pass

        # ---- Save dataset globally
        if df is not None and not df.empty:
            self.active_df = df
            self._dbg("Dataset stored globally")

        # ---- Convert dataset → knowledge sentences
        if not isinstance(docs, list):
            try:
                df2 = self.load_dataset(docs)
                docs = df_to_sentences(df2)
            except:
                docs = []

        return self.llm_engine.create_llm(config, docs, df)

    # ------------------------------------------------------------
    # ENTERPRISE DATA SUMMARY ENGINE (NUMERIC SAFE)
    # ------------------------------------------------------------
    def _data_summary(self, df: pd.DataFrame):

        df_clean = df.copy()

        # Clean & convert numeric safely
        for col in df_clean.columns:

            # Remove commas and whitespace
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )

            # Try numeric conversion
            df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")

        out = []

        # BASIC INFO
        out.append("### 📊 Dataset Summary")
        out.append(f"- Rows: {df_clean.shape[0]}")
        out.append(f"- Columns: {df_clean.shape[1]}")

        out.append("\n### 📌 Columns")
        out.append(", ".join(df_clean.columns.astype(str)))

        # COLUMN ANALYSIS
        out.append("\n### 🧱 Column-Level Analysis")

        for col in df_clean.columns:
            s = df_clean[col]

            if pd.api.types.is_numeric_dtype(s):
                out.append(
                    f"**{col}** → mean={s.mean():.2f}, min={s.min()}, max={s.max()}, std={s.std():.2f}"
                )
            else:
                vc = s.value_counts().head(3).to_dict()
                out.append(f"**{col}** → top values: {vc}")

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
            "trend", "trends", "profit", "revenue", "dataset", "structure"
        ]

        is_data_query = any(k in prompt for k in DATA_KEYWORDS)

        # Dataset-related question → summary engine
        if df is not None and not df.empty and is_data_query:
            self._dbg("DATA QUERY → Summary Engine")
            return {"status": "success", "reply": self._data_summary(df)}

        # Dataset available → hybrid LLM response
        if df is not None and not df.empty:
            self._dbg("General Query → LLM explain()")
            return {
                "status": "success",
                "reply": self.llm_engine.explain(prompt, df)
            }

        # No dataset → pure LLM inference
        self._dbg("No dataset → Vector LLM")
        raw = self.llm_engine.inference(llm_pkg, prompt)

        return raw if isinstance(raw, dict) else {"status": "success", "reply": str(raw)}

    # ------------------------------------------------------------
    # FE-Compatible AutoML Wrapper
    # ------------------------------------------------------------
    def _handle_automl(self, ctx):

        df = self.load_dataset(ctx.get("dataset"))
        self.active_df = df

        meta = self.modeler.run(df)

        # AutoML returns nested result sometimes
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
    # Full Brain Pipeline (HDP + HDS + NAREX + CRE + DMAO)
    # ------------------------------------------------------------
    def _handle_brain(self, ctx):
        df = self.load_dataset(ctx.get("dataset"))
        self.active_df = df

        brain = self.core.run("analyze", df)

        return {"status": "success", "response": brain}
# ============================================================