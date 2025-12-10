# ============================================================
#   SIFRA Unified Intelligence Engine v10.0 (FE-Compatible Mode)
#   NEW FORMAT OUTPUT FOR FRONTEND (SUMMARY + INSIGHTS + VISUALS)
# ============================================================

import pandas as pd
from io import StringIO
import traceback

from core.sifra_core import SifraCore
from utils.logger import SifraLogger

from tasks.auto_modeler import AutoModeler
from tasks.auto_visualize import AutoVisualize
from tasks.auto_insights import AutoInsights
from core.sifra_llm_engine import SifraLLMEngine
from tasks.dataset_to_knowledge import df_to_sentences


class SIFRAUnifiedEngine:

    def __init__(self):
        self.debug = True
        self.log = SifraLogger("SIFRA_UNIFIED_10_FE")

        self.core = SifraCore()
        self.modeler = AutoModeler()
        self.visualizer = AutoVisualize()
        self.insighter = AutoInsights()
        self.llm_engine = SifraLLMEngine()

        self.active_df = None
        self._dbg("Unified Engine Loaded (FE-Compatible Mode)")

    def _dbg(self, *msg):
        if self.debug:
            print("[DEBUG]", *msg)

    # ============================================================
    # UNIVERSAL DATA LOADER
    # ============================================================
    def load_dataset(self, src):

        self._dbg("load_dataset() src type:", type(src))

        try:
            if src is None:
                return pd.DataFrame()

            if isinstance(src, pd.DataFrame):
                return src.copy()

            if isinstance(src, dict) and "columns" in src and "data" in src:
                return pd.DataFrame(src["data"], columns=src["columns"])

            if isinstance(src, list) and len(src) > 0 and isinstance(src[0], dict):
                return pd.DataFrame(src)

            if isinstance(src, list) and len(src) > 0 and isinstance(src[0], list):
                max_len = max(len(row) for row in src)
                fixed = [(row + [None] * (max_len - len(row))) for row in src]
                cols = [f"col_{i+1}" for i in range(max_len)]
                return pd.DataFrame(fixed, columns=cols)

            if isinstance(src, str) and "," in src and "\n" in src:
                return pd.read_csv(StringIO(src))

        except Exception as e:
            self._dbg("load_dataset ERROR:", e)

        return pd.DataFrame()

    # ============================================================
    # MAIN ROUTER
    # ============================================================
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

    # ============================================================
    # LLM GENERATION
    # ============================================================
    def _handle_create_llm(self, ctx):

        docs = ctx.get("documents")
        dataset = ctx.get("dataset")
        config = ctx.get("config", {})

        df = None

        if dataset is not None:
            df = self.load_dataset(dataset)

        if df is None and isinstance(docs, list) and len(docs) > 3:
            try:
                if "," in docs[0]:
                    df = pd.DataFrame([r.split(",") for r in docs])
            except:
                pass

        if df is not None and not df.empty:
            self.active_df = df

        if not isinstance(docs, list):
            try:
                df2 = self.load_dataset(docs)
                docs = df_to_sentences(df2)
            except:
                docs = []

        return self.llm_engine.create_llm(config, docs, df)

    # ============================================================
    # LLM INFERENCE
    # ============================================================
    def _handle_test_llm(self, ctx):

        llm_pkg = ctx.get("llm_package")
        prompt = ctx.get("prompt", "").lower()
        df = self.active_df

        DATA_KEYWORDS = [
            "summarize", "summary", "columns", "insights",
            "analysis", "stats", "trend", "dataset"
        ]

        is_data_question = any(k in prompt for k in DATA_KEYWORDS)

        if df is not None and not df.empty and is_data_question:
            return {
                "status": "success",
                "reply": self._data_summary(df)
            }

        if df is not None and not df.empty:
            return {
                "status": "success",
                "reply": self.llm_engine.explain(prompt, df)
            }

        raw = self.llm_engine.inference(llm_pkg, prompt)
        return raw if isinstance(raw, dict) else {"status": "success", "reply": str(raw)}

    # ============================================================
    # BRAIN PIPELINE (FE FORMAT OUTPUT)
    # ============================================================
    def _handle_brain(self, ctx):

        df = self.load_dataset(ctx.get("dataset"))
        self.active_df = df

        # 1) Raw SIFRA Brain
        brain = self.core.run("analyze", df)

        # 2) Auto Insights module
        insights = self.insighter.run(df)

        # 3) Auto Visualization module
        visuals = self.visualizer.run(df)

        # 4) Summary for FE
        summary = (
            f"SIFRA detected {len(df.columns)} features and {len(df)} rows. "
            f"Trend score: {brain.get('HDS',{}).get('trend_score',0)}. "
            f"Key insights generated automatically."
        )

        # 5) AI Explanation from CRE
        ai_explain = brain.get("CRE", {}).get("final_decision", "No CRE explanation available.")

        # FE-Compatible Payload
        return {
            "summary": summary,
            "visuals": visuals.get("visual_plan"),
            "insights": insights.get("insights"),
            "ai_explanation": ai_explain,
            "raw_brain": brain
        }

    # ============================================================
    # AUTOML TRAINING
    # ============================================================
    def _handle_automl(self, ctx):

        df = self.load_dataset(ctx.get("dataset"))
        self.active_df = df

        meta = self.modeler.run(df)
        result = meta.get("result", meta)

        return {
            "status": "success",
            "mode": "train",
            "result": result
        }

    # ============================================================
    # INTERNAL — SUMMARY BUILDER (FIXED)
    # ============================================================
    def _data_summary(self, df):

        # Convert column names to string to avoid TypeError
        cols = [str(c) for c in df.columns]

        out = []
        out.append(f"Rows: {df.shape[0]}")
        out.append(f"Columns: {df.shape[1]}")

        # Safe Top Columns
        top_cols = cols[:5]
        out.append("Top columns: " + ", ".join(top_cols))

        return "\n".join(out)

