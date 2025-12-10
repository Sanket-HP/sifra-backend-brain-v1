# ============================================================
#   SIFRA Unified Intelligence Engine v10.1 
#   FE-Compatible + Human LLM Mode + Cognitive RAG Integration
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
        self.log = SifraLogger("SIFRA_UNIFIED_FE_v10_1")

        # Core components
        self.core = SifraCore()
        self.modeler = AutoModeler()
        self.visualizer = AutoVisualize()
        self.insighter = AutoInsights()
        self.llm_engine = SifraLLMEngine()

        self.active_df = None        # Memory for data-aware LLM
        self.last_llm_package = None # Persist last created LLM

        self._dbg("Unified Engine Loaded (v10.1 FE + Human LLM Mode)")

    # ------------------------------------------------------------
    def _dbg(self, *msg):
        if self.debug:
            print("[DEBUG]", *msg)

    # ============================================================
    # UNIVERSAL DATA LOADER (Robust for FE)
    # ============================================================
    def load_dataset(self, src):

        self._dbg("load_dataset() src:", type(src))

        try:
            if src is None:
                return pd.DataFrame()

            if isinstance(src, pd.DataFrame):
                return src.copy()

            if isinstance(src, dict) and "columns" in src and "data" in src:
                return pd.DataFrame(src["data"], columns=src["columns"])

            if isinstance(src, list) and len(src) > 0:
                if isinstance(src[0], dict):
                    return pd.DataFrame(src)
                if isinstance(src[0], list):
                    max_len = max(len(row) for row in src)
                    fixed = [(row + [None] * (max_len - len(row))) for row in src]
                    cols = [f"col_{i+1}" for i in range(max_len)]
                    return pd.DataFrame(fixed, columns=cols)

            if isinstance(src, str) and "," in src and "\n" in src:
                return pd.read_csv(StringIO(src))

        except Exception as e:
            self._dbg("Dataset load ERROR:", e)

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
                    return {"status": "error", "detail": f"Unknown goal {goal}"}

        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "detail": str(e)}

    # ============================================================
    # CREATE LLM PACKAGE (Human-like + RAG)
    # ============================================================
    def _handle_create_llm(self, ctx):

        docs = ctx.get("documents")
        dataset = ctx.get("dataset")
        config = ctx.get("config", {})

        df = None

        # Load dataset if given
        if dataset is not None:
            df = self.load_dataset(dataset)

        # Auto-detect CSV-style document list
        if df is None and isinstance(docs, list) and len(docs) > 3:
            try:
                if "," in docs[0]:
                    df = pd.DataFrame([r.split(",") for r in docs])
            except:
                pass

        # Store dataset for LLM memory
        if df is not None and not df.empty:
            self.active_df = df

        # Convert dataset→sentences if needed
        if not isinstance(docs, list):
            try:
                df2 = self.load_dataset(docs)
                docs = df_to_sentences(df2)
            except:
                docs = []

        # Create LLM
        result = self.llm_engine.create_llm(config, docs, df)
        self.last_llm_package = result.get("llm_package")

        return result

    # ============================================================
    # LLM INFERENCE (Human-like + RAG + Data-Aware)
    # ============================================================
    def _handle_test_llm(self, ctx):

        llm_pkg = ctx.get("llm_package") or self.last_llm_package
        prompt = ctx.get("prompt", "").strip()

        if not llm_pkg:
            return {"status": "error", "reply": "LLM not created yet."}

        df = self.active_df
        p = prompt.lower()

        # Detect if the user wants dataset-level reasoning
        DATA_KEYWORDS = [
            "summarize", "summary", "columns", "describe",
            "insights", "analysis", "stats", "trend",
            "dataset", "explain data", "data overview"
        ]

        is_data_query = any(k in p for k in DATA_KEYWORDS)

        # Data-aware summary mode
        if df is not None and not df.empty and is_data_query:
            return {
                "status": "success",
                "reply": self._data_summary(df)
            }

        # Human-like + RAG inference
        result = self.llm_engine.inference(llm_pkg, prompt)

        return {
            "status": "success",
            "reply": result.get("reply", ""),
            "model": result.get("llm_mode", "LLM-HUMAN-v10.1"),
            "confidence": result.get("confidence", 0.88)
        }

    # ============================================================
    # BRAIN PIPELINE (Produces FE-Compatible Structured Output)
    # ============================================================
    def _handle_brain(self, ctx):

        df = self.load_dataset(ctx.get("dataset"))
        self.active_df = df

        # Run SIFRA Brain
        brain = self.core.run("analyze", df)

        # Insights
        insights = self.insighter.run(df)

        # Visual Strategy
        visuals = self.visualizer.run(df)

        # Summary text for FE
        summary = (
            f"The dataset contains **{len(df)} rows** and **{len(df.columns)} columns**. "
            f"Trend Score: **{brain.get('HDS', {}).get('trend_score', 0)}**. "
            f"SIFRA automatically generated insights and visualization plans."
        )

        # CRE Explanation
        ai_explain = brain.get("CRE", {}).get("final_decision", "No CRE explanation available.")

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
    # HUMAN-FRIENDLY DATA SUMMARY
    # ============================================================
    def _data_summary(self, df):

        cols = [str(c) for c in df.columns]

        text = "### 📊 Dataset Summary\n"
        text += f"- **Rows:** {df.shape[0]}\n"
        text += f"- **Columns:** {df.shape[1]}\n"
        text += f"- **Top Columns:** {', '.join(cols[:5])}\n\n"

        # Numeric data overview
        nums = df.select_dtypes(include=['int64', 'float64']).columns
        if len(nums):
            text += "### 📈 Numeric Stats\n"
            for c in nums:
                s = df[c]
                text += f"- **{c}** → mean={s.mean():.2f}, min={s.min()}, max={s.max()}\n"

        return text

# ============================================================
# END OF FILE
# ============================================================
