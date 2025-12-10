# ============================================================
#   SIFRA Unified Intelligence Engine v10.0 (COGNITIVE MODE)
#
#   FIXED FOR:
#       ✓ AI Explain
#       ✓ Insights
#       ✓ Visuals (AutoVisualize)
#       ✓ FE-Compatible JSON
# ============================================================

import pandas as pd
from io import StringIO
import traceback

from core.sifra_core import SifraCore
from utils.logger import SifraLogger

from tasks.auto_modeler import AutoModeler
from core.sifra_llm_engine import SifraLLMEngine
from tasks.dataset_to_knowledge import df_to_sentences
from tasks.auto_visualize import AutoVisualize


class SIFRAUnifiedEngine:

    def __init__(self):
        self.debug = True
        self.log = SifraLogger("SIFRA_UNIFIED_10_0")

        self.core = SifraCore()
        self.modeler = AutoModeler()
        self.visualizer = AutoVisualize()
        self.llm_engine = SifraLLMEngine()

        self.active_df = None
        self._dbg("Unified Engine Loaded (v10.0 Cognitive Mode)")

    def _dbg(self, *msg):
        if self.debug:
            print("[DEBUG]", *msg)

    # ------------------------------------------------------------
    # UNIVERSAL DATASET LOADER
    # ------------------------------------------------------------
    def load_dataset(self, src):

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
                cols = [f"col_{i+1}" for i in range(len(src[0]))]
                return pd.DataFrame(src, columns=cols)

            if isinstance(src, str) and "," in src and "\n" in src:
                return pd.read_csv(StringIO(src))

        except Exception as e:
            self._dbg("load_dataset ERROR:", e)

        return pd.DataFrame()

    # ------------------------------------------------------------
    # MAIN ROUTER
    # ------------------------------------------------------------
    def run(self, goal, ctx):
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
    # CREATE LLM
    # ------------------------------------------------------------
    def _handle_create_llm(self, ctx):

        docs = ctx.get("documents")
        dataset = ctx.get("dataset")
        config = ctx.get("config", {})

        df = None

        if dataset is not None:
            df = self.load_dataset(dataset)

        if df is None and isinstance(docs, list) and len(docs) > 3:
            if "," in docs[0]:
                df = pd.DataFrame([r.split(",") for r in docs])

        if df is not None and not df.empty:
            self.active_df = df

        if not isinstance(docs, list):
            df2 = self.load_dataset(docs)
            docs = df_to_sentences(df2)

        return self.llm_engine.create_llm(config, docs, df)

    # ------------------------------------------------------------
    # DATA SUMMARY ENGINE
    # ------------------------------------------------------------
    def _data_summary(self, df):

        dfc = df.copy()

        for col in dfc.columns:
            dfc[col] = (
                dfc[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            dfc[col] = pd.to_numeric(dfc[col], errors="ignore")

        out = []
        out.append(f"Rows: {dfc.shape[0]}")
        out.append(f"Columns: {dfc.shape[1]}")

        for col in dfc.columns:
            s = dfc[col]
            if pd.api.types.is_numeric_dtype(s):
                out.append(f"[{col}] mean={s.mean():.2f}, min={s.min()}, max={s.max()}")
            else:
                out.append(f"[{col}] common: {s.value_counts().head(3).to_dict()}")

        return "\n".join(out)

    # ------------------------------------------------------------
    # TEST LLM
    # ------------------------------------------------------------
    def _handle_test_llm(self, ctx):

        prompt = ctx.get("prompt", "").lower()
        llm_pkg = ctx.get("llm_package")
        df = self.active_df

        DATA_WORDS = [
            "column", "columns", "summary", "describe",
            "analyze", "analysis", "stats", "statistics",
            "trend", "structure", "dataset"
        ]

        if df is not None and any(k in prompt for k in DATA_WORDS):
            return {"status": "success", "reply": self._data_summary(df)}

        if df is not None:
            return {"status": "success", "reply": self.llm_engine.explain(prompt, df)}

        raw = self.llm_engine.inference(llm_pkg, prompt)
        return raw if isinstance(raw, dict) else {"status": "success", "reply": str(raw)}

    # ------------------------------------------------------------
    # AUTOML TRAIN
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # BRAIN PIPELINE (FIXED FOR UI)
    # ------------------------------------------------------------
    def _handle_brain(self, ctx):

        df = self.load_dataset(ctx.get("dataset"))
        self.active_df = df

        brain = self.core.run("analyze", df)

        # ========== VISUALIZATION FIX ==========
        visual = self.visualizer.run(df)

        # ========== INSIGHTS ==========
        insights = brain.get("insights", [])
        if isinstance(insights, dict):
            insights = [insights]

        # ========== AI EXPLANATION ==========
        ai_explain = brain.get("CRE", {}).get("final_decision") or \
                     brain.get("DMAO", {}).get("agent_output", {}).get("natural_language_response") or \
                     "No explanation generated."

        return {
            "status": "success",
            "response": {
                "summary": self._data_summary(df),
                "visuals": visual.get("visual_plan"),
                "insights": insights,
                "ai_explanation": ai_explain,
                "raw_brain": brain
            }
        }
