# ============================================================
#   AUTO ANALYZE v4.5 — Cognitive Analysis Task
#
#   Fully compatible with SIFRA Core v4.5:
#     ✔ CRE reasoning extraction
#     ✔ DMAO agent reporting
#     ✔ HDP + HDS signal profiles
#     ✔ Learning feedback from ALL engine
# ============================================================

from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor


class AutoAnalyze:
    """
    Autonomous cognitive analysis task for SIFRA AI v4.5.
    Produces:
      • HDP meaning signals
      • HDS statistical signals
      • CRE reasoning summary
      • DMAO agent path
      • Natural insight summary via NARE-X
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto Analyze Module v4.5 Ready")

    # --------------------------------------------------------
    # RUN ANALYSIS TASK
    # --------------------------------------------------------
    def run(self, dataset):

        print("\n[AUTO ANALYZE] Running Cognitive Analysis...")

        # Step 1 — Clean dataset
        clean_data = self.preprocessor.clean(dataset)

        # Step 2 — Execute core brain in analysis mode
        result = self.core.run("analyze", clean_data)

        # Extract safe blocks
        hdp = result.get("HDP", {})
        hds = result.get("HDS", {})
        cre = result.get("CRE", {})
        dmao = result.get("DMAO", {})
        learn = result.get("ALL", {})

        # Natural language (via DMAO → NARE-X)
        natural = dmao.get("agent_output", {}).get("natural_language_response", "")

        # CRE reasoning summary
        reasoning_summary = cre.get("final_decision", "")

        return {
            "task": "auto_analyze",
            "status": "success",

            # --------------------------------------
            # HDP SIGNALS (Intent, Context, Meaning)
            # --------------------------------------
            "HDP": {
                "intent_vector": hdp.get("intent_vector"),
                "context_vector": hdp.get("context_vector"),
                "meaning_vector": hdp.get("meaning_vector"),
                "emotion_score": hdp.get("emotion_score"),
            },

            # --------------------------------------
            # HDS SIGNALS (Trend, Correlation, Variation)
            # --------------------------------------
            "HDS": {
                "trend_score": hds.get("trend_score"),
                "correlation_score": hds.get("correlation_score"),
                "variation_score": hds.get("variation_score"),
                "memory_signature": hds.get("memory_signature"),
            },

            # --------------------------------------
            # COGNITIVE REASONING (CRE)
            # --------------------------------------
            "CRE_reasoning": reasoning_summary,
            "CRE_steps": cre.get("steps", []),

            # --------------------------------------
            # MULTI-AGENT ROUTING
            # --------------------------------------
            "DMAO_agent_used": dmao.get("agent_selected"),
            "DMAO_output": dmao.get("agent_output"),

            # --------------------------------------
            # ADAPTIVE LEARNING UPDATE
            # --------------------------------------
            "learning_update": learn,

            # --------------------------------------
            # NATURAL LANGUAGE INSIGHT
            # --------------------------------------
            "insight_summary": natural or "No natural-language insight generated.",

            "message": "Cognitive analysis completed successfully (v4.5)"
        }
