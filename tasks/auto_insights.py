# ============================================================
#   AUTO INSIGHTS v4.5 — Cognitive Insights Engine
#
#   New Features:
#     ✔ HDP Semantic Meaning Insights (intent, context, emotion)
#     ✔ HDS Statistical Insights (trend, variation, correlation)
#     ✔ CRE Reasoning Summary (why these insights matter)
#     ✔ DMAO Insights Agent Output (multi-agent insights)
#     ✔ NARE-X natural-language insight generation
#     ✔ Adaptive Learning (ALL) included
# ============================================================

import numpy as np
import pandas as pd
from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor


class AutoInsights:
    """
    Generates insights using:
        • HDP semantic understanding
        • HDS statistical signals
        • CRE reasoning engine
        • DMAO insights agent
        • NARE-X natural-language summarization
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto Insights Module v4.5 Ready")

    # --------------------------------------------------------
    # RUN INSIGHTS ENGINE
    # --------------------------------------------------------
    def run(self, dataset):

        print("\n[AUTO INSIGHTS] Running Cognitive Insights Pipeline...")

        # Step 1 — Clean dataset
        clean_data = self.preprocessor.clean(dataset)
        if not isinstance(clean_data, pd.DataFrame):
            clean_data = pd.DataFrame(clean_data)

        # Step 2 — Run SIFRA Brain (insights mode)
        result = self.core.run("insights", clean_data)

        # Extract components
        hdp = result.get("HDP", {})
        hds = result.get("HDS", {})
        cre = result.get("CRE", {})
        dmao = result.get("DMAO", {})
        learning = result.get("ALL", {})

        # ----------------------------------------------
        # Basic numeric statistics
        # ----------------------------------------------
        avg_val = float(clean_data.mean().mean())
        max_val = float(clean_data.max().max())
        min_val = float(clean_data.min().min())
        std_val = float(clean_data.std().mean())

        # Trend from HDS engine
        trend_score = float(hds.get("trend_score", 0.0))
        correlation_score = float(hds.get("correlation_score", 0.0))
        variation_score = float(hds.get("variation_score", 0.0))

        # ----------------------------------------------
        # Insight bullets (cognitive + numeric)
        # ----------------------------------------------
        insight_list = [
            f"📊 The dataset exhibits a {'positive' if trend_score > 0 else 'negative'} trend trajectory.",
            f"📈 Average global value: **{avg_val:.2f}**",
            f"🔼 Highest value observed: **{max_val}**",
            f"🔽 Lowest value observed: **{min_val}**",
            f"📉 Overall volatility (std): **{std_val:.2f}**",
            f"🔗 Correlation signal strength: **{correlation_score:.2f}**",
            f"📡 Variation intensity: **{variation_score:.2f}**",
            f"🧠 CRE reasoning summary: {cre.get('final_decision', 'No reasoning available.')}"
        ]

        # ----------------------------------------------
        # DMAO Natural-Language Insights (NARE-X)
        # ----------------------------------------------
        natural = dmao.get("agent_output", {}).get("natural_language_response", "")
        if not natural:
            natural = (
                "SIFRA identified key insights using semantic signals (intent, meaning) "
                "and statistical variation. Trend indicates "
                f"{'upward' if trend_score > 0 else 'downward'} movement with notable patterns."
            )

        return {
            "task": "auto_insights",
            "status": "success",

            # -------------------------
            # HDP Semantic Understanding
            # -------------------------
            "HDP": {
                "intent_vector": hdp.get("intent_vector"),
                "context_vector": hdp.get("context_vector"),
                "meaning_vector": hdp.get("meaning_vector"),
                "emotion_score": hdp.get("emotion_score"),
            },

            # -------------------------
            # HDS Statistical Signals
            # -------------------------
            "HDS": {
                "trend_score": trend_score,
                "correlation_score": correlation_score,
                "variation_score": variation_score,
                "memory_signature": hds.get("memory_signature"),
            },

            # -------------------------
            # CRE Reasoning
            # -------------------------
            "CRE_reasoning": cre.get("final_decision"),
            "CRE_steps": cre.get("steps", []),

            # -------------------------
            # Multi-Agent System
            # -------------------------
            "agent_used": dmao.get("agent_selected", "Unknown"),
            "dmao_output": dmao.get("agent_output"),

            # -------------------------
            # Adaptive Learning
            # -------------------------
            "learning_update": learning,

            # -------------------------
            # Extracted Insights
            # -------------------------
            "numeric_insights": {
                "average": avg_val,
                "max_value": max_val,
                "min_value": min_val,
                "std_dev": std_val,
            },

            "insights_list": insight_list,
            "natural_language_insights": natural,

            "message": "Insights extracted via SIFRA v4.5 Cognitive Engine."
        }


# ------------------------------------------------------------
# DEMO
# ------------------------------------------------------------
if __name__ == "__main__":
    auto_insights = AutoInsights()
    sample_data = {
        "feature1": [10, 12, 14, 16, 18],
        "feature2": [20, 22, 24, 26, 28]
    }
    insights_result = auto_insights.run(sample_data)
    print("\nInsights Result:", insights_result)
