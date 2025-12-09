# ============================================================
#   AUTO PREDICT v4.5 — Cognitive Prediction Engine
#
#   Upgrades:
#     ✔ Uses HDS Trend + Correlation + Variation
#     ✔ CRE reasoning explains the prediction
#     ✔ DMAO Predict Agent integrated
#     ✔ NARE-X natural prediction summary
#     ✔ Adaptive Learning (ALL) included
# ============================================================

import pandas as pd
from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor


class AutoPredict:
    """
    Predicts the next value using:
        • HDS trend score
        • HDS variation & correlation
        • CRE reasoning
        • DMAO Predict Agent
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto Predict Module v4.5 Ready")

    # --------------------------------------------------------
    # RUN PREDICTION TASK
    # --------------------------------------------------------
    def run(self, dataset):

        print("\n[AUTO PREDICT] Running Cognitive Prediction Pipeline...")

        # Step 1 — Clean & ensure DataFrame
        clean_data = self.preprocessor.clean(dataset)
        if not isinstance(clean_data, pd.DataFrame):
            clean_data = pd.DataFrame(clean_data)

        # Step 2 — Run SIFRA brain (predict mode)
        result = self.core.run("predict", clean_data)

        # Extract components
        hds = result.get("HDS", {})
        cre = result.get("CRE", {})
        dmao = result.get("DMAO", {})
        learning = result.get("ALL", {})

        # ---- HDS Trend Score ----
        trend = float(hds.get("trend_score", 0.0))

        # ---- Calculate last value ----
        try:
            last_val = float(clean_data.mean(axis=1).mean())
        except Exception:
            last_val = 0.0

        # ---- Compute prediction ----
        prediction = float(last_val + trend)

        # ---- Natural Language Explanation from DMAO → NARE-X ----
        natural = dmao.get("agent_output", {}).get("natural_language_response", "")
        if not natural:
            natural = (
                f"SIFRA predicts the next value to move by {trend:.2f} "
                f"based on trend and variation signals."
            )

        # ---- CRE Reasoning Summary ----
        reasoning_summary = cre.get("final_decision", "No CRE reasoning available.")

        return {
            "task": "auto_predict",
            "status": "success",

            # -------------------------
            # Prediction Outputs
            # -------------------------
            "last_value": last_val,
            "trend_score": trend,
            "predicted_value": prediction,

            # -------------------------
            # Cognitive Reasoning
            # -------------------------
            "cre_reasoning": reasoning_summary,
            "cre_steps": cre.get("steps", []),

            # -------------------------
            # Agent Path
            # -------------------------
            "agent_used": dmao.get("agent_selected", "Unknown"),
            "dmao_output": dmao.get("agent_output"),

            # -------------------------
            # Adaptive Learning
            # -------------------------
            "learning_update": learning,

            # -------------------------
            # Natural Language Insight
            # -------------------------
            "insight_summary": natural,

            "message": "Prediction generated via SIFRA v4.5 Cognitive Engine."
        }


# ------------------------------------------------------------
# DEMO (optional)
# ------------------------------------------------------------
if __name__ == "__main__":
    auto_predict = AutoPredict()
    sample_data = {
        "feature1": [10, 12, 14, 16, 18],
        "feature2": [20, 22, 24, 26, 28]
    }
    prediction_result = auto_predict.run(sample_data)
    print("\nPrediction Result:", prediction_result)
