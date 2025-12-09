# ============================================================
#   AUTO FORECAST v4.5 — Cognitive Forecasting Agent
#
#   Upgrades:
#     ✔ HDP + HDS Forecast Signals
#     ✔ CRE reasoning integrated
#     ✔ DMAO Forecast Agent output
#     ✔ Natural-language predictions via NARE-X
#     ✔ Adaptive learning (ALL) included
# ============================================================

from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor
import numpy as np


class AutoForecast:
    """
    Forecast future points using:
        • HDS Trend Score (Primary Signal)
        • DMAO Forecast Agent
        • CRE Reasoning for explanation
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto Forecast Module v4.5 Ready")

    # --------------------------------------------------------
    # RUN FORECAST TASK
    # --------------------------------------------------------
    def run(self, dataset, steps=5):

        print("\n[AUTO FORECAST] Running Cognitive Forecast Pipeline...")

        # Step 1 — Clean dataset
        clean_data = self.preprocessor.clean(dataset)

        # Step 2 — Run SIFRA Brain (forecast mode)
        result = self.core.run("forecast", clean_data)

        # Extract components
        hds = result.get("HDS", {})
        cre = result.get("CRE", {})
        dmao = result.get("DMAO", {})
        learning = result.get("ALL", {})

        # --- Trend Score (core prediction driver)
        trend_score = hds.get("trend_score", 0.0)

        # --- Compute last numeric value for base
        # If dataset is a DataFrame
        try:
            last_value = clean_data.mean(axis=1).mean()
        except Exception:
            last_value = 0.0

        # ----------------------------------------------------
        #   FORECAST CURVE GENERATION
        # ----------------------------------------------------
        future_vals = []
        curr = last_value

        for _ in range(steps):
            curr += trend_score
            future_vals.append(float(curr))

        # ----------------------------------------------------
        #   NATURAL LANGUAGE FORECAST (from DMAO → NARE-X)
        # ----------------------------------------------------
        natural = ""
        agent_output = dmao.get("agent_output", {})

        if isinstance(agent_output, dict):
            natural = agent_output.get("natural_language_response", "")

        if not natural:
            natural = f"Forecast generated using SIFRA Trend Model. Expect a consistent change of {trend_score:.2f} per step."

        # CRE reasoning summary
        reasoning_summary = cre.get("final_decision", "No CRE reasoning available.")

        return {
            "task": "auto_forecast",
            "status": "success",

            # --- HDS Forecasting Signals ---
            "trend_score": trend_score,

            # --- Forecast Results ---
            "steps_requested": steps,
            "forecast_values": future_vals,

            # --- Brain Agent Path ---
            "agent_used": dmao.get("agent_selected", "Unknown"),
            "dmao_output": dmao.get("agent_output"),

            # --- CRE Reasoning ---
            "cre_reasoning": reasoning_summary,
            "cre_steps": cre.get("steps", []),

            # --- Learning Engine Log ---
            "learning_update": learning,

            # --- Natural Language Output ---
            "insight_summary": natural,

            "message": "Forecasting completed via SIFRA v4.5 Cognitive Engine."
        }


# ------------------------------------------------------------
# DEMO USAGE (SAFE)
# ------------------------------------------------------------
if __name__ == "__main__":
    auto_forecast = AutoForecast()
    sample_data = {
        "feature1": [10, 12, 14, 16, 18],
        "feature2": [20, 22, 24, 26, 28]
    }
    forecast_result = auto_forecast.run(sample_data, steps=5)
    print("\nForecast Result:", forecast_result)
