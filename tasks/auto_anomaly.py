# ============================================================
#   AUTO ANOMALY v4.5 — Cognitive Anomaly Detection Engine
#
#   Upgrades:
#     ✔ Uses HDS variation & deviation signals
#     ✔ CRE reasoning explains WHY anomalies exist
#     ✔ DMAO anomaly agent output included
#     ✔ NARE-X natural-language description
#     ✔ ALL adaptive learning included
# ============================================================

import numpy as np
import pandas as pd
from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor


class AutoAnomaly:
    """
    Detects anomalies using:
        • HDS Variation Score
        • Statistical deviation
        • CRE reasoning explanation
        • DMAO Anomaly Agent
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto Anomaly Detector v4.5 Ready")

    # --------------------------------------------------------
    # RUN ANOMALY DETECTION
    # --------------------------------------------------------
    def run(self, dataset):

        print("\n[AUTO ANOMALY] Running Cognitive Anomaly Detection...")

        # Step 1 — Clean dataset
        clean_data = self.preprocessor.clean(dataset)

        # Convert to DataFrame if needed
        if not isinstance(clean_data, pd.DataFrame):
            clean_data = pd.DataFrame(clean_data)

        # Step 2 — Execute SIFRA Brain (anomaly mode)
        result = self.core.run("anomaly", clean_data)

        # Extract brain outputs
        hds = result.get("HDS", {})
        cre = result.get("CRE", {})
        dmao = result.get("DMAO", {})
        learning = result.get("ALL", {})

        # HDS statistical signals
        variation_score = float(hds.get("variation_score", 0.0))
        corr_score = float(hds.get("correlation_score", 0.0))

        # Basic statistical measures
        numeric_df = clean_data.select_dtypes(include=["int64", "float64"])
        mean_val = float(numeric_df.mean().mean())
        std_val = float(numeric_df.std().mean())

        # --------------------------------------------------------
        # ANOMALY SCANNING (Z-score method)
        # --------------------------------------------------------
        anomalies = []

        # Loop through rows & columns to detect anomalies
        for row_idx, row in numeric_df.iterrows():
            for col in numeric_df.columns:
                val = float(row[col])
                if abs(val - mean_val) > 2 * std_val:
                    anomalies.append({
                        "row": int(row_idx),
                        "column": str(col),
                        "value": val,
                        "z_score": round(abs(val - mean_val) / (std_val + 1e-7), 3)
                    })

        # --------------------------------------------------------
        # DMAO → NARE-X natural-language explanation
        # --------------------------------------------------------
        natural = dmao.get("agent_output", {}).get("natural_language_response", "")
        if not natural:
            natural = (
                f"SIFRA detected {len(anomalies)} anomalies based on variation score "
                f"{variation_score:.3f} and statistical deviation."
            )

        # CRE reasoning summary
        reasoning_summary = cre.get("final_decision", "No CRE reasoning available.")

        return {
            "task": "auto_anomaly",
            "status": "success",

            # --- Statistical Scores ---
            "variation_score": variation_score,
            "correlation_score": corr_score,
            "mean_value": mean_val,
            "std_value": std_val,

            # --- Detected Anomalies ---
            "anomalies_found": anomalies,
            "total_anomalies": len(anomalies),

            # --- Brain Agents ---
            "agent_used": dmao.get("agent_selected", "Unknown"),
            "dmao_output": dmao.get("agent_output"),

            # --- Cognitive Reasoning ---
            "cre_reasoning": reasoning_summary,
            "cre_steps": cre.get("steps", []),

            # --- Learning Update ---
            "learning_update": learning,

            # --- Human-friendly Explanation ---
            "insight_summary": natural,

            "message": "Anomaly detection completed via SIFRA v4.5 Cognitive Engine."
        }


# ------------------------------------------------------------
# DEMO USAGE (SAFE)
# ------------------------------------------------------------
if __name__ == "__main__":
    auto_anomaly = AutoAnomaly()
    sample_data = {
        "feature1": [10, 12, 14, 100, 18],
        "feature2": [20, 22, 24, 26, -50]
    }
    anomaly_result = auto_anomaly.run(sample_data)
    print("\nAnomaly Detection Result:", anomaly_result)
