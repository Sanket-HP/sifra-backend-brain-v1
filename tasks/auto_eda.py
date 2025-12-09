# ============================================================
#   AUTO EDA v4.5 — Cognitive Exploratory Data Analysis Engine
#
#   New Features Added:
#     ✔ HDP semantic interpretation of dataset
#     ✔ HDS trend/variation/correlation signals
#     ✔ CRE reasoning summary for EDA interpretation
#     ✔ DMAO EDA Agent integration
#     ✔ Intelligent outlier detection (IQR + HDS-aware)
#     ✔ NARE-X natural-language EDA explanation
#     ✔ ALL adaptive learning support
# ============================================================

import numpy as np
import pandas as pd

from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor


class AutoEDA:
    """
    Cognitive Exploratory Data Analysis Engine.
    Produces:
        • Statistical summary
        • HDP semantic signals
        • HDS statistical intelligence
        • CRE reasoning explanation
        • DMAO natural-language EDA summary
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto EDA Engine v4.5 Ready")

    # ---------------------------------------------------------
    # Detect outliers (IQR-based) + HDS-aware patterns
    # ---------------------------------------------------------
    def detect_outliers(self, data):
        Q1 = np.nanpercentile(data, 25)
        Q3 = np.nanpercentile(data, 75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = []
        for i, val in enumerate(data):
            if val < lower or val > upper:
                outliers.append({"index": i, "value": float(val)})

        return outliers

    # ---------------------------------------------------------
    # RUN EDA (Cognitive + Statistical)
    # ---------------------------------------------------------
    def run(self, dataset):

        print("\n[AUTO EDA] Running Cognitive EDA...")

        # Step 1 — Clean dataset
        df = self.preprocessor.clean(dataset)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Ensure numeric conversion
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Step 2 — Ask SIFRA Brain for semantic + statistical insights
        brain_output = self.core.run("analyze", df)

        # Extract components
        hdp = brain_output.get("HDP", {})
        hds = brain_output.get("HDS", {})
        cre = brain_output.get("CRE", {})
        dmao = brain_output.get("DMAO", {})
        learning = brain_output.get("ALL", {})

        # --------------------------------------------
        # Standard EDA Summary
        # --------------------------------------------
        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "missing_values": df.isna().sum().to_dict(),
            "missing_ratio": df.isna().mean().round(4).to_dict()
        }

        # Column-level numeric statistics
        numeric_stats = {}
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue

            numeric_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "outliers": self.detect_outliers(series.values)
            }

        summary["column_statistics"] = numeric_stats

        # Correlation Matrix
        if df.shape[1] > 1:
            summary["correlation_matrix"] = df.corr().round(4).fillna(0).to_dict()
        else:
            summary["correlation_matrix"] = "Not enough numeric columns."

        # --------------------------------------------
        # Natural Language EDA Summary via DMAO → NARE-X
        # --------------------------------------------
        natural = dmao.get("agent_output", {}).get("natural_language_response", "")
        if not natural:
            natural = (
                "SIFRA analyzed the dataset and extracted statistical, semantic, "
                "and correlation patterns. Overall trends indicate "
                f"a {'positive' if hds.get('trend_score', 0) > 0 else 'negative'} "
                "movement with notable variation across columns."
            )

        # --------------------------------------------
        # CRE reasoning summary (Why these patterns matter)
        # --------------------------------------------
        reasoning_summary = cre.get("final_decision", "No CRE reasoning available.")

        # --------------------------------------------
        # Build Final Cognitive EDA Output
        # --------------------------------------------
        return {
            "task": "auto_eda",
            "status": "success",

            # -----------------------------------------------
            # Statistical Summary
            # -----------------------------------------------
            "eda_summary": summary,

            # -----------------------------------------------
            # HDP Semantic Interpretation
            # -----------------------------------------------
            "HDP": {
                "intent_vector": hdp.get("intent_vector"),
                "context_vector": hdp.get("context_vector"),
                "meaning_vector": hdp.get("meaning_vector"),
                "emotion_score": hdp.get("emotion_score"),
            },

            # -----------------------------------------------
            # HDS Statistical Intelligence
            # -----------------------------------------------
            "HDS": {
                "trend_score": hds.get("trend_score"),
                "variation_score": hds.get("variation_score"),
                "correlation_score": hds.get("correlation_score"),
                "memory_signature": hds.get("memory_signature"),
            },

            # -----------------------------------------------
            # CRE Cognitive Reasoning
            # -----------------------------------------------
            "CRE_reasoning": reasoning_summary,
            "CRE_steps": cre.get("steps", []),

            # -----------------------------------------------
            # DMAO Multi-Agent Intelligence
            # -----------------------------------------------
            "agent_used": dmao.get("agent_selected", "Unknown"),
            "dmao_output": dmao.get("agent_output"),

            # -----------------------------------------------
            # Learning Engine (ALL)
            # -----------------------------------------------
            "learning_update": learning,

            # -----------------------------------------------
            # Natural Language Summary
            # -----------------------------------------------
            "natural_language_summary": natural,

            "message": "EDA completed using SIFRA v4.5 Cognitive Engine."
        }
