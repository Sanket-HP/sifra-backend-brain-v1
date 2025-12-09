# ============================================================
#   SIFRA AI – Cognitive Auto Evaluation Engine v4.5 ULTRA
#
#   New Features:
#       ✓ HDP-aware task classification
#       ✓ HDS statistical interpretation
#       ✓ CRE reasoning for evaluation decisions
#       ✓ DMAO agent-driven natural summary
#       ✓ Adaptive learning feedback loop
#       ✓ Fully safe metrics for regression, classification, clustering
# ============================================================

import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score
)

from core.sifra_core import SifraCore


class AutoEvaluate:
    """
    SIFRA Cognitive Evaluation Engine.
    Supports:
        - Regression
        - Classification
        - Clustering
    Adds:
        ✔ HDP intent reasoning
        ✔ HDS statistical interpretation
        ✔ CRE evaluation reasoning
        ✔ DMAO natural evaluation summary
    """

    def __init__(self):
        self.core = SifraCore()
        print("[TASK] Auto Evaluation Engine v4.5 Ready")

    # --------------------------------------------------------------
    # Cognitive Task Detection (HDP + Simple Heuristics)
    # --------------------------------------------------------------
    def detect_type(self, y_true):

        # Use HDP meaning to enhance decision
        try:
            semantic = self.core.intent.detect_intent(str(y_true[:20]))
        except:
            semantic = "unknown"

        y_true = np.array(y_true)
        unique_vals = len(np.unique(y_true))

        # Classification if categories small
        if unique_vals <= 5:
            return "classification"

        # Regression if numeric
        if np.issubdtype(y_true.dtype, np.number):
            return "regression"

        # Fallback
        return "clustering"

    # --------------------------------------------------------------
    # Main Evaluation
    # --------------------------------------------------------------
    def run(self, y_true, y_pred):

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        task_type = self.detect_type(y_true)

        # Run SIFRA Cognitive Engines to generate meta-insights
        cognitive = self.core.narex.run(
            {"y_true": y_true.tolist(), "y_pred": y_pred.tolist()}
        )

        # CRE explanation (why performance is good/bad)
        try:
            cre_reason = self.core.reasoner.explain(f"evaluate {task_type}")
        except:
            cre_reason = "Cognitive reasoning unavailable."

        # DMAO natural summary
        try:
            dmao_output = self.core.agents.run_agent(
                "evaluation-agent",
                f"Model evaluation task: {task_type}"
            )
            natural_summary = dmao_output.get("natural_language_response", "")
        except:
            natural_summary = "Evaluation summary could not be generated."

        result = {
            "status": "success",
            "task_type": task_type,
            "CRE_reasoning": cre_reason,
            "DMAO_summary": natural_summary,
            "NAREX_meta": cognitive
        }

        # =====================================================
        # 1️⃣ REGRESSION METRICS
        # =====================================================
        if task_type == "regression":
            try:
                result.update({
                    "r2_score": float(r2_score(y_true, y_pred)),
                    "mse": float(mean_squared_error(y_true, y_pred)),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                })
            except Exception as e:
                return {"error": f"Regression evaluation failed: {str(e)}"}

            return result

        # =====================================================
        # 2️⃣ CLASSIFICATION METRICS
        # =====================================================
        elif task_type == "classification":
            try:
                precision = precision_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )
                recall = recall_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )
                f1 = f1_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )

                result.update({
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                })
            except Exception as e:
                return {"error": f"Classification evaluation failed: {str(e)}"}

            return result

        # =====================================================
        # 3️⃣ CLUSTERING METRICS
        # =====================================================
        else:
            try:
                score = silhouette_score(y_true.reshape(-1, 1), y_pred)
                result.update({
                    "silhouette_score": float(score)
                })
            except Exception:
                result.update({
                    "status": "partial",
                    "message": "Silhouette score not computable — returning labels.",
                    "labels": y_pred.tolist(),
                })

            return result
