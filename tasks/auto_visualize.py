# ============================================================
#   AUTO VISUALIZE v4.5 — Cognitive Visualization Engine
#
#   New Features:
#     ✔ HDP semantics for better chart selection
#     ✔ HDS-aware pattern detection (trend, correlation, variation)
#     ✔ CRE reasoning for visualization decisions
#     ✔ DMAO Visualization Agent + NARE-X explanation
#     ✔ Adaptive visualization learning (ALL)
# ============================================================

import numpy as np
import pandas as pd

from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor


class AutoVisualize:
    """
    Cognitive Visualization Engine for SIFRA AI.
    Produces visualization BLUEPRINTS instead of charts.
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto Visualization Module v4.5 Ready")

    # ------------------------------------------------------------
    # Determine the best chart type based on semantics + stats
    # ------------------------------------------------------------
    def decide_chart(self, df, hds, hdp):
        trend = hds.get("trend_score", 0)
        corr = hds.get("correlation_score", 0)
        variation = hds.get("variation_score", 0)

        num_cols = df.select_dtypes(include=[np.number]).shape[1]

        # Strong correlation → heatmap
        if num_cols > 1 and abs(corr) > 0.6:
            return "correlation_heatmap"

        # Large variation → boxplot or histogram
        if variation > 0.2:
            return "boxplot"

        # Clear trend → trend line
        if abs(trend) > 0.1:
            return "trend_line"

        # 2-column numeric → scatter chart
        if num_cols == 2:
            return "scatter"

        # 1-column numeric → line chart
        if num_cols == 1:
            return "line"

        # Fallback
        return "bar"

    # ------------------------------------------------------------
    # Build Visualization Plan Blueprint
    # ------------------------------------------------------------
    def create_visual_plan(self, df, chart_type):
        plan = {
            "chart_type": chart_type,
            "description": f"Recommended chart type: {chart_type}",
            "x": [],
            "y": []
        }

        if chart_type in ["line", "trend_line"]:
            col = df.select_dtypes(include=[np.number]).columns[0]
            plan["x"] = list(range(len(df[col])))
            plan["y"] = df[col].tolist()

        elif chart_type == "scatter":
            cols = df.select_dtypes(include=[np.number]).columns[:2]
            plan["x"] = df[cols[0]].tolist()
            plan["y"] = df[cols[1]].tolist()

        elif chart_type == "boxplot":
            plan["columns"] = df.select_dtypes(include=[np.number]).columns.tolist()

        elif chart_type == "correlation_heatmap":
            plan["matrix"] = df.corr().round(4).fillna(0).to_dict()

        elif chart_type == "bar":
            col = df.select_dtypes(include=[np.number]).columns[0]
            plan["x"] = list(range(len(df[col])))
            plan["y"] = df[col].tolist()

        return plan

    # ------------------------------------------------------------
    # MAIN VISUALIZATION ENGINE
    # ------------------------------------------------------------
    def run(self, dataset):

        print("\n[AUTO VISUALIZE] Running Cognitive Visualization Engine...")

        df = self.preprocessor.clean(dataset)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Run SIFRA Brain to detect deeper insights
        result = self.core.run("analyze", df)

        # Extract intelligence modules
        hdp = result.get("HDP", {})
        hds = result.get("HDS", {})
        cre = result.get("CRE", {})
        dmao = result.get("DMAO", {})
        learning = result.get("ALL", {})

        # Decide chart type using SIFRA AI intelligence
        chart_type = self.decide_chart(df, hds, hdp)

        # Build visualization blueprint
        plan = self.create_visual_plan(df, chart_type)

        # Natural language explanation
        natural = dmao.get("agent_output", {}).get("natural_language_response", "")
        if not natural:
            natural = (
                f"The dataset shows patterns suggesting that a **{chart_type}** is the most "
                "effective visualization. Trends and correlations were analyzed to choose this chart."
            )

        # CRE reasoning
        reasoning_summary = cre.get("final_decision", "No CRE reasoning available.")

        return {
            "task": "auto_visualize",
            "status": "success",

            # Visualization Blueprint
            "visual_plan": plan,

            # HDP Semantic Understanding
            "HDP": hdp,

            # HDS Statistical Understanding
            "HDS": hds,

            # CRE Cognitive Reasoning
            "CRE_reasoning": reasoning_summary,
            "CRE_steps": cre.get("steps", []),

            # DMAO Visualization Agent Output
            "agent_used": dmao.get("agent_selected", "Visualization-Agent"),
            "dmao_output": dmao.get("agent_output"),

            # Adaptive Visualization Learning
            "learning_update": learning,

            # Human-readable summary
            "natural_language_summary": natural,

            "message": "Visualization blueprint generated using SIFRA v4.5 Cognitive Engine."
        }
