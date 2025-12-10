# ============================================================
#   AUTO VISUALIZE v4.6 — Cognitive Visualization Engine
#
#   Completely Frontend-Compatible Version
#   Fixes:
#     ✔ Overlapping charts
#     ✔ Incorrect JSON format
#     ✔ Missing visualization metadata
# ============================================================

import numpy as np
import pandas as pd

from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor


class AutoVisualize:
    """
    Cognitive Visualization Engine for SIFRA AI.
    Produces visualization BLUEPRINTS instead of charts.
    Frontend-normalized JSON output.
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto Visualization Module v4.6 Ready")

    # ------------------------------------------------------------
    # Determine the best chart type
    # ------------------------------------------------------------
    def decide_chart(self, df, hds, hdp):
        trend = hds.get("trend_score", 0)
        corr = hds.get("correlation_score", 0)
        variation = hds.get("variation_score", 0)
        num_cols = df.select_dtypes(include=[np.number]).shape[1]

        if num_cols > 1 and abs(corr) > 0.6:
            return "correlation_heatmap"

        if variation > 0.2:
            return "boxplot"

        if abs(trend) > 0.1:
            return "line_chart"

        if num_cols == 2:
            return "scatter"

        if num_cols == 1:
            return "line_chart"

        return "bar"

    # ------------------------------------------------------------
    # Create standardized visual output
    # ------------------------------------------------------------
    def build_visual(self, df, chart_type):
        """
        Returns frontend normalized JSON:
        {
            "type": "line_chart",
            "x": [...],
            "y": [...],
            "label": "ColumnName",
            "meta": { ... }
        }
        """

        visual = {"type": chart_type, "meta": {}}

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # ---------------- LINE & TREND ----------------
        if chart_type in ["line_chart"]:
            col = num_cols[0]
            visual["label"] = col
            visual["x"] = list(range(df.shape[0]))
            visual["y"] = df[col].tolist()
            return visual

        # ---------------- SCATTER ---------------------
        if chart_type == "scatter":
            cols = num_cols[:2]
            visual["label"] = f"{cols[0]} vs {cols[1]}"
            visual["x"] = df[cols[0]].tolist()
            visual["y"] = df[cols[1]].tolist()
            return visual

        # ---------------- BAR -------------------------
        if chart_type == "bar":
            col = num_cols[0]
            visual["label"] = col
            visual["x"] = list(range(df.shape[0]))
            visual["y"] = df[col].tolist()
            return visual

        # ---------------- BOXPLOT ----------------------
        if chart_type == "boxplot":
            visual["columns"] = num_cols
            return visual

        # ---------------- CORRELATION HEATMAP ----------
        if chart_type == "correlation_heatmap":
            corr_matrix = df.corr().fillna(0).round(4)
            visual["matrix"] = corr_matrix.to_dict()
            return visual

        return visual

    # ------------------------------------------------------------
    # MAIN VISUAL ENGINE
    # ------------------------------------------------------------
    def run(self, dataset):

        print("\n[AUTO VISUALIZE] Cognitive Visualization Engine Running...")

        df = self.preprocessor.clean(dataset)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Run SIFRA Cognitive Brain
        result = self.core.run("analyze", df)

        hdp = result.get("HDP", {})
        hds = result.get("HDS", {})
        cre = result.get("CRE", {})
        dmao = result.get("DMAO", {})
        learning = result.get("ALL", {})

        # Decide chart
        chart_type = self.decide_chart(df, hds, hdp)

        # Create frontend-compatible output
        visual_output = self.build_visual(df, chart_type)

        # Natural Language Explanation
        natural = dmao.get("agent_output", {}).get("natural_language_response", "")
        if not natural:
            natural = (
                f"The system detected patterns suggesting a **{chart_type}** "
                f"is the most meaningful visualization. Statistical and semantic "
                f"signals guided this decision."
            )

        return {
            "status": "success",
            "task": "visualize",

            # Frontend-ready visuals
            "visual_plan": visual_output,

            # Cognitive outputs
            "HDP": hdp,
            "HDS": hds,
            "CRE_reasoning": cre.get("final_decision", "No reasoning generated."),
            "CRE_steps": cre.get("steps", []),

            # Agent Output
            "agent_used": dmao.get("agent_selected", "Visualization-Agent"),
            "dmao_output": dmao.get("agent_output", {}),

            # Adaptive Learning
            "learning_update": learning,

            # Human explanation
            "natural_language_summary": natural,

            "message": "Visualization blueprint generated successfully (v4.6)."
        }
