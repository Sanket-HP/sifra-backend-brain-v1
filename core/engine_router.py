# ============================================================
#  ENGINE ROUTER v4.6 (UPDATED FOR SIFRA AI + EXCELON)
#
#  New Capabilities:
#    ✔ Smart Goal Classifier 3.0
#    ✔ CRE + DMAO aware routing
#    ✔ NLP fuzzy goal correction
#    ✔ Excelon™ Spreadsheet Engine support
#    ✔ Multi-agent safe routing
#    ✔ Unified fallback handler
# ============================================================

import difflib
from core.sifra_core import SifraCore


class EngineRouter:
    """
    Routes user goals to the SIFRA Core Engine.
    Enhanced with Excelon™ (algorithm-based sheet generation).
    """

    VALID_GOALS = {
        # -------------------------
        # ANALYTICS
        # -------------------------
        "analyze": "analytics",
        "analysis": "analytics",
        "auto_analyze": "analytics",
        "insights": "analytics",
        "auto_insights": "analytics",
        "trend": "analytics",
        "pattern": "analytics",
        "statistics": "analytics",

        # -------------------------
        # TIME SERIES
        # -------------------------
        "predict": "time_series",
        "prediction": "time_series",
        "auto_predict": "time_series",
        "forecast": "time_series",
        "future": "time_series",
        "auto_forecast": "time_series",

        # -------------------------
        # ANOMALY
        # -------------------------
        "anomaly": "anomaly",
        "anomalies": "anomaly",
        "auto_anomaly": "anomaly",

        # -------------------------
        # ML MODELING
        # -------------------------
        "model": "ml_model",
        "train_model": "ml_model",
        "build_model": "ml_model",

        # -------------------------
        # EXCELON (NEW)
        # -------------------------
        "excelon": "excelon",
        "sheet": "excelon",
        "sheets": "excelon",
        "excel": "excelon",
        "report": "excelon",
        "generate_report": "excelon",
    }

    def __init__(self):
        self.core = SifraCore()
        print("[ENGINE ROUTER] SIFRA Router v4.6 Ready (Excelon enabled).")

    # --------------------------------------------------------
    #  NLP FUZZY MATCHING
    # --------------------------------------------------------
    def auto_correct_goal(self, text):
        """
        Fixes minor typos (ex: 'anlyze' → 'analyze').
        """
        candidates = list(self.VALID_GOALS.keys())
        match = difflib.get_close_matches(text, candidates, n=1, cutoff=0.6)
        return match[0] if match else None

    # --------------------------------------------------------
    #  SMART ROUTER
    # --------------------------------------------------------
    def route(self, goal, dataset, context=None):
        """
        Routes tasks to SIFRA Core or Excelon.
        """
        print(f"[ROUTER] Received goal: {goal}")

        if not goal or not isinstance(goal, str):
            return {
                "status": "error",
                "message": "Invalid goal format.",
                "goal": str(goal)
            }

        clean_goal = goal.lower().strip()
        context = context or {}

        # 1️⃣ DIRECT MATCH
        if clean_goal in self.VALID_GOALS:
            mapped_goal = self.VALID_GOALS[clean_goal]
            print(f"[ROUTER] Direct match → {mapped_goal}")

            # Excelon bypasses SifraCore
            if mapped_goal == "excelon":
                from tasks.auto_excelon import run_excelon
                return run_excelon(
                    dataset_path=dataset,
                    context=context
                )

            return self.core.run(mapped_goal, dataset)

        # 2️⃣ FUZZY MATCH
        corrected = self.auto_correct_goal(clean_goal)
        if corrected:
            mapped_goal = self.VALID_GOALS[corrected]
            print(f"[ROUTER] Auto-corrected '{clean_goal}' → '{corrected}'")

            if mapped_goal == "excelon":
                from tasks.auto_excelon import run_excelon
                return run_excelon(
                    dataset_path=dataset,
                    context=context
                )

            return self.core.run(mapped_goal, dataset)

        # 3️⃣ CLASSIFIER FALLBACK
        print("[ROUTER] No rule match → using classifier.")
        brain_mode = self.core.classify_goal(clean_goal)
        return self.core.run(brain_mode, dataset)

    # --------------------------------------------------------
    #  SAFE FALLBACK
    # --------------------------------------------------------
    def fallback(self, goal):
        return {
            "status": "error",
            "message": "Unrecognized task",
            "suggestion": self.auto_correct_goal(goal),
            "goal": goal
        }
