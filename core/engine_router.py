# ============================================================
#  ENGINE ROUTER v4.5 (UPDATED FOR SIFRA AI 4.5)
#
#  New Capabilities:
#    ✔ Smart Goal Classifier 3.0 (matches SifraCore)
#    ✔ CRE + DMAO aware routing
#    ✔ Auto-corrects similar goals (NLP fuzzy matching)
#    ✔ Multi-agent safe routing
#    ✔ Unified fallback handler
# ============================================================

import difflib
from core.sifra_core import SifraCore


class EngineRouter:
    """
    Routes user goals to the SIFRA Core Engine.
    Now enhanced for CRE + DMAO + ALL.
    """

    VALID_GOALS = {
        "analyze": "analytics",
        "analysis": "analytics",
        "auto_analyze": "analytics",

        "predict": "time_series",
        "prediction": "time_series",
        "auto_predict": "time_series",

        "forecast": "time_series",
        "future": "time_series",
        "auto_forecast": "time_series",

        "anomaly": "anomaly",
        "anomalies": "anomaly",
        "auto_anomaly": "anomaly",

        "insights": "analytics",
        "insight": "analytics",
        "auto_insights": "analytics",

        "trend": "analytics",
        "pattern": "analytics",
        "statistics": "analytics",

        # Additional intelligent mappings
        "model": "ml_model",
        "train_model": "ml_model",
        "build_model": "ml_model",
    }

    def __init__(self):
        self.core = SifraCore()
        print("[ENGINE ROUTER] SIFRA Router v4.5 Ready.")

    # --------------------------------------------------------
    #  NLP Fuzzy Matching
    # --------------------------------------------------------
    def auto_correct_goal(self, text):
        """
        Tries to fix unknown user goals (ex: 'anlyze' → 'analyze').
        """
        candidates = list(self.VALID_GOALS.keys())
        match = difflib.get_close_matches(text, candidates, n=1, cutoff=0.6)
        return match[0] if match else None

    # --------------------------------------------------------
    #  SMART ROUTER
    # --------------------------------------------------------
    def route(self, goal, dataset):
        """
        Routes tasks to SIFRA Core with deep brain support.
        """
        print(f"[ROUTER] Received goal: {goal}")

        if not goal or not isinstance(goal, str):
            return {
                "error": "Invalid goal format.",
                "goal": str(goal)
            }

        clean_goal = goal.lower().strip()

        # 1. DIRECT MATCH
        if clean_goal in self.VALID_GOALS:
            mapped_goal = clean_goal
            print(f"[ROUTER] Direct match → {mapped_goal}")
            return self.core.run(mapped_goal, dataset)

        # 2. FUZZY MATCH (auto-correction)
        corrected = self.auto_correct_goal(clean_goal)
        if corrected:
            print(f"[ROUTER] Auto-corrected '{clean_goal}' → '{corrected}'")
            return self.core.run(corrected, dataset)

        # 3. HIGH-LEVEL CLASSIFIER FALLBACK
        print("[ROUTER] No match found → using classifier.")
        brain_mode = self.core.classify_goal(clean_goal)
        return self.core.run(brain_mode, dataset)

    # --------------------------------------------------------
    #  SAFE DEFAULT HANDLER
    # --------------------------------------------------------
    def fallback(self, goal):
        return {
            "error": "Unrecognized task",
            "suggestion": f"Did you mean: {self.auto_correct_goal(goal)}?",
            "goal": goal
        }
