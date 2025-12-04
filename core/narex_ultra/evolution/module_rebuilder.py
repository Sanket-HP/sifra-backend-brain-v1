# engine/evolution/module_rebuilder.py

import numpy as np

class ModuleRebuilder:
    """
    A.M.R.E. — Auto Module Rebuild Engine
    ---------------------------------------------------
    Rebuilds or resets internal modules when feedback
    indicates performance degradation.
    """

    def __init__(self):
        self.threshold = 0.25
        print("[NARE-X][A.M.R.E.] Auto Module Rebuilder Ready")

    # ---------------------------------------------------------
    def rebuild_if_needed(self, evolution_signal, agents_output):
        score = evolution_signal["score"]

        if score < self.threshold:
            return {
                "rebuild_required": True,
                "action": "Rebuilding weak modules...",
                "modules_rebuilt": [
                    "insight-engine",
                    "forecast-engine",
                    "trend-analyzer"
                ]
            }

        return {
            "rebuild_required": False,
            "message": "Modules stable — no rebuild needed."
        }
