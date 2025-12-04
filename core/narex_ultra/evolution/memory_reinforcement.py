# evolution/memory_reinforcement.py

import numpy as np

class MemoryReinforcement:
    """
    Strengthens memory stability across runs.
    Adds reinforcement signals based on evolving patterns.
    """

    def __init__(self):
        print("[NARE-X][M.R.E.] Memory Reinforcement Engine Ready")
        self.previous_signature = None

    def reinforce(self, memory_state):
        """
        Internal reinforcement logic (existing).
        """
        signature = np.array(memory_state["memory_signature"])
        if signature.size == 0:
            return {"reinforced": False}

        # If first run
        if self.previous_signature is None:
            self.previous_signature = signature
            return {
                "reinforced": True,
                "stability": 1.0,
                "note": "First memory reinforcement stored."
            }

        # Compute similarity ratio
        similarity = 1.0 - (np.linalg.norm(signature - self.previous_signature) /
                            (np.linalg.norm(self.previous_signature) + 1e-9))

        # Update stored signature
        self.previous_signature = signature

        return {
            "reinforced": True,
            "stability": round(float(similarity), 4),
            "note": "Memory reinforced with similarity tracking."
        }

    # 🔥 REQUIRED METHOD → NarexUltra calls this!
    def update(self, memory_state):
        """
        Wrapping function so NarexUltra can call update() safely.
        """
        return self.reinforce(memory_state)
