# evolution/meta_learning_loop.py

import random
import numpy as np

class MetaLearningLoop:
    """
    Meta-learning engine for NARE-X.
    Learns from:
      - core features
      - agent outputs
      - past memory states
      - evolution results
    and improves the intelligence loop dynamically.
    """

    def __init__(self):
        print("[NARE-X][M.L.L.] Meta Learning Loop Ready")
        self.internal_weight = 1.0
        self.learning_rate = 0.05
        self.history = []

    def adapt_weights(self, core_features):
        """
        Learns based on volatility & direction.
        """
        vol = core_features.get("volatility_index", 0)
        direction = core_features.get("direction_score", 0)

        # Adjust weights based on observed patterns
        adjustment = (direction - (vol * 0.01)) * self.learning_rate
        self.internal_weight += adjustment

        # Clamp to avoid explosion
        self.internal_weight = max(0.1, min(self.internal_weight, 10))

        return round(float(self.internal_weight), 4)

    def remember(self, memory_state):
        """
        Stores memory signatures (like meta-memory).
        """
        signature = memory_state.get("memory_signature", [])
        if len(signature) > 0:
            self.history.append(signature)

        if len(self.history) > 50:
            self.history = self.history[-50:]  # limit memory

        return len(self.history)

    # 🔥 REQUIRED — NARE-X Ultra calls this
    def run(self, core_features, agent_outputs, memory_state):
        """
        Full meta-learning routine executed every cycle.
        """

        new_weight = self.adapt_weights(core_features)
        mem_len = self.remember(memory_state)

        return {
            "meta_weight": new_weight,
            "memory_history_length": mem_len,
            "note": "Meta-learning updated internal model weights."
        }
