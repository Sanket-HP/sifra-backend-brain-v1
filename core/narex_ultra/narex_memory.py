# engine/narex_memory.py

import numpy as np


class NAREXMemory:
    """
    Pattern memory storage:
    - Stores last vector
    - Computes similarity
    """

    def __init__(self):
        self.last_signature = None
        print("[NARE-X] Memory Module Ready")

    def update(self, core_features):
        vector = np.array(list(core_features.values()), dtype=float)

        if self.last_signature is None:
            similarity = 1.0
        else:
            similarity = float(
                np.dot(self.last_signature, vector)
                / (np.linalg.norm(self.last_signature) * np.linalg.norm(vector) + 1e-9)
            )

        self.last_signature = vector

        return {
            "memory_signature": vector.tolist(),
            "similarity_to_previous": similarity
        }
