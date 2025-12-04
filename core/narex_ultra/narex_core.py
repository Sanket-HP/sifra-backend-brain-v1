# engine/narex_core.py

import numpy as np
import pandas as pd


class NAREXCore:
    """
    Core reasoning engine:
    Extracts the 5 core intelligence features:
    - Intent Signal
    - Volatility Index
    - Direction Score
    - Stability Metric
    - Information Density
    """

    def __init__(self):
        print("[NARE-X] Core Engine Ready")

    def extract(self, df):
        arr = df.iloc[:, 0].values

        return {
            "intent_signal": float(np.mean(arr)),
            "volatility_index": float(np.std(arr)),
            "direction_score": float(np.mean(np.diff(arr))),
            "stability_metric": float(1 / (np.std(arr) + 1e-9)),
            "information_density": float(np.var(arr)),
        }
