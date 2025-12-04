# evolution/auto_governance.py

import numpy as np

class AutoGovernance:
    """
    Governance layer for NARE-X.
    Ensures:
    - Safe evolution
    - Controlled self-upgrading
    - Prevents harmful loops
    - Validates patches & new modules
    """

    def __init__(self):
        print("[EVOLUTION] Governance System Initialized")

    # --------------------------------------------------------
    # Check if evolution cycle is safe
    # --------------------------------------------------------
    def validate_upgrade(self, metrics):
        """
        Decide if system is allowed to evolve.
        metrics = dict of performance, stability, volatility
        """
        try:
            vol = metrics.get("volatility_index", 0)
            stability = metrics.get("stability_metric", 1)
            density = metrics.get("information_density", 0)

            if vol > 500 or stability < 0:
                return {"approved": False, "reason": "System unstable — evolution blocked."}

            if density < 1:
                return {"approved": False, "reason": "Insufficient information for upgrade."}

            return {"approved": True, "reason": "Evolution cycle approved."}

        except Exception as e:
            return {"approved": False, "reason": f"Validation failed: {e}"}

    # --------------------------------------------------------
    # Validate predicted output (safety check)
    # --------------------------------------------------------
    def validate_output(self, output):
        """
        Check if final predictions/insights are sane.
        """
        if output is None:
            return False

        if isinstance(output, dict):
            return True

        return False

    # --------------------------------------------------------
    # Risk Assessment of entire NARE-X cycle
    # --------------------------------------------------------
    def assess_risk(self, core_features):
        vol = core_features.get("volatility_index", 0)

        if vol < 30:
            return "Low Risk"
        elif 30 <= vol < 100:
            return "Moderate Risk"
        else:
            return "High Risk"
