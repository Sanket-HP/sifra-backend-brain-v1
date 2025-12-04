# engine/narex_strategy.py

class NAREXStrategy:
    """
    Strategic reasoning layer for NARE-X Ultra.
    Converts agent outputs into actionable decisions.
    """

    def __init__(self):
        print("[NARE-X] Strategy Layer Active")

    def safe_get_std(self, agents):
        """
        Safely extract standard deviation regardless of structure.
        """
        try:
            # Case 1: summary["0"]["std"]
            return agents["analysis"]["summary"]["0"]["std"]
        except:
            pass

        try:
            # Case 2: summary["std"]
            return agents["analysis"]["summary"]["std"]
        except:
            pass

        try:
            # Case 3: summary["0"] exists but has no std
            return agents["analysis"]["summary"]["0"].get("std", 0)
        except:
            pass

        # Default fallback
        return 0

    def generate(self, agents, memory):

        volatility = self.safe_get_std(agents)

        # ---------------------------
        # Decision Logic
        # ---------------------------
        if volatility < 5:
            decision = "Stable pattern detected — minimal change expected."
        elif volatility < 15:
            decision = "Moderate fluctuations — monitor trends closely."
        else:
            decision = "High volatility — take caution and evaluate anomalies."

        # Recommendation based on trend direction
        trend = agents.get("core", {}).get("trend_direction", "neutral")

        if trend == "up":
            rec = "Upward momentum — potential growth opportunity."
        elif trend == "down":
            rec = "Downward trend — consider risk mitigation."
        else:
            rec = "Neutral movement — maintain observation."

        return {
            "strategy": decision,
            "recommendation": rec,
            "memory_recalled": memory.get("related", []),
            "volatility": volatility
        }
