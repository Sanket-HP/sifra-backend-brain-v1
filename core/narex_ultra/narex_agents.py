# engine/narex_agents.py

import numpy as np
import pandas as pd


class NAREXAgents:
    """
    Multi-agent reasoning layer:
    1. Analysis Agent
    2. Causality Agent
    3. Forecast Agent
    4. Insights Agent
    5. Strategy Agent Helper
    """

    def __init__(self):
        print("[NARE-X] Agents Loaded")

    def run(self, df, core):
        return {
            "analysis": self.analysis_agent(df),
            "causality": self.causality_agent(df),
            "forecast": self.forecast_agent(df),
            "insights": self.insights_agent(df),
        }

    # ----------------------------------------------------
    # 1️⃣ Analysis Agent
    # ----------------------------------------------------
    def analysis_agent(self, df):
        desc = df.describe().to_dict()
        trend = float((df.diff().mean()).fillna(0).mean())
        return {
            "summary": desc,
            "trend": trend
        }

    # ----------------------------------------------------
    # 2️⃣ Causality Agent
    # ----------------------------------------------------
    def causality_agent(self, df):
        if df.shape[1] > 1:
            corr = df.corr().to_dict()
        else:
            corr = "Not enough columns"

        return {
            "correlations": corr
        }

    # ----------------------------------------------------
    # 3️⃣ Forecast Agent
    # ----------------------------------------------------
    def forecast_agent(self, df, steps=5):
        series = df.iloc[:, 0].values
        slope = np.mean(np.diff(series))

        future = [float(series[-1] + slope * (i + 1)) for i in range(steps)]

        return {
            "forecast_steps": steps,
            "forecast_values": future
        }

    # ----------------------------------------------------
    # 4️⃣ Insights Agent
    # ----------------------------------------------------
    def insights_agent(self, df):
        arr = df.iloc[:, 0].values
        insights = [
            f"Average value is {arr.mean():.2f}",
            f"Volatility (std) = {arr.std():.2f}",
            f"Growth trend = {np.mean(np.diff(arr)):.2f}",
        ]
        return {"insights": insights}
