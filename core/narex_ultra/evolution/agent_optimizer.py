# engine/evolution/agent_optimizer.py

import numpy as np

class AgentOptimizer:
    """
    D.A.O. — Dynamic Agent Optimizer
    -------------------------------------
    Adjusts agent confidence & weights automatically.
    """

    def __init__(self):
        self.agent_weights = {
            "analysis": 1.0,
            "forecast": 1.0,
            "causality": 1.0,
            "insights": 1.0
        }
        print("[NARE-X][D.A.O.] Agent Optimizer Active")

    # ---------------------------------------------------------
    def evaluate_agents(self, agents_output):
        """
        Scoring each agent:
        - higher trend = stronger analysis
        - lower volatility = stronger forecast
        """
        scores = {}

        if "analysis" in agents_output:
            scores["analysis"] = abs(
                agents_output["analysis"].get("trend", 0)
            )

        if "forecast" in agents_output:
            f = agents_output["forecast"].get("forecast_values", [])
            if f:
                scores["forecast"] = np.std(f)

        if "insights" in agents_output:
            scores["insights"] = len(agents_output["insights"].get("insights", []))

        if "causality" in agents_output:
            scores["causality"] = 1  # small baseline

        return scores

    # ---------------------------------------------------------
    def optimize(self, agents_output):
        scores = self.evaluate_agents(agents_output)

        for agent, score in scores.items():
            self.agent_weights[agent] += np.tanh(score / 50)

        return {
            "new_agent_weights": self.agent_weights.copy(),
            "raw_scores": scores
        }
