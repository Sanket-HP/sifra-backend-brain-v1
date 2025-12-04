# engine/nlg/template_generator.py

import random

class TemplateGenerator:
    """
    Generates dynamic text templates based on numerical insights.
    NOT static — templates are adaptive and evolve with use.
    """

    def __init__(self):
        self.intro_phrases = [
            "Here’s what I discovered:",
            "After analyzing the pattern, this stands out:",
            "The data reveals something interesting:",
            "From the latest analysis, here's what I see:"
        ]

        self.trend_phrases = [
            "The trend shows a strong upward movement.",
            "There is a noticeable rise over time.",
            "The signal indicates progressive growth.",
            "The pattern points to consistent improvement."
        ]

        self.volatility_phrases = [
            "Volatility remains stable and moderate.",
            "The variance is slightly fluctuating.",
            "Data shows low instability overall.",
            "Values oscillate but stay controlled."
        ]

        self.future_phrases = [
            "Future movement is likely to continue in this direction.",
            "The upcoming pattern seems predictable.",
            "Forward projection indicates steady momentum.",
            "Upcoming data is expected to follow this trend."
        ]

    def build(self, trend_score, volatility):
        """Generate a dynamic, evolving template-like structure."""

        intro = random.choice(self.intro_phrases)
        trend = random.choice(self.trend_phrases)
        vol = random.choice(self.volatility_phrases)
        future = random.choice(self.future_phrases)

        return {
            "intro": intro,
            "trend": trend,
            "volatility": vol,
            "future": future,
            "trend_value": trend_score,
            "volatility_value": volatility
        }
