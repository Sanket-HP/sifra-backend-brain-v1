# engine/nlg/phrase_mutation.py

import random

class PhraseMutation:
    """
    Mutates final generated text:
    - Random synonyms
    - Style shifting
    - Adds human-like variation
    """

    synonym_map = {
        "strong": ["powerful", "significant", "notable"],
        "stable": ["consistent", "steady", "balanced"],
        "future": ["upcoming", "forthcoming", "expected"],
        "trend": ["pattern", "signal", "trajectory"],
        "growth": ["increase", "rise", "expansion"],
    }

    def mutate(self, text: str) -> str:
        words = text.split()
        for i, w in enumerate(words):
            key = w.lower()
            if key in self.synonym_map and random.random() < 0.25:
                words[i] = random.choice(self.synonym_map[key])
        return " ".join(words)
