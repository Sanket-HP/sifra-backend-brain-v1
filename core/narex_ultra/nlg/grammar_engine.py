# engine/nlg/grammar_engine.py

# engine/nlg/grammar_engine.py

import re

class GrammarEngine:
    """
    Light grammar correction + readability enhancement.
    """

    def __init__(self):
        print("[NLG] Grammar Engine Ready")

    def clean(self, text: str) -> str:
        # Remove repeated words like "the the"
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text)

        # Fix spacing around punctuation
        text = re.sub(r"\s+([.,!?])", r"\1", text)

        # Remove double punctuation
        text = text.replace("..", ".")
        text = text.replace(",,", ",")

        # Remove stray symbols
        text = text.replace("→", "")

        # Fix repeated phrases produced by random selection
        text = text.replace("This gives a clearer view of", "This provides clarity on")
        text = text.replace("reflecting internal consistency reflecting internal consistency",
                            "reflecting internal consistency")

        # Clean spacing
        text = " ".join(text.split())

        return text.strip()

