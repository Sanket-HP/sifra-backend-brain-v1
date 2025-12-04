import random

# engine/nlg/markov_writer.py

class MarkovWriter:
    """
    Minimal smoother. 
    NO random rewriting.
    Only removes duplicates + makes flow cleaner.
    """

    def __init__(self):
        print("[NLG] Markov Writer Ready")

    def smooth(self, text: str) -> str:
        """
        Removes:
        - repeated lines
        - repeated fragments
        - accidental duplicates
        """
        if not text:
            return text

        lines = [l.strip() for l in text.split("\n") if l.strip()]

        cleaned = []
        seen = set()

        for line in lines:
            if line not in seen:
                cleaned.append(line)
                seen.add(line)

        # Join intelligently
        return " ".join(cleaned)

