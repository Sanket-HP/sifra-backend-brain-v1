# engine/nlg/response_builder.py

from .auto_template_synthesizer import AutoTemplateSynthesizer
from .markov_writer import MarkovWriter
from .grammar_engine import GrammarEngine


class ResponseBuilder:
    """
    PURE NLG MODE (Mode A)
    Produces ONLY human-like natural language text.
    No raw data, no statistics block, no duplication.
    """

    def __init__(self):
        print("[NLG] Response Builder Ready")

        self.synth = AutoTemplateSynthesizer()
        self.markov = MarkovWriter()
        self.grammar = GrammarEngine()

    # ------------------------------------------------------------------
    # Main NLG Pipeline
    # ------------------------------------------------------------------
    def generate(self, payload):
        """
        payload = {
            core_features, memory_state, agents, strategic_intelligence, evolution
        }

        Returns only human-style text for SIFRA AI.
        """

        core = payload["core_features"]
        memory = payload["memory_state"]
        agents = payload["agents"]

        # 1. Auto-synthesize the explanation
        base_text = self.synth.generate(core, agents, memory)

        # 2. Markov smoothing (human-like flow)
        smoothed = self.markov.smooth(base_text)

        # 3. Grammar cleanup
        final_text = self.grammar.clean(smoothed)

        # OUTPUT: ONLY ONE NATURAL LANGUAGE MESSAGE
        return final_text
