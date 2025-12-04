# engine/narex_ultra.py

import numpy as np
import pandas as pd

# -----------------------------------------------------------
# Base Intelligence Systems
# -----------------------------------------------------------
from .narex_core import NAREXCore
from .narex_memory import NAREXMemory
from .narex_agents import NAREXAgents
from .narex_strategy import NAREXStrategy
from .narex_utils import clean_numeric

# -----------------------------------------------------------
# Evolution Systems (relative imports)
# -----------------------------------------------------------
from .evolution.self_evolution_engine import SelfEvolutionEngine
from .evolution.meta_learning_loop import MetaLearningLoop
from .evolution.auto_governance import AutoGovernance
from .evolution.self_benchmarking import SelfBenchmark
from .evolution.memory_reinforcement import MemoryReinforcement
from .evolution.orchestrator import NAREXOrchestrator

# -----------------------------------------------------------
# Natural Language Generation Layer (NLG)
# -----------------------------------------------------------
from .nlg.response_builder import ResponseBuilder


class NarexUltra:
    """
    NARE-X ULTRA ENGINE
    Autonomous Multi-Agent + Self-Evolving Reasoning System
    With Integrated Natural Language Output (NLG).
    """

    def __init__(self):
        print("[NARE-X] Ultra Engine Initialized")

        # ---------------------------------------------------
        # Base Modules
        # ---------------------------------------------------
        print("[NARE-X] Loading Core Modules...")
        self.core = NAREXCore()
        self.memory = NAREXMemory()
        self.agents = NAREXAgents()
        self.strategy = NAREXStrategy()

        # ---------------------------------------------------
        # Evolution Engine Initialization
        # ---------------------------------------------------
        print("[NARE-X] Initializing Evolution System...")

        self.evolution = SelfEvolutionEngine()
        self.meta_learning = MetaLearningLoop()
        self.govern = AutoGovernance()
        self.benchmark = SelfBenchmark()
        self.reinforce = MemoryReinforcement()
        self.orchestrator = NAREXOrchestrator()

        # ---------------------------------------------------
        # NLG Engine Initialization
        # ---------------------------------------------------
        print("[NARE-X] Loading Natural Language Engine...")
        self.nlg = ResponseBuilder()

        print("[NARE-X] Evolution + NLG System Ready\n")

    # -----------------------------------------------------------------------
    # MAIN EXECUTION PIPELINE
    # -----------------------------------------------------------------------
    def run(self, data, mode=None):
        """
        Complete NARE-X ULTRA workflow:
        Includes multi-agent reasoning + evolution + NLG.

        mode = analyze / predict / forecast / anomaly / insights / model
        (Optional — used only for future reasoning upgrades)
        """

        try:
            # Convert input to DataFrame
            df = pd.DataFrame(data)
            numeric_df = clean_numeric(df)

            # ---------------------------------------------------
            # STEP 1 — Core Feature Extraction
            # ---------------------------------------------------
            core_features = self.core.extract(numeric_df)

            # ---------------------------------------------------
            # STEP 2 — Memory Engine
            # ---------------------------------------------------
            memory_state = self.memory.update(core_features)

            # ---------------------------------------------------
            # STEP 3 — Multi-Agent Reasoning
            # ---------------------------------------------------
            agent_outputs = self.agents.run(numeric_df, core_features)

            # ---------------------------------------------------
            # STEP 4 — Strategy Layer
            # ---------------------------------------------------
            strategy_out = self.strategy.generate(agent_outputs, memory_state)

            # ---------------------------------------------------
            # STEP 5 — Evolution + Meta Learning
            # ---------------------------------------------------
            benchmark_results = self.benchmark.benchmark(numeric_df)
            governance_decision = self.govern.validate_upgrade(core_features)
            memory_recall = self.reinforce.update(memory_state)
            orchestrator_plan = self.orchestrator.run(
                core_features,
                agent_outputs,
                memory_state
            )
            meta_out = self.meta_learning.run(
                core_features,
                agent_outputs,
                memory_state
            )
            evolution_output = self.evolution.run(
                core_features,
                agent_outputs,
                memory_state,
                orchestrator_plan,
                governance_decision,
                benchmark_results,
                meta_out
            )

            # ---------------------------------------------------
            # STEP 6 — Natural Language Output (NLG)
            # ---------------------------------------------------
            natural_language_output = self.nlg.generate({
                "core_features": core_features,
                "memory_state": memory_state,
                "agents": agent_outputs,
                "strategic_intelligence": strategy_out,
                "evolution": {
                    "governance": governance_decision,
                    "benchmark": benchmark_results,
                    "memory_reinforcement": memory_recall,
                    "orchestrator_plan": orchestrator_plan,
                    "meta_learning": meta_out,
                    "evolution_engine": evolution_output
                }
            })

            # ---------------------------------------------------
            # FINAL STRUCTURED OUTPUT
            # ---------------------------------------------------
            return {
                "status": "success",
                "core_features": core_features,
                "memory_state": memory_state,
                "agents": agent_outputs,
                "strategic_intelligence": strategy_out,

                "evolution": {
                    "governance": governance_decision,
                    "benchmark": benchmark_results,
                    "memory_reinforcement": memory_recall,
                    "orchestrator_plan": orchestrator_plan,
                    "meta_learning": meta_out,
                    "evolution_engine": evolution_output
                },

                "natural_language_response": natural_language_output
            }

        except Exception as e:
            return {
                "error": f"NARE-X error: {str(e)}"
            }
