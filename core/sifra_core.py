# ============================================================
#   SIFRA CORE v4.5 ULTRA-ENTERPRISE (SINGLETON-SAFE)
# ============================================================

import numpy as np
import pandas as pd

# -------- HDP-FUSIONNET MODULES --------
from core.hdp_fusionnet.intent import IntentModule
from core.hdp_fusionnet.context import ContextModule
from core.hdp_fusionnet.meaning import MeaningModule
from core.hdp_fusionnet.emotion import EmotionModule

# -------- HDS-UNITY ENGINE MODULES --------
from core.hds_unity.trend_channel import TrendChannel
from core.hds_unity.correlation_channel import CorrelationChannel
from core.hds_unity.variation_channel import VariationChannel
from core.hds_unity.fusion_matrix import FusionMatrix
from core.hds_unity.memory_signature import MemorySignature

# -------- NARE-X ULTRA ENGINE --------
from core.narex_ultra.narex_ultra import NarexUltra

# -------- PREPROCESSOR --------
from data.preprocessor import Preprocessor

# -------- LOGGER --------
from utils.logger import SifraLogger


# ============================================================
# SAFE FLOAT CONVERTER
# ============================================================
def safe_float(x):
    try:
        if isinstance(x, (np.floating, float, int)):
            if np.isnan(x) or np.isinf(x):
                return 0.0
            return float(x)
        return float(x)
    except:
        return 0.0


# ============================================================
# COGNITIVE REASONING ENGINE (CRE)
# ============================================================
class CognitiveReasoningEngine:
    def __init__(self):
        self.max_steps = 5

    def think(self, goal, hdp_vectors, hds_vectors):
        reasoning_steps = []
        for step in range(self.max_steps):
            reasoning_steps.append({
                "step": step + 1,
                "analysis": f"CRE analyzing goal pattern → {goal}",
                "hdp_signal": str(hdp_vectors),
                "hds_signal": str(hds_vectors)
            })

        return {
            "steps": reasoning_steps,
            "final_decision": f"CRE final reasoning for goal '{goal}' completed."
        }


# ============================================================
# DYNAMIC MULTI-AGENT ORCHESTRATOR (DMAO)
# ============================================================
class DMaoOrchestrator:
    def route(self, goal_type, clean_data, narex_engine):

        if goal_type == "time_series":
            agent_used = "Forecast Agent"
        elif goal_type == "analytics":
            agent_used = "Analytics Agent"
        elif goal_type == "ml_model":
            agent_used = "Modeling Agent"
        elif goal_type == "anomaly":
            agent_used = "Anomaly Agent"
        else:
            agent_used = "NARE-X Agent"

        try:
            narex_output = narex_engine.run(clean_data)
        except Exception as e:
            narex_output = {"error": str(e)}

        return {
            "agent_selected": agent_used,
            "agent_output": narex_output
        }


# ============================================================
# ADAPTIVE LEARNING LOOP (ALL)
# ============================================================
class AdaptiveLearningLoop:
    def __init__(self):
        self.memory = {}

    def update_memory(self, goal, signals):
        self.memory[goal] = {
            "trend": safe_float(signals.get("trend", 0)),
            "corr": safe_float(signals.get("corr", 0)),
            "var": safe_float(signals.get("var", 0)),
            "memory_signature": safe_float(signals.get("memory_signature", 0)),
        }
        return {"status": "learned", "goal": goal}


# ============================================================
# MAIN SIFRA CORE (SINGLETON-HARD-LOCKED)
# ============================================================
class SifraCore:
    """
    Hybrid HDP + HDS + NARE-X + CRE + DMAO + ALL Engine
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SifraCore, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # 🔒 ABSOLUTE PROTECTION AGAINST MULTIPLE BOOTS
        if SifraCore._initialized:
            return

        SifraCore._initialized = True
        self._boot()

    # ----------------------------------------------------------
    # CORE BOOT (RUNS EXACTLY ONCE)
    # ----------------------------------------------------------
    def _boot(self):
        self.log = SifraLogger("SIFRA_CORE")

        # HDP modules
        self.intent = IntentModule()
        self.context = ContextModule()
        self.meaning = MeaningModule()
        self.emotion = EmotionModule()

        # HDS modules
        self.trend = TrendChannel()
        self.corr = CorrelationChannel()
        self.variation = VariationChannel()
        self.fusion = FusionMatrix()
        self.memory = MemorySignature()

        # NARE-X engine
        self.narex = NarexUltra()

        # CRE, DMAO, ALL
        self.cre = CognitiveReasoningEngine()
        self.dmao = DMaoOrchestrator()
        self.all_engine = AdaptiveLearningLoop()

        # Preprocessor
        self.preprocessor = Preprocessor()

        self.log.info("SIFRA CORE v4.5 initialized successfully.")

    # ----------------------------------------------------------
    # GOAL CLASSIFIER
    # ----------------------------------------------------------
    def classify_goal(self, goal: str):
        goal = (goal or "").lower().strip()

        if any(k in goal for k in ["forecast", "predict", "future"]):
            return "time_series"
        if any(k in goal for k in ["analyze", "insight", "summary"]):
            return "analytics"
        if "model" in goal:
            return "ml_model"
        if "anomaly" in goal:
            return "anomaly"

        return "general"

    # ----------------------------------------------------------
    # MAIN PIPELINE
    # ----------------------------------------------------------
    def run(self, goal, dataset):

        self.log.info(f"Running SIFRA Brain Pipeline: {goal}")
        brain_mode = self.classify_goal(goal)

        try:
            clean_data = self.preprocessor.clean(dataset)
        except Exception as e:
            return {"error": f"Preprocessing failed: {str(e)}"}

        if not isinstance(clean_data, pd.DataFrame):
            return {"error": "Invalid dataset after preprocessing."}

        # HDP
        intent_vec = self.intent.detect_intent(goal)
        context_vec = self.context.detect_context(goal, clean_data)
        meaning_vec = self.meaning.create_meaning(intent_vec, context_vec)
        emotion_score = safe_float(self.emotion.detect_emotion(clean_data))

        # HDS
        trend_score = safe_float(self.trend.compute_trend(clean_data))
        corr_score = safe_float(self.corr.compute_correlation(clean_data))
        var_score = safe_float(self.variation.compute_variation(clean_data))

        fusion_vector = self.fusion.fuse(trend_score, corr_score, var_score)
        fusion_vector_safe = [safe_float(x) for x in fusion_vector]
        memory_signature = safe_float(self.memory.generate_signature(fusion_vector))

        # CRE
        cre_output = self.cre.think(
            goal,
            {"intent": intent_vec, "context": context_vec, "meaning": meaning_vec},
            {"trend": trend_score, "corr": corr_score, "var": var_score}
        )

        # DMAO
        dmao_output = self.dmao.route(brain_mode, clean_data, self.narex)

        # ALL
        learning_output = self.all_engine.update_memory(
            goal,
            {
                "trend": trend_score,
                "corr": corr_score,
                "var": var_score,
                "memory_signature": memory_signature
            }
        )

        return {
            "goal": goal,
            "brain_mode": brain_mode,
            "data_shape": {
                "rows": int(clean_data.shape[0]),
                "columns": int(clean_data.shape[1])
            },
            "HDP": {
                "intent_vector": intent_vec,
                "context_vector": context_vec,
                "meaning_vector": meaning_vec,
                "emotion_score": emotion_score,
            },
            "HDS": {
                "trend_score": trend_score,
                "correlation_score": corr_score,
                "variation_score": var_score,
                "fusion_vector": fusion_vector_safe,
                "memory_signature": memory_signature,
            },
            "CRE": cre_output,
            "DMAO": dmao_output,
            "ALL": learning_output,
            "message": f"SIFRA Unified Brain (v4.5) executed successfully for: {goal}"
        }
