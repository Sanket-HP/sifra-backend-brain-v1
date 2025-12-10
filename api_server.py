# ============================================================
#  SIFRA AI v10.1 ENTERPRISE (COGNITIVE + HUMAN LLM EDITION)
#  AutoML • Cognitive RAG • Human Reply LLM • Brain Pipeline
# ============================================================

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import traceback
import pandas as pd
import pickle
import math
from io import StringIO

from core.sifra_unified import SIFRAUnifiedEngine
from core.sifra_llm_engine import SifraLLMEngine
from tasks.dataset_to_knowledge import df_to_sentences


# ------------------------------------------------------------
# FASTAPI INSTANCE
# ------------------------------------------------------------
app = FastAPI(
    title="SIFRA AI Backend",
    version="10.1-Human-Cognitive",
    description="SIFRA Enterprise Cognitive Engine v10.1 (AutoML + LLM + Brain)"
)

# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# GLOBAL ENGINE OBJECTS
# ------------------------------------------------------------
engine = SIFRAUnifiedEngine()
llm_engine = SifraLLMEngine()
LLM_CACHE = None


# ------------------------------------------------------------
# NORMALIZE DATASET (Strongest Version)
# ------------------------------------------------------------
def normalize_dataset(ds):
    try:
        # JSON structured dataset
        if isinstance(ds, dict) and "columns" in ds and "data" in ds:
            cols = ds["columns"]
            fixed = []
            for row in ds["data"]:
                # trim / pad rows safely
                if len(row) < len(cols):
                    row = row + [None] * (len(cols) - len(row))
                elif len(row) > len(cols):
                    row = row[:len(cols)]
                fixed.append(row)
            return pd.DataFrame(fixed, columns=cols)

        # CSV string
        if isinstance(ds, str) and "," in ds and "\n" in ds:
            return pd.read_csv(StringIO(ds))

        # List of lists
        if isinstance(ds, list) and len(ds) and isinstance(ds[0], list):
            longest = max(len(r) for r in ds)
            cols = [f"col_{i+1}" for i in range(longest)]
            fixed = []
            for r in ds:
                if len(r) < longest:
                    r = r + [None] * (longest - len(r))
                elif len(r) > longest:
                    r = r[:longest]
                fixed.append(r)
            return pd.DataFrame(fixed, columns=cols)

        return pd.DataFrame()

    except:
        traceback.print_exc()
        return pd.DataFrame()


# ------------------------------------------------------------
# SANITIZER
# ------------------------------------------------------------
def sanitize(v):
    if isinstance(v, float):
        return 0 if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, dict):
        return {k: sanitize(x) for k, x in v.items()}
    if isinstance(v, list):
        return [sanitize(x) for x in v]
    return v


# ============================================================
# CREATE MODEL (AutoML)
# ============================================================
@app.post("/create_model")
async def create_model(request: Request):
    try:
        body = await request.json()
        df = normalize_dataset(body.get("dataset") or body.get("data"))

        if df.empty or df.shape[1] < 2:
            return {"status": "fail", "detail": "Dataset invalid"}

        automl = engine.run("automl_train", {"dataset": df})
        result = automl.get("result", automl)

        # Decode preprocessor metadata
        pre_hex = result.get("preprocessor_hex")
        if pre_hex:
            pre = pickle.loads(bytes.fromhex(pre_hex))
            try:
                result["feature_count"] = pre.n_features_in_
                result["feature_names"] = (
                    list(pre.feature_names_in_)
                    if hasattr(pre, "feature_names_in_")
                    else [f"feature_{i+1}" for i in range(pre.n_features_in_)]
                )
            except:
                result["feature_names"] = list(df.columns[:-1])
                result["feature_count"] = len(result["feature_names"])
        else:
            result["feature_names"] = list(df.columns[:-1])
            result["feature_count"] = len(result["feature_names"])

        return sanitize({
            "status": "success",
            "mode": "automl",
            "result": result
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/create_model failed: {str(e)}")


# ============================================================
# CREATE LLM (Synthetic + Human Mode)
# ============================================================
@app.post("/create_llm")
async def create_llm(request: Request):
    try:
        body = await request.json()
        docs = body.get("documents")
        config = body.get("config", {})

        if docs is None:
            df = normalize_dataset(body.get("dataset") or body.get("data"))
            docs = df_to_sentences(df)

        if isinstance(docs, str):
            docs = [x.strip() for x in docs.split("\n") if x.strip()]

        result = engine.run("create_llm", {
            "documents": docs,
            "config": config
        })

        global LLM_CACHE
        LLM_CACHE = result.get("llm_package")

        return {"status": "success", "mode": "synthetic_llm", "result": result}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/create_llm failed: {str(e)}")


# ============================================================
# LLM INFERENCE (Human LLM + RAG)
# ============================================================
@app.post("/llm_inference")
async def llm_inference(request: Request):
    try:
        body = await request.json()

        llm_package = body.get("llm_package") or LLM_CACHE
        if not llm_package:
            raise Exception("LLM Package missing")

        prompt = body.get("prompt") or body.get("query")
        if not prompt:
            raise Exception("Prompt missing")

        response = engine.run("test_llm", {
            "llm_package": llm_package,
            "prompt": prompt
        })

        return {"status": "success", "response": response}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/llm_inference failed: {str(e)}")


# ============================================================
# DATASET → KNOWLEDGE SENTENCES
# ============================================================
@app.post("/dataset_to_knowledge")
async def dataset_to_knowledge(request: Request):
    try:
        body = await request.json()
        df = normalize_dataset(body.get("dataset") or body.get("data"))

        if df.empty:
            return {"status": "success", "sentences": []}

        return {"status": "success", "sentences": df_to_sentences(df)}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/dataset_to_knowledge failed: {str(e)}")


# ============================================================
# BRAIN PIPELINE (Final FE-Compatible Output)
# ============================================================
@app.post("/run")
async def run_brain(request: Request):
    try:
        body = await request.json()
        mode = (body.get("mode") or "").lower()

        df = normalize_dataset(body.get("dataset") or body.get("data"))
        if df.empty:
            raise Exception("Dataset empty")

        mode_map = {
            "analyze": "Dataset analysis",
            "visualize": "Create visualization plan",
            "forecast": "Forecast future values",
            "anomaly": "Detect anomalies",
            "insights": "Extract insights"
        }

        query = mode_map.get(mode)
        if not query:
            raise Exception(f"Unknown mode '{mode}'")

        # FULL FE-compatible object returned directly
        return engine.run("brain_pipeline", {
            "dataset": df,
            "query": query
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/run failed: {str(e)}")


# ============================================================
# PREDICT (AutoML Model Inference)
# ============================================================
@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()

        model_hex = body.get("model_hex")
        pre_hex = body.get("preprocessor_hex")
        sample = body.get("sample") or body.get("features")

        if not model_hex or not pre_hex:
            raise Exception("Model or Preprocessor missing")

        model = pickle.loads(bytes.fromhex(model_hex))
        pre = pickle.loads(bytes.fromhex(pre_hex))

        sample_df = pd.DataFrame([sample])
        X = pre.transform(sample_df)
        pred = model.predict(X).tolist()

        return {"status": "success", "prediction": pred}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/predict failed: {str(e)}")


# ============================================================
# HEALTH
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "version": "10.1-Human-Cognitive"}
