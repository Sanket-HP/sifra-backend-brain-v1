# ============================================================
#  SIFRA AI v10.0 ENTERPRISE (COGNITIVE ENGINE EDITION)
#  AutoML + LLM + Brain Pipeline + Knowledge + Insights
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
    version="10.0-Cognitive-Enterprise",
    description="SIFRA Enterprise Cognitive Engine (AutoML + LLM + Brain)"
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
# GLOBAL ENGINES
# ------------------------------------------------------------
engine = SIFRAUnifiedEngine()
llm_engine = SifraLLMEngine()
LLM_CACHE = None


# ------------------------------------------------------------
# NORMALIZE DATASET (robust)
# ------------------------------------------------------------
def normalize_dataset(ds):
    try:
        if isinstance(ds, dict) and "columns" in ds and "data" in ds:
            cols = ds["columns"]
            fixed = []
            for row in ds["data"]:
                if len(row) > len(cols):
                    row = row[:len(cols)]
                elif len(row) < len(cols):
                    row = row + [None] * (len(cols) - len(row))
                fixed.append(row)
            return pd.DataFrame(fixed, columns=cols)

        if isinstance(ds, str) and "," in ds and "\n" in ds:
            return pd.read_csv(StringIO(ds))

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
        if math.isnan(v) or math.isinf(v):
            return 0
        return v
    if isinstance(v, dict):
        return {k: sanitize(x) for k, x in v.items()}
    if isinstance(v, list):
        return [sanitize(x) for x in v]
    return v


# ============================================================
# CREATE MODEL
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
# CREATE LLM
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

        result = engine.run("create_llm", {"documents": docs, "config": config})

        global LLM_CACHE
        LLM_CACHE = result.get("llm_package")

        return {"status": "success", "mode": "synthetic_llm", "result": result}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/create_llm failed: {str(e)}")


# ============================================================
# LLM INFERENCE
# ============================================================
@app.post("/llm_inference")
async def llm_inference(request: Request):
    try:
        body = await request.json()

        llm_package = body.get("llm_package") or LLM_CACHE
        if not llm_package:
            raise Exception("LLM package missing")

        prompt = body.get("prompt") or body.get("query")
        if not prompt:
            raise Exception("Prompt missing")

        raw = engine.run("test_llm", {
            "llm_package": llm_package,
            "prompt": prompt
        })

        return {"status": "success", "response": raw}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/llm_inference failed: {str(e)}")


# ============================================================
# DATASET → KNOWLEDGE
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
# BRAIN PIPELINE (FIXED — NO DOUBLE WRAP)
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
        if query is None:
            raise Exception(f"Unknown analysis mode '{mode}'")

        # NEW: engine already returns full FE-compatible object
        unified = engine.run("brain_pipeline", {"dataset": df, "query": query})

        # unified = {
        #   "status": "success",
        #   "response": {summary, visuals, insights, ai_explanation, raw_brain}
        # }

        return unified   # DO NOT WRAP AGAIN

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/run failed: {str(e)}")


# ============================================================
# PREDICT
# ============================================================
@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()

        model_hex = body.get("model_hex")
        pre_hex = body.get("preprocessor_hex")
        sample = body.get("sample") or body.get("features")

        if not model_hex or not pre_hex:
            raise Exception("Model or preprocessor missing")

        model = pickle.loads(bytes.fromhex(model_hex))
        pre = pickle.loads(bytes.fromhex(pre_hex))

        df = pd.DataFrame([sample])

        X = pre.transform(df)
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
    return {"status": "ok", "version": "10.0-Cognitive-Enterprise"}
