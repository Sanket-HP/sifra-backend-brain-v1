# ============================================================
#  SIFRA AI v8.1 ENTERPRISE (JSON DATASET EDITION)
#  AutoML + LLM + Knowledge + Insights
#  FULL FIXED VERSION FOR RENDER + GITHUB
# ============================================================

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import traceback
import pandas as pd
import numpy as np
import pickle
import math
from io import StringIO

# Core Engines
from core.sifra_unified import SIFRAUnifiedEngine
from core.sifra_llm_engine import SifraLLMEngine
from tasks.dataset_to_knowledge import df_to_sentences

# FastAPI Init
app = FastAPI(
    title="SIFRA AI Backend",
    version="8.1-JSON-STABLE-FULL",
    description="SIFRA Enterprise AutoML + LLM Engine"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SIFRAUnifiedEngine()
llm_engine = SifraLLMEngine()

LLM_CACHE = None


# ============================================================
# HOME ROUTE (Fixes Render 404)
# ============================================================
@app.get("/")
def index():
    return {
        "status": "running",
        "message": "SIFRA AI Backend Online",
        "version": "8.1-JSON-STABLE-FULL"
    }


# ============================================================
# UNIVERSAL NORMALIZER
# ============================================================
def normalize_dataset(ds):

    try:
        if isinstance(ds, dict) and "columns" in ds and "data" in ds:
            return pd.DataFrame(ds["data"], columns=ds["columns"])

        if isinstance(ds, list) and len(ds) > 0 and isinstance(ds[0], list):
            cols = [f"col_{i+1}" for i in range(len(ds[0]))]
            return pd.DataFrame(ds, columns=cols)

        if isinstance(ds, str) and "," in ds and "\n" in ds:
            return pd.read_csv(StringIO(ds))

        return pd.DataFrame()

    except Exception:
        traceback.print_exc()
        return pd.DataFrame()


# ============================================================
# SANITIZER
# ============================================================
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
# CREATE MODEL (AutoML)
# ============================================================
@app.post("/create_model")
async def create_model(request: Request):

    try:
        body = await request.json()

        ds = body.get("dataset") or body.get("data") or body
        df = normalize_dataset(ds)

        if df.empty:
            return {"status": "fail", "detail": "Dataset empty or invalid"}

        automl_raw = engine.run("automl_train", {"dataset": df})

        # unwrap AutoML wrapper
        result = automl_raw.get("result", automl_raw)

        # decode preprocessor to extract true raw feature names
        pre_hex = result.get("preprocessor_hex")
        if pre_hex:
            pre = pickle.loads(bytes.fromhex(pre_hex))

            # TRUE COUNT of features
            result["feature_count"] = pre.n_features_in_

            # TRUE RAW feature names
            if hasattr(pre, "feature_names_in_"):
                result["feature_names"] = list(pre.feature_names_in_)
            else:
                result["feature_names"] = [
                    f"feature_{i+1}" for i in range(pre.n_features_in_)
                ]

        else:
            # fallback
            result["feature_names"] = list(df.columns[:-1])
            result["feature_count"] = len(df.columns) - 1

        return sanitize({
            "status": "success",
            "mode": "automl",
            "result": result
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/create_model failed: {str(e)}")


# ============================================================
# LLM CREATION
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
            raise Exception("LLM Package missing")

        prompt = body.get("prompt") or body.get("message") or body.get("query")
        if not prompt:
            raise Exception("Prompt missing")

        raw = engine.run("test_llm", {"llm_package": llm_package, "prompt": prompt})

        if isinstance(raw, str):
            raw = {"reply": raw}

        return {"status": "success", "response": raw}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/llm_inference failed: {str(e)}")


# ============================================================
# KNOWLEDGE GENERATION
# ============================================================
@app.post("/dataset_to_knowledge")
async def dataset_to_knowledge(request: Request):

    try:
        body = await request.json()

        df = normalize_dataset(body.get("dataset") or body.get("data"))

        if df.empty:
            return {"status": "success", "sentences": []}

        sentences = df_to_sentences(df)

        return {"status": "success", "sentences": sentences}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/dataset_to_knowledge failed: {str(e)}")


# ============================================================
# PREDICT (Fixed)
# ============================================================
@app.post("/predict")
async def predict(request: Request):

    try:
        body = await request.json()

        model_hex = body.get("model_hex")
        pre_hex = body.get("preprocessor_hex")
        sample_list = body.get("sample") or body.get("features")

        if not model_hex:
            raise Exception("model_hex missing")
        if not pre_hex:
            raise Exception("preprocessor_hex missing")
        if sample_list is None:
            raise Exception("Sample missing")

        # convert to DataFrame — FIXES your previous error
        sample_df = pd.DataFrame([sample_list])

        model = pickle.loads(bytes.fromhex(model_hex))
        pre = pickle.loads(bytes.fromhex(pre_hex))

        if sample_df.shape[1] != pre.n_features_in_:
            raise Exception(
                f"Invalid input size. Expected {pre.n_features_in_}, got {sample_df.shape[1]}"
            )

        # transform + predict
        X = pre.transform(sample_df)
        pred = model.predict(X)

        return {"status": "success", "prediction": pred.tolist()}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/predict failed: {str(e)}")


# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "version": "8.1-JSON-STABLE-FULL"}
