# ============================================================
#  SIFRA AI v8.1 ENTERPRISE (JSON DATASET EDITION)
#  AutoML + Synthetic LLM + Knowledge + Insights
#  FULLY FIXED VERSION
#  → FIXED create_model (true feature names from preprocessor)
#  → FIXED predict (raw feature alignment)
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

app = FastAPI(title="SIFRA AI Backend", version="8.1-JSON-STABLE-FULL")

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
# UNIVERSAL NORMALIZER v8.1
# ============================================================
def normalize_dataset(ds):
    print("\n========================")
    print("DEBUG: NORMALIZE DATASET")
    print("TYPE:", type(ds))
    print("PREVIEW:", str(ds)[:200])
    print("========================\n")

    try:
        # JSON {columns, data}
        if isinstance(ds, dict) and "columns" in ds and "data" in ds:
            print("[DEBUG] FE JSON (columns + data)")
            return pd.DataFrame(ds["data"], columns=ds["columns"])

        # List-of-lists
        if isinstance(ds, list) and len(ds) > 0 and isinstance(ds[0], list):
            print("[DEBUG] FE list-of-lists")
            cols = [f"col_{i+1}" for i in range(len(ds[0]))]
            return pd.DataFrame(ds, columns=cols)

        # CSV fallback
        if isinstance(ds, str) and "," in ds and "\n" in ds:
            print("[DEBUG] CSV fallback")
            return pd.read_csv(StringIO(ds))

        print("[DEBUG] Unsupported → empty DF")
        return pd.DataFrame()

    except Exception as e:
        print("[NORMALIZE ERROR]:", e)
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
# CREATE MODEL (AutoML) — FIXED WITH TRUE RAW FEATURES
# ============================================================
@app.post("/create_model")
async def create_model(request: Request):
    try:
        body = await request.json()

        print("\n========== /create_model ==========")
        print("BODY:", body)
        print("===================================\n")

        ds = body.get("dataset") or body.get("data") or body
        df = normalize_dataset(ds)

        print("[DEBUG] Normalized DF shape =", df.shape)

        if df.empty:
            return {"status": "fail", "detail": "Dataset empty or invalid"}

        print("[DEBUG] Running AutoML...")
        automl_raw = engine.run("automl_train", {"dataset": df})

        # unwrap internal object
        result = automl_raw.get("result", automl_raw)

        # --- DECODE PREPROCESSOR ---
        pre_hex = result.get("preprocessor_hex")
        if pre_hex:
            pre = pickle.loads(bytes.fromhex(pre_hex))

            # TRUE raw feature count expected by predictor
            raw_feature_count = pre.n_features_in_
            result["feature_count"] = raw_feature_count

            # TRUE raw feature names
            if hasattr(pre, "feature_names_in_"):
                raw_features = list(pre.feature_names_in_)
            else:
                raw_features = [f"feature_{i+1}" for i in range(raw_feature_count)]

            result["feature_names"] = raw_features

            print("[DEBUG] TRUE RAW FEATURE COUNT:", raw_feature_count)
            print("[DEBUG] TRUE RAW FEATURE NAMES:", raw_features)

        else:
            print("[ERROR] Preprocessor hex missing from AutoML result!")
            result["feature_names"] = list(df.columns[:-1])
            result["feature_count"] = len(df.columns) - 1

        print("[DEBUG] Final Feature Names Returned to FE:", result["feature_names"])

        return sanitize({
            "status": "success",
            "mode": "automl",
            "result": result
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"create_model failed: {str(e)}")


# ============================================================
# LLM CREATION
# ============================================================
@app.post("/create_llm")
async def create_llm(request: Request):
    try:
        body = await request.json()
        ds = body.get("dataset") or body.get("data")

        docs = body.get("documents")
        config = body.get("config", {})

        if docs is None:
            df = normalize_dataset(ds)
            docs = df_to_sentences(df)

        result = engine.run("create_llm", {"documents": docs, "config": config})

        global LLM_CACHE
        LLM_CACHE = result.get("llm_package")

        return {"status": "success", "mode": "synthetic_llm", "result": result}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"create_llm failed: {str(e)}")


# ============================================================
# LLM INFERENCE
# ============================================================
@app.post("/llm_inference")
async def llm_inference(request: Request):
    try:
        body = await request.json()

        llm_package = body.get("llm_package") or LLM_CACHE
        if llm_package is None:
            raise Exception("LLM Package missing. Use /create_llm first.")

        prompt = body.get("prompt") or body.get("message")
        if not prompt:
            raise Exception("Prompt missing")

        raw = engine.run("test_llm", {"llm_package": llm_package, "prompt": prompt})

        if isinstance(raw, str):
            raw = {"reply": raw}

        return {"status": "success", "response": raw}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"LLM inference failed: {str(e)}")


# ============================================================
# KNOWLEDGE GENERATOR
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
        raise HTTPException(500, f"dataset_to_knowledge failed: {str(e)}")


# ============================================================
# PREDICT — FULLY FIXED
# ============================================================
@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()

        print("\n========== /predict ==========")
        print("BODY:", body)
        print("==============================\n")

        model_hex = body.get("model_hex")
        pre_hex = body.get("preprocessor_hex")
        sample_list = body.get("sample") or body.get("features")

        if not model_hex:
            raise Exception("model_hex missing")
        if not pre_hex:
            raise Exception("preprocessor_hex missing")
        if sample_list is None:
            raise Exception("No sample or features provided")

        sample = np.array(sample_list).reshape(1, -1)

        model = pickle.loads(bytes.fromhex(model_hex))
        pre = pickle.loads(bytes.fromhex(pre_hex))

        expected = pre.n_features_in_

        print("[DEBUG] Expected raw feature count:", expected)
        print("[DEBUG] Provided sample count:", sample.shape[1])

        if sample.shape[1] != expected:
            raise Exception(f"Invalid feature count. Expected {expected}, got {sample.shape[1]}")

        X = pre.transform(sample)
        pred = model.predict(X)

        return {"status": "success", "prediction": pred.tolist()}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"predict failed: {str(e)}")


# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "version": "8.1-JSON-STABLE-FULL"}
