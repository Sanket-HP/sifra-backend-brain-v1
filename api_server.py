# ============================================================
#  SIFRA AI v10.3 ENTERPRISE (COGNITIVE + EXCELON EDITION)
#  AutoML • Cognitive RAG • Human LLM • Excelon™
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import traceback
import pandas as pd
import pickle
import math
from io import StringIO
from typing import Any, Optional, List
import os

from pydantic import BaseModel

from core.sifra_unified import SIFRAUnifiedEngine
from core.sifra_llm_engine import SifraLLMEngine
from tasks.dataset_to_knowledge import df_to_sentences
from tasks.auto_excelon import run_excelon


# ------------------------------------------------------------
# FASTAPI INSTANCE
# ------------------------------------------------------------
app = FastAPI(
    title="SIFRA AI Backend",
    version="10.3-Enterprise-Excelon",
    description="SIFRA Enterprise Engine v10.3 (AutoML + LLM + Excelon)"
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


# ============================================================
# Pydantic Models (IMPORTANT)
# ============================================================

class ExcelonRequest(BaseModel):
    dataset_path: str
    context: Optional[dict] = {}


class ExcelonResponse(BaseModel):
    status: str
    engine: str
    file_name: str
    file_path: str
    sheets_created: List[str]


class CreateLLMRequest(BaseModel):
    documents: Optional[Any] = None
    dataset: Optional[Any] = None
    data: Optional[Any] = None
    config: Optional[dict] = {}


class LLMInferenceRequest(BaseModel):
    llm_package: Optional[Any] = None
    prompt: Optional[str] = None
    query: Optional[str] = None


class CreateModelRequest(BaseModel):
    dataset: Optional[Any] = None
    data: Optional[Any] = None


# ------------------------------------------------------------
# NORMALIZE DATASET
# ------------------------------------------------------------
def normalize_dataset(ds):
    try:
        if isinstance(ds, dict) and "columns" in ds and "data" in ds:
            cols = ds["columns"]
            fixed = []
            for row in ds["data"]:
                row = row + [None] * (len(cols) - len(row))
                fixed.append(row[:len(cols)])
            return pd.DataFrame(fixed, columns=cols)

        if isinstance(ds, str) and "," in ds and "\n" in ds:
            return pd.read_csv(StringIO(ds))

        if isinstance(ds, list) and ds and isinstance(ds[0], list):
            longest = max(len(r) for r in ds)
            cols = [f"col_{i+1}" for i in range(longest)]
            fixed = []
            for r in ds:
                r = r + [None] * (longest - len(r))
                fixed.append(r[:longest])
            return pd.DataFrame(fixed, columns=cols)

        return pd.DataFrame()

    except:
        traceback.print_exc()
        return pd.DataFrame()


# ------------------------------------------------------------
# JSON SAFE SANITIZER
# ------------------------------------------------------------
def json_safe(obj: Any):
    try:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, float):
            return 0 if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {str(k): json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_safe(x) for x in obj]
        if isinstance(obj, tuple):
            return [json_safe(x) for x in obj]
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)
    except:
        return str(obj)


# ============================================================
# CREATE MODEL (AUTOML)
# ============================================================
@app.post("/create_model")
async def create_model(req: CreateModelRequest):
    try:
        df = normalize_dataset(req.dataset or req.data)

        if df.empty or df.shape[1] < 2:
            return {"status": "fail", "detail": "Dataset invalid"}

        automl = engine.run("automl_train", {"dataset": df})
        result = automl.get("result", automl)

        pre_hex = result.get("preprocessor_hex")
        if pre_hex:
            pre = pickle.loads(bytes.fromhex(pre_hex))
            result["feature_names"] = list(df.columns[:-1])
            result["feature_count"] = len(result["feature_names"])

        return json_safe({
            "status": "success",
            "mode": "automl",
            "result": result
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/create_model failed: {str(e)}")


# ============================================================
# EXCELON™ – CREATE EXCEL REPORT (FIXED)
# ============================================================
@app.post("/excelon/create", response_model=ExcelonResponse)
async def create_excelon(req: ExcelonRequest):
    try:
        if not os.path.exists(req.dataset_path):
            raise Exception("Dataset path does not exist")

        result = run_excelon(
            dataset_path=req.dataset_path,
            context=req.context
        )

        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/excelon/create failed: {str(e)}")


# ============================================================
# CREATE LLM (HUMAN + RAG)
# ============================================================
@app.post("/create_llm")
async def create_llm(req: CreateLLMRequest):
    try:
        docs = req.documents

        if docs is None:
            df = normalize_dataset(req.dataset or req.data)
            docs = df_to_sentences(df)

        if isinstance(docs, str):
            docs = [x.strip() for x in docs.split("\n") if x.strip()]

        result = engine.run("create_llm", {
            "documents": docs,
            "config": req.config or {}
        })

        global LLM_CACHE
        LLM_CACHE = result.get("llm_package")

        return json_safe({
            "status": "success",
            "mode": "synthetic_llm",
            "result": result
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/create_llm failed: {str(e)}")


# ============================================================
# LLM INFERENCE
# ============================================================
@app.post("/llm_inference")
async def llm_inference(req: LLMInferenceRequest):
    try:
        llm_package = req.llm_package or LLM_CACHE
        prompt = req.prompt or req.query

        if not llm_package or not prompt:
            raise Exception("LLM package or prompt missing")

        response = engine.run("test_llm", {
            "llm_package": llm_package,
            "prompt": prompt
        })

        return json_safe({
            "status": "success",
            "response": response
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/llm_inference failed: {str(e)}")


# ============================================================
# HEALTH
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "version": "10.3-Enterprise-Excelon"}
