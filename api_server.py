# ============================================================
#  SIFRA AI v10.3 ENTERPRISE (COGNITIVE + EXCELON EDITION)
# ============================================================

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
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


# ============================================================
# SINGLETON ENGINE REGISTRY
# ============================================================

_ENGINE = None
_LLM_ENGINE = None
_LLM_CACHE = None


def get_engine() -> SIFRAUnifiedEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SIFRAUnifiedEngine()
    return _ENGINE


def get_llm_engine() -> SifraLLMEngine:
    global _LLM_ENGINE
    if _LLM_ENGINE is None:
        _LLM_ENGINE = SifraLLMEngine()
    return _LLM_ENGINE


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_engine()
    get_llm_engine()
    yield


# ============================================================
# FASTAPI INSTANCE
# ============================================================

app = FastAPI(
    title="SIFRA AI Backend",
    version="10.3-Enterprise-Excelon",
    description="SIFRA Enterprise Engine v10.3 (AutoML + LLM + Excelon)",
    lifespan=lifespan
)

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PYDANTIC MODELS
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


# ============================================================
# DATA NORMALIZATION
# ============================================================

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

    except Exception:
        traceback.print_exc()
        return pd.DataFrame()


# ============================================================
# JSON SAFE SERIALIZER
# ============================================================

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
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)
    except Exception:
        return str(obj)


# ============================================================
# ROOT (OPTIONAL BUT CLEAN)
# ============================================================

@app.get("/")
def root():
    return {
        "name": "SIFRA AI Backend",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================
# UNIVERSAL RUN ENDPOINT (🔥 FIXES 404 /run 🔥)
# ============================================================

@app.post("/run")
async def run_engine(request: Request):
    try:
        payload = await request.json()
        engine = get_engine()

        result = engine.run(
            payload.get("goal", "analyze"),
            payload.get("dataset")
        )

        return {
            "status": "success",
            "result": json_safe(result)
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


# ============================================================
# CREATE MODEL (AUTOML)
# ============================================================

@app.post("/create_model")
async def create_model(req: CreateModelRequest):
    try:
        df = normalize_dataset(req.dataset or req.data)

        if df.empty or df.shape[1] < 2:
            return {"status": "fail", "detail": "Dataset invalid"}

        engine = get_engine()
        automl = engine.run("automl_train", {"dataset": df})
        result = automl.get("result", automl)

        return json_safe({
            "status": "success",
            "mode": "automl",
            "result": result
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/create_model failed: {str(e)}")


# ============================================================
# EXCELON™
# ============================================================

@app.post("/excelon/create", response_model=ExcelonResponse)
async def create_excelon(req: ExcelonRequest):
    try:
        if not os.path.exists(req.dataset_path):
            raise Exception("Dataset path does not exist")

        return run_excelon(
            dataset_path=req.dataset_path,
            context=req.context
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"/excelon/create failed: {str(e)}")


# ============================================================
# CREATE LLM
# ============================================================

@app.post("/create_llm")
async def create_llm(req: CreateLLMRequest):
    global _LLM_CACHE
    try:
        docs = req.documents

        if docs is None:
            df = normalize_dataset(req.dataset or req.data)
            docs = df_to_sentences(df)

        if isinstance(docs, str):
            docs = [x.strip() for x in docs.split("\n") if x.strip()]

        engine = get_engine()
        result = engine.run("create_llm", {
            "documents": docs,
            "config": req.config or {}
        })

        _LLM_CACHE = result.get("llm_package")

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
        llm_package = req.llm_package or _LLM_CACHE
        prompt = req.prompt or req.query

        if not llm_package or not prompt:
            raise Exception("LLM package or prompt missing")

        engine = get_engine()
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
