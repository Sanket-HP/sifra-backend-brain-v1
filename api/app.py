import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# -----------------------------------------------------------
# FIX PYTHON PATH FOR VERCEL (NO folder structure changes)
# -----------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend


# -----------------------------------------------------------
# LAZY LOADING — optimized for Vercel
# -----------------------------------------------------------
def load_engines():
    """Load all engines on-demand to avoid Vercel cold boot crashes."""

    # Core modules
    from data.dataset_loader import DatasetLoader
    from core.engine_router import EngineRouter

    # Old core engines
    from tasks.auto_analyze import AutoAnalyze
    from tasks.auto_predict import AutoPredict
    from tasks.auto_forecast import AutoForecast
    from tasks.auto_anomaly import AutoAnomaly
    from tasks.auto_insights import AutoInsights

    # New SIFRA 2.0 modules
    from tasks.auto_visualize import AutoVisualize
    from tasks.auto_eda import AutoEDA
    from tasks.auto_feature_engineering import AutoFeatureEngineering
    from tasks.auto_modeler import AutoModeler
    from tasks.auto_evaluate import AutoEvaluate
    from tasks.auto_bigdata import AutoBigData

    return {
        "loader": DatasetLoader(),
        "router": EngineRouter(),

        # Old modules
        "analyzer": AutoAnalyze(),
        "predictor": AutoPredict(),
        "forecaster": AutoForecast(),
        "anomaly": AutoAnomaly(),
        "insight": AutoInsights(),

        # New modules
        "visualize": AutoVisualize(),
        "eda": AutoEDA(),
        "feature_eng": AutoFeatureEngineering(),
        "modeler": AutoModeler(),
        "evaluate": AutoEvaluate(),
        "bigdata": AutoBigData(),
    }


# -----------------------------------------------------------
# ROOT CHECK
# -----------------------------------------------------------
@app.get("/")
def home():
    return {"status": "SIFRA AI API Running", "version": "2.0.0"}


# -----------------------------------------------------------
# FILE UPLOAD HANDLER (CSV only)
# -----------------------------------------------------------
@app.post("/upload")
def upload():
    try:
        if "file" not in request.files:
            return {"error": "No file uploaded"}, 400

        file = request.files["file"]

        import pandas as pd
        df = pd.read_csv(file)

        return {
            "status": "success",
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict(orient="records")
        }

    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}, 500


# -----------------------------------------------------------
# COMMON DATASET PARSER
# -----------------------------------------------------------
def extract_dataset(req, loader):
    if not req.json or "dataset" not in req.json:
        return None, {"error": "Missing 'dataset'"}, 400

    try:
        dataset = loader.load_raw(req.json["dataset"])
        return dataset, None, None

    except Exception as e:
        return None, {"error": f"Dataset error: {str(e)}"}, 400


# -----------------------------------------------------------
# OLD ENGINE ROUTES
# -----------------------------------------------------------
@app.post("/analyze")
def analyze():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err: return err, code
    return jsonify(eng["analyzer"].run(dataset))


@app.post("/predict")
def predict():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err: return err, code
    return jsonify(eng["predictor"].run(dataset))


@app.post("/forecast")
def forecast():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err: return err, code

    steps = request.json.get("steps", 5)
    try: steps = int(steps)
    except: steps = 5

    return jsonify(eng["forecaster"].run(dataset, steps))


@app.post("/anomaly")
def anomaly():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err: return err, code
    return jsonify(eng["anomaly"].run(dataset))


@app.post("/insights")
def insights():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err: return err, code
    return jsonify(eng["insight"].run(dataset))


@app.post("/trend")
def trend():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err: return err, code
    return jsonify({"trend_score": eng["router"].route("trend", dataset)})


# -----------------------------------------------------------
# NEW ADVANCED MODULE ROUTES
# -----------------------------------------------------------
@app.post("/visualize")
def visualize():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err: return err, code
    return jsonify(eng["visualize"].run(dataset))


@app.post("/eda")
def eda():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err: return err, code
    return jsonify(eng["eda"].run(dataset))


@app.post("/feature_engineering")
def feature_engineering():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err: return err, code
    return jsonify(eng["feature_eng"].run(dataset))


@app.post("/modeler")
def modeler():
    eng = load_engines()
    body = request.json

    if "X" not in body or "y" not in body:
        return {"error": "Provide 'X' and 'y'"}, 400

    return jsonify(eng["modeler"].run(body["X"], body["y"]))


@app.post("/evaluate")
def evaluate():
    eng = load_engines()
    body = request.json

    if "y_true" not in body or "y_pred" not in body:
        return {"error": "Provide y_true & y_pred"}, 400

    return jsonify(eng["evaluate"].run(body["y_true"], body["y_pred"]))


@app.post("/bigdata")
def bigdata():
    eng = load_engines()
    body = request.json

    if "file_path" not in body:
        return {"error": "Missing 'file_path'"}, 400

    return jsonify(eng["bigdata"].run(body["file_path"]))


# -----------------------------------------------------------
# No app.run() — Vercel handles execution
# -----------------------------------------------------------
