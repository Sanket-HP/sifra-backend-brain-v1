from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable cross-origin for frontend


# -----------------------------------------------------------
# LAZY LOADING (Fixes Vercel cold-start crash issues)
# -----------------------------------------------------------
def load_engines():

    # Core & preprocessing
    from data.dataset_loader import DatasetLoader
    from core.engine_router import EngineRouter

    # Old Engines
    from tasks.auto_analyze import AutoAnalyze
    from tasks.auto_predict import AutoPredict
    from tasks.auto_forecast import AutoForecast
    from tasks.auto_anomaly import AutoAnomaly
    from tasks.auto_insights import AutoInsights

    # New Engines (Data science expansion pack)
    from tasks.auto_visualize import AutoVisualize
    from tasks.auto_eda import AutoEDA
    from tasks.auto_feature_engineering import AutoFeatureEngineering
    from tasks.auto_modeler import AutoModeler
    from tasks.auto_evaluate import AutoEvaluate
    from tasks.auto_bigdata import AutoBigData

    return {
        # Loader & Router
        "loader": DatasetLoader(),
        "router": EngineRouter(),

        # Old Modules
        "analyzer": AutoAnalyze(),
        "predictor": AutoPredict(),
        "forecaster": AutoForecast(),
        "anomaly": AutoAnomaly(),
        "insight": AutoInsights(),

        # New Modules
        "visualize": AutoVisualize(),
        "eda": AutoEDA(),
        "feature_eng": AutoFeatureEngineering(),
        "modeler": AutoModeler(),
        "evaluate": AutoEvaluate(),
        "bigdata": AutoBigData(),
    }


# -----------------------------------------------------------
# HOME ROUTE
# -----------------------------------------------------------
@app.get("/")
def home():
    return {"status": "SIFRA AI API Running", "version": "2.0.0"}


# -----------------------------------------------------------
# CSV FILE UPLOAD (Used by Frontend)
# -----------------------------------------------------------
@app.post("/upload")
def upload():
    try:
        if "file" not in request.files:
            return {"error": "No file provided"}, 400

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
        return {"error": str(e)}, 500


# -----------------------------------------------------------
# COMMON DATASET EXTRACTOR
# -----------------------------------------------------------
def extract_dataset(req, loader):
    if not req.json or "dataset" not in req.json:
        return None, {"error": "Missing 'dataset' in request"}, 400

    try:
        dataset = loader.load_raw(req.json["dataset"])
        return dataset, None, None

    except Exception as e:
        return None, {"error": f"Invalid dataset: {str(e)}"}, 400


# -----------------------------------------------------------
# OLD ENGINE ROUTES (Analyze, Predict, Forecast, etc.)
# -----------------------------------------------------------
@app.post("/analyze")
def analyze():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err:
        return err, code
    return jsonify(eng["analyzer"].run(dataset))


@app.post("/predict")
def predict():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err:
        return err, code
    return jsonify(eng["predictor"].run(dataset))


@app.post("/forecast")
def forecast():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err:
        return err, code

    steps = request.json.get("steps", 5)
    try:
        steps = int(steps)
    except:
        steps = 5

    return jsonify(eng["forecaster"].run(dataset, steps))


@app.post("/anomaly")
def anomaly():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err:
        return err, code
    return jsonify(eng["anomaly"].run(dataset))


@app.post("/insights")
def insights():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err:
        return err, code
    return jsonify(eng["insight"].run(dataset))


@app.post("/trend")
def trend():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err:
        return err, code

    return jsonify({"trend_score": eng["router"].route("trend", dataset)})


# -----------------------------------------------------------
# NEW ADVANCED ENGINE ROUTES
# -----------------------------------------------------------

@app.post("/visualize")
def visualize():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err:
        return err, code
    return jsonify(eng["visualize"].run(dataset))


@app.post("/eda")
def eda():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err:
        return err, code
    return jsonify(eng["eda"].run(dataset))


@app.post("/feature_engineering")
def feature_engineering():
    eng = load_engines()
    dataset, err, code = extract_dataset(request, eng["loader"])
    if err:
        return err, code
    return jsonify(eng["feature_eng"].run(dataset))


@app.post("/modeler")
def modeler():
    eng = load_engines()
    data = request.json

    if "X" not in data or "y" not in data:
        return {"error": "Provide 'X' and 'y' for model training"}, 400

    return jsonify(eng["modeler"].run(data["X"], data["y"]))


@app.post("/evaluate")
def evaluate():
    eng = load_engines()
    data = request.json

    if "y_true" not in data or "y_pred" not in data:
        return {"error": "Provide 'y_true' and 'y_pred' for evaluation"}, 400

    return jsonify(eng["evaluate"].run(data["y_true"], data["y_pred"]))


@app.post("/bigdata")
def bigdata():
    eng = load_engines()

    if "file_path" not in request.json:
        return {"error": "Missing 'file_path'"}, 400

    return jsonify(eng["bigdata"].run(request.json["file_path"]))


# -----------------------------------------------------------
# No app.run() — Vercel handles this automatically
# -----------------------------------------------------------
