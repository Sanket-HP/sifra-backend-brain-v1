from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Lazy imports inside a function (prevents Vercel cold start crashes)
def load_engines():
    from tasks.auto_analyze import AutoAnalyze
    from tasks.auto_predict import AutoPredict
    from tasks.auto_forecast import AutoForecast
    from tasks.auto_anomaly import AutoAnomaly
    from tasks.auto_insights import AutoInsights
    from data.dataset_loader import DatasetLoader
    from core.engine_router import EngineRouter

    engines = {
        "loader": DatasetLoader(),
        "router": EngineRouter(),
        "analyzer": AutoAnalyze(),
        "predictor": AutoPredict(),
        "forecaster": AutoForecast(),
        "anomaly": AutoAnomaly(),
        "insight": AutoInsights()
    }
    return engines


@app.get("/")
def home():
    return {"status": "SIFRA AI API Running", "version": "1.0.0"}


def extract_dataset(req, loader):
    """ Extract and validate dataset payload """
    if not req.json or "dataset" not in req.json:
        return None, {"error": "Missing 'dataset' in request"}, 400

    try:
        dataset = loader.load_raw(req.json["dataset"])
        return dataset, None, None

    except Exception as e:
        return None, {"error": f"Invalid dataset: {str(e)}"}, 400


@app.post("/analyze")
def analyze():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code
    return jsonify(engines["analyzer"].run(dataset))


@app.post("/predict")
def predict():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code
    return jsonify(engines["predictor"].run(dataset))


@app.post("/forecast")
def forecast():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code

    steps = request.json.get("steps", 5)
    try:
        steps = int(steps)
    except:
        steps = 5

    return jsonify(engines["forecaster"].run(dataset, steps))


@app.post("/anomaly")
def anomaly():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code
    return jsonify(engines["anomaly"].run(dataset))


@app.post("/insights")
def insights():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code
    return jsonify(engines["insight"].run(dataset))


@app.post("/trend")
def trend():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code
    score = engines["router"].route("trend", dataset)
    return jsonify({"trend_score": score})


# ❗ VERY IMPORTANT: Do NOT add `app.run()` — Vercel will crash!
