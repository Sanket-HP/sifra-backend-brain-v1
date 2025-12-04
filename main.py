# main.py

import json
from data.dataset_loader import DatasetLoader

# OLD engines
from tasks.auto_analyze import AutoAnalyze
from tasks.auto_predict import AutoPredict
from tasks.auto_forecast import AutoForecast
from tasks.auto_anomaly import AutoAnomaly
from tasks.auto_insights import AutoInsights

# NEW engines
from tasks.auto_visualize import AutoVisualize
from tasks.auto_eda import AutoEDA
from tasks.auto_feature_engineering import AutoFeatureEngineering
from tasks.auto_modeler import AutoModeler
from tasks.auto_evaluate import AutoEvaluate
from tasks.auto_bigdata import AutoBigData

from ui.dashboard import Dashboard
from core.engine_router import EngineRouter

# UNIFIED ENGINE (HDP + HDS + NAREX + ML)
from core.sifra_unified import SIFRAUnifiedEngine


# ===============================================================
# SAFE input evaluator
# ===============================================================
def safe_eval(expr):
    """
    Only evaluates Python list or dict.
    Otherwise returns raw string (file path or query).
    """
    expr = expr.strip()

    if (expr.startswith("[") and expr.endswith("]")) or \
       (expr.startswith("{") and expr.endswith("}")):
        try:
            return eval(expr)
        except Exception:
            raise ValueError("Invalid list/dict format.")

    return expr  # treat as file path string


# ===============================================================
# MAIN APPLICATION
# ===============================================================
def main():

    print("\n===============================")
    print("     SIFRA AI - Autonomous")
    print("    Data Scientist Engine")
    print("===============================\n")

    dashboard = Dashboard()
    loader = DatasetLoader()

    # OLD engines
    analyzer = AutoAnalyze()
    predictor = AutoPredict()
    forecaster = AutoForecast()
    anomaly_detector = AutoAnomaly()
    insight_engine = AutoInsights()

    # NEW engines
    visualizer = AutoVisualize()
    eda_engine = AutoEDA()
    feature_engineer = AutoFeatureEngineering()
    model_engine = AutoModeler()
    evaluator = AutoEvaluate()
    bigdata_engine = AutoBigData()

    router = EngineRouter()

    # UNIFIED BRAIN ENGINE
    unified_engine = SIFRAUnifiedEngine()

    # ===============================================================
    # MAIN LOOP
    # ===============================================================
    while True:

        print("\n========== SIFRA AI DASHBOARD ==========")
        print(" 1.  Auto Analyze")
        print(" 2.  Auto Predict")
        print(" 3.  Auto Forecast")
        print(" 4.  Auto Anomaly Detection")
        print(" 5.  Auto Insights")
        print(" 6.  Trend Extraction")
        print(" 7.  Auto Visualization")
        print(" 8.  Auto EDA")
        print(" 9.  Auto Feature Engineering")
        print("10.  Auto Model Builder")
        print("11.  Auto Evaluation")
        print("12.  Auto Big Data Processing")
        print("13.  Load Dataset File")
        print("14.  Exit")
        print("15.  SIFRA Full-Brain Pipeline (HDP + HDS + NAREX + AutoML)")
        print("========================================")

        choice = input("\nEnter choice: ").strip()

        # ===========================================================
        # OLD TOOLS
        # ===========================================================

        if choice == "1":
            print("\n[INPUT] Enter dataset:")
            try:
                dataset = loader.load_raw(safe_eval(input("Dataset: ")))
                result = analyzer.run(dataset)
                dashboard.show_analysis_result(result)
            except Exception as e:
                print("[ERROR] Auto-Analyze error:", e)

        elif choice == "2":
            print("\n[INPUT] Dataset for prediction")
            try:
                dataset = loader.load_raw(safe_eval(input("Dataset: ")))
                print(json.dumps(predictor.run(dataset), indent=2))
            except Exception as e:
                print("[ERROR] Prediction error:", e)

        elif choice == "3":
            print("\n[INPUT] Dataset for forecasting")
            try:
                data = safe_eval(input("Dataset: "))
                steps = input("Steps (default=5): ").strip()
                dataset = loader.load_raw(data)
                steps = int(steps) if steps else 5
                print(json.dumps(forecaster.run(dataset, steps), indent=2))
            except Exception as e:
                print("[ERROR] Forecast error:", e)

        elif choice == "4":
            print("\n[INPUT] Dataset for anomaly detection")
            try:
                dataset = loader.load_raw(safe_eval(input("Dataset: ")))
                print(json.dumps(anomaly_detector.run(dataset), indent=2))
            except Exception as e:
                print("[ERROR] Anomaly error:", e)

        elif choice == "5":
            print("\n[INPUT] Dataset for insights")
            try:
                dataset = loader.load_raw(safe_eval(input("Dataset: ")))
                result = insight_engine.run(dataset)
                for line in result["insights"]:
                    print("-", line)
            except Exception as e:
                print("[ERROR] Insight error:", e)

        elif choice == "6":
            print("\n[INPUT] Dataset for trend extraction")
            try:
                dataset = loader.load_raw(safe_eval(input("Dataset: ")))
                score = router.route("trend", dataset)
                print("Trend Score:", score)
            except Exception as e:
                print("[ERROR] Trend error:", e)

        elif choice == "7":
            print("\n[INPUT] Dataset for visualization")
            try:
                dataset = loader.load_raw(safe_eval(input("Dataset: ")))
                print(json.dumps(visualizer.run(dataset), indent=2))
            except Exception as e:
                print("[ERROR] Visualization error:", e)

        elif choice == "8":
            print("\n[INPUT] Dataset for EDA")
            try:
                dataset = loader.load_raw(safe_eval(input("Dataset: ")))
                print(json.dumps(eda_engine.run(dataset), indent=2))
            except Exception as e:
                print("[ERROR] EDA error:", e)

        elif choice == "9":
            print("\n[INPUT] Dataset for Feature Engineering")
            try:
                dataset = loader.load_raw(safe_eval(input("Dataset: ")))
                print(json.dumps(feature_engineer.run(dataset), indent=2))
            except Exception as e:
                print("[ERROR] Feature engineering error:", e)

        elif choice == "10":
            print("\nEnter X (features):")
            X = safe_eval(input("X: "))
            print("\nEnter y (labels):")
            y = safe_eval(input("y: "))
            try:
                print(json.dumps(model_engine.run(X, y), indent=2))
            except Exception as e:
                print("[ERROR] Model builder error:", e)

        elif choice == "11":
            print("\nEnter true labels:")
            y_true = safe_eval(input("y_true: "))
            print("Enter predicted labels:")
            y_pred = safe_eval(input("y_pred: "))
            try:
                print(json.dumps(evaluator.run(y_true, y_pred), indent=2))
            except Exception as e:
                print("[ERROR] Evaluation error:", e)

        elif choice == "12":
            print("\nEnter big CSV file path:")
            try:
                print(json.dumps(bigdata_engine.run(input("File path: ").strip()), indent=2))
            except Exception as e:
                print("[ERROR] Big Data error:", e)

        elif choice == "13":
            print("\nChoose file: 1=CSV, 2=Excel, 3=JSON")
            ft = input("File type: ")
            path = input("File path: ")

            try:
                if ft == "1": data = loader.load_csv(path)
                elif ft == "2": data = loader.load_excel(path)
                elif ft == "3": data = loader.load_json(path)
                else:
                    print("[ERROR] Invalid file type.")
                    continue

                print("\nLoaded dataset:")
                print(data)
            except Exception as e:
                print("[ERROR] File load error:", e)

        elif choice == "14":
            print("\n[EXIT] Goodbye!")
            break

        # ===========================================================
        # FULL BRAIN PIPELINE (HDP + HDS + NAREX + ML)
        # ===========================================================
        elif choice == "15":
            print("\n[UNIFIED] Enter dataset (list, dict, API config, DB config, or file path):")
            raw = input("Dataset: ").strip()

            print("\nEnter GOAL (analyze / predict / forecast / anomaly / model / insights):")
            goal = input("Goal: ").strip().lower()

            try:
                data = safe_eval(raw)
                result = unified_engine.run(goal, data)

                print("\n===== SIFRA FULL-BRAIN OUTPUT =====")
                print(json.dumps(result, indent=2))

            except Exception as e:
                print("[ERROR] Unified pipeline error:", e)

        else:
            print("\n[ERROR] Invalid choice. Try again.")


if __name__ == "__main__":
    main()
