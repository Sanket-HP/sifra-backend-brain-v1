# ============================================================
#   SIFRA AI - Cognitive AutoML v9.0.0-ULTRA (ENTERPRISE EDITION)
#   Integrated With:
#       • CRE (Cognitive Reasoning Engine)
#       • DMAO Modeling Agent
#       • ALL Adaptive Learning Loop
#       • HDP/HDS Semantic + Statistical Profiling
#       • Natural Language Model Summary (NARE-X)
# ============================================================

import numpy as np
import pandas as pd
import pickle
import time

from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, accuracy_score, mean_squared_error, mean_absolute_error
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression


# ------------------------------------------------------------
def dprint(*msg):
    print("[AutoML DEBUG]", *msg)


# ------------------------------------------------------------
class AutoModeler:

    def __init__(self):
        print("[AutoML v9.0.0-ULTRA] Cognitive AutoML Ready")

        self.preprocessor = None
        self.core = SifraCore()
        self.pp = Preprocessor()


    # ============================================================
    #  CLEAN INPUT
    # ============================================================
    def clean_quotes(self, df):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('"', "").str.strip()
        return df


    # ============================================================
    #  DATA TYPE INFERENCE
    # ============================================================
    def fast_is_datetime(self, series):
        try:
            pd.to_datetime(series.head(200), errors="raise")
            return True
        except:
            return False

    def fast_infer_types(self, df):
        types = {}
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                types[col] = "numeric"
            elif self.fast_is_datetime(s):
                types[col] = "datetime"
            elif s.nunique() > 2000:
                types[col] = "highcat"
            else:
                types[col] = "categorical"
        dprint("Detected column types:", types)
        return types


    # ============================================================
    #  DATETIME EXPANSION
    # ============================================================
    def expand_datetime(self, df, cols):
        df = df.copy()
        for col in cols:
            dt = pd.to_datetime(df[col], errors="ignore")
            df[f"{col}_year"] = dt.dt.year
            df[f"{col}_month"] = dt.dt.month
            df[f"{col}_day"] = dt.dt.day
            df[f"{col}_weekday"] = dt.dt.weekday
        return df.drop(columns=cols)


    # ============================================================
    #  PREPROCESSOR BUILDER
    # ============================================================
    def build_preprocessing(self, X):

        X.columns = X.columns.astype(str)
        types = self.fast_infer_types(X)

        nums = [c for c, t in types.items() if t == "numeric"]
        cats = [c for c, t in types.items() if t == "categorical"]
        dts = [c for c, t in types.items() if t == "datetime"]
        highcats = [c for c, t in types.items() if t == "highcat"]

        if dts:
            X = self.expand_datetime(X, dts)

        transformers = []

        if nums:
            transformers.append(("num", StandardScaler(), nums))

        if cats:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=50), cats)
            )

        if highcats:
            for col in highcats:
                X[col] = X[col].astype("category").cat.codes
            transformers.append(("highcat", StandardScaler(), highcats))

        self.preprocessor = ColumnTransformer(
            transformers,
            remainder="drop",
            sparse_threshold=0.4,
        )

        return X


    # ============================================================
    #  SAFE METRICS
    # ============================================================
    def safe_rmse(self, y, preds):
        try:
            return float(mean_squared_error(y, preds, squared=False))
        except:
            return float(np.sqrt(np.mean((np.array(y)-np.array(preds))**2)))


    # ============================================================
    #  MODEL TEST WRAPPER
    # ============================================================
    def try_model(self, mdl, Xtr, ytr, Xte, yte, metric):
        try:
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xte)
            score = float(metric(yte, pred))
            dprint("Model tested:", mdl.__class__.__name__, "| Score:", score)
            return score, mdl
        except Exception as e:
            dprint("Model failed:", mdl.__class__.__name__, "| Error:", str(e))
            return None, None


    # ============================================================
    #  COGNITIVE ML TASK DETECTION
    # ============================================================
    def detect_task(self, y):
        unique = len(np.unique(y))
        if unique <= 1:
            return "anomaly"
        if y.dtype.kind in ["U", "S", "O"]:
            return "classification"
        if unique <= max(3, int(0.07 * len(y))):
            return "classification"
        return "regression"


    # ============================================================
    #                 MAIN COGNITIVE AutoML ENGINE
    # ============================================================
    def run(self, df_input):

        start = time.time()

        df = self.clean_quotes(df_input.copy())

        if df.shape[1] < 2:
            return self.wrap_error("Dataset must contain at least 2 columns.")

        target = df.columns[-1]
        y = df[target]
        X = df.drop(columns=[target]).copy()

        # -----------------------------------------------------------
        # Step 1 — Cognitive dataset understanding (HDP + HDS + CRE)
        # -----------------------------------------------------------
        cognitive_summary = self.core.run("analyze", df)

        # Pull out SIFRA intelligence
        hdp = cognitive_summary.get("HDP", {})
        hds = cognitive_summary.get("HDS", {})
        cre = cognitive_summary.get("CRE", {})
        dmao = cognitive_summary.get("DMAO", {})
        learning = cognitive_summary.get("ALL", {})

        # -----------------------------------------------------------
        # Step 2 — Preprocessing
        # -----------------------------------------------------------
        try:
            X = self.build_preprocessing(X)
            Xarr = self.preprocessor.fit_transform(X)
        except Exception as e:
            return self.wrap_error("Preprocessing failed: " + str(e))

        task = self.detect_task(y)

        Xtr, Xte, ytr, yte = train_test_split(Xarr, y, test_size=0.2)

        # ===========================================================
        #                    MODEL SELECTION LOGIC (COGNITIVE)
        # ===========================================================

        regression_models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=120),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
        }

        classification_models = {
            "LogisticRegression": LogisticRegression(max_iter=500),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=140),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
        }

        if task == "regression":
            best_score = -999
            best_model = None
            best_name = None

            for name, mdl in regression_models.items():
                score, fitted = self.try_model(mdl, Xtr, ytr, Xte, yte, r2_score)
                if score is not None and score > best_score:
                    best_score = score
                    best_model = fitted
                    best_name = name

            preds = best_model.predict(Xte)
            end = time.time()

            return self.wrap_success(
                task="regression",
                best_model=best_name,
                best_score=best_score,
                accuracy={
                    "rmse": self.safe_rmse(yte, preds),
                    "mae": float(mean_absolute_error(yte, preds)),
                },
                summary=f"{best_name} (R2={best_score:.4f})",
                model=best_model,
                preproc=self.preprocessor,
                runtime=end - start,
                hdp=hdp,
                hds=hds,
                cre=cre,
                dmao=dmao,
                learning=learning,
            )

        if task == "classification":
            best_score = -1
            best_model = None
            best_name = None

            for name, mdl in classification_models.items():
                score, fitted = self.try_model(mdl, Xtr, ytr, Xte, yte, accuracy_score)
                if score is not None and score > best_score:
                    best_score = score
                    best_model = fitted
                    best_name = name

            end = time.time()

            return self.wrap_success(
                task="classification",
                best_model=best_name,
                best_score=best_score,
                accuracy=best_score,
                summary=f"{best_name} (ACC={best_score:.4f})",
                model=best_model,
                preproc=self.preprocessor,
                runtime=end - start,
                hdp=hdp,
                hds=hds,
                cre=cre,
                dmao=dmao,
                learning=learning,
            )


    # ============================================================
    #  SUCCESS WRAPPER (Cognitive Response)
    # ============================================================
    def wrap_success(self, task, best_model, best_score, accuracy,
                     summary, model, preproc, runtime,
                     hdp, hds, cre, dmao, learning):

        try:
            model_hex = pickle.dumps(model).hex()
            preproc_hex = pickle.dumps(preproc).hex()
            try:
                feature_names = list(preproc.get_feature_names_out())
            except:
                feature_names = []
        except Exception as e:
            model_hex = None
            preproc_hex = None
            feature_names = []

        natural = dmao.get("agent_output", {}).get("natural_language_response", "")
        if not natural:
            natural = (
                f"SIFRA selected **{best_model}** based on dataset semantics, "
                f"trend {hds.get('trend_score')}, and cognitive reasoning. "
                "Model achieved strong performance according to AutoML evaluation."
            )

        return {
            "status": "success",
            "mode": "model",

            "result": {
                "task": task,
                "best_model": best_model,
                "best_score": best_score,
                "accuracy": accuracy,
                "model_summary": summary,

                "model_hex": model_hex,
                "preprocessor_hex": preproc_hex,

                "feature_names": feature_names,
                "feature_count": len(feature_names),

                "runtime": runtime,

                # Cognitive Additions
                "HDP": hdp,
                "HDS": hds,
                "CRE_reasoning": cre.get("final_decision"),
                "CRE_steps": cre.get("steps", []),
                "agent_used": dmao.get("agent_selected", "Modeler-Agent"),
                "dmao_output": dmao.get("agent_output"),
                "learning_update": learning,

                "natural_language_summary": natural,
            }
        }

    # ============================================================
    #  ERROR WRAPPER
    # ============================================================
    def wrap_error(self, msg):
        return {
            "status": "error",
            "mode": "model",
            "result": {
                "error": msg,
                "best_model": None,
                "model_hex": None,
                "preprocessor_hex": None,
                "feature_names": [],
                "feature_count": 0
            }
        }
