# ============================================================
#   SIFRA AI - Cognitive AutoML v10.0-PRO (FINAL ENTERPRISE)
#   Stable Version for SIFRA Unified Engine v10 + Frontend V10
# ============================================================

import numpy as np
import pandas as pd
import pickle
import time
import traceback

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


def dprint(*msg):
    print("[AutoML DEBUG]", *msg)


class AutoModeler:

    def __init__(self):
        print("[AutoML v10.0-PRO] Cognitive AutoML Ready")

        self.preprocessor = None
        self.core = SifraCore()
        self.pp = Preprocessor()


    # ------------------------------------------------------------
    # CLEAN INPUT
    # ------------------------------------------------------------
    def clean_quotes(self, df):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('"', "").str.strip()
        return df


    # ------------------------------------------------------------
    # DATA TYPE INFERENCE
    # ------------------------------------------------------------
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
        return types


    # ------------------------------------------------------------
    # DATETIME EXPANSION
    # ------------------------------------------------------------
    def expand_datetime(self, df, cols):
        df = df.copy()
        for col in cols:
            dt = pd.to_datetime(df[col], errors="ignore")
            df[f"{col}_year"] = dt.dt.year
            df[f"{col}_month"] = dt.dt.month
            df[f"{col}_day"] = dt.dt.day
            df[f"{col}_weekday"] = dt.dt.weekday
        return df.drop(columns=cols)


    # ------------------------------------------------------------
    # PREPROCESSOR BUILDER
    # ------------------------------------------------------------
    def build_preprocessing(self, X):

        X.columns = X.columns.astype(str)
        types = self.fast_infer_types(X)

        nums = [c for c, t in types.items() if t == "numeric"]
        cats = [c for c, t in types.items() if t == "categorical"]
        dts = [c for c, t in types.items() if t == "datetime"]
        highcats = [c for c, t in types.items() if t == "highcat"]

        if dts:
            X = self.expand_datetime(X, dts)

        # Categorical & numeric pipelines
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


    # ------------------------------------------------------------
    # METRICS SAFE WRAPPERS
    # ------------------------------------------------------------
    def safe_rmse(self, y, preds):
        try:
            return float(mean_squared_error(y, preds, squared=False))
        except:
            return float(np.sqrt(np.mean((np.array(y)-np.array(preds))**2)))

    def safe_r2(self, y, preds):
        try:
            s = r2_score(y, preds)
            if np.isnan(s):
                return -999
            return float(s)
        except:
            return -999


    # ------------------------------------------------------------
    # MODEL TEST WRAPPER
    # ------------------------------------------------------------
    def try_model(self, mdl, Xtr, ytr, Xte, yte, metric):
        try:
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xte)
            score = float(metric(yte, pred))
            return score, mdl
        except Exception as e:
            dprint("Model failed:", mdl.__class__.__name__, str(e))
            return None, None


    # ------------------------------------------------------------
    # DETECT ML TASK
    # ------------------------------------------------------------
    def detect_task(self, y):
        unique = len(np.unique(y))
        if unique <= 1:
            return "anomaly"
        if y.dtype.kind in ["U", "S", "O"]:
            return "classification"
        if unique <= max(3, int(0.07 * len(y))):
            return "classification"
        return "regression"


    # ------------------------------------------------------------
    # MAIN ENGINE
    # ------------------------------------------------------------
    def run(self, df_input):

        try:
            start = time.time()
            df = self.clean_quotes(df_input.copy())

            if df.shape[1] < 2:
                return self.wrap_error("Dataset must contain at least 2 columns.")

            target = df.columns[-1]
            y = df[target]
            X = df.drop(columns=[target]).copy()

            # Cognitive Summary (never allowed to break AutoML)
            try:
                cognitive = self.core.run("analyze", df)
            except:
                cognitive = {}

            hdp = cognitive.get("HDP", {})
            hds = cognitive.get("HDS", {})
            cre = cognitive.get("CRE", {})
            dmao = cognitive.get("DMAO", {})
            learning = cognitive.get("ALL", {})

            # -------------------------------------------
            # PREPROCESSING
            # -------------------------------------------
            X = self.build_preprocessing(X)
            Xarr = self.preprocessor.fit_transform(X)

            task = self.detect_task(y)
            Xtr, Xte, ytr, yte = train_test_split(Xarr, y, test_size=0.2)

            # -------------------------------------------
            # MODEL CANDIDATES
            # -------------------------------------------
            reg_models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(n_estimators=120),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
            }

            clf_models = {
                "LogisticRegression": LogisticRegression(max_iter=500),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=140),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
            }

            # -------------------------------------------
            # TRAINING LOOP
            # -------------------------------------------
            best_score = -999
            best_model = None
            best_name = None

            if task == "regression":
                for name, mdl in reg_models.items():
                    score, fitted = self.try_model(mdl, Xtr, ytr, Xte, yte, self.safe_r2)
                    if score is not None and score > best_score:
                        best_score = score
                        best_model = fitted
                        best_name = name

                preds = best_model.predict(Xte)

                accuracy = {
                    "rmse": self.safe_rmse(yte, preds),
                    "mae": float(mean_absolute_error(yte, preds)),
                }

                summary = f"{best_name} (R2={best_score:.4f})"

            else:
                for name, mdl in clf_models.items():
                    score, fitted = self.try_model(mdl, Xtr, ytr, Xte, yte, accuracy_score)
                    if score is not None and score > best_score:
                        best_score = score
                        best_model = fitted
                        best_name = name

                accuracy = best_score
                summary = f"{best_name} (ACC={best_score:.4f})"

            end = time.time()

            return self.wrap_success(
                task=task,
                best_model=best_name,
                best_score=best_score,
                accuracy=accuracy,
                summary=summary,
                model=best_model,
                preproc=self.preprocessor,
                runtime=end - start,
                hdp=hdp,
                hds=hds,
                cre=cre,
                dmao=dmao,
                learning=learning
            )

        except Exception as e:
            traceback.print_exc()
            return self.wrap_error(str(e))


    # ------------------------------------------------------------
    # SUCCESS WRAPPER
    # ------------------------------------------------------------
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
        except:
            model_hex = None
            preproc_hex = None
            feature_names = []

        natural = dmao.get("agent_output", {}).get("natural_language_response", "")
        if not natural:
            natural = (
                f"SIFRA selected **{best_model}** based on dataset semantics, "
                f"trend={hds.get('trend_score')}, CRE reasoning, and Adaptive Learning."
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

                "HDP": hdp,
                "HDS": hds,
                "CRE": cre,
                "DMAO": dmao,
                "learning": learning,
                "natural_language_summary": natural,
            }
        }


    # ------------------------------------------------------------
    # ERROR WRAPPER
    # ------------------------------------------------------------
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
