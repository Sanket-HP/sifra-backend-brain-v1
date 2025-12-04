# ============================================================
#   SIFRA AI - AutoML v8.1.2-PRO-STABLE (FULL DEBUG MODE)
#   FINAL VERSION – FIXED FEATURE EXTRACTION + PREDICTION SUPPORT
# ============================================================

import numpy as np
import pandas as pd
import pickle
import time

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


def clean_quotes(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace('"', "", regex=False).str.strip()
    return df


def fast_is_datetime(series):
    sample = series.head(500)
    try:
        pd.to_datetime(sample, errors="raise")
        return True
    except:
        return False


def fast_infer_types(df):
    types = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            types[col] = "numeric"
        elif fast_is_datetime(s):
            types[col] = "datetime"
        elif s.nunique() > 2000:
            types[col] = "highcat"
        else:
            types[col] = "categorical"
    dprint("Type inference:", types)
    return types


def expand_datetime(df, cols):
    df = df.copy()
    for col in cols:
        dt = pd.to_datetime(df[col], errors="ignore")
        df[f"{col}_year"] = dt.dt.year
        df[f"{col}_month"] = dt.dt.month
        df[f"{col}_day"] = dt.dt.day
        df[f"{col}_weekday"] = dt.dt.weekday
    dprint("Expanded datetime cols:", cols)
    return df.drop(columns=cols)


# ============================================================
class AutoModeler:

    def __init__(self):
        print("[AutoML v8.1.2-PRO-STABLE] Ready")
        self.preprocessor = None

    # --------------------------------------------------------
    def detect_task(self, y):
        unique = len(np.unique(y))
        task = (
            "anomaly" if unique <= 1 else
            "classification" if y.dtype.kind in ["U", "S", "O"] else
            "classification" if unique <= max(3, int(0.07 * len(y))) else
            "regression"
        )
        dprint("Detected ML Task:", task)
        return task

    # --------------------------------------------------------
    def build_preprocessing(self, X):

        X.columns = X.columns.astype(str)
        types = fast_infer_types(X)

        nums = [c for c, t in types.items() if t == "numeric"]
        cats = [c for c, t in types.items() if t == "categorical"]
        dts = [c for c, t in types.items() if t == "datetime"]
        highcats = [c for c, t in types.items() if t == "highcat"]

        dprint("Preprocessing columns:",
               {"nums": nums, "cats": cats, "dts": dts, "highcats": highcats})

        if dts:
            X = expand_datetime(X, dts)

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

        dprint("Preprocessor created")
        return X

    # --------------------------------------------------------
    def safe_rmse(self, y, preds):
        try:
            return float(mean_squared_error(y, preds, squared=False))
        except:
            return float(np.sqrt(np.mean((np.array(y)-np.array(preds))**2)))

    # --------------------------------------------------------
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

    # --------------------------------------------------------
    def run(self, df):

        start = time.time()
        df = clean_quotes(df.copy())

        dprint("Starting AutoML. Dataset shape:", df.shape)

        if df.shape[1] < 2:
            return self.wrap_error("Dataset must contain at least 2 columns.")

        target = df.columns[-1]
        y = df[target]
        X = df.drop(columns=[target]).copy()

        dprint("Target column:", target)

        try:
            X = self.build_preprocessing(X)
            Xarr = self.preprocessor.fit_transform(X)
        except Exception as e:
            return self.wrap_error("Preprocessing failed: " + str(e))

        dprint("Preprocessing complete. Data shape:", Xarr.shape)

        task = self.detect_task(y)
        Xtr, Xte, ytr, yte = train_test_split(Xarr, y, test_size=0.2)

        # ============ REGRESSION ==================================
        if task == "regression":

            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(n_estimators=120),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
            }

            best_score = -999
            best_model = None
            best_name = None

            for name, mdl in models.items():
                score, trained = self.try_model(mdl, Xtr, ytr, Xte, yte, r2_score)
                if score is not None and score > best_score:
                    best_score = score
                    best_name = name
                    best_model = trained
                    dprint("New BEST model:", name, "Score:", score)

            if best_model is None:
                return self.wrap_error("Training failed — No model returned a valid score.")

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
            )

        # ============ CLASSIFICATION ==============================
        if task == "classification":

            models = {
                "LogisticRegression": LogisticRegression(max_iter=500),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=140),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
            }

            best_score = -1
            best_model = None
            best_name = None

            for name, mdl in models.items():
                score, trained = self.try_model(mdl, Xtr, ytr, Xte, yte, accuracy_score)
                if score is not None and score > best_score:
                    best_score = score
                    best_name = name
                    best_model = trained
                    dprint("New BEST model:", name, "Score:", score)

            if best_model is None:
                return self.wrap_error("Training failed — Classification models could not train.")

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
            )

    # ============================================================
    # WRAPPERS
    # ============================================================
    def wrap_success(self, task, best_model, best_score,
                     accuracy, summary, model, preproc, runtime):

        dprint("WRAP SUCCESS:", best_model, best_score)

        try:
            model_hex = pickle.dumps(model).hex()
            preproc_hex = pickle.dumps(preproc).hex()

            # MOST IMPORTANT FIX → REAL FEATURE NAMES
            try:
                feature_names = list(preproc.get_feature_names_out())
            except:
                feature_names = []
        except Exception as e:
            dprint("Serialization ERROR:", e)
            model_hex = None
            preproc_hex = None
            feature_names = []

        payload = {
            "task": task,
            "best_model": best_model,
            "best_score": best_score,
            "accuracy": accuracy,
            "model_summary": summary,
            "model_hex": model_hex,
            "preprocessor_hex": preproc_hex,

            # CRITICAL FIX FOR FRONTEND
            "feature_names": feature_names,
            "feature_count": len(feature_names),

            "runtime": runtime,
        }

        dprint("FINAL PAYLOAD SENT TO FE:", payload)

        return {
            "status": "success",
            "mode": "model",
            "result": payload,
        }

    def wrap_error(self, msg):
        dprint("WRAP ERROR:", msg)
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
            },
        }
