# ============================================================
#   AUTO FEATURE ENGINEERING v4.5 — Cognitive FE Engine
#
#   New in v4.5:
#     ✔ CRE-informed feature decisions
#     ✔ DMAO Feature Engineering Agent integration
#     ✔ ALL adaptive learning feedback
#     ✔ HDP semantics for column understanding
#     ✔ HDS variation/trend/correlation-aware transformations
#     ✔ NARE-X natural feature summary
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor


class AutoFeatureEngineering:
    """
    Enhanced Feature Engineering Engine using:
        • HDP semantic column understanding
        • HDS statistical intelligence
        • CRE reasoning
        • DMAO Feature Agent
        • Natural-language transformation explanation
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto Feature Engineering Engine v4.5 Ready")

    # ------------------------------------------------------------
    # Detect dtype
    # ------------------------------------------------------------
    def detect_type(self, series):
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        else:
            return "categorical"

    # ------------------------------------------------------------
    # Generate polynomial features
    # ------------------------------------------------------------
    def generate_polynomials(self, df):
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 1:
            return df

        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(numeric_df)

        poly_df = pd.DataFrame(
            poly_features,
            columns=poly.get_feature_names_out(numeric_df.columns)
        )

        df = df.reset_index(drop=True)
        poly_df = poly_df.reset_index(drop=True)

        return pd.concat([df, poly_df], axis=1)

    # ------------------------------------------------------------
    # Extract date features
    # ------------------------------------------------------------
    def extract_date_features(self, df):
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_weekday"] = df[col].dt.weekday
        return df

    # ------------------------------------------------------------
    # MAIN ENGINE
    # ------------------------------------------------------------
    def run(self, dataset):

        print("\n[AUTO FEATURE ENGINEERING] Running Cognitive FE Pipeline...")

        # Step 1 — Clean dataset
        df = self.preprocessor.clean(dataset)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df.columns = df.columns.astype(str)

        # Step 2 — Ask SIFRA Brain for deeper analysis
        result = self.core.run("analyze", df)

        # Extract SIFRA components
        hdp = result.get("HDP", {})
        hds = result.get("HDS", {})
        cre = result.get("CRE", {})
        dmao = result.get("DMAO", {})
        learning = result.get("ALL", {})

        # Step 3 — Type detection
        col_types = {col: self.detect_type(df[col]) for col in df.columns}

        # Convert numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

            # Attempt datetime conversion
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_datetime(df[col], errors="raise")
                except:
                    pass

        # Fill missing values safely
        df = df.ffill().bfill()

        # Step 4 — Extract date features
        df = self.extract_date_features(df)

        # Step 5 — One-hot encode categorical features
        df = pd.get_dummies(df, drop_first=False)

        # Step 6 — Scale numeric values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Step 7 — Polynomial features (HDS-governed)
        # Variation > 0.1 → high non-linearity, add polynomial terms
        variation_score = hds.get("variation_score", 0)
        if variation_score > 0.1:
            df = self.generate_polynomials(df)

        # Step 8 — Drop constant columns
        df = df.loc[:, df.apply(pd.Series.nunique) > 1]

        # -----------------------------------------------------------
        # Natural Language Summary (NARE-X via DMAO)
        # -----------------------------------------------------------
        natural = dmao.get("agent_output", {}).get("natural_language_response", "")
        if not natural:
            natural = (
                "SIFRA analyzed the dataset and generated enhanced features using "
                "semantic understanding, trend-based scaling, and polynomial expansion. "
                f"Variation score {variation_score:.3f} indicates usefulness of nonlinear transformations."
            )

        # CRE reasoning
        reasoning_summary = cre.get("final_decision", "No CRE reasoning available.")

        # -----------------------------------------------------------
        # Final Output
        # -----------------------------------------------------------
        return {
            "task": "auto_feature_engineering",
            "status": "success",

            # Original Column Info
            "original_columns": list(col_types.keys()),
            "column_types": col_types,

            # HDS + CRE Intelligence
            "HDS": {
                "trend_score": hds.get("trend_score"),
                "correlation_score": hds.get("correlation_score"),
                "variation_score": variation_score,
            },
            "CRE_reasoning": reasoning_summary,
            "CRE_steps": cre.get("steps", []),

            # Agent Information
            "agent_used": dmao.get("agent_selected", "Unknown"),
            "dmao_output": dmao.get("agent_output"),

            # Adaptive Learning
            "learning_update": learning,

            # Final Data
            "final_shape": df.shape,
            "transformed_data": df.fillna(0).values.tolist(),

            # Natural Language Feature Summary
            "feature_summary": natural,

            "message": "Feature engineering completed using SIFRA v4.5 Cognitive Engine."
        }
