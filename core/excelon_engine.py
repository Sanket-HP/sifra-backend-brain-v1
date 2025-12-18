# core/excelon_engine.py
# SIFRA Excelon™ - Algorithm Driven Spreadsheet Engine
# NO LLM USED

import pandas as pd
import numpy as np
from datetime import datetime


class ExcelonEngine:
    def __init__(self, df: pd.DataFrame, context: dict | None = None):
        self.df = df.copy()
        self.context = context or {}
        self.sheets = {}

    # -----------------------------
    # STEP 1: DATA PROFILING
    # -----------------------------
    def profile_data(self) -> pd.DataFrame:
        profile_rows = []

        for col in self.df.columns:
            series = self.df[col]
            profile_rows.append({
                "column": col,
                "dtype": str(series.dtype),
                "missing_%": round(series.isna().mean() * 100, 2),
                "unique_values": series.nunique(dropna=True),
                "mean": series.mean() if pd.api.types.is_numeric_dtype(series) else None,
                "std": series.std() if pd.api.types.is_numeric_dtype(series) else None,
            })

        return pd.DataFrame(profile_rows)

    # -----------------------------
    # STEP 2: OUTLIER DETECTION
    # -----------------------------
    def detect_outliers(self) -> pd.DataFrame:
        outlier_frames = []

        for col in self.df.select_dtypes(include=np.number).columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            if not outliers.empty:
                temp = outliers[[col]].copy()
                temp["outlier_column"] = col
                outlier_frames.append(temp)

        if outlier_frames:
            return pd.concat(outlier_frames)
        return pd.DataFrame()

    # -----------------------------
    # STEP 3: INSIGHT GENERATION
    # -----------------------------
    def generate_insights(self, profile_df: pd.DataFrame) -> pd.DataFrame:
        insights = []

        total_rows = len(self.df)
        insights.append({
            "insight": f"Dataset contains {total_rows} rows and {self.df.shape[1]} columns."
        })

        high_missing = profile_df[profile_df["missing_%"] > 10]
        if not high_missing.empty:
            cols = ", ".join(high_missing["column"].tolist())
            insights.append({
                "insight": f"High missing values detected (>10%) in columns: {cols}."
            })

        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            insights.append({
                "insight": f"{len(numeric_cols)} numeric columns detected, suitable for statistical analysis."
            })

        return pd.DataFrame(insights)

    # -----------------------------
    # STEP 4: SHEET DECISION ENGINE
    # -----------------------------
    def decide_sheets(self):
        self.sheets["raw_data"] = self.df

        profile_df = self.profile_data()
        self.sheets["data_profile"] = profile_df

        insights_df = self.generate_insights(profile_df)
        self.sheets["key_insights"] = insights_df

        outliers_df = self.detect_outliers()
        if not outliers_df.empty:
            self.sheets["anomalies"] = outliers_df

    # -----------------------------
    # STEP 5: FILE METADATA
    # -----------------------------
    def generate_filename(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.context.get("name", "report")
        return f"sifra_excelon_{name}_{timestamp}.xlsx"

    # -----------------------------
    # FINAL BUILD PAYLOAD
    # -----------------------------
    def build(self) -> dict:
        self.decide_sheets()
        return {
            "sheets": self.sheets,
            "file_name": self.generate_filename()
        }
