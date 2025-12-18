# tasks/auto_excelon.py
# SIFRA Excelon™ Task
# Responsibility: Task-level orchestration
# NO LLM USED

import os
import pandas as pd
from core.excelon_engine import ExcelonEngine
from utils.excelon_writer import write_excel


SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")


def _load_dataset(dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    ext = dataset_path.lower()

    if ext.endswith(".csv"):
        return pd.read_csv(dataset_path)

    if ext.endswith((".xlsx", ".xls")):
        return pd.read_excel(dataset_path)

    raise ValueError("Unsupported file format. Use CSV or Excel.")


def run_excelon(
    dataset_path: str,
    context: dict | None = None,
    output_dir: str = "data/excelon_outputs"
) -> dict:
    """
    Main Excelon execution entry point

    Parameters:
    - dataset_path: input dataset path (CSV / Excel)
    - context: metadata (project name, user, etc.)
    - output_dir: where to store generated Excel files
    """

    try:
        context = context or {}

        # 1️⃣ Load dataset
        df = _load_dataset(dataset_path)

        if df.empty:
            return {
                "status": "error",
                "message": "Dataset is empty."
            }

        # 2️⃣ Run Excelon Engine
        engine = ExcelonEngine(df, context)
        payload = engine.build()

        # 3️⃣ Prepare output directory
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, payload["file_name"])

        # 4️⃣ Write Excel file
        write_excel(payload["sheets"], file_path)

        # 5️⃣ Success response
        return {
            "status": "success",
            "engine": "excelon",
            "file_name": payload["file_name"],
            "file_path": file_path,
            "sheets_created": list(payload["sheets"].keys())
        }

    except Exception as e:
        return {
            "status": "error",
            "engine": "excelon",
            "message": str(e)
        }
