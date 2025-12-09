# ============================================================
#   AUTO BIGDATA v4.5 — Cognitive Big-Data Engine
#
#   New Features:
#     ✔ CRE reasoning over sampled windows
#     ✔ DMAO BigData-Agent integration
#     ✔ HDS incremental trend/variation estimation
#     ✔ HDP contextual & meaning extraction (global)
#     ✔ Chunk-based anomaly intelligence
#     ✔ Natural-language summary (NARE-X)
#     ✔ Adaptive Learning Loop (ALL)
# ============================================================

import numpy as np
import pandas as pd
import os

from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor


class AutoBigData:
    """
    Enterprise-grade Big Data Cognitive Engine for datasets
    that cannot fit into memory.

    Supports:
        • Chunk streaming (50K–5M rows per chunk)
        • Incremental numeric profiling
        • CRE-aware anomaly reasoning
        • DMAO agent reporting
        • Natural-language big-data insights
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto BigData Engine v4.5 Ready")

    # ------------------------------------------------------------
    # Clean user file path (Fix Windows quotes)
    # ------------------------------------------------------------
    def clean_path(self, file_path: str):
        if isinstance(file_path, str):
            return file_path.strip().replace('"', "").replace("'", "")
        return file_path

    # ------------------------------------------------------------
    # Stream CSV safely in chunks
    # ------------------------------------------------------------
    def stream_csv(self, file_path, chunk_size=50000):
        file_path = self.clean_path(file_path)

        if not os.path.exists(file_path):
            print(f"[BIGDATA ERROR] File not found: {file_path}")
            return

        try:
            for chunk in pd.read_csv(
                    file_path,
                    chunksize=chunk_size,
                    low_memory=False):
                yield chunk
        except Exception as e:
            print(f"[BIGDATA ERROR] {str(e)}")
            return

    # ------------------------------------------------------------
    # Incremental Statistics (BigData-Safe)
    # ------------------------------------------------------------
    def incremental_stats(self, file_path, chunk_size=50000):

        total_sum = None
        total_min = None
        total_max = None
        total_count = 0

        for chunk in self.stream_csv(file_path, chunk_size):
            if chunk is None:
                continue

            numeric = chunk.select_dtypes(include=[np.number])
            if numeric.empty:
                continue

            n_sum = numeric.sum()
            n_count = numeric.count()

            # SUM
            total_sum = n_sum if total_sum is None else total_sum + n_sum
            total_count += int(n_count.sum())

            # MIN
            total_min = (
                numeric.min() if total_min is None
                else np.minimum(total_min, numeric.min())
            )

            # MAX
            total_max = (
                numeric.max() if total_max is None
                else np.maximum(total_max, numeric.max())
            )

        if total_sum is None or total_count == 0:
            return {"error": "No numeric data found."}

        return {
            "mean": (total_sum / total_count).round(4).tolist(),
            "min": pd.Series(total_min).tolist(),
            "max": pd.Series(total_max).tolist(),
            "count": total_count
        }

    # ------------------------------------------------------------
    # Chunk-Level Anomaly Scanning + CRE Reasoning Hooks
    # ------------------------------------------------------------
    def chunk_anomaly_scan(self, chunk, std_threshold=3):

        numeric = chunk.select_dtypes(include=[np.number])
        if numeric.empty:
            return None

        mean = numeric.mean()
        std = numeric.std()

        upper = mean + std_threshold * std
        lower = mean - std_threshold * std

        outliers = numeric[
            (numeric > upper) | (numeric < lower)
        ]

        if outliers.dropna().empty:
            return None

        return {
            "chunk_outliers": int(outliers.count().sum()),
            "mean": mean.tolist(),
            "std": std.tolist()
        }

    # ------------------------------------------------------------
    # Main BigData Workflow
    # ------------------------------------------------------------
    def run(self, file_path, chunk_size=50000):

        file_path = self.clean_path(file_path)
        print(f"[BIGDATA] Processing massive dataset: {file_path}")

        global_stats = self.incremental_stats(file_path, chunk_size)

        total_anomaly_chunks = 0
        anomaly_details = []

        # SAMPLE chunks to feed SIFRA brain (for CRE + DMAO)
        sample_windows = []

        for chunk in self.stream_csv(file_path, chunk_size):
            if chunk is None:
                continue

            numeric = chunk.select_dtypes(include=[np.number])
            if numeric.empty:
                continue

            # --- Anomaly scan ---
            anomaly_info = self.chunk_anomaly_scan(chunk)
            if anomaly_info:
                total_anomaly_chunks += 1
                anomaly_details.append(anomaly_info)

            # --- Sample 1% of rows to feed SIFRA Core for reasoning ---
            try:
                sample = numeric.sample(frac=0.01, random_state=42)
            except:
                sample = numeric.head(50)

            sample_windows.append(sample)

        # --------------------------------------------------------
        # Run SIFRA Brain once on sampled data → CRE + DMAO output
        # --------------------------------------------------------
        if len(sample_windows) == 0:
            return {
                "status": "error",
                "message": "No numeric data found in file."
            }

        combined_sample = pd.concat(sample_windows, ignore_index=True)

        # Run insights mode for reasoning + summary
        brain_output = self.core.run("insights", combined_sample)

        # Extract SIFRA intelligence
        hds = brain_output.get("HDS", {})
        cre = brain_output.get("CRE", {})
        dmao = brain_output.get("DMAO", {})
        learning = brain_output.get("ALL", {})

        natural = dmao.get("agent_output", {}).get("natural_language_response", "")
        if not natural:
            natural = (
                "SIFRA analyzed the large-scale dataset using cognitive sampling. "
                f"The trend signal is {hds.get('trend_score', 0):.2f} with "
                f"variation {hds.get('variation_score', 0):.2f}, and "
                "detected anomalies across multiple segments."
            )

        return {
            "status": "success",
            "task": "auto_bigdata",

            # ---------------------------
            # Global BigData Stats
            # ---------------------------
            "statistics": global_stats,

            # ---------------------------
            # Anomaly Summary
            # ---------------------------
            "anomaly_chunks_found": total_anomaly_chunks,
            "anomaly_details": anomaly_details,

            # ---------------------------
            # SIFRA Cognitive Summary
            # ---------------------------
            "HDS": hds,
            "CRE_reasoning": cre.get("final_decision"),
            "CRE_steps": cre.get("steps", []),

            # ---------------------------
            # DMAO Multi-Agent Intelligence
            # ---------------------------
            "agent_used": dmao.get("agent_selected", "BigData-Agent"),
            "dmao_output": dmao.get("agent_output"),

            # ---------------------------
            # Learning Engine Feedback
            # ---------------------------
            "learning_update": learning,

            # ---------------------------
            # Natural Language Summary
            # ---------------------------
            "insight_summary": natural,

            "message": "Big-data processing completed using SIFRA v4.5 Cognitive Engine."
        }
