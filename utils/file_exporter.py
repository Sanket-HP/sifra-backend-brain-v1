# utils/file_exporter.py
# SIFRA In-Memory File Exporter
# DataFrame → CSV / JSON / XLSX (NO DISK STORAGE)
# Stateless • Secure • Production-Ready

import pandas as pd
from typing import Dict, List
from io import BytesIO, StringIO
from utils.excelon_writer import write_excel


class FileExporter:
    """
    Exports DataFrame into multiple formats entirely in-memory.
    Returns bytes that can be sent directly to frontend.
    """

    def export(
        self,
        df: pd.DataFrame,
        file_name: str,
        formats: List[str]
    ) -> Dict[str, Dict]:
        """
        Returns:
        {
          "csv":  { "filename": "...csv",  "content": bytes, "mime": "text/csv" },
          "json": { "filename": "...json", "content": bytes, "mime": "application/json" },
          "xlsx": { "filename": "...xlsx", "content": bytes, "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }
        }
        """

        if df is None or df.empty:
            raise ValueError("DataFrame is empty. Cannot export.")

        outputs = {}
        safe_name = file_name.replace(" ", "_").lower()

        for fmt in formats:
            fmt = fmt.lower()

            # ---------------- CSV ----------------
            if fmt == "csv":
                buffer = StringIO()
                df.to_csv(buffer, index=False)
                outputs["csv"] = {
                    "filename": f"{safe_name}.csv",
                    "content": buffer.getvalue().encode("utf-8"),
                    "mime": "text/csv"
                }

            # ---------------- JSON ----------------
            elif fmt == "json":
                buffer = StringIO()
                df.to_json(buffer, orient="records", indent=2)
                outputs["json"] = {
                    "filename": f"{safe_name}.json",
                    "content": buffer.getvalue().encode("utf-8"),
                    "mime": "application/json"
                }

            # ---------------- EXCEL (Excelon) ----------------
            elif fmt in ("xlsx", "excel"):
                buffer = BytesIO()
                write_excel({"data": df}, buffer)
                buffer.seek(0)
                outputs["xlsx"] = {
                    "filename": f"{safe_name}.xlsx",
                    "content": buffer.read(),
                    "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                }

            else:
                raise ValueError(f"Unsupported format: {fmt}")

        return outputs
