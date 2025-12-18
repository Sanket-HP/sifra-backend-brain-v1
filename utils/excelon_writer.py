# utils/excelon_writer.py
# SIFRA Excelon™ - Excel Writer
# Writes Excel to FILE PATH or IN-MEMORY BUFFER
# NO intelligence / NO LLM

import pandas as pd
from typing import Dict, Union
from io import BytesIO


def write_excel(
    sheets: Dict[str, pd.DataFrame],
    output: Union[str, BytesIO]
):
    """
    Writes multiple pandas DataFrames into a single Excel file.

    Parameters:
    - sheets: dict -> {sheet_name: DataFrame}
    - output: str (file path) OR BytesIO (in-memory)
    """

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Common formats
        header_format = workbook.add_format({
            "bold": True,
            "text_wrap": True,
            "valign": "middle",
            "align": "center",
            "border": 1
        })

        cell_format = workbook.add_format({
            "border": 1,
            "valign": "top"
        })

        number_format = workbook.add_format({
            "border": 1,
            "num_format": "#,##0.00"
        })

        percent_format = workbook.add_format({
            "border": 1,
            "num_format": "0.00%"
        })

        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue

            safe_sheet_name = sheet_name[:31]  # Excel limit
            df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

            worksheet = writer.sheets[safe_sheet_name]

            # Header formatting
            for col_num, col_name in enumerate(df.columns):
                worksheet.write(0, col_num, col_name, header_format)

            # Column width + formatting
            for col_num, col_name in enumerate(df.columns):
                series = df[col_name]

                max_len = max(
                    series.astype(str).map(len).max(),
                    len(col_name)
                ) + 2
                max_len = min(max_len, 40)

                # Choose format
                if pd.api.types.is_float_dtype(series):
                    fmt = number_format
                elif pd.api.types.is_integer_dtype(series):
                    fmt = cell_format
                else:
                    fmt = cell_format

                worksheet.set_column(col_num, col_num, max_len, fmt)

            # UX improvements
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(
                0, 0,
                len(df),
                len(df.columns) - 1
            )

    # Important for in-memory buffers
    if isinstance(output, BytesIO):
        output.seek(0)
