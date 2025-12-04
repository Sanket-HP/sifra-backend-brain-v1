# engine/narex_utils.py

import pandas as pd
import numpy as np

def clean_numeric(df):
    """
    Cleans raw dataset:
    - Converts all columns to numeric
    - Replaces errors with NaN
    - Forward fill + back fill
    """
    new_df = pd.DataFrame()

    for col in df.columns:
        new_df[col] = pd.to_numeric(df[col], errors="coerce")

    new_df = new_df.fillna(method="ffill").fillna(method="bfill")

    return new_df
