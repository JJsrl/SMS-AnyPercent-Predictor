import pandas as pd
import numpy as np

# Convert IL time strings like "1:18.24" → seconds as float
def time_to_seconds(t):
    if pd.isna(t):
        return np.nan

    t = str(t).strip()

    # Case 1: "H:MM:SS.xx" or "H:MM:SS"
    if t.count(":") == 2:
        try:
            parts = t.split(":")
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except:
            return np.nan
    
    # Case 2: "M:SS.xx" or "M:SS"
    elif t.count(":") == 1:
        try:
            minutes, seconds = t.split(":")
            return int(minutes) * 60 + float(seconds)
        except:
            return np.nan

    # Case 3: "SS.xx" (just seconds)
    else:
        try:
            return float(t)
        except:
            return np.nan


def preprocess_dataframe(df):
    """Cleans the dataframe: drop unnamed columns, convert IL times, fix missing values."""

    # 1. Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    # 1b. Drop non-numeric "Player" column
    if "Player" in df.columns:
        df = df.drop(columns=["Player"])

    # 2. Convert Personal Best → seconds
    df["Personal Best"] = df["Personal Best"].apply(time_to_seconds)

    # 3. Convert ALL OTHER level columns → seconds (even if float!)
    for col in df.columns:
        if col != "Personal Best":
            df[col] = df[col].apply(time_to_seconds)

    # 4. Fill missing numeric values with median
    df = df.fillna(df.median(numeric_only=True))

    # 5. Drop ANY rows that STILL have NaN in Personal Best
    df = df.dropna(subset=["Personal Best"])

    return df
