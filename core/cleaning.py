"""
cleaning.py — understand and clean ANY table.

The functions here never assume fixed column names. They look at what
each column contains (numbers, text or dates) and act on that.
"""

import re
import warnings

import numpy as np
import pandas as pd

from core.config import (END_DATE_WORDS, MAX_CATEGORIES, START_DATE_WORDS, TARGET_WORDS, YES_NO)

DATE_PATTERN = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


# ---------- Column types ----------
def numeric_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def date_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def text_columns(df: pd.DataFrame) -> list:
    skip = set(numeric_columns(df)) | set(date_columns(df))
    return [c for c in df.columns if c not in skip]


def category_columns(df: pd.DataFrame) -> list:
    """Text columns with few enough values to group by."""
    return [c for c in text_columns(df) if 1 < df[c].nunique() <= MAX_CATEGORIES]


def find_column(columns, *words):
    """First column whose name contains one of the words (case-insensitive), or None."""
    for col in columns:
        if any(word in str(col).lower() for word in words):
            return col
    return None


def duration_pair(df: pd.DataFrame, columns=None):
    """Find (start date, end date) columns, e.g. planting date and harvest date."""
    dates = [c for c in date_columns(df) if columns is None or c in columns]
    start = find_column(dates, *START_DATE_WORDS)
    end = find_column([c for c in dates if c != start], *END_DATE_WORDS)
    if start and end:
        return start, end
    if len(dates) >= 2:                      # no helpful names: earliest column is the start
        ordered = sorted(dates, key=lambda c: df[c].median())
        return ordered[0], ordered[-1]
    return None


# ---------- Cleaning ----------
def day_comes_first(values: pd.Series) -> bool:
    """Is 05/03/2024 the 5th of March (True) or May 3rd (False)?

    If any first number is above 12, the first number must be the day.
    The whole column is then read the same way.
    """
    parts = values.str.extract(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.]\d{2,4}")
    first = pd.to_numeric(parts[0], errors="coerce").dropna()
    return bool(len(first) and first.max() > 12)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Turn text columns that really hold numbers or dates into the right type."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].map({True: "Yes", False: "No"})
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        values = df[col].astype("string").str.strip()
        filled = values.dropna()
        if filled.empty:
            continue
        # Numbers stored as text, such as "1,200"
        numbers = pd.to_numeric(filled.str.replace(",", "", regex=False), errors="coerce")
        if numbers.notna().mean() >= 0.9:
            df[col] = pd.to_numeric(values.str.replace(",", "", regex=False), errors="coerce")
            continue
        # Dates stored as text, such as "2024-05-17" or "17/05/2024"
        hinted = any(w in str(col).lower() for w in ("date", "time") + START_DATE_WORDS + END_DATE_WORDS)
        if filled.head(200).str.match(DATE_PATTERN).mean() > 0.8 or hinted:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dates = pd.to_datetime(values, errors="coerce", dayfirst=day_comes_first(filled))
            if dates.notna().sum() >= 0.8 * len(filled):
                df[col] = dates
    return df


def prepare_dataset(raw: pd.DataFrame):
    """Safe cleaning for any dataset. Returns (clean table, report)."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]     # index columns saved by mistake
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = coerce_types(df)

    for col in text_columns(df):
        values = df[col].astype("string").str.strip()
        lowered = set(values.dropna().str.lower().unique())
        if lowered and lowered <= set(YES_NO):
            values = values.str.lower().map(YES_NO)             # y / yes / TRUE -> Yes
        elif values.nunique() <= MAX_CATEGORIES:
            values = values.str.title()                         # "  rice " -> "Rice"
        df[col] = values.astype(object).where(values.notna(), np.nan)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, {"duplicates": before - len(df)}


def clean_for_training(df: pd.DataFrame, target: str, features: list):
    """Remove rows that would damage training. Returns (clean table, report)."""
    report, clean = {}, df.copy()

    n = len(clean)
    clean = clean.dropna(subset=[target])
    report["missing target"] = n - len(clean)

    # A few zero or negative values in a column that is otherwise positive are mistakes
    n = len(clean)
    for col in [target] + [f for f in features if f in numeric_columns(clean)]:
        bad = clean[col] <= 0 if col == target else clean[col] < 0
        if 0 < bad.mean() < 0.01:
            clean = clean[~bad]
    report["impossible values"] = n - len(clean)

    # Extreme outliers in the target (for example a typing mistake such as 9999)
    q1, q3 = clean[target].quantile([0.25, 0.75])
    n = len(clean)
    if q3 > q1:
        clean = clean[(clean[target] <= q3 + 5 * (q3 - q1)) & (clean[target] >= q1 - 5 * (q3 - q1))]
    report["extreme target values"] = n - len(clean)

    return clean.reset_index(drop=True), report


# ---------- Guessing roles ----------
def guess_target(df: pd.DataFrame):
    """Guess the column to predict. Returns (column, confident?)."""
    numbers = numeric_columns(df)
    if not numbers:
        return None, False
    for word in TARGET_WORDS:
        for col in numbers:
            low = col.lower()
            if word in low and "prev" not in low and "last" not in low:
                return col, True
    return numbers[-1], False            # fall back to the last numeric column


def default_features(df: pd.DataFrame, target: str) -> list:
    """Columns suggested as model inputs."""
    features = []
    per_area = any(w in target.lower() for w in ("yield", "per_ha", "/ha", "per ha"))
    for col in df.columns:
        low = col.lower()
        if col == target or low == "id" or low.endswith("_id") or low.endswith(" id"):
            continue
        if per_area and "area" in low:
            continue                      # yield is already per hectare
        if col in text_columns(df) and not (1 < df[col].nunique() <= MAX_CATEGORIES):
            continue                      # free text, names, codes
        if df[col].nunique() <= 1:
            continue                      # the same value everywhere teaches nothing
        features.append(col)
    return features


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, missing values, unique values and an example."""
    rows = []
    for col in df.columns:
        kind = "Number" if col in numeric_columns(df) else "Date" if col in date_columns(df) else "Text"
        example = df[col].dropna().iloc[0] if df[col].notna().any() else ""
        if kind == "Date":
            example = pd.Timestamp(example).date().isoformat()
        rows.append({"Column": col, "Type": kind, "Missing": f"{df[col].isna().mean() * 100:.1f}%",
                     "Different values": int(df[col].nunique()), "Example": str(example)})
    return pd.DataFrame(rows)
