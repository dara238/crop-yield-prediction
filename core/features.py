"""
features.py — turn date columns into numbers a model can learn from.

A model cannot use "2024-05-17" directly. From each date we create:
    * the month (1-12)            -> season
    * the day of the year (1-366) -> exact time in the season
and, when there is a start date and an end date (planting -> harvest):
    * the number of days between them (the growing period)
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.cleaning import date_columns, duration_pair, numeric_columns


def duration_name(pair) -> str:
    return f"days from {pair[0]} to {pair[1]}"


class DateFeatures(BaseEstimator, TransformerMixin):
    """First step of every model pipeline: replaces date columns with numbers."""

    def __init__(self, dates=(), pair=None):
        self.dates = dates
        self.pair = pair

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        parsed = {col: pd.to_datetime(X[col], errors="coerce") for col in self.dates}
        for col, values in parsed.items():
            X[f"{col} (month)"] = values.dt.month
            X[f"{col} (day of year)"] = values.dt.dayofyear
        if self.pair:
            X[duration_name(self.pair)] = (parsed[self.pair[1]] - parsed[self.pair[0]]).dt.days
        return X.drop(columns=list(self.dates))


def build_spec(df: pd.DataFrame, features: list) -> dict:
    """Describe how the chosen input columns are used."""
    dates = [c for c in features if c in date_columns(df)]
    numeric = [c for c in features if c in numeric_columns(df)]
    categorical = [c for c in features if c not in dates and c not in numeric]
    pair = duration_pair(df, features)

    engineered = [name for col in dates for name in (f"{col} (month)", f"{col} (day of year)")]
    if pair:
        engineered.append(duration_name(pair))
    return {"features": list(features), "numeric": numeric, "categorical": categorical,
            "dates": dates, "pair": pair, "model_numeric": numeric + engineered}
