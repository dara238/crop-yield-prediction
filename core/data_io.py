"""
data_io.py — read uploaded files and export data.

Supported uploads: CSV, TSV/TXT, Excel, JSON and Jupyter notebooks (.ipynb).
Every reader raises ValueError with a message the user can understand.
"""

import io
import json
import re

import pandas as pd
import streamlit as st

from core.cleaning import numeric_columns
from core.config import MIN_ROWS, SAMPLE_PATH


def read_notebook(data: bytes) -> pd.DataFrame:
    """Take the largest table shown in a Jupyter notebook's outputs.

    A notebook holds code, not the dataset itself. The only data inside it are
    the tables that were displayed when it was last run (for example df.head()).
    """
    try:
        notebook = json.loads(data.decode("utf-8", errors="ignore"))
        cells = notebook["cells"]
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError("This file is not a valid Jupyter notebook.") from error

    tables, referenced = [], []
    for cell in cells:
        source = "".join(cell.get("source", []))
        referenced += re.findall(r"read_(?:csv|excel|json|table)\(\s*[rf]?['\"]([^'\"]+)['\"]", source)
        for output in cell.get("outputs", []):
            html_text = "".join(output.get("data", {}).get("text/html", []))
            if "<table" in html_text:
                try:
                    tables += pd.read_html(io.StringIO(html_text))
                except ValueError:
                    pass

    hint = f" The notebook loads data from: {', '.join(sorted(set(referenced)))}. Upload that file instead." \
        if referenced else " Upload the CSV or Excel file that the notebook reads."
    if not tables:
        raise ValueError("The notebook has no data table in its outputs." + hint)

    table = max(tables, key=lambda t: t.shape[0] * t.shape[1])
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [" ".join(str(p) for p in col if "Unnamed" not in str(p)).strip() for col in table.columns]
    table = table[~table.astype(str).eq("...").all(axis=1)]          # "..." rows of a shortened preview
    if len(table) < MIN_ROWS:
        raise ValueError(f"The largest table in the notebook has only {len(table)} rows (a preview)." + hint)
    return table


def read_table(file) -> pd.DataFrame:
    """Read an uploaded file into a table."""
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        if name.endswith((".tsv", ".txt")):
            return pd.read_csv(file, sep=None, engine="python")      # detects tab, comma or semicolon
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file)
        if name.endswith(".json"):
            content = json.load(file)
            if isinstance(content, dict):                            # {"data": [...]} or column -> values
                lists = [v for v in content.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
                content = lists[0] if lists else content
            return pd.json_normalize(content) if isinstance(content, list) else pd.DataFrame(content)
        if name.endswith(".ipynb"):
            return read_notebook(file.read())
    except ValueError:
        raise
    except Exception as error:                                       # broken or unreadable file
        raise ValueError(f"The file could not be read ({type(error).__name__}: {error}).") from error
    raise ValueError("Unsupported file type. Upload a CSV, Excel, JSON or Jupyter notebook file.")


def validate_dataset(df: pd.DataFrame) -> list:
    """Problems that make a dataset unusable (empty list = OK)."""
    if df.empty:
        return ["The file has no rows."]
    problems = []
    if df.shape[1] < 2:
        problems.append("The dataset needs at least 2 columns (one input and one value to predict).")
    if len(df) < MIN_ROWS:
        problems.append(f"The dataset has only {len(df)} rows. At least {MIN_ROWS} are needed.")
    if not numeric_columns(df):
        problems.append("No numeric column was found. The value to predict must be a number.")
    return problems


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


@st.cache_data
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, sheet_name="data")
    return buffer.getvalue()
