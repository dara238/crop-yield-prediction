"""
upload.py — Upload & Train page.

Step 1  upload any dataset (CSV, Excel, JSON, Jupyter notebook)
Step 2  check how the app understood each column
Step 3  train the models on it (same panel as the Train Model page)
Step 4  or upload a model that is already trained
"""

import streamlit as st

from core import ui
from core.cleaning import date_columns, duration_pair, numeric_columns, profile, text_columns
from core.config import MODEL_TYPES, SAMPLE_NAME, UPLOAD_TYPES
from core.data_io import load_sample, read_table, validate_dataset
from core.models import read_model_file
from views.train import training_panel

STEPS = ["1. Upload data", "2. Check columns", "3. Train models", "4. Upload a model"]


def step_upload() -> None:
    st.write("Upload **any table**: each row is one record, and one numeric column is the value to predict. "
             "The other columns can be numbers, categories or dates (for example planting and harvest dates).")
    st.caption("Accepted files: CSV, TSV, Excel (.xlsx), JSON and Jupyter notebook (.ipynb). From a notebook, "
               "the app takes the largest table shown in its outputs.")
    file = st.file_uploader("Dataset file", type=UPLOAD_TYPES)
    if file is not None:
        signature = (file.name, file.size)
        if signature != st.session_state.upload_sig:          # handle each new file once
            st.session_state.upload_sig = signature
            try:
                raw = read_table(file)
                problems = validate_dataset(raw)
            except ValueError as error:
                problems = [str(error)]
            if problems:
                for problem in problems:
                    ui.notify("error", f"Upload failed: {problem}")
            else:
                ui.set_dataset(raw, file.name)
                df = st.session_state.df
                ui.flash("success", f"Dataset **{file.name}** loaded: {len(df):,} rows, {df.shape[1]} columns. "
                                    "The Home page now shows this data.")
                st.session_state.pending_step = STEPS[1]
                st.rerun()

    if st.session_state.data_name != SAMPLE_NAME:
        if st.button("↩️ Go back to the sample dataset"):
            ui.set_dataset(load_sample(), SAMPLE_NAME)
            st.session_state.upload_sig = None
            st.rerun()
    df = st.session_state.df
    st.markdown(f"**Current dataset:** {st.session_state.data_name} — {len(df):,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(8), hide_index=True)


def step_check() -> None:
    df = st.session_state.df
    guess, confident = st.session_state.target_guess
    numbers, dates, texts = numeric_columns(df), date_columns(df), text_columns(df)
    ui.metric_cards([
        ("Number columns", len(numbers), "Can be inputs or the value to predict"),
        ("Text columns", len(texts), "Used as categories"),
        ("Date columns", len(dates), "Turned into month, day of year and durations"),
        ("Suggested target", guess or "—", "Detected by name" if confident else "Last numeric column (check it)"),
    ])
    pair = duration_pair(df)
    if pair:
        st.success(f"📅 Growing period detected: from **{pair[0]}** to **{pair[1]}**. "
                   "The number of days between them is used as an input.")
    if st.session_state.prep_report["duplicates"]:
        st.caption(f"{st.session_state.prep_report['duplicates']} duplicate rows were removed automatically.")
    st.markdown('<div class="chart-title">How the app understood each column</div>', unsafe_allow_html=True)
    st.dataframe(profile(df), hide_index=True)
    st.caption("Missing inputs are filled in automatically during training. Rows without a target value are removed.")
    st.button("Continue to training →", type="primary", on_click=lambda: st.session_state.update(pending_step=STEPS[2]))


def step_model() -> None:
    st.write("Upload a model saved by this app (**.pkl**), or a scikit-learn model or pipeline "
             "trained on a pandas DataFrame (**.pkl** or **.joblib**).")
    st.warning("Only upload model files that you created or trust. A model file can contain "
               "program code that runs when the file is opened.")
    trusted = st.checkbox("I created this file or I trust its source")
    file = st.file_uploader("Model file", type=MODEL_TYPES, disabled=not trusted)
    if file is not None and trusted:
        signature = (file.name, file.size)
        if signature != st.session_state.model_sig:
            st.session_state.model_sig = signature
            try:
                st.session_state.uploaded = read_model_file(file)
                st.session_state.last = None
                ui.flash("success", f"Model **{file.name}** loaded. You can now open Prediction.")
                st.rerun()
            except ValueError as error:
                st.session_state.uploaded = None
                ui.notify("error", f"Upload failed: {error}")
    uploaded = st.session_state.uploaded
    if uploaded:
        st.markdown(f"**Uploaded model:** {uploaded['name']}")
        st.write("Input columns it needs: " + ", ".join(f"`{f}`" for f in uploaded["features"]))
        missing = [f for f in uploaded["features"] if f not in st.session_state.df.columns]
        if missing:
            st.warning("The current dataset does not have: " + ", ".join(missing)
                       + ". Upload the matching dataset before predicting.")
        c1, c2, _ = st.columns([1.2, 1.2, 2])
        c1.button("📈 Go to Prediction →", type="primary", on_click=ui.go_to, args=("📈 Prediction",))
        if c2.button("🗑️ Remove the uploaded model"):
            st.session_state.uploaded = None
            st.session_state.model_sig = None
            st.rerun()


def render() -> None:
    ui.page_header("📤", "Upload & Train", "Bring your own dataset or your own model. The whole app adapts "
                                           "to the columns in your file.")
    ui.stepper()
    if st.session_state.pending_step:          # set before a reload to move to the next step
        st.session_state.upload_step = st.session_state.pending_step
        st.session_state.pending_step = None
    ui.show_flash()
    # A radio keeps its choice when the page reloads (tabs would jump back to the first tab)
    step = st.radio("Step", STEPS, horizontal=True, label_visibility="collapsed", key="upload_step")
    if step == STEPS[0]:
        step_upload()
    elif step == STEPS[1]:
        step_check()
    elif step == STEPS[2]:
        training_panel("upload")
    else:
        step_model()
