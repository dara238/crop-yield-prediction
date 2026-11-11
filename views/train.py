"""
train.py — Train Model page.

training_panel() is also used by the Upload & Train page, so both pages
train in exactly the same way.
"""

import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score

from core import charts, ui
from core.cleaning import (category_columns, clean_for_training, date_columns, default_features,
                           find_column, numeric_columns, text_columns)
from core.config import MAX_CATEGORIES, MIN_ROWS
from core.models import bundle_bytes, train_models


def check_setup(df: pd.DataFrame, target: str, features: list):
    """Return (problems, clean data, cleaning report) before training starts."""
    problems, clean, report = [], None, {}
    if not features:
        problems.append("Choose at least one input column.")
    too_many = [c for c in features if c in text_columns(df) and df[c].nunique() > MAX_CATEGORIES]
    if too_many:
        problems.append(f"These text columns have too many different values to use as inputs: "
                        f"**{', '.join(too_many)}**. Remove them.")
    if not problems:
        clean, report = clean_for_training(df, target, features)
        if len(clean) < MIN_ROWS:
            problems.append(f"Only {len(clean)} usable rows remain after cleaning. At least {MIN_ROWS} are needed.")
        elif clean[target].nunique() < 5:
            problems.append(f"**{target}** has only {clean[target].nunique()} different values. The value to "
                            "predict must be a continuous number, such as yield.")
    return problems, clean, report


def show_results(results: dict) -> None:
    metrics = results["metrics"]
    st.markdown('<div class="section-title">Model accuracy on test data</div>', unsafe_allow_html=True)
    st.write(f"Trained on {results['n_train']:,} rows and tested on {results['n_test']:,} rows "
             "the models did not see.")
    removed = {k: v for k, v in results.get("cleaning", {}).items() if v}
    if removed:
        st.caption("Rows removed before training: " + ", ".join(f"{k} ({v})" for k, v in removed.items()) + ".")

    table = metrics.copy()
    table.insert(0, "", ["🏆" if m == results["best"] else "" for m in table["Model"]])
    st.dataframe(table.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}",
                                     "Training time (s)": "{:.1f}", "Accuracy (R² %)": "{:.1f}%"}), hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="chart-title">Accuracy (R² %) — higher is better</div>', unsafe_allow_html=True)
        charts.show(charts.bar_chart(metrics["Model"], metrics["Accuracy (R² %)"], "Model", "R² (%)",
                                     highlight=results["best"]))
    with right:
        st.markdown('<div class="chart-title">Error (RMSE) — lower is better</div>', unsafe_allow_html=True)
        charts.show(charts.bar_chart(metrics["Model"], metrics["RMSE"], "Model", "RMSE", highlight=results["best"]))
    st.caption("For a regression problem, 'accuracy' is the R² score: the share of the differences in the "
               "target that the model explains. MAE and RMSE are the average error in the target's own units.")

    left, right = st.columns(2)
    with left:
        importance = results["importance"]
        if importance is not None and importance.sum() > 0:
            share = (importance / importance.sum() * 100).head(10)
            st.markdown(f'<div class="chart-title">Which inputs matter most to {results["best"]}?</div>',
                        unsafe_allow_html=True)
            charts.show(charts.bar_chart([ui.nice(i) for i in share.index], share.values, "Input",
                                         "Share of importance (%)", horizontal=True))
    with right:
        group_col = find_column([f for f in results["features"] if f in category_columns(results["test"])], "crop")
        if group_col:
            rows = []
            for name, group in results["test"].groupby(group_col):
                if len(group) >= 5:
                    predicted = group[f"__pred__{results['best']}"]
                    rows.append({ui.nice(group_col): name, "Test rows": len(group),
                                 "MAE": round(mean_absolute_error(group["__actual__"], predicted), 3),
                                 "R²": round(r2_score(group["__actual__"], predicted), 3)})
            st.markdown(f'<div class="chart-title">Accuracy of {results["best"]} for each '
                        f'{ui.nice(group_col).lower()}</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("The overall R² is high partly because the groups have very different values. "
                       "These per-group numbers are the more honest measure.")

    c1, c2, _ = st.columns([1.3, 1.6, 2])
    c1.button("📈 Continue to Prediction →", type="primary", on_click=ui.go_to, args=("📈 Prediction",))
    c2.download_button(f"⬇️ Download the best model ({results['best']})", bundle_bytes(results, results["best"]),
                       "best_model.pkl", "application/octet-stream")


def training_panel(key: str) -> None:
    """Target and input selection, the Train button and the results."""
    df = st.session_state.df
    numbers = numeric_columns(df)
    if not numbers:
        ui.notify("error", "This dataset has no numeric column to predict.")
        return

    guess, confident = st.session_state.target_guess
    target = st.selectbox("Value to predict (target column)", numbers,
                          index=numbers.index(guess) if guess in numbers else 0, key=f"{key}_target")
    if not confident and target == guess:
        st.warning("The app could not find a column named like 'yield', so it chose the last numeric column. "
                   "Check that this is the value you want to predict.")

    candidates = [c for c in df.columns if c != target]
    features = st.multiselect("Input columns (features)", candidates, default=default_features(df, target),
                              key=f"{key}_features_{target}",
                              help="Any column can be an input: numbers, categories or dates.")
    dates = [c for c in features if c in date_columns(df)]
    if dates:
        st.caption(f"📅 Date columns ({', '.join(dates)}) are turned into month, day of year and the number "
                   "of days between the first and last date.")
    leaks = [c for c in features if c in numbers and abs(df[c].corr(df[target])) > 0.98]
    if leaks:
        st.warning(f"**{', '.join(leaks)}** is almost identical to the target. If it is calculated from the "
                   "target (for example total production), remove it, or the accuracy will look better than it is.")
    test_percent = st.slider("Share of data kept for testing (%)", 10, 40, 20, step=5, key=f"{key}_test")
    st.caption("Models compared: **Random Forest**, **Gradient Boosting** and **SVM (Support Vector Regression)**.")

    if st.button("🚀 Train the 3 models", type="primary", key=f"{key}_train"):
        problems, clean, report = check_setup(df, target, features)
        if problems:
            for problem in problems:
                ui.notify("error", problem)
        else:
            try:
                with st.spinner("Training Random Forest, Gradient Boosting and SVM..."):
                    results = train_models(clean, target, features, test_percent / 100)
                results["cleaning"] = report
                st.session_state.results = results
                st.session_state.last = None
                ui.flash("success", f"Training finished. Best model: **{results['best']}**.")
                st.rerun()
            except Exception as error:
                ui.notify("error", f"Training failed ({type(error).__name__}: {error}).")

    if st.session_state.results:
        show_results(st.session_state.results)


def render() -> None:
    ui.page_header("🧠", "Train Model", "Train three models on the current dataset and compare their accuracy. "
                                        "A model must be trained before you can predict.")
    ui.stepper()
    ui.show_flash()
    df = st.session_state.df
    st.markdown(f"**Current dataset:** {st.session_state.data_name} — {len(df):,} rows × {df.shape[1]} columns. "
                "To use other data, open **Upload & Train**.")
    training_panel("train")
