"""
prediction.py — Prediction page.

The page is locked until a model exists (trained or uploaded).
The form is built from the model's input columns, so it works for any dataset:
    number column -> number box, text column -> dropdown, date column -> date picker.
"""

import datetime

import numpy as np
import pandas as pd
import streamlit as st

from core import charts, ui
from core.cleaning import clean_for_training, date_columns, duration_pair, find_column, numeric_columns, text_columns
from core.config import CROP_TIPS, GROWTH_STAGES, START_DATE_WORDS, known_crop
from core.recommend import make_recommendations

MIN_DATE, MAX_DATE = datetime.date(1990, 1, 1), datetime.date(2100, 12, 31)   # limits of the date pickers


def locked() -> None:
    ui.render_html("""
        <div class="lock-card">
            <div class="lock-icon">🔒</div>
            <h3>Train a model first</h3>
            <p>Prediction needs a trained model. Train the three models on the current dataset,
            or upload your own data or model.</p>
        </div>
    """)
    ui.notify("warning", "No model is ready yet. Train a model before predicting.")
    c1, c2, _ = st.columns([1, 1, 2])
    c1.button("🧠 Go to Train Model", type="primary", on_click=ui.go_to, args=("🧠 Train Model",))
    c2.button("📤 Go to Upload & Train", on_click=ui.go_to, args=("📤 Upload & Train",))


def input_form(df, reference, features, crop_col, per_area):
    """Draw the form. Returns (inputs, area, submitted, rows of the chosen group)."""
    inputs, subset = {}, reference
    if crop_col:      # chosen outside the form so the default values can follow the crop
        crop = st.selectbox(ui.nice(crop_col), sorted(reference[crop_col].dropna().unique()))
        inputs[crop_col] = crop
        subset = reference[reference[crop_col] == crop]

    with st.form("prediction_form"):
        st.markdown('<div class="form-section">🌱 Field information</div>', unsafe_allow_html=True)
        columns, position = st.columns(3), 0
        for feature in features:
            if feature == crop_col:
                continue
            box, position = columns[position % 3], position + 1
            key = f"in_{feature}_{inputs.get(crop_col, '')}"
            values = subset[feature].dropna()
            if values.empty:
                values = reference[feature].dropna()
            if feature in date_columns(df):
                inputs[feature] = box.date_input(ui.nice(feature), value=values.median().date(), key=key,
                                                 min_value=MIN_DATE, max_value=MAX_DATE)
            elif feature in numeric_columns(df):
                if pd.api.types.is_integer_dtype(df[feature]):
                    start = int(values.max()) if "year" in feature.lower() else int(values.median())
                    inputs[feature] = box.number_input(ui.nice(feature), value=start, step=1, key=key)
                else:
                    inputs[feature] = box.number_input(
                        ui.nice(feature), value=round(float(values.median()), 2), format="%.2f", key=key,
                        help=f"In the data: {values.min():,.1f} to {values.max():,.1f}")
            else:
                inputs[feature] = box.selectbox(ui.nice(feature), sorted(values.astype(str).unique()), key=key)
        area = 1.0
        if per_area:
            area = columns[position % 3].number_input("Field area (hectares)", value=1.0, step=0.5, format="%.2f")
        submitted = st.form_submit_button("🌾 Predict Yield", type="primary")
    return inputs, area, submitted, subset


def check_inputs(inputs, area, features, df, reference, pair) -> list:
    """Mistakes in the form (empty list = OK)."""
    problems = []
    if area <= 0:
        problems.append("**Field area** must be greater than 0 hectares.")
    for feature in features:
        if feature in numeric_columns(df) and inputs[feature] < 0 and reference[feature].min() >= 0:
            problems.append(f"**{ui.nice(feature)}** cannot be negative.")
    if pair and inputs[pair[1]] <= inputs[pair[0]]:
        problems.append(f"**{ui.nice(pair[1])}** must be after **{ui.nice(pair[0]).lower()}**.")
    return problems


def render() -> None:
    ui.page_header("📈", "Prediction", "Enter the field information, click <b>Predict Yield</b>, then read "
                                       "the charts and the recommendations.")
    ui.stepper()
    if st.session_state.get("toast"):
        st.toast(st.session_state.pop("toast"), icon="✅")
    models = ui.get_models()
    if not models:
        locked()
        return

    results, df = st.session_state.results, st.session_state.df
    names = list(models)
    model_name = st.selectbox("Model used for the prediction", names,
                              index=names.index(results["best"]) if results else 0,
                              help="The best model from training is selected by default.")
    active = models[model_name]
    model, features, target = active["model"], active["features"], active["target"]

    missing = [f for f in features if f not in df.columns]
    if missing:
        ui.notify("error", f"This model needs columns that are not in the current dataset: "
                           f"**{', '.join(missing)}**. Upload the matching dataset in **Upload & Train**.")
        return

    reference = df
    if target and target in df.columns:
        reference, _ = clean_for_training(df, target, features)
    crop_col = find_column([f for f in features if f in text_columns(df)], "crop")
    pair = duration_pair(df, features)
    low_target = (target or "").lower()
    per_area = any(word in low_target for word in ("yield", "per_ha", "/ha", "per ha"))
    unit = "t/ha" if "t_per_ha" in low_target else ""

    inputs, area, submitted, subset = input_form(df, reference, features, crop_col, per_area)

    if submitted:
        problems = check_inputs(inputs, area, features, df, reference, pair)
        if problems:
            for problem in problems:
                ui.notify("error", problem)
            st.session_state.last = None
            return
        st.session_state.last = {"inputs": inputs, "area": float(area), "model": model_name}
        st.session_state.toast = "Prediction ready"
        st.rerun()                      # reload so the step bar at the top shows "Predict" as done

    last = st.session_state.last
    if not last or last["model"] != model_name or set(last["inputs"]) != set(features):
        st.info("Fill in the form and click **Predict Yield** to see the result.")
        return
    inputs, area = last["inputs"], last["area"]
    if crop_col:
        subset = reference[reference[crop_col] == inputs[crop_col]]

    row = pd.DataFrame([inputs])[features]
    try:
        prediction = float(model.predict(row)[0])
    except Exception as error:
        ui.notify("error", f"The model could not predict with these inputs ({type(error).__name__}: {error}).")
        return

    # ---------- Result card ----------
    name = ui.nice(target) if target else "value"
    title = str(inputs[crop_col]) if crop_col else "your inputs"
    second = ""
    if per_area:
        second = f"""
            <div class="result-metric">
                <div class="result-label">Estimated total production</div>
                <div class="result-value">{ui.fmt(prediction * area)} <span>{'tonnes' if unit else ''}</span></div>
                <div class="result-sub">for {area:,.2f} hectares</div>
            </div>
        """
    ui.render_html(f"""
        <div class="result-card">
            <div class="result-header">✅ Prediction result — {ui.esc(title)} · {ui.esc(model_name)}</div>
            <div class="result-grid">
                <div class="result-metric">
                    <div class="result-label">Predicted {ui.esc(name.lower())}</div>
                    <div class="result-value">{ui.fmt(prediction)} <span>{unit}</span></div>
                    <div class="result-sub">{'per hectare' if per_area else 'model estimate'}</div>
                </div>
                {second}
            </div>
        </div>
    """)

    # ---------- Warnings for unusual inputs ----------
    for feature in features:
        if feature in numeric_columns(df) and "year" not in feature.lower() and len(subset):
            low, high = subset[feature].min(), subset[feature].max()
            if inputs[feature] < low or inputs[feature] > high:
                ui.notify("warning", f"**{ui.nice(feature)}** = {inputs[feature]:,.1f} is outside the data the model "
                                     f"learned from ({low:,.1f} to {high:,.1f}). The prediction may be less reliable.")

    # ---------- Growth timeline with real dates ----------
    crop = known_crop(inputs.get(crop_col, "")) if crop_col else None
    start_col = pair[0] if pair else find_column([f for f in features if f in date_columns(df)], *START_DATE_WORDS)
    if crop and start_col:
        st.markdown('<div class="section-title">Growth calendar for this field</div>', unsafe_allow_html=True)
        ui.growth_timeline(crop, start=inputs[start_col])
        if pair:
            entered = (pd.Timestamp(inputs[pair[1]]) - pd.Timestamp(inputs[pair[0]])).days
            typical = sum(days for _, days in GROWTH_STAGES[crop])
            st.caption(f"You entered a growing period of **{entered} days**. A typical {crop.lower()} crop "
                       f"takes about {typical} days.")

    # ---------- Charts ----------
    st.markdown('<div class="section-title">Understand the result</div>', unsafe_allow_html=True)
    y_title = f"Predicted {name.lower()}"
    left, right = st.columns(2)
    with left:
        sweepable = [f for f in features if f in numeric_columns(df) and subset[f].nunique() > 5
                     and "year" not in f.lower()]
        if sweepable:
            start = find_column(sweepable, "fertil", "rain") or sweepable[0]
            feature = st.selectbox("Line plot: what happens if this input changes?", sweepable,
                                   index=sweepable.index(start), format_func=ui.nice)
            grid = np.linspace(subset[feature].min(), subset[feature].max(), 25)
            trial = pd.concat([row] * len(grid), ignore_index=True).astype({feature: float})
            trial[feature] = grid
            charts.show(charts.line_chart(grid, model.predict(trial), ui.nice(feature), y_title,
                                          point=(inputs[feature], prediction)))
            st.caption("The orange dot is your field. All other inputs stay the same.")

        st.markdown('<div class="chart-title">Bar chart: the same inputs predicted by each model</div>',
                    unsafe_allow_html=True)
        compare = {}
        for other_name, item in models.items():
            if set(item["features"]) <= set(inputs):
                try:
                    compare[other_name] = float(item["model"].predict(pd.DataFrame([inputs])[item["features"]])[0])
                except Exception:
                    pass
        charts.show(charts.bar_chart(compare.keys(), compare.values(), "Model", y_title, highlight=model_name))
        st.caption("The orange bar is the model you selected. Similar bars mean the models agree.")
    with right:
        st.markdown('<div class="chart-title">Scatter plot: predicted vs. actual on test data</div>',
                    unsafe_allow_html=True)
        if results and model_name in results["models"]:
            test = results["test"]
            if crop_col:
                test = test[test[crop_col] == inputs[crop_col]]
            charts.show(charts.scatter_chart(test["__actual__"], test[f"__pred__{model_name}"],
                                             "Actual", "Predicted", diagonal=True))
            st.caption("Each dot is a row the model never saw in training. Dots near the dashed line "
                       "are good predictions.")
        else:
            st.info("Not available for an uploaded model, because it has no test set in this app.")
        if target and target in subset.columns and len(subset):
            st.markdown('<div class="chart-title">Histogram: your prediction compared with the data</div>',
                        unsafe_allow_html=True)
            charts.show(charts.histogram(subset[target], name, marker=prediction))
            share = (subset[target] < prediction).mean() * 100
            st.caption(f"The orange line is your prediction. It is higher than {share:.0f}% of the "
                       f"{str(inputs[crop_col]).lower() + ' ' if crop_col else ''}rows in the data.")

    # ---------- Recommendations ----------
    st.markdown('<div class="section-title">Recommendations for the farmer</div>', unsafe_allow_html=True)
    recommendations, potential = make_recommendations(model, row, prediction, subset, features)
    items = ""
    for rec in recommendations[:5]:
        items += f"""
            <div class="factor-row">
                <div><div class="factor-name">{ui.esc(rec['text'])}</div>
                <div class="factor-detail">{'Action you can take' if rec['kind'] == 'control' else 'Weather condition'}</div></div>
                <div class="factor-effect effect-pos">▲ +{ui.fmt(rec['gain'])} {unit}</div>
            </div>
        """
    if not recommendations:
        items = ("<p>No clear improvement was found. Either the inputs are already close to the best "
                 "combination in the data, or the dataset has no inputs a farmer can change "
                 "(such as fertilizer or irrigation).</p>")
    summary = ""
    if potential and potential > prediction:
        summary = (f'<p class="factor-summary">If you follow the actions below together, the model predicts about '
                   f'<b>{ui.fmt(potential)} {unit}</b> instead of {ui.fmt(prediction)} {unit} '
                   f'(<b>+{(potential / prediction - 1) * 100:.0f}%</b>).</p>')
    tip = CROP_TIPS.get(crop, "")
    tip_html = f'<p style="margin-top:0.9rem"><b>Crop tip:</b> {ui.esc(tip)}</p>' if tip else ""
    ui.render_html(f"""
        <div class="card factors-card">
            <h3>🌾 What to do next</h3>
            {summary}
            {items}
            {tip_html}
            <p class="small-note">Recommendations come from patterns the model found in the data, not from
            field trials. Check them with a local agricultural officer, and consider the cost of each action.</p>
        </div>
    """)
