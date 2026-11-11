"""
home.py — Home page.

Everything here is built from the CURRENT dataset, so the metrics and charts
change automatically when the user uploads different data.
"""

import pandas as pd
import streamlit as st

from core import charts, ui
from core.cleaning import (category_columns, date_columns, duration_pair, find_column,
                           numeric_columns, text_columns)
from core.config import APP_TITLE, GROWTH_STAGES, known_crop
from core.data_io import to_csv_bytes, to_excel_bytes


def build_cards(df: pd.DataFrame, target, group_col) -> list:
    """Metric cards that fit whatever columns the dataset has."""
    numbers, dates, texts = numeric_columns(df), date_columns(df), text_columns(df)
    cards = [
        ("Records", f"{len(df):,}", "Rows after automatic cleaning"),
        ("Columns", f"{df.shape[1]}", f"{len(numbers)} number · {len(texts)} text · {len(dates)} date"),
    ]
    if target:
        cards.append((f"Average {ui.nice(target).lower()}", ui.fmt(df[target].mean()),
                      f"From {ui.fmt(df[target].min())} to {ui.fmt(df[target].max())}"))
    if group_col:
        names = df[group_col].value_counts().index[:5]
        cards.append((ui.nice(group_col), f"{df[group_col].nunique()}", ", ".join(str(n) for n in names)))
    if dates:
        first, last = df[dates[0]].min(), df[dates[0]].max()
        cards.append(("Period", f"{first:%Y}–{last:%Y}" if first.year != last.year else f"{first:%Y}",
                      f"{ui.nice(dates[0])}: {first:%d %b %Y} to {last:%d %b %Y}"))
    pair = duration_pair(df)
    if pair:
        days = (df[pair[1]] - df[pair[0]]).dt.days
        cards.append(("Growing period", f"{days.mean():,.0f} days", f"Average from {ui.nice(pair[0]).lower()} "
                                                                    f"to {ui.nice(pair[1]).lower()}"))
    missing = df.isna().sum().sum() / max(df.size, 1) * 100
    cards.append(("Missing values", f"{missing:.1f}%", "Share of empty cells in the table"))

    results = st.session_state.results
    if results:
        row = results["metrics"].set_index("Model").loc[results["best"]]
        cards.append(("Model accuracy", f"{row['R²'] * 100:.1f}%", f"{results['best']} (R² on test data)"))
    else:
        cards.append(("Model accuracy", "—", "Not trained yet. Open Train Model."))
    return cards[:8]


def time_series(df: pd.DataFrame, value_col: str, group_col):
    """Average of a value over time, if the data has a date or a year column."""
    dates = date_columns(df)
    year_col = find_column([c for c in numeric_columns(df) if c != value_col], "year")
    if dates:
        col = dates[-1] if len(dates) > 1 else dates[0]           # the latest date, e.g. harvest
        span = (df[col].max() - df[col].min()).days
        period = df[col].dt.year if span > 3 * 365 else df[col].dt.to_period("M").dt.to_timestamp()
        temporal, title = span <= 3 * 365, f"{ui.nice(col)} ({'year' if span > 3 * 365 else 'month'})"
    elif year_col:
        period, temporal, title = df[year_col], False, ui.nice(year_col)
    else:
        return None
    data = df.assign(__period__=period).dropna(subset=["__period__"])
    by = ["__period__"]
    if group_col:
        top = data[group_col].value_counts().index[:6]
        data, by = data[data[group_col].isin(top)], ["__period__", group_col]
    trend = data.groupby(by)[value_col].mean().reset_index()
    return charts.line_chart(trend["__period__"], trend[value_col], title, f"Average {ui.nice(value_col).lower()}",
                             group=trend[group_col] if group_col else None, temporal=temporal)


def growth_section(df: pd.DataFrame, target) -> None:
    """Growth stages of the crops in the data, plus real growing periods if dates exist."""
    st.markdown('<div class="section-title">Growth of the crop</div>', unsafe_allow_html=True)
    crop_col = find_column(text_columns(df), "crop") or next(
        (c for c in category_columns(df) if any(known_crop(v) for v in df[c].dropna().unique())), None)
    crops = []
    if crop_col:
        crops = sorted({known_crop(v) for v in df[crop_col].dropna().unique() if known_crop(v)})

    pair = duration_pair(df)
    left, right = st.columns([3, 2])
    with left:
        if crops:
            st.markdown('<div class="chart-title">Growth stages from planting to harvest</div>',
                        unsafe_allow_html=True)
            for crop in crops:
                ui.growth_timeline(crop)
            st.caption("Typical stage lengths. Real values depend on variety, season and location.")
        else:
            st.info("This dataset has no crop the app knows (rice, chilli, watermelon, tomato, banana), "
                    "so reference growth stages are shown.")
            with st.expander("Reference: growth stages of five common crops", expanded=not pair):
                for crop in GROWTH_STAGES:
                    ui.growth_timeline(crop)
    with right:
        if pair:
            days = (df[pair[1]] - df[pair[0]]).dt.days
            data = df.assign(__days__=days)
            st.markdown(f'<div class="chart-title">Days from {ui.nice(pair[0]).lower()} to '
                        f'{ui.nice(pair[1]).lower()} in this dataset</div>', unsafe_allow_html=True)
            if crop_col:
                table = data.groupby(crop_col)["__days__"].agg(["mean", "min", "max"]).round(0).astype(int)
                table.columns = ["Average days", "Shortest", "Longest"]
                typical = {c: sum(d for _, d in s) for c, s in GROWTH_STAGES.items()}
                table["Typical"] = pd.array([typical.get(known_crop(name)) for name in table.index], dtype="Int64")
                st.dataframe(table.reset_index().rename(columns={crop_col: ui.nice(crop_col)}), hide_index=True)
            else:
                charts.show(charts.histogram(days, "Days"))
        elif date_columns(df):
            st.caption("Only one date column was found, so the growing period cannot be measured.")
        else:
            st.caption("No date columns in this dataset. Add planting and harvest dates to see real "
                       "growing periods here.")


def render() -> None:
    df, target = ui.view_data()
    numbers, groups = numeric_columns(df), category_columns(df)
    group_default = find_column(groups, "crop") or (groups[0] if groups else None)

    ui.render_html(f"""
        <div class="hero">
            <div class="hero-badge">Machine Learning · Agriculture</div>
            <div class="hero-title">{APP_TITLE}</div>
            <p class="hero-text">Explore your data, train three models, predict the yield and get
            recommendations for the field.</p>
            <div class="stat-row">
                <div class="stat-chip">Dataset: <span>{ui.esc(st.session_state.data_name)}</span></div>
                <div class="stat-chip"><span>{len(df):,}</span> rows</div>
                <div class="stat-chip"><span>{df.shape[1]}</span> columns</div>
            </div>
        </div>
    """)
    ui.stepper()

    # ---------- Current metrics ----------
    st.markdown('<div class="section-title">Current metrics</div>', unsafe_allow_html=True)
    ui.metric_cards(build_cards(df, target, group_default))
    if not st.session_state.results:
        st.button("🧠 Train the models on this dataset →", type="primary", on_click=ui.go_to,
                  args=("🧠 Train Model",))

    # ---------- Visualization ----------
    st.markdown('<div class="section-title">Visualization</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    value_col = c1.selectbox("Value to analyse", numbers, index=numbers.index(target) if target in numbers else 0,
                             format_func=ui.nice)
    group_col = c2.selectbox("Group by", groups, index=groups.index(group_default), format_func=ui.nice) \
        if groups else None
    shown, choice = df, "All"
    if group_col:
        choice = c3.selectbox("Show data for", ["All"] + sorted(df[group_col].dropna().unique()))
        if choice != "All":
            shown = df[df[group_col] == choice]
    if shown.empty:
        ui.notify("warning", "No rows match this filter.")
        return

    left, right = st.columns(2)
    with left:
        if group_col and choice == "All":
            means = shown.groupby(group_col)[value_col].mean().sort_values(ascending=False).head(12)
            st.markdown(f'<div class="chart-title">Bar chart: average {ui.nice(value_col).lower()} by '
                        f'{ui.nice(group_col).lower()}</div>', unsafe_allow_html=True)
            charts.show(charts.bar_chart(means.index, means.values, ui.nice(group_col),
                                         f"Average {ui.nice(value_col).lower()}"))
        elif groups and len(groups) > 1:
            other = next(g for g in groups if g != group_col)
            means = shown.groupby(other)[value_col].mean().sort_values(ascending=False).head(12)
            st.markdown(f'<div class="chart-title">Bar chart: average {ui.nice(value_col).lower()} by '
                        f'{ui.nice(other).lower()}</div>', unsafe_allow_html=True)
            charts.show(charts.bar_chart(means.index, means.values, ui.nice(other),
                                         f"Average {ui.nice(value_col).lower()}"))
        st.markdown(f'<div class="chart-title">Histogram: how {ui.nice(value_col).lower()} is spread</div>',
                    unsafe_allow_html=True)
        charts.show(charts.histogram(shown[value_col], ui.nice(value_col)))
    with right:
        others = [c for c in numbers if c != value_col]
        if others:
            not_ids = [c for c in others if not c.lower().endswith("id")] or others
            default = find_column(others, "rain", "fertil") or not_ids[0]
            x_col = st.selectbox(f"Scatter plot: compare {ui.nice(value_col).lower()} with", others,
                                 index=others.index(default), format_func=ui.nice)
            charts.show(charts.scatter_chart(shown[x_col], shown[value_col], ui.nice(x_col), ui.nice(value_col),
                                             group=shown[group_col] if group_col else None))
        line = time_series(shown, value_col, group_col if (group_col and choice == "All") else None)
        if line is not None:
            st.markdown(f'<div class="chart-title">Line plot: average {ui.nice(value_col).lower()} over time</div>',
                        unsafe_allow_html=True)
            charts.show(line)

    if len(numbers) >= 3:
        with st.expander("Heatmap: how the numeric columns move together"):
            charts.show(charts.heatmap(shown[numbers[:12]].corr().round(2)))
            st.caption("Values near 1 or -1 mean two columns change together. Values near 0 mean no clear link.")

    # ---------- Growth ----------
    growth_section(df, target)

    # ---------- Download ----------
    st.markdown('<div class="section-title">Download the dataset</div>', unsafe_allow_html=True)
    st.write(f"{len(df):,} rows × {df.shape[1]} columns (cleaned version of the current dataset).")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("⬇️ Download CSV", to_csv_bytes(df), "dataset.csv", "text/csv")
    c2.download_button("⬇️ Download Excel", to_excel_bytes(df), "dataset.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with st.expander("Preview the first 20 rows"):
        st.dataframe(df.head(20), hide_index=True)
