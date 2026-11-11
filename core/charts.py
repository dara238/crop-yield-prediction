"""
charts.py — every chart in the app (Altair).

Data is copied into simple column names (x, y, group, ...) before drawing,
so unusual column names in uploaded files cannot break a chart.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core.config import ACCENT, GREEN, PALETTE


def show(chart) -> None:
    st.altair_chart(chart.configure(background="#ffffff").configure_view(strokeWidth=0), width="stretch")


def bar_chart(labels, values, x_title, y_title, highlight=None, horizontal=False):
    data = pd.DataFrame({"label": [str(v) for v in labels], "value": list(values)})
    color = alt.value(GREEN)
    if highlight is not None:
        color = alt.condition(alt.datum.label == str(highlight), alt.value(ACCENT), alt.value(GREEN))
    tooltip = [alt.Tooltip("label:N", title=x_title), alt.Tooltip("value:Q", title=y_title, format=",.2f")]
    if horizontal:
        return alt.Chart(data).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            y=alt.Y("label:N", title=None, sort=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("value:Q", title=y_title), color=color, tooltip=tooltip,
        ).properties(height=max(120, 30 * len(data)))
    return alt.Chart(data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("label:N", title=x_title, sort=None,
                axis=alt.Axis(labelAngle=0 if len(data) <= 6 else -40, labelLimit=120)),
        y=alt.Y("value:Q", title=y_title), color=color, tooltip=tooltip,
    ).properties(height=300)


def histogram(values, title, marker=None, marker_label="Your prediction"):
    counts, edges = np.histogram(pd.Series(values).dropna(), bins=25)
    data = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "count": counts})
    chart = alt.Chart(data).mark_bar(color=GREEN, opacity=0.85).encode(
        x=alt.X("start:Q", title=title, bin="binned"), x2="end:Q",
        y=alt.Y("count:Q", title="Number of rows"),
        tooltip=[alt.Tooltip("start:Q", format=",.2f", title="From"),
                 alt.Tooltip("end:Q", format=",.2f", title="To"), alt.Tooltip("count:Q", title="Rows")])
    if marker is not None:
        rule = alt.Chart(pd.DataFrame({"x": [marker], "label": [marker_label]})).mark_rule(
            color=ACCENT, strokeWidth=3).encode(x="x:Q", tooltip=["label:N", alt.Tooltip("x:Q", format=",.2f")])
        chart = chart + rule
    return chart.properties(height=300)


def scatter_chart(x, y, x_title, y_title, group=None, diagonal=False):
    data = pd.DataFrame({"x": list(x), "y": list(y)})
    if group is not None:
        data["group"] = [str(g) for g in group]
    data = data.dropna()
    if len(data) > 2000:
        data = data.sample(2000, random_state=1)
    encoding = dict(
        x=alt.X("x:Q", title=x_title, scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title=y_title, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("x:Q", title=x_title, format=",.2f"), alt.Tooltip("y:Q", title=y_title, format=",.2f")])
    if group is not None and data["group"].nunique() <= len(PALETTE):
        encoding["color"] = alt.Color("group:N", title=None, scale=alt.Scale(range=PALETTE),
                                      legend=alt.Legend(orient="bottom"))
        encoding["tooltip"].insert(0, alt.Tooltip("group:N", title="Group"))
    else:
        encoding["color"] = alt.value(GREEN)
    chart = alt.Chart(data).mark_circle(size=40, opacity=0.55).encode(**encoding)
    if diagonal and len(data):
        low, high = min(data.x.min(), data.y.min()), max(data.x.max(), data.y.max())
        line = alt.Chart(pd.DataFrame({"x": [low, high], "y": [low, high]})).mark_line(
            color="gray", strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
        chart = line + chart
    return chart.properties(height=300)


def line_chart(x, y, x_title, y_title, group=None, point=None, temporal=False):
    data = pd.DataFrame({"x": list(x), "y": list(y)})
    x_enc = alt.X("x:T", title=x_title) if temporal else \
        alt.X("x:Q", title=x_title, scale=alt.Scale(zero=False), axis=alt.Axis(format="~f"))
    encoding = dict(x=x_enc, y=alt.Y("y:Q", title=y_title, scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("x:T" if temporal else "x:Q", title=x_title),
                             alt.Tooltip("y:Q", title=y_title, format=",.2f")])
    if group is not None:
        data["group"] = [str(g) for g in group]
        encoding["color"] = alt.Color("group:N", title=None, scale=alt.Scale(range=PALETTE),
                                      legend=alt.Legend(orient="bottom"))
        encoding["tooltip"].insert(0, alt.Tooltip("group:N", title="Group"))
    else:
        encoding["color"] = alt.value(GREEN)
    chart = alt.Chart(data).mark_line(point=True, strokeWidth=2.5).encode(**encoding)
    if point is not None:
        dot = alt.Chart(pd.DataFrame({"x": [point[0]], "y": [point[1]]})).mark_point(
            color=ACCENT, size=220, filled=True, opacity=1).encode(
            x="x:Q", y="y:Q", tooltip=[alt.Tooltip("x:Q", title="Your value", format="~f"),
                                       alt.Tooltip("y:Q", title="Predicted", format=",.2f")])
        chart = chart + dot
    return chart.properties(height=300)


def heatmap(corr: pd.DataFrame):
    """Correlation between numeric columns (-1 to 1)."""
    data = corr.reset_index(names="a").melt(id_vars="a", var_name="b", value_name="value")
    base = alt.Chart(data).encode(
        x=alt.X("a:N", title=None, sort=list(corr.columns), axis=alt.Axis(labelAngle=-40, labelLimit=110)),
        y=alt.Y("b:N", title=None, sort=list(corr.columns), axis=alt.Axis(labelLimit=150)))
    squares = base.mark_rect().encode(
        color=alt.Color("value:Q", title="Correlation",
                        scale=alt.Scale(domain=[-1, 0, 1], range=["#9e2a2b", "#f4f6f5", "#2d6a4f"])),
        tooltip=["a:N", "b:N", alt.Tooltip("value:Q", format=".2f")])
    numbers = base.mark_text(fontSize=11).encode(
        text=alt.Text("value:Q", format=".2f"),
        color=alt.condition("abs(datum.value) > 0.55", alt.value("white"), alt.value("#1d2b24")))
    return (squares + numbers).properties(height=max(260, 34 * len(corr)))
