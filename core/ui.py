"""
ui.py — shared interface helpers: styles, notifications, session memory
and small HTML components used by several pages.
"""

import html

import pandas as pd
import streamlit as st

from core.cleaning import clean_for_training, guess_target, prepare_dataset
from core.config import ASSETS_DIR, GROWTH_STAGES, SAMPLE_NAME
from core.data_io import load_sample

esc = html.escape


# ---------- Basics ----------
def load_css() -> None:
    """Add every stylesheet in assets/ to the page."""
    css = "\n".join((ASSETS_DIR / name).read_text(encoding="utf-8")
                    for name in ("base.css", "components.css", "animations.css")
                    if (ASSETS_DIR / name).exists())
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_html(block: str) -> None:
    """Show HTML. Indentation is removed because Markdown treats it as code."""
    lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


def notify(kind: str, message: str) -> None:
    """Notification: a pop-up toast AND a message on the page.

    kind is "error", "warning", "success" or "info".
    """
    icons = {"error": "🚫", "warning": "⚠️", "success": "✅", "info": "ℹ️"}
    st.toast(message, icon=icons[kind])
    getattr(st, kind)(message)


def flash(kind: str, message: str) -> None:
    """Save a notification to show after the page reloads."""
    st.session_state.flash = (kind, message)


def show_flash() -> None:
    if st.session_state.get("flash"):
        notify(*st.session_state.pop("flash"))


def nice(name) -> str:
    """Readable label: 'rainfall_mm' -> 'Rainfall mm'."""
    return str(name).replace("_", " ").strip().capitalize()


def fmt(value) -> str:
    """Format a number for display: 158858.37 -> '158,858', 17.523 -> '17.52'."""
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.2f}"


def go_to(page: str) -> None:
    """Button callback: open another page."""
    st.session_state.nav = page


# ---------- Session memory ----------
def set_dataset(raw: pd.DataFrame, name: str) -> None:
    """Make a table the current dataset. Old models and predictions are cleared."""
    df, report = prepare_dataset(raw)
    st.session_state.df = df
    st.session_state.data_name = name
    st.session_state.prep_report = report
    st.session_state.target_guess = guess_target(df)      # (column, confident?)
    st.session_state.results = None
    st.session_state.last = None


def init_state() -> None:
    for key in ("uploaded", "upload_sig", "model_sig", "last", "flash", "pending_step"):
        st.session_state.setdefault(key, None)
    if "df" not in st.session_state:
        set_dataset(load_sample(), SAMPLE_NAME)


def current_target():
    """Column being predicted: the trained target, otherwise the best guess."""
    results = st.session_state.results
    return results["target"] if results else st.session_state.target_guess[0]


def view_data():
    """Current dataset with invalid target rows removed. Returns (table, target)."""
    df, target = st.session_state.df, current_target()
    if target and target in df.columns:
        df, _ = clean_for_training(df, target, [c for c in df.columns if c != target])
    return df, target


def get_models() -> dict:
    """Every model that can be used for prediction right now."""
    models = {}
    results = st.session_state.results
    if results:
        for name, model in results["models"].items():
            models[name] = {"model": model, "features": results["features"], "target": results["target"]}
    uploaded = st.session_state.uploaded
    if uploaded:
        models[f"Uploaded: {uploaded['name']}"] = uploaded
    return models


# ---------- HTML components ----------
def page_header(icon: str, title: str, subtitle: str) -> None:
    render_html(f"""
        <div class="page-header fade-up">
            <div class="page-title">{icon} {esc(title)}</div>
            <p>{subtitle}</p>
        </div>
    """)


def stepper() -> None:
    """Shows where the user is: data -> train -> predict."""
    trained = bool(get_models())
    predicted = bool(st.session_state.last)
    steps = [("1", "Data ready", True), ("2", "Train model", trained), ("3", "Predict", predicted)]
    active = next((i for i, step in enumerate(steps) if not step[2]), len(steps))
    parts = []
    for i, (number, text, done) in enumerate(steps):
        state = "done" if done else "active" if i == active else "todo"
        parts.append(f'<div class="step {state}"><span class="step-dot">{"✓" if done else number}</span>'
                     f'<span class="step-text">{text}</span></div>')
    render_html('<div class="stepper fade-up">' + '<div class="step-line"></div>'.join(parts) + "</div>")


def metric_cards(cards: list) -> None:
    """cards = [(name, value, note), ...] shown four per row."""
    for start in range(0, len(cards), 4):
        row = cards[start:start + 4]
        for i, (col, (name, value, note)) in enumerate(zip(st.columns(4), row)):
            with col:
                render_html(f"""
                    <div class="metric-card fade-up delay-{i + 1}">
                        <div class="metric-name">{esc(str(name))}</div>
                        <div class="metric-value{' small' if len(str(value)) > 8 else ''}">{esc(str(value))}</div>
                        <p>{esc(str(note))}</p>
                    </div>
                """)


def growth_timeline(crop: str, start=None, note: str = "") -> None:
    """Animated bar of the crop's growth stages. With a start date, real dates are shown."""
    stages = GROWTH_STAGES[crop]
    total = sum(days for _, days in stages)
    segments, labels, day = "", "", 0
    for i, (stage, days) in enumerate(stages):
        when = f"{days} days"
        if start is not None:
            first = pd.Timestamp(start) + pd.Timedelta(days=day)
            last = first + pd.Timedelta(days=days)
            when = f"{first:%d %b} – {last:%d %b %Y}"
        segments += (f'<div class="stage-seg stage-{i % 4 + 1}" style="flex:{days};'
                     f'animation-delay:{i * 0.18:.2f}s" title="{esc(stage)}: {days} days"></div>')
        labels += (f'<div class="stage-label" style="flex:{days}"><b>{esc(stage)}</b><span>{when}</span></div>')
        day += days
    harvest = ""
    if start is not None:
        harvest = f" · expected harvest around <b>{pd.Timestamp(start) + pd.Timedelta(days=total):%d %b %Y}</b>"
    render_html(f"""
        <div class="stage-block fade-up">
            <div class="stage-head"><b>{esc(crop)}</b><span>about {total} days{harvest}{esc(note)}</span></div>
            <div class="stage-track">{segments}</div>
            <div class="stage-labels">{labels}</div>
        </div>
    """)
