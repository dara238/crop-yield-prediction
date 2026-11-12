import streamlit as st

from core import ui
from core.config import APP_TITLE, PAGES
from views import about, home, prediction, train, upload

# Which function draws which page
ROUTES = dict(zip(PAGES, [home.render, train.render, prediction.render, upload.render, about.render]))


def sidebar() -> str:
    with st.sidebar:
        ui.render_html("""
            <div class="sidebar-brand">
                <div class="sidebar-logo">🌾</div>
                <div>
                    <div class="sidebar-title">Crop Yield</div>
                    <div class="sidebar-subtitle">Prediction</div>
                </div>
            </div>
        """)
        page = st.radio("Menu", PAGES, label_visibility="collapsed", key="nav")
        results = st.session_state.results
        if results:
            status = f"Best model: {results['best']}"
        elif st.session_state.uploaded:
            status = "Uploaded model ready"
        else:
            status = "No model trained yet"
        ui.render_html(f"""
            <div class="sidebar-note">
                <b>Dataset:</b> {ui.esc(st.session_state.data_name)}<br>
                <b>Status:</b> {ui.esc(status)}
            </div>
        """)
    return page


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🌾", layout="wide")
    ui.load_css()
    ui.init_state()
    ROUTES[sidebar()]()


if __name__ == "__main__":
    main()


