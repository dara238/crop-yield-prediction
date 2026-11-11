"""about.py — About page: the purpose of the project."""

import streamlit as st

from core import ui
from core.config import GROWTH_STAGES


def render() -> None:
    ui.page_header("ℹ️", "About this project", "Crop Yield Prediction System Using Machine Learning and Streamlit")
    crops = ", ".join(GROWTH_STAGES)
    ui.render_html(f"""
        <div class="card intro-card">
            <h3>Main purpose</h3>
            <p>The purpose of this project is to help farmers, agricultural officers and planners
            <b>estimate how much a field will produce before harvest</b>, and to turn that estimate into
            <b>practical recommendations</b>. Knowing the expected yield early helps with planning
            inputs, storage, sales and food security.</p>
            <p>The system learns from past field records. It compares three machine-learning models,
            chooses the most accurate one, predicts the yield, and shows which actions could raise it.</p>
        </div>
        <div class="card">
            <h3>How to use the app</h3>
            <ol>
                <li><b>Home</b> — see the current metrics and charts, and download the data as CSV or Excel.
                The page starts with sample data and changes when you upload your own.</li>
                <li><b>Train Model</b> — train the three models on the current dataset and compare their accuracy.
                This step comes before prediction.</li>
                <li><b>Prediction</b> — enter field information to get the predicted yield, a growth calendar,
                four charts and recommendations.</li>
                <li><b>Upload &amp; Train</b> — upload any dataset (CSV, Excel, JSON or a Jupyter notebook) or a
                trained model. The app detects number, text and date columns by itself.</li>
            </ol>
        </div>
        <div class="card">
            <h3>Models</h3>
            <ul>
                <li><b>Random Forest</b> — many decision trees whose answers are averaged. Stable and hard to overfit.</li>
                <li><b>Gradient Boosting</b> — trees built one after another, each fixing the errors of the last.
                Often the most accurate.</li>
                <li><b>SVM (Support Vector Regression)</b> — fits a smooth curve through the data. Works well on
                small datasets.</li>
            </ul>
            <p>Each model is tested on data it did not see during training. The app reports <b>MAE</b>,
            <b>RMSE</b> and <b>R²</b>, and selects the model with the lowest RMSE.</p>
        </div>
        <div class="card">
            <h3>Dates and growth stages</h3>
            <p>If your data has dates, such as a planting date and a harvest date, the app turns them into
            the month, the day of the year and the <b>number of growing days</b>, and uses them as inputs.
            Growth-stage calendars are included for: <b>{crops}</b>.</p>
        </div>
        <div class="card">
            <h3>Limitations</h3>
            <ul>
                <li>The included sample dataset is <b>synthetic</b>. Upload real field records before using
                results in practice.</li>
                <li>The models only know what is in the data. Pests, diseases, floods or seed variety are
                ignored unless your file has columns for them.</li>
                <li>Predictions for inputs far outside the training data are less reliable. The app warns
                you when this happens.</li>
                <li>Recommendations show how the model reacts to inputs. They are not proof of cause and effect.</li>
                <li>Growth-stage lengths are typical values and vary by variety and season.</li>
            </ul>
        </div>
        <div class="disclaimer">
            <b>⚠️ Disclaimer:</b> This system is for <b>educational and decision-support purposes only</b>.
            Predictions are estimates and actual yield may vary. Combine the results with local knowledge
            and advice from agricultural experts.
        </div>
    """)
    
