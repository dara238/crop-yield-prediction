# Crop Yield Prediction

A Streamlit app that explores any tabular dataset, trains three machine-learning models, predicts crop yield and gives recommendations for farmers.

## How to run

Python 3.11 or newer is required.

```bash
cd crop_yield_app
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Project structure

```
crop_yield_app/
├── app.py                    # entry point: page setup and menu only
├── requirements.txt
├── README.md
├── .streamlit/config.toml    # theme colours
│
├── core/                     # logic (no page layout here)
│   ├── config.py             # settings, crop growth stages, crop tips
│   ├── data_io.py            # read CSV / Excel / JSON / notebook, export CSV / Excel
│   ├── cleaning.py           # detect column types, clean data, guess the target
│   ├── features.py           # turn dates into month, day of year, growing days
│   ├── models.py             # Random Forest, Gradient Boosting, SVM: train, compare, save
│   ├── recommend.py          # recommendations for the farmer
│   ├── charts.py             # every chart
│   └── ui.py                 # notifications, session memory, shared HTML components
│
├── views/                    # one file per page
│   ├── home.py
│   ├── train.py
│   ├── prediction.py
│   ├── upload.py
│   └── about.py
│
├── assets/                   # stylesheets
│   ├── base.css              # colours, cards, form, result card
│   ├── components.css        # menu, step bar, growth timeline, buttons
│   └── animations.css        # all motion effects (delete to switch off)
│
└── data/
    ├── crop_yield_dataset.csv     # sample data (synthetic), 5 crops with dates
    └── generate_sample_data.py    # script that created it
```

### Where to change things

| I want to... | Edit |
|---|---|
| Add a crop's growth stages or tip | `core/config.py` (`GROWTH_STAGES`, `CROP_ALIASES`, `CROP_TIPS`) |
| Add or tune a model | `core/models.py` (`make_models`) |
| Accept another file type | `core/data_io.py` (`read_table`) and `UPLOAD_TYPES` in `core/config.py` |
| Change a chart | `core/charts.py` |
| Change colours or animations | `assets/*.css` and `.streamlit/config.toml` |
| Add a page | new file in `views/`, then add it to `PAGES` in `core/config.py` and `ROUTES` in `app.py` |

## Pages

| Page | What it does |
|---|---|
| **Home** | Current metrics, charts, crop growth stages and Download CSV / Excel. Starts with the sample data and rebuilds itself from whatever dataset is loaded. |
| **Train Model** | Trains Random Forest, Gradient Boosting and SVM on the current dataset and compares MAE, RMSE and R². |
| **Prediction** | Locked until a model exists. Then: input form, predicted yield, growth calendar, four charts, recommendations. |
| **Upload & Train** | Upload any dataset or a trained model, check how the columns were understood, then train. |
| **About** | Purpose of the project, models, limitations. |

The order is: **data → train → predict**. The step bar at the top of each page shows where you are.

## Uploading your own data

Accepted files: **CSV, TSV, Excel (.xlsx), JSON, Jupyter notebook (.ipynb)**.

The app does not need fixed column names. It looks at each column and decides:

- **Number** → input, or the value to predict
- **Text** → category (for example crop, province, soil type)
- **Date** → turned into month and day of year. With a start and an end date (for example planting and harvest), the number of growing days is added as an input.

Requirements: at least 30 rows, and at least one numeric column to predict.

**About Jupyter notebooks:** a notebook holds code, not the dataset. The app takes the largest table shown in the notebook's saved outputs. If the notebook only shows a short preview, the app tells you which data file the notebook reads, so you can upload that file instead.

## Sample data and crops

The sample dataset is **synthetic** (2,500 fields, 2019–2024) with five crops: **Rice, Chilli, Watermelon, Tomato, Banana**. Each row has a planting date and a harvest date. Growth-stage calendars for these five crops are in `core/config.py`. Stage lengths are typical values and vary by variety and season.

## Notifications

The app shows a pop-up and a message on the page when something is wrong, for example:

- unreadable file, wrong file type, or fewer than 30 rows
- a notebook with no data table
- no numeric column to predict, or no input column selected
- a target with too few different values
- an input column that is almost identical to the target (possible data leak)
- invalid model file
- opening Prediction before a model is trained
- field area of 0, a negative input, or a harvest date before the planting date
- an input outside the range of the training data (warning)

## Notes

- "Accuracy" for a regression problem is the R² score. The per-crop table on the Train Model page is the more honest measure, because crops have very different yields.
- Models learn the logarithm of the target when it is always positive, so low-yield crops such as rice are fitted as carefully as high-yield crops.
- Recommendations come from patterns the model found in the data, not from field trials.
- A `.pkl` model file can contain program code that runs when opened. Only upload model files you created or trust.
- Trained models are kept while the app session is open. Use **Download the best model** to save one.
