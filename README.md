# MHT-CET College Cutoff Predictor

An end-to-end machine learning web application that predicts the **minimum percentile** a student needs to get admission into a specific college, branch, and category (seat type) under Maharashtra's engineering admission process (MHT-CET / JEE(Main) / Merit-based entry).

Instead of manually scrolling through hundreds of pages of official cutoff PDFs, a student can pick their preferred **college**, **branch**, **category**, and **score type** on a simple web form and instantly get an estimated cutoff percentile, based on a model trained on real historical admission data.

---

## What this project does

- **Explores and cleans** real MHT-CET admission cutoff data (326 colleges, 95 branches, 77 seat categories, ~28,000 records) through a full EDA notebook
- **Engineers features**, removes data leakage, and reduces high-cardinality categorical noise
- **Trains and compares 6 regression models** (Linear Regression, Ridge, Lasso, Random Forest, XGBoost, CatBoost) via `GridSearchCV`, automatically selecting and saving the best performer
- **Serves live predictions** through a Flask web app with dropdown selectors populated directly from the training data (so users can only ever choose combinations the model actually understands)
- **Validates input** — if a selected branch was never offered at the selected college, the app explicitly tells the user instead of guessing
- **Deployed and live** on Render

---

## How it works, under the hood

1. **`notebook/1. EDA and Feature Engineering.ipynb`** — explores the raw cutoff data, removes columns that leak the answer (`max`, `mean`, `sum`, etc. — all derived from the target), groups rare colleges/branches/categories into `'Other'` to control dimensionality, and exports a clean CSV.
2. **`src/components/data_ingestion.py`** — reads that clean CSV, splits it into train/test sets, and saves raw copies to `artifacts/`.
3. **`src/components/data_transformation.py`** — builds a `ColumnTransformer` pipeline (One-Hot Encoding for categoricals, scaling for the numeric `count` feature) and saves it as `artifacts/preprocessor.pkl`.
4. **`src/components/model_trainer.py`** — grid-searches 6 regression models, picks the best by R² score, and saves it as `artifacts/model.pkl`.
5. **`src/pipeline/predict_pipeline.py`** — loads both artifacts, auto-fills the `count` feature from historical averages (since a real user wouldn't know this value), validates the college/branch combination exists, and returns a prediction.
6. **`app.py`** — a Flask app tying it all together, with `templates/index.html` (landing page) and `templates/home.html` (the prediction form).

**Current model performance:** CatBoost Regressor, R² ≈ 0.71 on the held-out test set.

---

## Tech stack

- **Data & ML:** pandas, numpy, scikit-learn, XGBoost, CatBoost
- **Web:** Flask, Gunicorn
- **Deployment:** Render

---

## Project structure

```
MHT-CET/
├── artifacts/                  # model.pkl, preprocessor.pkl, data.csv (tracked for deployment)
├── notebook/
│   ├── data/
│   │   ├── kaggle_pivot_min_descending.csv
│   │   └── college_cutoff_processed.csv
│   └── 1. EDA and Feature Engineering.ipynb
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/
│   ├── index.html
│   └── home.html
├── app.py
├── requirements.txt
├── runtime.txt
├── Procfile
└── setup.py
```

---

## How to run this on your own system

### 1. Clone the repository

```bash
git clone https://github.com/quirkyharsh/MHT-CET.git
cd MHT-CET
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the EDA & feature engineering notebook

Open `notebook/1. EDA and Feature Engineering.ipynb` in Jupyter/VS Code and run all cells top to bottom. This produces `notebook/data/college_cutoff_processed.csv`, which the rest of the pipeline depends on.

### 5. Run the training pipeline

```bash
python src/components/data_ingestion.py
```

This chains together data ingestion → transformation → model training, and populates `artifacts/` with `train.csv`, `test.csv`, `preprocessor.pkl`, and `model.pkl`. Expect this step to take a while (tree-based models with grid search on this data can take 15–30+ minutes depending on your machine).

### 6. Run the web app

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser, click through to the prediction form, and try it out.

---

## Live demo

Deployed on Render: https://mht-cet-1.onrender.com/

---

## Notes for anyone forking this

- `artifacts/train.csv` and `artifacts/test.csv` are intentionally **not** committed (they're regenerable training byproducts) — but `model.pkl`, `preprocessor.pkl`, and `data.csv` **are** committed, since the deployed app needs them and retraining on every deploy isn't practical.
- If you retrain the model, make sure your local `scikit-learn`/`xgboost`/`catboost` versions match what's pinned in `requirements.txt` — pickled models don't always unpickle cleanly across major library version changes.
