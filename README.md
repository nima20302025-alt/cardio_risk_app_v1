<div align="center">

# ❤️ Cardiovascular Disease Risk Predictor

**An end-to-end ML system that predicts cardiovascular disease risk from routine clinical measurements — trained, explained, and deployed as a live web app.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-8A2BE2)](https://github.com/shap/shap)
[![Deployed on Railway](https://img.shields.io/badge/Deployed%20on-Railway-0B0D0E?logo=railway&logoColor=white)]([https://railway.app/]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[🔗 Live demo](https://cardiorisk-app-v1-production.up.railway.app/)** · [Features](#-features) · [Results](#-results) · [Run locally](#-run-locally) · [Deploy](#-deploy-to-railway)

</div>

---

## Overview

This project builds a full machine learning pipeline — from raw, noisy clinical
data to a deployed, explainable prediction tool — for estimating a patient's
risk of cardiovascular disease (CVD).

It's trained on the [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
(Kaggle, 70,000 patient records: age, blood pressure, cholesterol, glucose,
BMI, and lifestyle factors). Five classifiers are trained and compared,
the best is selected by validation ROC-AUC, and the exact fitted pipeline
is served through an interactive Streamlit app with per-patient
explainability, batch scoring, and dataset exploration — not just a static
notebook.

## 📸 Screenshots

<div align="center">
<i>Add a screenshot or short GIF of the Predict tab, the SHAP explanation panel, and the Explore Data tab here.</i>
</div>

## 🚀 Features

| | |
|---|---|
| 🔍 **Live prediction** | Enter a patient's vitals in the sidebar and get an instant risk score with a gauge visualization. |
| 🧠 **Per-patient explainability** | Every prediction includes a SHAP breakdown of exactly which of *that patient's* measurements pushed their risk up or down — not just global feature importance. |
| 📄 **Batch prediction** | Upload a CSV of many patients, score them all at once, and download the results (with a ready-made CSV template). |
| 📈 **Dataset exploration** | Interactive charts of age, BMI, blood pressure, and cholesterol against CVD outcomes, plus a feature/target correlation chart. |
| 📊 **Model performance dashboard** | Validation comparison across five model families, feature importance, and a confusion matrix on a held-out test set. |
| 🧼 **Reproducible pipeline** | Documented cleaning of physiologically implausible readings and feature engineering, encapsulated in a single `train_model.py` script. |
| 🚀 **One-command deploy** | Ships to Railway with a pre-deploy step that retrains the model against the exact dependency versions installed at deploy time (see [Engineering notes](#-engineering-notes)). |

## 🏗️ Architecture

```
┌──────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│  cardio_train.csv│────▶│   train_model.py   │────▶│ cardio_pipeline     │
│  (Kaggle, 70k     │     │  clean → engineer  │     │ .joblib + metrics   │
│  patient records) │     │  → compare 5 models│     │      .json          │
└──────────────────┘     │  → select best     │     └──────────┬──────────┘
                          └───────────────────┘                │
                                                                 ▼
                                                      ┌────────────────────┐
                                                      │       app.py        │
                                                      │  Streamlit UI:       │
                                                      │  • Predict + SHAP    │
                                                      │  • Batch scoring     │
                                                      │  • Data exploration  │
                                                      │  • Model dashboard   │
                                                      └──────────┬──────────┘
                                                                 │
                                                                 ▼
                                                        Deployed on Railway
```

## 📊 Results

**Best model: Gradient Boosting** (selected by validation ROC-AUC)

| Model | Validation Accuracy | Validation ROC-AUC |
|---|---|---|
| **Gradient Boosting** ⭐ | 72.9% | **0.796** |
| Logistic Regression | 71.9% | 0.787 |
| Decision Tree | 71.7% | 0.783 |
| K-Nearest Neighbors | 71.8% | 0.773 |
| Random Forest | 70.5% | 0.764 |

**Held-out test set** (10,292 patients, evaluated once):

| Metric | Score |
|---|---|
| Accuracy | 74.2% |
| ROC-AUC | 0.810 |
| Precision (CVD) | 76.2% |
| Recall (CVD) | 69.5% |
| F1-score (CVD) | 72.7% |

Top predictors of risk (by feature importance): **systolic blood pressure**
(74%), **age** (12%), and **cholesterol level** (7%) dominate the model —
consistent with established cardiovascular risk factors, which was a useful
sanity check on the pipeline.

## 🛠️ Tech stack

| Layer | Tools |
|---|---|
| Data & modeling | pandas, NumPy, scikit-learn |
| Explainability | SHAP |
| Visualization | Plotly |
| App / UI | Streamlit |
| Deployment | Railway, Nixpacks |
| Model persistence | joblib |

## 📁 Project structure

```
.
├── app.py                    # Streamlit app (UI + inference + explainability)
├── train_model.py            # Data cleaning, feature engineering, model training & selection
├── requirements.txt
├── Procfile                  # Railway/Heroku-style start command
├── railway.json              # Railway build/deploy config (incl. pre-deploy retrain step)
├── runtime.txt                # Pinned Python version
├── .streamlit/config.toml    # Streamlit server/theme config
├── data/
│   └── cardio_train.csv      # Training data (Kaggle)
└── models/
    ├── cardio_pipeline.joblib  # Trained, ready-to-serve pipeline
    └── metrics.json             # Saved evaluation metrics, read by the app
```

## 🧪 Methodology

1. **Cleaning** — drop rows with physiologically impossible readings (e.g.
   diastolic ≥ systolic blood pressure, extreme heights/weights); removes
   ~2% of rows.
2. **Feature engineering**
   - `age_years` — age converted from days to years
   - `bmi` — weight (kg) / height (m)²
   - `pulse_pressure` — systolic − diastolic blood pressure
3. **Model comparison** — Logistic Regression, Decision Tree, Random
   Forest, Gradient Boosting, and k-NN, each in a `StandardScaler` +
   classifier pipeline, evaluated on a stratified 70/15/15 train/val/test
   split. Class weights are balanced on tree models rather than relying on
   an external oversampling library, since the two classes are naturally
   near 50/50.
4. **Selection** — best model chosen by validation ROC-AUC, then scored
   once, and only once, on the held-out test set to avoid leakage from
   model selection.
5. **Explainability** — a SHAP explainer is built against the selected
   model and a sampled, scaled background of the training data, giving
   per-patient contribution breakdowns at inference time.
6. **Deployment** — the exact selected pipeline is serialized with
   `joblib`; Railway retrains it against the pinned dependency versions
   immediately before each deploy (see below).

## 🔧 Engineering notes

A few real decisions worth calling out (useful context if this comes up in
an interview):

- **Dropped the `imbalanced-learn` dependency.** The two classes in this
  dataset are naturally close to 50/50, so oversampling wasn't needed —
  `class_weight="balanced"` on the tree models is enough. One less
  dependency to break in production.
- **numpy/joblib version pinning across environments.** A model pickled
  with one numpy version can fail to unpickle on a host with a different
  one (`joblib`/`pickle` serialize numpy's internal RNG state). Rather
  than trying to pin identical versions everywhere, `railway.json` runs a
  **pre-deploy command** that retrains the model against whatever versions
  `requirements.txt` actually resolves to on the deploy host, so the
  pickle and the runtime are always self-consistent. `app.py` also has a
  defensive fallback: if loading the committed model ever fails for any
  reason, it retrains on the fly.

## ▶️ Run locally

```bash
git clone https://github.com/<your-username>/cardio-risk-app.git
cd cardio-risk-app
pip install -r requirements.txt

# (optional) retrain the model from scratch
python train_model.py --data data/cardio_train.csv --out models

streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## ☁️ Deploy to Railway

1. Push this repo to GitHub.
2. On [railway.app](https://railway.app), click **New Project → Deploy from
   GitHub repo** and select this repository.
3. Railway auto-detects Python via `requirements.txt` and uses the
   `Procfile` / `railway.json` start command:
   ```
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
4. The pre-deploy step in `railway.json` retrains the model against the
   installed dependency versions — no manual steps needed, no environment
   variables, no database.
5. Once deployed, Railway gives you a public URL — add it to the badges
   section at the top of this README.

## 🗺️ Roadmap

- [ ] FastAPI `/predict` REST endpoint alongside the Streamlit UI
- [ ] Population-percentile comparison ("this patient vs. their age group")
- [ ] Downloadable PDF risk report
- [ ] Unit tests for the cleaning/feature-engineering functions

## ⚠️ Disclaimer

This project is for educational and portfolio purposes. It is **not** a
medical device and should not be used to make real clinical decisions.

## 📚 Dataset & credit

- Dataset: [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
  by Svetlana Ulianova, Kaggle (CC0 license).

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
