"""
Streamlit app for the Cardiovascular Disease Risk Predictor.

Loads the pre-trained pipeline in models/cardio_pipeline.joblib and lets a
user enter patient measurements to get a live risk prediction (with a
per-patient SHAP explanation), score a batch of patients from a CSV, and
explore the training dataset.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = Path("models/cardio_pipeline.joblib")
METRICS_PATH = Path("models/metrics.json")
DATA_PATH = "data/cardio_train.csv"

FEATURE_COLUMNS = [
    "age_years",
    "gender",
    "height",
    "weight",
    "bmi",
    "ap_hi",
    "ap_lo",
    "pulse_pressure",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]

FEATURE_LABELS = {
    "age_years": "Age",
    "gender": "Gender",
    "height": "Height",
    "weight": "Weight",
    "bmi": "BMI",
    "ap_hi": "Systolic BP",
    "ap_lo": "Diastolic BP",
    "pulse_pressure": "Pulse pressure",
    "cholesterol": "Cholesterol",
    "gluc": "Glucose",
    "smoke": "Smoking",
    "alco": "Alcohol",
    "active": "Physical activity",
}

st.set_page_config(
    page_title="Cardiovascular Risk Predictor",
    page_icon="❤️",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the saved pipeline, retraining on the fly if it's missing or
    was pickled by an incompatible numpy/scikit-learn version (this can
    happen if the model file was trained locally and the deploy host
    installs different dependency versions than requirements.txt pins)."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.warning(
                f"Saved model could not be loaded ({e.__class__.__name__}); "
                "retraining now. This only happens once."
            )

    with st.spinner("No compatible model found — training now (~30s)..."):
        import train_model

        train_model.main(csv_path=DATA_PATH, out_dir="models")
        load_metrics.clear()  # force the metrics cache to pick up the fresh file

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading dataset for exploration...")
def load_clean_dataset():
    """Cleaned training data, used for both the EDA tab and the SHAP
    background sample. Downsampled for plotting/explaining performance."""
    import train_model as tm

    df = tm.load_and_clean(DATA_PATH)
    if len(df) > 10000:
        df = df.sample(10000, random_state=42)
    return df


@st.cache_resource
def get_shap_explainer(_pipeline):
    """Build a SHAP explainer for the pipeline's model, using a small
    scaled background sample from the training data. Returns None if
    SHAP isn't available or the model type isn't supported."""
    try:
        import shap
    except ImportError:
        return None

    try:
        df = load_clean_dataset()
        background_raw = df[FEATURE_COLUMNS].sample(
            n=min(100, len(df)), random_state=42
        )
        scaler = _pipeline.named_steps["scale"]
        model = _pipeline.named_steps["model"]
        background_scaled = scaler.transform(background_raw)
        explainer = shap.Explainer(
            model, background_scaled, feature_names=FEATURE_COLUMNS
        )
        return explainer
    except Exception:
        return None


def explain_patient(pipeline, patient: dict):
    """Return a Series of per-feature SHAP contributions (sorted) for one
    patient, or None if an explanation could not be computed."""
    explainer = get_shap_explainer(pipeline)
    if explainer is None:
        return None
    try:
        scaler = pipeline.named_steps["scale"]
        input_df = pd.DataFrame([patient], columns=FEATURE_COLUMNS)
        scaled_input = scaler.transform(input_df)
        shap_values = explainer(scaled_input)
        values = shap_values.values
        if values.ndim == 3:
            # (n_samples, n_features, n_classes) -> take the positive class
            values = values[0, :, -1]
        else:
            values = values[0]
        labels = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLUMNS]
        return pd.Series(values, index=labels).sort_values()
    except Exception:
        return None


def predict(pipeline, patient: dict):
    input_df = pd.DataFrame([patient], columns=FEATURE_COLUMNS)
    proba = pipeline.predict_proba(input_df)[0]
    pred = pipeline.predict(input_df)[0]
    return pred, proba


def sidebar_inputs():
    st.sidebar.header("Patient information")

    age_years = st.sidebar.slider("Age (years)", 18, 100, 50)
    gender = st.sidebar.radio("Gender", options=["Female", "Male"], horizontal=True)
    height = st.sidebar.slider("Height (cm)", 120, 220, 170)
    weight = st.sidebar.slider("Weight (kg)", 30, 200, 75)

    st.sidebar.subheader("Blood pressure")
    ap_hi = st.sidebar.slider("Systolic BP (ap_hi, mmHg)", 80, 250, 120)
    ap_lo = st.sidebar.slider("Diastolic BP (ap_lo, mmHg)", 40, 200, 80)

    st.sidebar.subheader("Labs & lifestyle")
    cholesterol = st.sidebar.selectbox(
        "Cholesterol level", options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[x],
    )
    gluc = st.sidebar.selectbox(
        "Glucose level", options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[x],
    )
    smoke = st.sidebar.checkbox("Smoker")
    alco = st.sidebar.checkbox("Drinks alcohol")
    active = st.sidebar.checkbox("Physically active", value=True)

    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo

    patient = {
        "age_years": float(age_years),
        "gender": 2 if gender == "Male" else 1,  # matches dataset encoding
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "pulse_pressure": pulse_pressure,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": int(smoke),
        "alco": int(alco),
        "active": int(active),
    }
    return patient, bmi, pulse_pressure


def render_predict_tab(pipeline, metrics):
    patient, bmi, pulse_pressure = sidebar_inputs()

    col_left, col_right = st.columns([1, 1.3])

    with col_left:
        st.subheader("Derived values")
        m1, m2 = st.columns(2)
        m1.metric("BMI", f"{bmi:.1f}")
        m2.metric("Pulse pressure", f"{pulse_pressure} mmHg")

        if bmi < 18.5:
            bmi_note = "Underweight"
        elif bmi < 25:
            bmi_note = "Normal range"
        elif bmi < 30:
            bmi_note = "Overweight"
        else:
            bmi_note = "Obese range"
        st.caption(f"BMI category: **{bmi_note}**")

        if patient["ap_hi"] >= 140 or patient["ap_lo"] >= 90:
            st.caption("⚠️ Blood pressure is in the hypertensive range (≥140/90).")

        predict_clicked = st.button("Predict risk", type="primary", use_container_width=True)

    with col_right:
        if predict_clicked or "last_patient" in st.session_state:
            if predict_clicked:
                st.session_state["last_patient"] = patient
            pred, proba = predict(pipeline, st.session_state["last_patient"])
            risk_pct = proba[1] * 100

            st.subheader("Prediction")
            if pred == 1:
                st.error(f"**High risk of cardiovascular disease** — {risk_pct:.1f}% estimated probability")
            else:
                st.success(f"**Low risk of cardiovascular disease** — {risk_pct:.1f}% estimated probability")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_pct,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#d62728" if pred == 1 else "#2ca02c"},
                        "steps": [
                            {"range": [0, 33], "color": "#eafaf1"},
                            {"range": [33, 66], "color": "#fff8e1"},
                            {"range": [66, 100], "color": "#fdecea"},
                        ],
                    },
                    title={"text": "Estimated CVD probability"},
                )
            )
            gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(gauge, use_container_width=True)

            st.caption(
                "This is a statistical estimate from a model trained on population data — "
                "not a medical diagnosis. Always consult a clinician."
            )

            with st.expander("🔍 Why this prediction? (per-patient explanation)", expanded=True):
                with st.spinner("Computing explanation..."):
                    contributions = explain_patient(pipeline, st.session_state["last_patient"])
                if contributions is None:
                    st.info(
                        "A per-patient explanation isn't available for this model/environment."
                    )
                else:
                    colors = ["#d62728" if v > 0 else "#2ca02c" for v in contributions.values]
                    fig_exp = go.Figure(
                        go.Bar(
                            x=contributions.values,
                            y=contributions.index,
                            orientation="h",
                            marker_color=colors,
                        )
                    )
                    fig_exp.update_layout(
                        title="Feature contributions to this patient's risk score",
                        xaxis_title="← pushes risk down · pushes risk up →",
                        height=380,
                        margin=dict(t=40, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig_exp, use_container_width=True)
                    st.caption(
                        "Red bars increase this patient's predicted risk; green bars decrease it. "
                        "Computed with SHAP against a sample of the training population."
                    )
        else:
            st.info("Set the patient's details in the sidebar, then click **Predict risk**.")


def render_batch_tab(pipeline):
    st.subheader("Score multiple patients from a CSV")
    st.write(
        "Upload a CSV with one row per patient to get risk predictions for "
        "all of them at once, and download the results."
    )

    template_df = pd.DataFrame(
        [
            {
                "age_years": 50, "gender": 2, "height": 170, "weight": 75,
                "ap_hi": 120, "ap_lo": 80, "cholesterol": 1, "gluc": 1,
                "smoke": 0, "alco": 0, "active": 1,
            }
        ]
    )
    st.download_button(
        "⬇️ Download CSV template",
        template_df.to_csv(index=False),
        file_name="patient_template.csv",
        mime="text/csv",
    )
    st.caption(
        "Required columns: age_years (or age in days), gender (1=female, 2=male), "
        "height (cm), weight (kg), ap_hi, ap_lo, cholesterol (1-3), gluc (1-3), "
        "smoke, alco, active (0/1). bmi and pulse_pressure are computed automatically "
        "if not provided."
    )

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Couldn't read that file: {e}")
        return

    df = raw.copy()
    if "age_years" not in df.columns and "age" in df.columns:
        df["age_years"] = (df["age"] / 365.25).round(1)
    if "bmi" not in df.columns and {"height", "weight"}.issubset(df.columns):
        df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    if "pulse_pressure" not in df.columns and {"ap_hi", "ap_lo"}.issubset(df.columns):
        df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    with st.spinner(f"Scoring {len(df):,} patients..."):
        X = df[FEATURE_COLUMNS]
        proba = pipeline.predict_proba(X)[:, 1]
        pred = pipeline.predict(X)

    result = raw.copy()
    result["risk_probability_pct"] = (proba * 100).round(1)
    result["prediction"] = np.where(pred == 1, "High risk", "Low risk")

    st.success(f"Scored {len(result):,} patients.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Patients scored", f"{len(result):,}")
    c2.metric("Flagged high risk", f"{(pred == 1).sum():,}")
    c3.metric("Average risk", f"{proba.mean() * 100:.1f}%")

    st.dataframe(result, use_container_width=True)

    fig = px.histogram(
        result, x="risk_probability_pct", nbins=20,
        title="Distribution of predicted risk across uploaded patients",
        labels={"risk_probability_pct": "Predicted risk (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "⬇️ Download results as CSV",
        result.to_csv(index=False),
        file_name="cardio_predictions.csv",
        mime="text/csv",
        type="primary",
    )


def render_eda_tab():
    st.subheader("Explore the training data")
    df = load_clean_dataset()
    if df is None or df.empty:
        st.warning("Dataset not available for exploration.")
        return

    st.caption(
        f"{len(df):,} patients shown (sampled from the cleaned training set) — "
        f"{df['cardio'].mean() * 100:.1f}% have cardiovascular disease."
    )
    df_plot = df.copy()
    df_plot["CVD status"] = df_plot["cardio"].map({0: "No CVD", 1: "CVD"})

    col1, col2 = st.columns(2)
    with col1:
        fig_age = px.histogram(
            df_plot, x="age_years", color="CVD status", barmode="overlay",
            nbins=40, opacity=0.7, title="Age distribution by CVD status",
        )
        st.plotly_chart(fig_age, use_container_width=True)
    with col2:
        fig_bmi = px.histogram(
            df_plot, x="bmi", color="CVD status", barmode="overlay",
            nbins=40, opacity=0.7, range_x=[15, 50],
            title="BMI distribution by CVD status",
        )
        st.plotly_chart(fig_bmi, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        sample = df_plot.sample(min(3000, len(df_plot)), random_state=1)
        fig_bp = px.scatter(
            sample, x="ap_hi", y="ap_lo", color="CVD status", opacity=0.4,
            title="Blood pressure by CVD status (sample of patients)",
            labels={"ap_hi": "Systolic BP", "ap_lo": "Diastolic BP"},
        )
        st.plotly_chart(fig_bp, use_container_width=True)
    with col4:
        chol_rate = df_plot.groupby("cholesterol")["cardio"].mean().reset_index()
        chol_rate["cholesterol"] = chol_rate["cholesterol"].map(
            {1: "Normal", 2: "Above normal", 3: "Well above normal"}
        )
        fig_chol = px.bar(
            chol_rate, x="cholesterol", y="cardio",
            title="CVD rate by cholesterol level",
            labels={"cardio": "CVD rate", "cholesterol": "Cholesterol level"},
        )
        fig_chol.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_chol, use_container_width=True)

    st.markdown("#### Feature correlation with CVD diagnosis")
    corr = df[FEATURE_COLUMNS + ["cardio"]].corr()["cardio"].drop("cardio").sort_values()
    corr.index = [FEATURE_LABELS.get(c, c) for c in corr.index]
    fig_corr = px.bar(corr, orientation="h", labels={"value": "correlation", "index": ""})
    fig_corr.update_layout(showlegend=False, title="Which measurements track most with CVD?")
    st.plotly_chart(fig_corr, use_container_width=True)


def render_model_tab(metrics):
    if metrics is None:
        st.warning("No metrics.json found. Run train_model.py first.")
        return

    st.subheader(f"Best model: {metrics['best_model']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
    c2.metric("Test ROC-AUC", f"{metrics['test_auc']:.3f}")
    c3.metric("Rows after cleaning", f"{metrics['n_rows_after_cleaning']:,}")

    st.markdown("#### Validation comparison across models")
    val_df = pd.DataFrame(metrics["validation_results"]).T.reset_index()
    val_df.columns = ["model", "accuracy", "auc"]
    fig = px.bar(
        val_df.melt(id_vars="model", var_name="metric", value_name="score"),
        x="model", y="score", color="metric", barmode="group",
        title="Validation accuracy & AUC by model",
    )
    st.plotly_chart(fig, use_container_width=True)

    if metrics.get("feature_importances"):
        st.markdown("#### Feature importance")
        fi = pd.Series(metrics["feature_importances"]).sort_values(ascending=True)
        fi.index = [FEATURE_LABELS.get(c, c) for c in fi.index]
        fig2 = px.bar(fi, orientation="h", labels={"value": "importance", "index": "feature"})
        fig2.update_layout(showlegend=False, title="What drives the model's predictions")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Confusion matrix (test set)")
    cm = np.array(metrics["confusion_matrix"])
    fig3 = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=["Predicted: No CVD", "Predicted: CVD"],
        y=["Actual: No CVD", "Actual: CVD"],
    )
    fig3.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Full classification report"):
        st.json(metrics["classification_report"])


def render_about_tab():
    st.markdown(
        """
### About this project

This app predicts the risk of cardiovascular disease (CVD) from routine
clinical measurements, using the
[Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
(70,000 patient records, Kaggle).

**Pipeline**
1. **Cleaning** — remove physiologically implausible readings (e.g.
   diastolic > systolic blood pressure, extreme heights/weights).
2. **Feature engineering** — age in years (from age in days), BMI, and
   pulse pressure (systolic − diastolic).
3. **Model selection** — Logistic Regression, Decision Tree, Random Forest,
   Gradient Boosting, and k-NN are compared on a held-out validation split;
   the model with the best ROC-AUC is kept.
4. **Evaluation** — the chosen model is scored once on a held-out test set
   (see the *Model performance* tab).

**App features**
- 🔍 Live single-patient prediction with a SHAP-based explanation of what
  drove that specific prediction.
- 📄 Batch prediction — upload a CSV of patients, get risk scores and a
  downloadable results file.
- 📈 Interactive exploration of the training dataset.
- 📊 Full model performance dashboard.

**Stack:** scikit-learn (model), SHAP (explainability), Streamlit (UI),
deployed on Railway.

**Disclaimer:** this tool is for educational/portfolio purposes only. It
is not a medical device and must not be used for real clinical decisions.
        """
    )


def main():
    st.title("❤️ Cardiovascular Disease Risk Predictor")
    st.caption("A machine-learning app that estimates CVD risk from patient measurements.")

    pipeline = load_model()
    metrics = load_metrics()

    if pipeline is None:
        st.error(
            "No trained model found at `models/cardio_pipeline.joblib`. "
            "Run `python train_model.py` first to train and save the model."
        )
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔍 Predict", "📄 Batch prediction", "📈 Explore data", "📊 Model performance", "ℹ️ About"]
    )
    with tab1:
        render_predict_tab(pipeline, metrics)
    with tab2:
        render_batch_tab(pipeline)
    with tab3:
        render_eda_tab()
    with tab4:
        render_model_tab(metrics)
    with tab5:
        render_about_tab()


if __name__ == "__main__":
    main()
