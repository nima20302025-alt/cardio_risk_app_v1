"""
Train and save the cardiovascular disease risk prediction model.

Dataset: Cardiovascular Disease Dataset (Kaggle, sulianova)
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Run this once to (re)produce models/cardio_pipeline.joblib and
models/metrics.json, which the Streamlit app (app.py) loads at runtime.

Usage:
    python train_model.py --data data/cardio_train.csv
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
TARGET_COLUMN = "cardio"


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the raw Kaggle CSV, engineer features, and drop implausible rows."""
    df = pd.read_csv(csv_path, sep=";")

    # age is given in days -> years (interpretable + a better model feature)
    df["age_years"] = (df["age"] / 365.25).round(1)

    # Body Mass Index from height (cm) and weight (kg)
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

    # Pulse pressure = systolic - diastolic (clinically meaningful)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

    # Drop physiologically impossible readings -- a known data-quality
    # issue in this dataset (e.g. ap_lo > ap_hi, height of 55cm, etc.)
    before = len(df)
    df = df[
        (df["ap_hi"] >= 80)
        & (df["ap_hi"] <= 250)
        & (df["ap_lo"] >= 40)
        & (df["ap_lo"] <= 200)
        & (df["ap_hi"] > df["ap_lo"])
        & (df["height"] >= 120)
        & (df["height"] <= 220)
        & (df["weight"] >= 30)
        & (df["weight"] <= 200)
    ].copy()
    removed = before - len(df)
    print(f"Cleaned data: removed {removed} implausible rows ({removed / before:.2%})")
    return df


def build_candidates() -> dict:
    """Return the model family used for validation-set comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            random_state=42, solver="liblinear", max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Random Forest": RandomForestClassifier(
            random_state=42, n_estimators=300, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
    }


def main(csv_path: str, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(csv_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 70 / 15 / 15 stratified split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=(0.15 / 0.85),
        random_state=42,
        stratify=y_train_full,
    )
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # The dataset's classes are naturally ~50/50, so we rely on
    # class_weight="balanced" on the tree models rather than an
    # external oversampling dependency (keeps deployment lean).
    candidates = build_candidates()
    val_results = {}
    fitted = {}

    for name, clf in candidates.items():
        t0 = time.time()
        pipe = Pipeline(steps=[("scale", StandardScaler()), ("model", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        preds = pipe.predict(X_val)
        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, proba)
        val_results[name] = {"accuracy": acc, "auc": auc}
        fitted[name] = pipe
        print(f"{name:22s} | val_acc={acc:.4f} | val_auc={auc:.4f} | {time.time() - t0:.1f}s")

    best_name = max(val_results, key=lambda k: val_results[k]["auc"])
    best_pipeline = fitted[best_name]
    print(f"\nBest model on validation AUC: {best_name}")

    # Final, only-touched-once evaluation on the held-out test set
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_preds = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_proba)
    report = classification_report(y_test, test_preds, output_dict=True)
    cm = confusion_matrix(y_test, test_preds).tolist()

    print(f"Test accuracy: {test_acc:.4f} | Test AUC: {test_auc:.4f}")

    # Feature importance (tree models only) for the app's explainability tab
    importances = None
    model = best_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = dict(
            zip(FEATURE_COLUMNS, [float(v) for v in model.feature_importances_])
        )

    joblib.dump(best_pipeline, out_path / "cardio_pipeline.joblib")

    metrics = {
        "best_model": best_name,
        "validation_results": val_results,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_columns": FEATURE_COLUMNS,
        "feature_importances": importances,
        "n_rows_after_cleaning": int(len(df)),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {out_path / 'cardio_pipeline.joblib'}")
    print(f"Saved metrics to {out_path / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/cardio_train.csv", help="Path to cardio_train.csv"
    )
    parser.add_argument(
        "--out", default="models", help="Output directory for model + metrics"
    )
    args = parser.parse_args()
    main(args.data, args.out)
