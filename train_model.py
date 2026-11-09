"""
Train the Student Service Request SLA Breach model.
The bundled CSV is synthetic demonstration data. Replace it with authorized
midterm data and preserve the exact target/features required by your course.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "student_data.csv"
MODEL_PATH = ROOT / "student_model.pkl"
METRICS_PATH = ROOT / "model_metrics.json"
TARGET = "sla_breach"

CATEGORICAL = ["request_type", "priority", "submission_channel"]
NUMERIC = ["assigned_staff_workload", "processing_days", "missing_documents"]
FEATURES = CATEGORICAL + NUMERIC

def main():
    data = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    data = data.dropna(subset=[TARGET]).copy()
    X = data[FEATURES]
    y = data[TARGET].astype(int)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    prep = ColumnTransformer([
        ("categorical", categorical_pipe, CATEGORICAL),
        ("numeric", numeric_pipe, NUMERIC)
    ])
    model = Pipeline([
        ("preprocessing", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42
        ))
    ])

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "model_name": "RandomForestClassifier pipeline",
        "target": TARGET,
        "features": FEATURES,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "data_note": "Bundled dataset is synthetic demonstration data; not validated for real student-service decisions."
    }
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
