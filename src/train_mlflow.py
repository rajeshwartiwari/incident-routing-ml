# src/train_mlflow.py
"""
Train the incident routing model with MLflow experiment tracking.

- Log all hyperparameters
- Log all metrics (train and test)
- Log the trained pipeline as an artifact
- Register the model in the Model Registry

Compare this to train.py to see the difference!
"""
from datetime import datetime
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from src.preprocess import TextCleaner, prepare_and_filter_data

TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
TEXT_COL = "description"
CATEGORICAL_COLS = ["urgency", "impact"]
TARGET_COL = "assignment_group"
FEATURE_COLS = [TEXT_COL] + CATEGORICAL_COLS
REGISTERED_MODEL_NAME = "incident-routing-model"

# Configure MLflow to use our tracking server
mlflow.set_tracking_uri("http://localhost:5001")

# All runs are grouped under this experiment name
mlflow.set_experiment("incident-routing")


def load_data():
    """Load, validate, and clean the training and test data."""
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Ensure no leakage columns and clean/filter invalid rows
    train_df = prepare_and_filter_data(train_df, text_col=TEXT_COL, target_col=TARGET_COL)
    test_df = prepare_and_filter_data(test_df, text_col=TEXT_COL, target_col=TARGET_COL)

    return (
        train_df[FEATURE_COLS],
        train_df[TARGET_COL],
        test_df[FEATURE_COLS],
        test_df[TARGET_COL],
    )


def build_pipeline(
    model_type: str = "LogisticRegression",
    C: float = 1.0,
    ngram_max: int = 2,
    min_df: int = 2,
) -> Pipeline:
    """Build preprocessing and classifier pipeline (LogisticRegression or LinearSVC)."""
    text_pipeline = Pipeline(
        [
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), min_df=min_df)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("text", text_pipeline, TEXT_COL),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    if model_type.lower() in ("linearsvc", "linear_svc", "svm"):
        # CalibratedClassifierCV wraps LinearSVC to output well-calibrated probabilities
        base_svc = LinearSVC(C=C, random_state=42, dual="auto")
        classifier = CalibratedClassifierCV(estimator=base_svc, cv=3)
    else:
        classifier = LogisticRegression(C=C, max_iter=1000, random_state=42)

    return Pipeline(
        [
            ("features", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_and_log_model(
    model_type: str = "LogisticRegression",
    C: float = 1.0,
    ngram_max: int = 2,
    min_df: int = 2,
    register: bool = True,
):
    """Train a model (LogReg or Linear SVM) and log everything to MLflow."""
    X_train, y_train, X_test, y_test = load_data()

    with mlflow.start_run():
        run_name = f"{model_type}_C{C}_ngram{ngram_max}_{datetime.now().strftime('%H%M%S')}"
        mlflow.set_tag("mlflow.runName", run_name)

        # Hyperparameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("C", C)
        mlflow.log_param("ngram_max", ngram_max)
        mlflow.log_param("min_df", min_df)

        # Data information
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        n_classes = y_train.nunique()
        mlflow.log_param("n_classes", n_classes)
        mlflow.log_param("majority_class_ratio", float(y_train.value_counts(normalize=True).max()))
        mlflow.log_param("features", ",".join(FEATURE_COLS))

        print(f"\nTraining {model_type}: C={C}, ngram_max={ngram_max}, min_df={min_df}")
        model = build_pipeline(model_type, C, ngram_max, min_df)
        model.fit(X_train, y_train)

        # Metrics for BOTH train and test sets to help detect overfitting
        classes = model.classes_
        for dataset_name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)

            top1_acc = accuracy_score(y, y_pred)
            k_val = min(3, len(classes))
            top3_acc = top_k_accuracy_score(y, y_prob, k=k_val, labels=classes)
            precision = precision_score(y, y_pred, average="macro", zero_division=0)
            recall = recall_score(y, y_pred, average="macro", zero_division=0)
            macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

            try:
                roc_auc = roc_auc_score(
                    y, y_prob, multi_class="ovr", average="macro", labels=classes
                )
            except ValueError:
                roc_auc = float("nan")

            mlflow.log_metric(f"{dataset_name}_top1_accuracy", top1_acc)
            mlflow.log_metric(f"{dataset_name}_top3_accuracy", top3_acc)
            mlflow.log_metric(f"{dataset_name}_precision_macro", precision)
            mlflow.log_metric(f"{dataset_name}_recall_macro", recall)
            mlflow.log_metric(f"{dataset_name}_macro_f1", macro_f1)
            mlflow.log_metric(f"{dataset_name}_weighted_f1", weighted_f1)
            mlflow.log_metric(f"{dataset_name}_roc_auc_ovr", roc_auc)

            print(
                f"  {dataset_name.upper()} - Top-1 Acc: {top1_acc:.4f}, "
                f"Top-3 Acc: {top3_acc:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}"
            )

        # Test confusion matrix as an artifact
        labels = list(classes)
        cm = pd.DataFrame(
            confusion_matrix(y_test, model.predict(X_test), labels=labels),
            index=labels,
            columns=labels,
        )
        mlflow.log_text(cm.to_csv(), "test_confusion_matrix.csv")

        if register:
            print(f"\nRegistering model '{REGISTERED_MODEL_NAME}' in MLflow Registry...")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
                input_example=X_train.iloc[:5],
            )

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print(f"View this run: http://localhost:5001/#/experiments/1/runs/{run_id}")

        return model


def run_experiment_sweep():
    """Run baseline models: Logistic Regression and Linear SVM with parameter variations."""
    print("=" * 60)
    print("RUNNING BASELINE EXPERIMENT SWEEP (LogReg & Linear SVM)")
    print("=" * 60)

    experiments = [
        {"model_type": "LogisticRegression", "C": 0.1, "ngram_max": 1},
        {"model_type": "LogisticRegression", "C": 1.0, "ngram_max": 2},
        {"model_type": "LogisticRegression", "C": 10.0, "ngram_max": 2},
        {"model_type": "LinearSVC", "C": 0.5, "ngram_max": 1},
        {"model_type": "LinearSVC", "C": 1.0, "ngram_max": 2},
    ]

    for i, params in enumerate(experiments, 1):
        print(f"\n--- Experiment {i}/{len(experiments)} ---")
        train_and_log_model(**params)

    print("\n" + "=" * 60)
    print("EXPERIMENT SWEEP COMPLETE!")
    print("=" * 60)
    print("\nView all experiments at: http://localhost:5001")
    print("Compare runs to find the best hyperparameters!")


if __name__ == "__main__":
    run_experiment_sweep()
