# src/evaluate.py
"""
Comprehensive model evaluation and error analysis for Incident Routing.

Assesses:
- Top-1 and Top-3 accuracy
- Macro F1 and Weighted F1 scores
- Confusion Matrix and per-assignment-group misclassification patterns
- Inspects problematic / misclassified incidents for targeted triage review
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)

from src.preprocess import prepare_and_filter_data

TEST_PATH = Path("data/raw/test.csv")
MODEL_PATH = Path("models/saved_models/model.pkl")

TEXT_COL = "description"
CATEGORICAL_COLS = ["urgency", "impact"]
TARGET_COL = "assignment_group"
FEATURE_COLS = [TEXT_COL] + CATEGORICAL_COLS


def evaluate_model(
    model_path: Path = MODEL_PATH, test_path: Path = TEST_PATH
) -> pd.DataFrame:
    """Evaluate trained model against test data with top-k metrics and misclassification audit."""
    print("=" * 60)
    print("INCIDENT ROUTING - MODEL EVALUATION & ERROR ANALYSIS")
    print("=" * 60)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train a model first!")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    test_df = pd.read_csv(test_path)
    test_df = prepare_and_filter_data(test_df, text_col=TEXT_COL, target_col=TARGET_COL)

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    print(f"Loaded {len(test_df)} test incidents across {y_test.nunique()} target groups.")

    # Predictions and class probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    classes = model.classes_
    k_val = min(3, len(classes))

    # Top-1 & Top-3 Accuracy
    top1_acc = (y_test.values == y_pred).mean()
    top3_acc = top_k_accuracy_score(y_test, y_prob, k=k_val, labels=classes)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\nOVERALL METRICS:")
    print(f"  Top-1 Accuracy:  {top1_acc:.4%}")
    print(f"  Top-3 Accuracy:  {top3_acc:.4%}")
    print(f"  Macro F1:        {macro_f1:.4f}")
    print(f"  Weighted F1:     {weighted_f1:.4f}")

    # Detailed classification report
    print("\nPER-GROUP CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    labels = sorted(y_test.unique())
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels
    )
    print("\nCONFUSION MATRIX (Rows: Actual, Columns: Predicted):")
    print(cm.to_string())

    # Error analysis & misclassified sample extraction
    test_df["predicted_group"] = y_pred
    test_df["top_confidence"] = y_prob.max(axis=1)

    # Top-3 recommendations extraction
    top3_indices = np.argsort(y_prob, axis=1)[:, -k_val:][:, ::-1]
    test_df["top3_recommendations"] = [
        [classes[idx] for idx in row_indices] for row_indices in top3_indices
    ]
    test_df["in_top3"] = [
        actual in recs
        for actual, recs in zip(test_df[TARGET_COL], test_df["top3_recommendations"])
    ]

    misclassified = test_df[test_df[TARGET_COL] != test_df["predicted_group"]]
    print(f"\nMISCLASSIFICATION SUMMARY:")
    print(f"  Total misclassified tickets: {len(misclassified)} ({len(misclassified)/len(test_df):.2%})")

    if not misclassified.empty:
        print("\nSample Misclassifications for Root Cause Review:")
        sample_cols = [
            "description",
            TARGET_COL,
            "predicted_group",
            "top_confidence",
            "top3_recommendations",
        ]
        print(misclassified[sample_cols].head(5).to_string(index=False))
    else:
        print("  All test instances correctly routed to Top-1 group!")

    return test_df


if __name__ == "__main__":
    evaluate_model()
