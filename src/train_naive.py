# src/train.py
"""
Train an incident routing model - NAIVE VERSION.

Predicts the `assignment_group` for an incident from its free-text
description plus urgency and impact. Category and subcategory are not used
because they are not reliably known when an incident is first raised and
would almost give away the routing label.

This is the "quick and dirty" approach:
- No experiment tracking
- No model versioning
- Just train and save to a pickle file
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.preprocess import TextCleaner, prepare_and_filter_data

TRAIN_PATH = Path("data/raw/train.csv")
TEST_PATH = Path("data/raw/test.csv")
MODEL_PATH = Path("models/saved_models/model.pkl")

TEXT_COL = "description"
CATEGORICAL_COLS = ["urgency", "impact"]
TARGET_COL = "assignment_group"


def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Validate, clean, and screen out potential data leakage
    train_df = prepare_and_filter_data(train_df, text_col=TEXT_COL, target_col=TARGET_COL)
    test_df = prepare_and_filter_data(test_df, text_col=TEXT_COL, target_col=TARGET_COL)

    print(f"Training samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    print(f"Number of assignment groups: {train_df[TARGET_COL].nunique()}")

    feature_cols = [TEXT_COL] + CATEGORICAL_COLS
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    # Preprocessor with TextCleaner to normalise incoming incident text
    text_pipeline = Pipeline(
        [
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("text", text_pipeline, TEXT_COL),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
    model = Pipeline(
        [
            ("features", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    print("\nTraining Logistic Regression model...")
    model.fit(X_train, y_train)
    print("Training complete!")

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    classes = model.classes_
    k_val = min(3, len(classes))

    print(f"\nTop-1 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Top-3 Accuracy: {top_k_accuracy_score(y_test, y_prob, k=k_val, labels=classes):.4f}")
    print(f"Macro F1:       {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Weighted F1:    {f1_score(y_test, y_pred, average='weighted'):.4f}")

    labels = sorted(y_train.unique())
    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels)
    print(cm.to_string())

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nMost influential terms per assignment group:")
    feature_names = model.named_steps["features"].get_feature_names_out()
    classifier = model.named_steps["classifier"]
    for group, coefs in zip(classifier.classes_, classifier.coef_):
        top = coefs.argsort()[-3:][::-1]
        print(f"  {group}: {', '.join(feature_names[i].split('__', 1)[1] for i in top)}")

    print(f"\nSaving model to {MODEL_PATH}...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("\nModel trained and saved successfully!")
    print("\nWARNING: This naive approach has several problems:")
    print("  - No record of hyperparameters or metrics")
    print("  - No model versioning")
    print("  - No way to reproduce this exact model")
    print("  - Synthetic data is templated, so scores are likely optimistic")


if __name__ == "__main__":
    main()
