# src/preprocess.py
"""
Data preprocessing, cleaning, and validation module for Incident Routing.

Provides functions to:
- Standardise and clean raw free-text descriptions
- Validate incident input schemas and check for missing/invalid fields
- Screen out fields that introduce data leakage (e.g. resolution notes, close codes)
- Build reproducible sklearn preprocessing transformers
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Allowed values for categorical attributes available at ticket creation
VALID_URGENCIES = {"Low", "Medium", "High"}
VALID_IMPACTS = {"Low", "Medium", "High"}
VALID_PRIORITIES = {"P1", "P2", "P3", "P4"}

# Fields known to introduce target or lookahead leakage if used during routing
LEAKAGE_COLUMNS = [
    "resolved_at",
    "closed_at",
    "close_code",
    "close_notes",
    "resolution_code",
    "resolution_notes",
    "reassignment_count",
    "assigned_to",
    "time_worked",
    "root_cause",
]

# Standard resolver groups targeted by the incident routing model
STANDARD_ASSIGNMENT_GROUPS = [
    "Application Support",
    "Database Operations",
    "End User Computing",
    "Identity and Access Management",
    "Messaging",
    "Network Operations",
    "Security Operations",
    "Service Desk",
]


def clean_text(text: Optional[str]) -> str:
    """Clean and standardise free-text incident descriptions.

    - Strips leading/trailing whitespace
    - Lowercases text
    - Removes ticket numbers/IDs (e.g., INC001234, PRB9921)
    - Replaces emails and URLs with placeholders
    - Collapses multiple whitespace characters
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = text.strip().lower()

    # Mask URLs and emails
    cleaned = re.sub(r"https?://\S+|www\.\S+", "<url>", cleaned)
    cleaned = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "<email>", cleaned)

    # Mask ticket IDs like INC-123456 or INC000123
    cleaned = re.sub(r"\b(inc|req|prb|chg)[-_]?\d+\b", "<ticket_id>", cleaned)

    # Collapse multiple whitespaces/tabs/newlines
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


class TextCleaner(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for text cleaning."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.apply(clean_text)
        elif isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].apply(clean_text)
        elif isinstance(X, list):
            return [clean_text(item) for item in X]
        return pd.Series(X).apply(clean_text)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return ["description"]
        return list(input_features)


def validate_incident_payload(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate incident fields available at creation time.

    Returns:
        (is_valid, list_of_errors)
    """
    errors: List[str] = []

    # Check for potential data leakage fields passed in payload
    leaked_fields = [k for k in data.keys() if k.lower() in LEAKAGE_COLUMNS]
    if leaked_fields:
        errors.append(f"Potential data leakage fields detected: {leaked_fields}")

    # Description validation
    desc = data.get("description")
    if not desc or not isinstance(desc, str) or not desc.strip():
        errors.append("description is required and must not be empty or whitespace-only")
    elif len(desc.strip()) < 5:
        errors.append("description is too short (must be at least 5 characters)")

    # Urgency validation
    urgency = data.get("urgency")
    if urgency not in VALID_URGENCIES:
        errors.append(f"urgency must be one of {sorted(VALID_URGENCIES)} (got '{urgency}')")

    # Impact validation
    impact = data.get("impact")
    if impact not in VALID_IMPACTS:
        errors.append(f"impact must be one of {sorted(VALID_IMPACTS)} (got '{impact}')")

    return (len(errors) == 0, errors)


def prepare_and_filter_data(
    df: pd.DataFrame,
    text_col: str = "description",
    target_col: Optional[str] = "assignment_group",
) -> pd.DataFrame:
    """Clean, deduplicate, and remove incomplete/leakage records from raw dataframe."""
    df_clean = df.copy()

    # Drop leakage columns if present
    drop_cols = [c for c in df_clean.columns if c.lower() in LEAKAGE_COLUMNS]
    if drop_cols:
        df_clean = df_clean.drop(columns=drop_cols)

    # Filter nulls or empty descriptions
    df_clean = df_clean.dropna(subset=[text_col])
    df_clean = df_clean[df_clean[text_col].astype(str).str.strip().str.len() >= 5]

    # Filter null targets if target column exists
    if target_col and target_col in df_clean.columns:
        df_clean = df_clean.dropna(subset=[target_col])
        df_clean = df_clean[df_clean[target_col].astype(str).str.strip() != ""]

    # Deduplicate on description and created_at if created_at exists
    subset = [text_col]
    if "created_at" in df_clean.columns:
        subset.append("created_at")
    df_clean = df_clean.drop_duplicates(subset=subset)

    return df_clean
