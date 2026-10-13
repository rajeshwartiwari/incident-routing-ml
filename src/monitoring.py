"""
Model and data monitoring module for Incident Routing.

Provides capabilities to:
- Detect input text and metadata drift using statistical testing (Kolmogorov-Smirnov / Chi-Square)
- Track prediction confidence distributions over time
- Calculate real-time agreement rates and reassignment metrics in shadow mode
- Check for drift alert thresholds that trigger model retraining
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.preprocess import clean_text


class IncidentDriftMonitor:
    """Monitor for detecting input distribution drift and performance decay in incident routing."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        text_column: str = "description",
        categorical_columns: Optional[List[str]] = None,
    ):
        self.reference = reference_data.copy()
        self.text_col = text_column
        self.cat_cols = categorical_columns or ["urgency", "impact"]

        # Compute reference features (e.g. text length, character count, word count)
        self.reference["text_length"] = self.reference[self.text_col].astype(str).str.len()
        self.reference["word_count"] = (
            self.reference[self.text_col].astype(str).apply(lambda t: len(t.split()))
        )

        self.history: List[Dict[str, Any]] = []

    def check_drift(
        self, current_data: pd.DataFrame, p_value_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Run statistical drift checks between reference and production data.

        - Two-sample Kolmogorov-Smirnov test on text length & word count distributions
        - Chi-Square test on categorical distributions (urgency, impact)
        """
        curr = current_data.copy()
        curr["text_length"] = curr[self.text_col].astype(str).str.len()
        curr["word_count"] = curr[self.text_col].astype(str).apply(lambda t: len(t.split()))

        feature_drift = {}
        drift_detected = False

        # 1. Text numerical characteristics (KS test)
        for num_feat in ["text_length", "word_count"]:
            ref_feat = self.reference[num_feat].dropna()
            curr_feat = curr[num_feat].dropna()

            if len(ref_feat) > 0 and len(curr_feat) > 0:
                ks_stat, p_val = stats.ks_2samp(ref_feat, curr_feat)
                is_drift = p_val < p_value_threshold
                if is_drift:
                    drift_detected = True
                feature_drift[num_feat] = {
                    "test": "Kolmogorov-Smirnov",
                    "statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_val), 4),
                    "drift_detected": bool(is_drift),
                }

        # 2. Categorical distribution shifts (Chi-square test)
        for cat_col in self.cat_cols:
            if cat_col in self.reference.columns and cat_col in curr.columns:
                ref_counts = self.reference[cat_col].value_counts()
                curr_counts = curr[cat_col].value_counts()

                categories = sorted(list(set(ref_counts.index) | set(curr_counts.index)))
                ref_freq = [ref_counts.get(c, 0) + 1 for c in categories]  # Laplace smoothing
                curr_freq = [curr_counts.get(c, 0) + 1 for c in categories]

                # Normalise to compare relative frequency
                ref_norm = np.array(ref_freq) / sum(ref_freq) * sum(curr_freq)
                chi2_stat, p_val = stats.chisquare(f_obs=curr_freq, f_exp=ref_norm)

                is_drift = p_val < p_value_threshold
                if is_drift:
                    drift_detected = True
                feature_drift[cat_col] = {
                    "test": "Chi-Square",
                    "statistic": round(float(chi2_stat), 4),
                    "p_value": round(float(p_val), 4),
                    "drift_detected": bool(is_drift),
                }

        summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_samples": len(self.reference),
            "current_samples": len(curr),
            "drift_detected": drift_detected,
            "feature_details": feature_drift,
            "retraining_recommended": drift_detected,
        }

        self.history.append(summary)
        return summary

    def audit_shadow_mode(
        self,
        predicted_groups: List[str],
        actual_analyst_groups: List[str],
        confidences: List[float],
        confidence_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """Audit shadow-mode deployment predictions against actual analyst decisions."""
        if len(predicted_groups) != len(actual_analyst_groups):
            raise ValueError("Predictions and analyst groups must have equal length")

        total = len(predicted_groups)
        agreements = [
            p.lower().strip() == a.lower().strip()
            for p, a in zip(predicted_groups, actual_analyst_groups)
        ]
        agreement_count = sum(agreements)

        # High confidence subgroup
        high_conf_mask = [c >= confidence_threshold for c in confidences]
        high_conf_total = sum(high_conf_mask)
        high_conf_agreement = sum(
            agree for agree, is_hc in zip(agreements, high_conf_mask) if is_hc
        )

        return {
            "total_incidents": total,
            "agreement_count": agreement_count,
            "overall_agreement_rate": round(agreement_count / total, 4) if total else 0.0,
            "high_confidence_total": high_conf_total,
            "high_confidence_coverage": round(high_conf_total / total, 4) if total else 0.0,
            "high_confidence_agreement_rate": round(
                high_conf_agreement / high_conf_total, 4
            )
            if high_conf_total
            else 0.0,
            "eligible_for_automated_routing": (
                high_conf_total / total >= 0.6 and (high_conf_agreement / high_conf_total) >= 0.90
            )
            if high_conf_total
            else False,
        }
