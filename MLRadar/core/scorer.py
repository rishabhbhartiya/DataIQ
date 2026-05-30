"""
scorer.py — ML Readiness Score (MRS)
=====================================
Scores every column AND the full dataset on a 0–100 scale,
broken into seven sub-dimensions, each with a letter grade A–F.

Seven Dimensions
-----------------
  1. Completeness    — missing value rate
  2. Distribution    — skewness + kurtosis combined
  3. Cardinality     — encoding complexity risk
  4. Leakage Risk    — name/correlation proximity to target
  5. Outlier Severity— IQR outlier rate
  6. Type Fitness    — how well dtype suits ML
  7. Consistency     — mixed types, suspicious values, near-zero variance

Grade Scale
-----------
  A  ≥ 90   — production-ready
  B  ≥ 75   — minor prep needed
  C  ≥ 60   — moderate prep needed
  D  ≥ 45   — significant issues
  F  < 45   — critical, fix before training

Dimension Weights (sum = 10)
-----------------------------
  Completeness     2.5   (most critical — missing = dead column)
  Leakage Risk     2.0   (silent killer)
  Outlier Severity 1.5   (distorts linear models, k-means, etc.)
  Distribution     1.0
  Cardinality      1.0
  Type Fitness     0.8
  Consistency      0.2
"""
from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────

DIMENSION_WEIGHTS: Dict[str, float] = {
    "completeness":    2.5,
    "leakage_risk":    2.0,
    "outlier_severity":1.5,
    "distribution":    1.0,
    "cardinality":     1.0,
    "type_fitness":    0.8,
    "consistency":     0.2,
}
_TOTAL_WEIGHT = sum(DIMENSION_WEIGHTS.values())   # 9.0

GRADE_THRESHOLDS: List[Tuple[float, str]] = [
    (90, "A"), (75, "B"), (60, "C"), (45, "D"), (0, "F"),
]

GRADE_COLORS: Dict[str, str] = {
    "A": "#22c55e",   # green
    "B": "#84cc16",   # lime
    "C": "#f59e0b",   # amber
    "D": "#f97316",   # orange
    "F": "#ef4444",   # red
}

GRADE_LABELS: Dict[str, str] = {
    "A": "Production-ready",
    "B": "Minor prep needed",
    "C": "Moderate prep needed",
    "D": "Significant issues",
    "F": "Critical — fix before training",
}

_ID_PATTERN = re.compile(
    r"^(id|uuid|key|index|rowid|row_id|record_id|pk|primary_key|seq|sequence)$",
    re.IGNORECASE,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def letter_grade(score: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


def grade_color(grade: str) -> str:
    return GRADE_COLORS.get(grade, "#94a3b8")


def grade_label(grade: str) -> str:
    return GRADE_LABELS.get(grade, "")


# ── main class ─────────────────────────────────────────────────────────────────

class ReadinessScorer:
    """
    Compute ML Readiness Scores for every column and for the full dataset.

    Parameters
    ----------
    df     : pd.DataFrame   — the dataset to score
    target : str, optional  — target column name (used for leakage detection)

    Example
    -------
    >>> scorer = ReadinessScorer(df, target="churn")
    >>> report = scorer.score()
    >>> print(report["dataset_score"], report["dataset_grade"])
    """

    def __init__(self, df: pd.DataFrame, target: Optional[str] = None):
        self.df     = df.copy()
        self.target = target
        self.n      = len(df)
        self.k      = len(df.columns)
        self._kinds = self._classify_columns()

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def score(self) -> Dict[str, Any]:
        """
        Run all scorers and return a full report dict.

        Returns
        -------
        {
            dataset_score   : float          — 0–100
            dataset_grade   : str            — A/B/C/D/F
            dataset_color   : str            — hex color
            dataset_label   : str            — human label
            dimension_avgs  : Dict[str,float]— avg score per dimension
            grade_dist      : Dict[str,int]  — count per grade
            columns         : List[Dict]     — per-column score records
            top_issues      : List[Dict]     — worst 5 columns (D or F)
            strong_cols     : List[Dict]     — best 5 columns (A)
        }
        """
        col_scores: List[Dict[str, Any]] = [
            self._score_column(col) for col in self.df.columns
        ]

        # Dataset score = weight numeric cols 1.5×, rest 1×
        col_weights = [
            1.5 if s["kind"] == "numeric" else 1.0
            for s in col_scores
        ]
        w_sum         = sum(col_weights)
        dataset_score = round(
            sum(s["total"] * w for s, w in zip(col_scores, col_weights)) / w_sum, 1
        )
        dataset_grade = letter_grade(dataset_score)

        # Dimension averages across all columns
        dim_names   = list(DIMENSION_WEIGHTS.keys())
        dim_avgs    = {
            d: round(float(np.mean([s["dimensions"][d] for s in col_scores])), 1)
            for d in dim_names
        }

        # Grade distribution
        grade_dist  = {g: sum(1 for s in col_scores if s["grade"] == g)
                       for g in ("A", "B", "C", "D", "F")}

        # Top issues (D or F, sorted worst first)
        top_issues  = sorted(
            [s for s in col_scores if s["grade"] in ("D", "F") and not s["is_target"]],
            key=lambda x: x["total"],
        )[:5]

        # Strongest columns (grade A, non-target)
        strong_cols = sorted(
            [s for s in col_scores if s["grade"] == "A" and not s["is_target"]],
            key=lambda x: -x["total"],
        )[:5]

        return {
            "dataset_score":  dataset_score,
            "dataset_grade":  dataset_grade,
            "dataset_color":  grade_color(dataset_grade),
            "dataset_label":  grade_label(dataset_grade),
            "dimension_avgs": dim_avgs,
            "grade_dist":     grade_dist,
            "columns":        col_scores,
            "top_issues":     top_issues,
            "strong_cols":    strong_cols,
        }

    # ═══════════════════════════════════════════════════════════════════
    # PER-COLUMN SCORER
    # ═══════════════════════════════════════════════════════════════════

    def _score_column(self, col: str) -> Dict[str, Any]:
        s    = self.df[col]
        kind = self._kinds.get(col, "categorical")

        dims: Dict[str, float] = {
            "completeness":     self._completeness(s),
            "distribution":     self._distribution(s, kind),
            "cardinality":      self._cardinality(s, kind),
            "leakage_risk":     self._leakage_risk(col, s, kind),
            "outlier_severity": self._outlier_severity(s, kind),
            "type_fitness":     self._type_fitness(s, kind),
            "consistency":      self._consistency(s, kind),
        }

        total = round(
            sum(dims[d] * DIMENSION_WEIGHTS[d] for d in dims) / _TOTAL_WEIGHT, 1
        )
        grade = letter_grade(total)

        return {
            "column":     col,
            "kind":       kind,
            "dtype":      str(s.dtype),
            "total":      total,
            "grade":      grade,
            "color":      grade_color(grade),
            "label":      grade_label(grade),
            "dimensions": dims,
            "verdicts":   self._verdicts(col, s, kind, dims),
            "is_target":  col == self.target,
            # quick stats always available for the HTML table
            "missing_pct": round(s.isnull().mean() * 100, 1),
            "unique":      int(s.nunique()),
        }

    # ═══════════════════════════════════════════════════════════════════
    # DIMENSION SCORERS  (all return float 0–100)
    # ═══════════════════════════════════════════════════════════════════

    # 1. Completeness ──────────────────────────────────────────────────
    def _completeness(self, s: pd.Series) -> float:
        pct = s.isnull().mean() * 100
        if pct == 0:    return 100.0
        if pct <   2:   return  95.0
        if pct <   5:   return  88.0
        if pct <  15:   return  70.0
        if pct <  30:   return  48.0
        if pct <  50:   return  22.0
        return 5.0

    # 2. Distribution (skewness + kurtosis) ────────────────────────────
    def _distribution(self, s: pd.Series, kind: str) -> float:
        if kind != "numeric":
            return 85.0   # neutral score for non-numeric
        clean = s.dropna()
        if len(clean) < 4:
            return 70.0
        try:
            sk = abs(float(clean.skew()))
            ku = abs(float(clean.kurt()))
        except Exception:
            return 70.0

        # Skewness score (0–70 points)
        sk_score = (
            70.0 if sk < 0.5  else
            60.0 if sk < 1.0  else
            42.0 if sk < 2.0  else
            22.0 if sk < 3.0  else
            10.0
        )
        # Kurtosis penalty (0–30 points)
        ku_score = (
            30.0 if ku < 1.0  else
            24.0 if ku < 3.0  else
            16.0 if ku < 7.0  else
            8.0
        )
        return round(sk_score + ku_score, 1)

    # 3. Cardinality ───────────────────────────────────────────────────
    def _cardinality(self, s: pd.Series, kind: str) -> float:
        if kind in ("numeric", "datetime", "boolean"):
            return 95.0
        if kind == "text":
            return 35.0   # always risky without NLP

        n_unique = s.nunique()
        n_total  = max(len(s.dropna()), 1)
        ratio    = n_unique / n_total

        if n_unique == 1:    return  20.0   # constant
        if n_unique == 2:    return 100.0   # binary — perfect
        if n_unique <= 10:   return  95.0
        if n_unique <= 20:   return  82.0
        if n_unique <= 50:   return  65.0
        if n_unique <= 100:  return  45.0
        if ratio > 0.90:     return  10.0   # near-unique → ID column
        return 28.0

    # 4. Leakage Risk  (high score = LOW risk) ─────────────────────────
    def _leakage_risk(self, col: str, s: pd.Series, kind: str) -> float:
        # Target column itself gets a neutral score
        if not self.target or col == self.target:
            return 100.0

        risk = 0.0
        col_l = col.lower()
        tgt_l = self.target.lower()

        # a) exact / substring name match
        if col_l == tgt_l:
            risk = max(risk, 98.0)
        elif tgt_l in col_l or col_l in tgt_l:
            risk = max(risk, 82.0)
        else:
            # shared long tokens (>3 chars)
            tgt_tokens = {t for t in re.split(r"[_\s\-]", tgt_l) if len(t) > 3}
            col_tokens = {t for t in re.split(r"[_\s\-]", col_l) if len(t) > 3}
            if tgt_tokens & col_tokens:
                risk = max(risk, 50.0)

        # b) numeric correlation with numeric target
        if kind == "numeric" and self.target in self.df.columns:
            tgt_s = self.df[self.target]
            if pd.api.types.is_numeric_dtype(tgt_s):
                try:
                    corr = abs(float(s.corr(tgt_s)))
                    if   corr > 0.97: risk = max(risk, 95.0)
                    elif corr > 0.90: risk = max(risk, 78.0)
                    elif corr > 0.80: risk = max(risk, 55.0)
                    elif corr > 0.70: risk = max(risk, 35.0)
                except Exception:
                    pass

        # c) ID-like column name
        if _ID_PATTERN.match(col):
            risk = max(risk, 72.0)

        return round(100.0 - risk, 1)

    # 5. Outlier Severity ──────────────────────────────────────────────
    def _outlier_severity(self, s: pd.Series, kind: str) -> float:
        if kind != "numeric":
            return 90.0
        clean = s.dropna()
        if len(clean) < 4:
            return 75.0
        try:
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                return 75.0
            fence_lo = q1 - 1.5 * iqr
            fence_hi = q3 + 1.5 * iqr
            n_out    = int(((clean < fence_lo) | (clean > fence_hi)).sum())
            pct      = n_out / len(clean) * 100
        except Exception:
            return 75.0

        if pct == 0:   return 100.0
        if pct <  0.5: return  95.0
        if pct <  1.0: return  88.0
        if pct <  3.0: return  72.0
        if pct <  7.0: return  50.0
        if pct < 15.0: return  28.0
        return 10.0

    # 6. Type Fitness ──────────────────────────────────────────────────
    def _type_fitness(self, s: pd.Series, kind: str) -> float:
        if kind == "boolean":   return 100.0
        if kind == "numeric":
            # Stored as object string → bad
            return 30.0 if s.dtype == object else 95.0
        if kind == "datetime":  return  68.0   # needs extraction
        if kind == "text":      return  32.0   # needs NLP
        # categorical
        if s.dtype.name == "category": return 95.0
        return 80.0

    # 7. Consistency ───────────────────────────────────────────────────
    def _consistency(self, s: pd.Series, kind: str) -> float:
        """
        Checks:
          - Near-constant (>95% same value)          → penalise
          - Mixed apparent types in object column    → penalise
          - All-zeros numeric                        → penalise
        """
        clean = s.dropna()
        if len(clean) == 0:
            return 50.0

        # Near-constant
        top_freq = clean.value_counts(normalize=True).iloc[0]
        if top_freq > 0.99:  return 15.0
        if top_freq > 0.95:  return 40.0

        # Numeric stored as object: check if values are mixed types
        if kind == "numeric" and s.dtype == object:
            try:
                pd.to_numeric(clean, errors="raise")
            except (ValueError, TypeError):
                return 25.0   # definitely mixed

        # All-zero numeric
        if kind == "numeric":
            try:
                if (clean == 0).all():
                    return 20.0
            except Exception:
                pass

        return 100.0

    # ═══════════════════════════════════════════════════════════════════
    # COLUMN TYPE CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════════

    def _classify_columns(self) -> Dict[str, str]:
        kinds: Dict[str, str] = {}
        for col in self.df.columns:
            s = self.df[col]
            if pd.api.types.is_bool_dtype(s):
                kinds[col] = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(s):
                kinds[col] = "datetime"
            elif pd.api.types.is_numeric_dtype(s):
                kinds[col] = "numeric"
            else:
                # Heuristic: long strings or near-unique → free text
                sample  = s.dropna().astype(str)
                avg_len = sample.str.len().mean() if len(sample) else 0
                uniq_r  = s.nunique() / max(len(sample), 1)
                if avg_len > 60 or uniq_r > 0.85:
                    kinds[col] = "text"
                else:
                    kinds[col] = "categorical"
        return kinds

    # ═══════════════════════════════════════════════════════════════════
    # HUMAN-READABLE VERDICTS
    # ═══════════════════════════════════════════════════════════════════

    def _verdicts(
        self,
        col: str,
        s: pd.Series,
        kind: str,
        dims: Dict[str, float],
    ) -> Dict[str, str]:
        v: Dict[str, str] = {}

        # --- Completeness ---
        miss_pct = s.isnull().mean() * 100
        if miss_pct == 0:
            v["completeness"] = "✓ Complete — no missing values"
        elif miss_pct < 5:
            v["completeness"] = f"{miss_pct:.1f}% missing — simple imputation sufficient"
        elif miss_pct < 15:
            v["completeness"] = f"{miss_pct:.1f}% missing — median/mode or KNN imputation recommended"
        elif miss_pct < 40:
            v["completeness"] = f"{miss_pct:.1f}% missing — significant; use MICE or add missing indicator"
        else:
            v["completeness"] = f"{miss_pct:.1f}% missing — critical; consider dropping this column"

        # --- Distribution ---
        if kind == "numeric":
            try:
                sk = abs(float(s.dropna().skew()))
                if sk < 0.5:
                    v["distribution"] = f"Normal distribution (skew={sk:.2f}) — no transform needed"
                elif sk < 1.0:
                    v["distribution"] = f"Moderate skew ({sk:.2f}) — usually acceptable"
                elif sk < 2.0:
                    v["distribution"] = f"High skew ({sk:.2f}) — log or Yeo-Johnson recommended"
                else:
                    v["distribution"] = f"Severe skew ({sk:.2f}) — transform required before linear models"
            except Exception:
                v["distribution"] = "Could not compute skewness"
        else:
            v["distribution"] = "N/A — non-numeric column"

        # --- Cardinality ---
        n_u = s.nunique()
        if kind == "text":
            v["cardinality"] = "Free-text — needs TF-IDF, tokenisation, or embeddings"
        elif n_u == 1:
            v["cardinality"] = "Constant column — zero variance, drop before training"
        elif n_u == 2:
            v["cardinality"] = f"Binary ({n_u} values) — Label Encode or leave as-is"
        elif n_u <= 10:
            v["cardinality"] = f"Low cardinality ({n_u}) — One-Hot Encoding safe"
        elif n_u <= 50:
            v["cardinality"] = f"Medium cardinality ({n_u}) — prefer Target or Frequency encoding"
        elif n_u <= 200:
            v["cardinality"] = f"High cardinality ({n_u}) — use Frequency/Hashing encoding"
        else:
            v["cardinality"] = f"Very high cardinality ({n_u}) — likely ID column; drop or hash"

        # --- Leakage Risk ---
        risk_score = 100.0 - dims["leakage_risk"]   # higher = more risk
        if risk_score >= 80:
            v["leakage_risk"] = "🔴 Critical leakage risk — column encodes the target"
        elif risk_score >= 55:
            v["leakage_risk"] = f"🟠 High leakage risk — strong target correlation or name overlap"
        elif risk_score >= 35:
            v["leakage_risk"] = f"🟡 Moderate leakage risk — monitor closely"
        else:
            v["leakage_risk"] = "✓ Low leakage risk — looks safe"

        # --- Outlier Severity ---
        if kind == "numeric":
            try:
                clean = s.dropna()
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr    = q3 - q1
                if iqr > 0:
                    n_out = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
                    pct   = n_out / len(clean) * 100
                    if pct == 0:
                        v["outlier_severity"] = "✓ No outliers detected"
                    elif pct < 1:
                        v["outlier_severity"] = f"{n_out} outliers ({pct:.1f}%) — negligible"
                    elif pct < 5:
                        v["outlier_severity"] = f"{n_out} outliers ({pct:.1f}%) — Winsorize recommended"
                    else:
                        v["outlier_severity"] = f"{n_out} outliers ({pct:.1f}%) — significant; cap or remove"
                else:
                    v["outlier_severity"] = "Zero IQR — constant or near-constant distribution"
            except Exception:
                v["outlier_severity"] = "Could not compute"
        else:
            v["outlier_severity"] = "N/A — non-numeric column"

        # --- Type Fitness ---
        score = dims["type_fitness"]
        if score >= 90:
            v["type_fitness"] = "✓ Optimal dtype for ML"
        elif score >= 65:
            v["type_fitness"] = "Good dtype — minor preprocessing needed (e.g. feature extraction)"
        elif score >= 45:
            v["type_fitness"] = "Suboptimal — datetime or high-cardinality; requires transformation"
        else:
            v["type_fitness"] = "Poor dtype — free-text or numeric stored as string; heavy prep needed"

        # --- Consistency ---
        score = dims["consistency"]
        if score >= 90:
            v["consistency"] = "✓ No consistency issues detected"
        elif score >= 50:
            v["consistency"] = "Near-constant distribution — very low variance, low information"
        else:
            v["consistency"] = "Consistency issue — constant, mixed types, or all-zero values detected"

        return v