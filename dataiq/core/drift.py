"""
drift.py — Drift Analyzer
==========================
Detects statistical drift between two DataFrames:
  reference (train)  vs  new (test / production)

Three detection methods per column
------------------------------------
  Numeric columns
    • PSI   (Population Stability Index)   — industry standard, bin-based
    • KS    (Kolmogorov-Smirnov test)      — distribution shape change
    • Mean/Std shift                        — simple magnitude change

  Categorical columns
    • PSI   (category frequency based)
    • Chi²  (chi-square independence test)
    • New / disappeared categories          — schema-level change

  All columns
    • Missing-rate drift                   — Δ null % between ref and new

PSI Severity Scale  (industry standard)
-----------------------------------------
  PSI < 0.10  → Negligible  — no action needed
  PSI < 0.20  → Minor       — monitor
  PSI < 0.25  → Moderate    — investigate, plan retraining
  PSI ≥ 0.25  → Major       — retrain immediately

Output
------
  Per-column drift records + dataset-level summary + schema diff +
  missing-rate changes. Everything the HTML drift report needs.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── PSI thresholds ────────────────────────────────────────────────────────────
_PSI_LEVELS: List[Tuple[float, str, str]] = [
    (0.25, "Major",      "#ef4444"),
    (0.20, "Moderate",   "#f59e0b"),
    (0.10, "Minor",      "#84cc16"),
    (0.00, "Negligible", "#22c55e"),
]

_SEV_ORDER: Dict[str, int] = {
    "Major": 0, "Moderate": 1, "Minor": 2, "Negligible": 3
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _psi_level(psi: float) -> Tuple[str, str]:
    """Return (level_label, hex_color) for a PSI value."""
    for threshold, label, color in _PSI_LEVELS:
        if psi >= threshold:
            return label, color
    return "Negligible", "#22c55e"


def _psi_numeric(ref: np.ndarray, new: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index for two numeric arrays.
    Uses equal-width bins spanning the combined range of both arrays.
    Smooths zero-count bins to 1e-4 to avoid log(0).
    """
    if len(ref) == 0 or len(new) == 0:
        return 0.0

    mn = min(ref.min(), new.min())
    mx = max(ref.max(), new.max())

    if mx == mn:
        return 0.0

    edges       = np.linspace(mn, mx, buckets + 1)
    ref_counts  = np.histogram(ref, bins=edges)[0].astype(float)
    new_counts  = np.histogram(new, bins=edges)[0].astype(float)

    # Convert to proportions and smooth
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    new_pct = new_counts / max(new_counts.sum(), 1)
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    new_pct = np.where(new_pct == 0, 1e-4, new_pct)

    psi = float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))
    return round(abs(psi), 6)   # abs() to avoid -0.0 artefacts


def _psi_categorical(
    ref: pd.Series, new: pd.Series
) -> float:
    """
    PSI for categorical columns using category frequency distributions.
    Categories present in one but not the other are smoothed.
    """
    all_cats = set(ref.unique()) | set(new.unique())

    ref_freq = ref.value_counts(normalize=True)
    new_freq = new.value_counts(normalize=True)

    ref_pct = np.array([ref_freq.get(c, 1e-4) for c in all_cats])
    new_pct = np.array([new_freq.get(c, 1e-4) for c in all_cats])

    # Re-normalise after smoothing
    ref_pct = ref_pct / ref_pct.sum()
    new_pct = new_pct / new_pct.sum()

    psi = float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))
    return round(abs(psi), 6)


def _classify_col(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "categorical"


# ══════════════════════════════════════════════════════════════════════════════

class DriftAnalyzer:
    """
    Compare a reference dataset (train) against a new dataset (test / prod).

    Parameters
    ----------
    df_ref          : pd.DataFrame   — reference / training data
    df_new          : pd.DataFrame   — new / test / production data
    target          : str, optional  — target column (excluded from drift)
    psi_buckets     : int            — bins for numeric PSI (default 10)

    Example
    -------
    >>> da = DriftAnalyzer(df_train, df_test, target="churn")
    >>> report = da.analyze()
    >>> print(report["verdict"])
    >>> for col in report["columns"]:
    ...     print(col["column"], col["psi"], col["drift_level"])
    """

    def __init__(
        self,
        df_ref: pd.DataFrame,
        df_new: pd.DataFrame,
        target:      Optional[str] = None,
        psi_buckets: int           = 10,
    ):
        self.ref         = df_ref.copy()
        self.new         = df_new.copy()
        self.target      = target
        self.psi_buckets = psi_buckets
        self.n_ref       = len(df_ref)
        self.n_new       = len(df_new)

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def analyze(self) -> Dict[str, Any]:
        """
        Run full drift analysis and return a report dict.

        Returns
        -------
        {
            n_ref           : int
            n_new           : int
            n_cols_analyzed : int
            avg_psi         : float
            drift_level     : str        — dataset-level PSI label
            drift_color     : str        — hex color for that level
            verdict         : str        — one-line human summary
            columns         : List[Dict] — per-column drift records (sorted worst first)
            n_major         : int
            n_moderate      : int
            n_minor         : int
            n_negligible    : int
            missing_drift   : List[Dict] — columns with significant Δ null rate
            schema          : Dict       — new_only / ref_only / changed_dtype cols
        }
        """
        # Columns to analyze: in both frames, not the target
        common = [
            c for c in self.ref.columns
            if c in self.new.columns and c != self.target
        ]

        col_records: List[Dict] = []
        for col in common:
            kind = _classify_col(self.ref[col])
            if kind == "numeric":
                col_records.append(self._drift_numeric(col))
            elif kind == "categorical":
                col_records.append(self._drift_categorical(col))
            # datetime columns: skipped from PSI but included in missing drift

        # Sort worst-first
        col_records.sort(
            key=lambda x: (
                _SEV_ORDER.get(x.get("drift_level", "Negligible"), 3),
                -(x.get("psi") or 0),
            )
        )

        # Dataset-level PSI = mean of all column PSIs
        psi_vals = [r["psi"] for r in col_records if r.get("psi") is not None]
        avg_psi  = round(float(np.mean(psi_vals)), 4) if psi_vals else 0.0
        level, color = _psi_level(avg_psi)

        # Grade counts
        counts = {lv: sum(1 for r in col_records if r.get("drift_level") == lv)
                  for lv in ("Major", "Moderate", "Minor", "Negligible")}

        return {
            "n_ref":            self.n_ref,
            "n_new":            self.n_new,
            "n_cols_analyzed":  len(col_records),
            "avg_psi":          avg_psi,
            "drift_level":      level,
            "drift_color":      color,
            "verdict":          self._verdict(level, counts),
            "columns":          col_records,
            "n_major":          counts["Major"],
            "n_moderate":       counts["Moderate"],
            "n_minor":          counts["Minor"],
            "n_negligible":     counts["Negligible"],
            "missing_drift":    self._missing_drift(common),
            "schema":           self._schema_diff(),
        }

    # ═══════════════════════════════════════════════════════════════════
    # NUMERIC DRIFT
    # ═══════════════════════════════════════════════════════════════════

    def _drift_numeric(self, col: str) -> Dict[str, Any]:
        ref_s = self.ref[col].dropna()
        new_s = self.new[col].dropna()

        base = {
            "column": col,
            "type":   "numeric",
            "n_ref":  int(len(ref_s)),
            "n_new":  int(len(new_s)),
        }

        if len(ref_s) < 5 or len(new_s) < 5:
            return {**base, "psi": None, "drift_level": None,
                    "drift_color": "#94a3b8", "error": "Insufficient data (<5 non-null rows)"}

        ref_arr = ref_s.values.astype(float)
        new_arr = new_s.values.astype(float)

        # ── PSI ───────────────────────────────────────────────────────
        psi_val           = _psi_numeric(ref_arr, new_arr, self.psi_buckets)
        drift_level, color = _psi_level(psi_val)

        # ── KS Test ───────────────────────────────────────────────────
        ks_stat, ks_p = stats.ks_2samp(ref_arr, new_arr)

        # ── Mean / Std shift ──────────────────────────────────────────
        ref_mean, new_mean = float(np.mean(ref_arr)), float(np.mean(new_arr))
        ref_std,  new_std  = float(np.std(ref_arr)),  float(np.std(new_arr))
        ref_med,  new_med  = float(np.median(ref_arr)), float(np.median(new_arr))

        denom_mean = max(abs(ref_mean), 1e-9)
        denom_std  = max(abs(ref_std),  1e-9)
        mean_shift_pct = round(abs(new_mean - ref_mean) / denom_mean * 100, 2)
        std_shift_pct  = round(abs(new_std  - ref_std)  / denom_std  * 100, 2)

        # ── Percentile snapshot ───────────────────────────────────────
        pcts = [0, 5, 25, 50, 75, 95, 100]
        ref_pct_vals = {f"p{p}": round(float(np.percentile(ref_arr, p)), 4) for p in pcts}
        new_pct_vals = {f"p{p}": round(float(np.percentile(new_arr, p)), 4) for p in pcts}

        return {
            **base,
            "psi":            psi_val,
            "drift_level":    drift_level,
            "drift_color":    color,
            # KS
            "ks_stat":        round(float(ks_stat), 4),
            "ks_pvalue":      round(float(ks_p), 4),
            "ks_significant": bool(ks_p < 0.05),
            # Mean / Std
            "ref_mean":       round(ref_mean, 4),
            "new_mean":       round(new_mean, 4),
            "mean_shift_pct": mean_shift_pct,
            "ref_std":        round(ref_std, 4),
            "new_std":        round(new_std, 4),
            "std_shift_pct":  std_shift_pct,
            "ref_median":     round(ref_med, 4),
            "new_median":     round(new_med, 4),
            # Percentiles
            "ref_percentiles": ref_pct_vals,
            "new_percentiles": new_pct_vals,
        }

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORICAL DRIFT
    # ═══════════════════════════════════════════════════════════════════

    def _drift_categorical(self, col: str) -> Dict[str, Any]:
        ref_s = self.ref[col].dropna().astype(str)
        new_s = self.new[col].dropna().astype(str)

        base = {
            "column": col,
            "type":   "categorical",
            "n_ref":  int(len(ref_s)),
            "n_new":  int(len(new_s)),
        }

        if len(ref_s) < 5 or len(new_s) < 5:
            return {**base, "psi": None, "drift_level": None,
                    "drift_color": "#94a3b8", "error": "Insufficient data (<5 non-null rows)"}

        # ── PSI ───────────────────────────────────────────────────────
        psi_val           = _psi_categorical(ref_s, new_s)
        drift_level, color = _psi_level(psi_val)

        # ── Chi-square ────────────────────────────────────────────────
        chi2_val = chi_p = None
        chi_significant  = False
        try:
            all_cats  = sorted(set(ref_s.unique()) | set(new_s.unique()))
            ref_cnts  = np.array([int((ref_s == c).sum()) for c in all_cats])
            new_cnts  = np.array([int((new_s == c).sum()) for c in all_cats])
            # Add +1 smoothing so chi2 doesn't fail on zero cells
            chi2_val, chi_p = stats.chisquare(new_cnts + 1, f_exp=ref_cnts + 1)[:2]
            chi_significant  = bool(chi_p < 0.05)
        except Exception:
            pass

        # ── Category changes ──────────────────────────────────────────
        ref_cats = set(ref_s.unique())
        new_cats = set(new_s.unique())
        appeared  = sorted(new_cats - ref_cats)    # new cats in production
        disappeared = sorted(ref_cats - new_cats)  # cats that vanished

        # Top-10 frequency table for each side
        ref_freq = ref_s.value_counts(normalize=True).head(10)
        new_freq = new_s.value_counts(normalize=True).head(10)

        ref_table = [{"label": k, "pct": round(v * 100, 2)}
                     for k, v in ref_freq.items()]
        new_table = [{"label": k, "pct": round(v * 100, 2)}
                     for k, v in new_freq.items()]

        return {
            **base,
            "psi":              psi_val,
            "drift_level":      drift_level,
            "drift_color":      color,
            # Chi-square
            "chi2":             round(float(chi2_val), 4) if chi2_val is not None else None,
            "chi_pvalue":       round(float(chi_p), 4)   if chi_p   is not None else None,
            "chi_significant":  chi_significant,
            # Category changes
            "appeared":         appeared,
            "disappeared":      disappeared,
            "n_appeared":       len(appeared),
            "n_disappeared":    len(disappeared),
            "ref_n_unique":     int(ref_s.nunique()),
            "new_n_unique":     int(new_s.nunique()),
            # Frequency tables
            "ref_freq_table":   ref_table,
            "new_freq_table":   new_table,
        }

    # ═══════════════════════════════════════════════════════════════════
    # MISSING-RATE DRIFT
    # ═══════════════════════════════════════════════════════════════════

    def _missing_drift(self, cols: List[str]) -> List[Dict[str, Any]]:
        """
        Return columns where the null rate changed by more than 3 percentage
        points between reference and new dataset.
        """
        results: List[Dict] = []

        for col in cols:
            ref_miss = self.ref[col].isnull().mean() * 100
            new_miss = self.new[col].isnull().mean() * 100
            delta    = new_miss - ref_miss          # signed: + means more missing in new

            if abs(delta) < 3.0:
                continue

            results.append({
                "column":   col,
                "ref_miss": round(ref_miss, 2),
                "new_miss": round(new_miss, 2),
                "delta":    round(delta, 2),
                "direction": "increased" if delta > 0 else "decreased",
                "severity": "high"   if abs(delta) > 20 else
                            "medium" if abs(delta) > 10 else "low",
            })

        return sorted(results, key=lambda x: -abs(x["delta"]))

    # ═══════════════════════════════════════════════════════════════════
    # SCHEMA DIFF
    # ═══════════════════════════════════════════════════════════════════

    def _schema_diff(self) -> Dict[str, Any]:
        """
        Detect:
          - Columns present in new but not in reference (appeared)
          - Columns present in reference but not in new (disappeared)
          - Columns present in both but with changed dtype
        """
        ref_cols = set(self.ref.columns)
        new_cols = set(self.new.columns)

        appeared    = sorted(new_cols - ref_cols)
        disappeared = sorted(ref_cols - new_cols)

        dtype_changed: List[Dict] = []
        for col in ref_cols & new_cols:
            rt = str(self.ref[col].dtype)
            nt = str(self.new[col].dtype)
            if rt != nt:
                dtype_changed.append({
                    "column":    col,
                    "ref_dtype": rt,
                    "new_dtype": nt,
                })

        return {
            "appeared":      appeared,
            "disappeared":   disappeared,
            "dtype_changed": dtype_changed,
            "has_changes":   bool(appeared or disappeared or dtype_changed),
        }

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════

    def _verdict(self, level: str, counts: Dict[str, int]) -> str:
        n_maj = counts["Major"]
        n_mod = counts["Moderate"]
        if level == "Major":
            return (
                f"MAJOR DRIFT — {n_maj} column(s) have PSI ≥ 0.25. "
                f"Immediate retraining is strongly recommended."
            )
        if level == "Moderate":
            return (
                f"MODERATE DRIFT — {n_mod} column(s) show meaningful distribution shift. "
                f"Monitor performance and plan retraining."
            )
        if level == "Minor":
            return (
                "MINOR DRIFT — Small distribution changes detected. "
                "Normal variation; continue monitoring."
            )
        return "STABLE — No significant drift detected. Dataset distributions are consistent."