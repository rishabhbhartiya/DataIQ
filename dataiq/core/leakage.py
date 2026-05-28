"""
leakage.py — Leakage Detective
================================
Automatically detects six categories of data leakage that silently
destroy real-world model performance.

Leakage Categories
------------------
  1. Target Correlation   — numeric feature correlated > threshold with target
  2. Name Proximity       — column name shares tokens with target name
  3. Derived Feature      — column appears post-hoc derived from target
                            (e.g. churn_flag, is_churned, churn_date)
  4. Future Data          — datetime column contains values after a cutoff
  5. ID / Primary Key     — high-cardinality identifier column
  6. Constant / Zero-Var  — single-value column (trivially fits anything)

Each finding contains
---------------------
  column    : str    — affected column name
  category  : str    — one of the six categories above
  severity  : str    — "critical" | "high" | "medium" | "low"
  evidence  : str    — quantitative proof
  fix       : str    — plain-English recommended action
  icon      : str    — emoji for the HTML report

Severity Scale
--------------
  critical  — almost certain leakage, drop immediately
  high      — strong evidence, investigate before training
  medium    — suspicious pattern, manually verify
  low       — minor concern, keep an eye on
"""
from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── severity ordering (lower int = worse) ─────────────────────────────────────
_SEV_ORDER: Dict[str, int] = {
    "critical": 0,
    "high":     1,
    "medium":   2,
    "low":      3,
}

# ── ID-like column name pattern ────────────────────────────────────────────────
_ID_RE = re.compile(
    r"^(id|uuid|guid|key|index|rowid|row_id|record_id|pk|primary_key|seq|sequence|ref|reference)$",
    re.IGNORECASE,
)

# ── suffixes that suggest a derived / post-hoc column ─────────────────────────
_DERIVED_SUFFIX_RE = re.compile(
    r"(flag|label|tag|outcome|result|status|class|category|group|"
    r"score|rank|bucket|bin|band|tier|segment|_date|_time|_ts|_at|_on)$",
    re.IGNORECASE,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _tokens(name: str) -> set:
    """Split a column name into meaningful tokens (len > 3)."""
    return {t for t in re.split(r"[_\s\-]", name.lower()) if len(t) > 3}


def _worst(a: Dict, b: Dict) -> Dict:
    """Return the finding with the higher (worse) severity."""
    return a if _SEV_ORDER[a["severity"]] <= _SEV_ORDER[b["severity"]] else b


# ══════════════════════════════════════════════════════════════════════════════
class LeakageDetector:
    """
    Detect data leakage between features and the target variable.

    Parameters
    ----------
    df              : pd.DataFrame   — dataset to inspect
    target          : str, optional  — target column name
    corr_threshold  : float          — |Pearson r| above which leakage is flagged
                                       (default 0.85)
    future_cutoff   : str, optional  — ISO date string; datetime cols with values
                                       after this date are flagged as future leakage

    Example
    -------
    >>> ld = LeakageDetector(df, target="churn", corr_threshold=0.85)
    >>> report = ld.detect()
    >>> print(report["verdict"])
    >>> for f in report["findings"]:
    ...     print(f["column"], f["severity"], f["evidence"])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        corr_threshold: float = 0.85,
        future_cutoff: Optional[str] = None,
    ):
        self.df             = df.copy()
        self.target         = target
        self.corr_threshold = corr_threshold
        self.n              = len(df)

        self.future_cutoff: Optional[pd.Timestamp] = None
        if future_cutoff:
            try:
                self.future_cutoff = pd.Timestamp(future_cutoff)
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def detect(self) -> Dict[str, Any]:
        """
        Run all six detectors and return a consolidated report.

        Returns
        -------
        {
            target      : str
            findings    : List[Dict]   — one entry per leaky column (worst sev wins)
            n_critical  : int
            n_high      : int
            n_medium    : int
            n_low       : int
            n_total     : int
            safe_cols   : List[str]    — columns with no finding
            verdict     : str          — one-line summary
            risk_level  : str          — "CRITICAL"|"HIGH"|"MODERATE"|"CLEAN"
        }
        """
        raw: List[Dict] = []
        raw += self._target_correlation()
        raw += self._name_proximity()
        raw += self._derived_feature()
        raw += self._future_data()
        raw += self._id_column()
        raw += self._constant_column()

        # Deduplicate: one finding per column, keep worst severity
        best: Dict[str, Dict] = {}
        for f in raw:
            col = f["column"]
            best[col] = _worst(f, best[col]) if col in best else f

        findings = sorted(best.values(), key=lambda x: _SEV_ORDER[x["severity"]])

        counts = {s: sum(1 for f in findings if f["severity"] == s)
                  for s in ("critical", "high", "medium", "low")}

        safe_cols = [
            c for c in self.df.columns
            if c not in best and c != self.target
        ]

        risk_level = (
            "CRITICAL" if counts["critical"] >= 1 else
            "HIGH"     if counts["high"]     >= 2 else
            "MODERATE" if len(findings)      >= 1 else
            "CLEAN"
        )

        verdict = {
            "CRITICAL": "CRITICAL — Definite leakage detected, do not train until resolved",
            "HIGH":     "HIGH RISK — Strong leakage signals, investigate before training",
            "MODERATE": "MODERATE — Suspicious patterns found, manually verify",
            "CLEAN":    "CLEAN — No leakage patterns detected",
        }[risk_level]

        return {
            "target":     self.target,
            "findings":   findings,
            "n_critical": counts["critical"],
            "n_high":     counts["high"],
            "n_medium":   counts["medium"],
            "n_low":      counts["low"],
            "n_total":    len(findings),
            "safe_cols":  safe_cols,
            "risk_level": risk_level,
            "verdict":    verdict,
        }

    # ═══════════════════════════════════════════════════════════════════
    # DETECTOR 1 — Target Correlation
    # ═══════════════════════════════════════════════════════════════════

    def _target_correlation(self) -> List[Dict]:
        """Flag numeric features with |Pearson r| ≥ corr_threshold vs target."""
        findings: List[Dict] = []

        if not self.target or self.target not in self.df.columns:
            return findings

        tgt = self.df[self.target]
        if not pd.api.types.is_numeric_dtype(tgt):
            return findings

        num_cols = [
            c for c in self.df.select_dtypes(include=[np.number]).columns
            if c != self.target
        ]

        for col in num_cols:
            try:
                r = float(self.df[col].corr(tgt))
                if np.isnan(r):
                    continue
                abs_r = abs(r)
            except Exception:
                continue

            if abs_r < self.corr_threshold:
                continue

            severity = "critical" if abs_r >= 0.97 else "high"
            direction = "positively" if r > 0 else "negatively"

            findings.append({
                "column":   col,
                "category": "Target Correlation Leakage",
                "severity": severity,
                "evidence": (
                    f"Pearson r = {r:.4f} (|r| = {abs_r:.4f}) — {direction} correlated "
                    f"with target '{self.target}'"
                ),
                "fix": (
                    f"Drop '{col}' — it is almost certainly a post-hoc feature that "
                    f"encodes the target answer directly. A real-world model would "
                    f"never have access to this at prediction time."
                    if abs_r >= 0.97 else
                    f"Investigate '{col}' — the {abs_r:.2f} correlation with target "
                    f"'{self.target}' is suspiciously high. Verify it is available "
                    f"at prediction time."
                ),
                "icon": "🔴" if severity == "critical" else "🟠",
            })

        return findings

    # ═══════════════════════════════════════════════════════════════════
    # DETECTOR 2 — Name Proximity
    # ═══════════════════════════════════════════════════════════════════

    def _name_proximity(self) -> List[Dict]:
        """
        Flag columns whose name is semantically close to the target name.
        Uses three levels:
          - exact / full containment  → critical / high
          - shared long tokens (>3ch) → medium
        """
        findings: List[Dict] = []

        if not self.target:
            return findings

        tgt_l      = self.target.lower()
        tgt_tokens = _tokens(self.target)

        for col in self.df.columns:
            if col == self.target:
                continue

            col_l = col.lower()

            # Level 1 — exact or full substring
            if col_l == tgt_l:
                sev  = "critical"
                note = f"Column name '{col}' is identical to target '{self.target}'"
            elif tgt_l in col_l or col_l in tgt_l:
                sev  = "high"
                note = f"Column name '{col}' contains / is contained in target name '{self.target}'"
            else:
                # Level 2 — shared meaningful tokens
                shared = _tokens(col) & tgt_tokens
                if not shared:
                    continue
                sev  = "medium"
                note = f"Column '{col}' shares name tokens {sorted(shared)} with target '{self.target}'"

            findings.append({
                "column":   col,
                "category": "Name Proximity Leakage",
                "severity": sev,
                "evidence": note,
                "fix": (
                    f"Remove or rename '{col}' — name similarity to the target "
                    f"strongly suggests it encodes target information. "
                    f"If it is legitimately predictive, verify it is available at inference time."
                ),
                "icon": "⚠️",
            })

        return findings

    # ═══════════════════════════════════════════════════════════════════
    # DETECTOR 3 — Derived Feature
    # ═══════════════════════════════════════════════════════════════════

    def _derived_feature(self) -> List[Dict]:
        """
        Flag columns that appear to be derived from the target after the fact.
        Condition: column name shares a target token AND ends with a
        'result-like' suffix (flag, label, status, _date, _time …).
        """
        findings: List[Dict] = []

        if not self.target:
            return findings

        tgt_tokens = _tokens(self.target)

        for col in self.df.columns:
            if col == self.target:
                continue

            col_l      = col.lower()
            col_tokens = _tokens(col)

            has_target_token = bool(tgt_tokens & col_tokens)
            has_result_suffix = bool(_DERIVED_SUFFIX_RE.search(col_l))

            if not (has_target_token and has_result_suffix):
                continue

            findings.append({
                "column":   col,
                "category": "Derived Feature Leakage",
                "severity": "high",
                "evidence": (
                    f"'{col}' name suggests it was derived from target '{self.target}' "
                    f"(shared tokens + result-like suffix)"
                ),
                "fix": (
                    f"Verify '{col}' was not created using the target variable. "
                    f"If it was computed after observing the outcome, drop it immediately."
                ),
                "icon": "🧬",
            })

        return findings

    # ═══════════════════════════════════════════════════════════════════
    # DETECTOR 4 — Future Data
    # ═══════════════════════════════════════════════════════════════════

    def _future_data(self) -> List[Dict]:
        """
        Flag datetime columns that contain values after future_cutoff.
        If no cutoff is set, warn about columns with very recent max dates.
        """
        findings: List[Dict] = []

        dt_cols = [
            c for c in self.df.select_dtypes(include=["datetime64"]).columns
            if c != self.target
        ]

        for col in dt_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue

            col_max = s.max()

            if self.future_cutoff is not None:
                n_future = int((s > self.future_cutoff).sum())
                if n_future == 0:
                    continue
                pct = round(n_future / self.n * 100, 2)
                findings.append({
                    "column":   col,
                    "category": "Future Data Leakage",
                    "severity": "high",
                    "evidence": (
                        f"{n_future:,} rows ({pct}%) in '{col}' have dates "
                        f"after cutoff {self.future_cutoff.date()}"
                    ),
                    "fix": (
                        f"Remove or clip rows where '{col}' > {self.future_cutoff.date()}. "
                        f"A deployed model will never see future timestamps at prediction time."
                    ),
                    "icon": "⏰",
                })
            else:
                # Heuristic: max date within the last 30 days → warn
                days_old = (pd.Timestamp.now() - col_max).days
                if days_old <= 30:
                    findings.append({
                        "column":   col,
                        "category": "Future Data Leakage",
                        "severity": "medium",
                        "evidence": (
                            f"'{col}' has values as recent as {col_max.date()} "
                            f"({days_old} days ago) — potential future leakage"
                        ),
                        "fix": (
                            f"Set future_cutoff='YYYY-MM-DD' in LeakageDetector to enforce "
                            f"a hard temporal boundary and recheck."
                        ),
                        "icon": "⏰",
                    })

        return findings

    # ═══════════════════════════════════════════════════════════════════
    # DETECTOR 5 — ID / Primary Key
    # ═══════════════════════════════════════════════════════════════════

    def _id_column(self) -> List[Dict]:
        """
        Flag columns that look like identifiers:
          - column name matches _ID_RE  OR
          - uniqueness ratio > 95% with more than 50 unique values
        """
        findings: List[Dict] = []

        for col in self.df.columns:
            if col == self.target:
                continue

            s        = self.df[col]
            n_vals   = len(s.dropna())
            n_uniq   = s.nunique()
            ratio    = n_uniq / max(n_vals, 1)

            is_id_name  = bool(_ID_RE.match(col))
            # Ratio check only fires for non-float columns (int, str, object)
            # Continuous float columns legitimately have near-100% uniqueness
            is_float_col = pd.api.types.is_float_dtype(s)
            is_id_ratio  = ratio > 0.95 and n_uniq > 50 and not is_float_col

            if not (is_id_name or is_id_ratio):
                continue

            severity = "high" if (is_id_name and is_id_ratio) else "medium"

            evidence_parts = []
            if is_id_name:
                evidence_parts.append(f"column name '{col}' matches ID pattern")
            if is_id_ratio:
                evidence_parts.append(
                    f"uniqueness ratio = {ratio:.1%} ({n_uniq:,} unique / {n_vals:,} rows)"
                )

            findings.append({
                "column":   col,
                "category": "ID / Primary-Key Leakage",
                "severity": severity,
                "evidence": "; ".join(evidence_parts),
                "fix": (
                    f"Drop '{col}' — identifier columns carry zero predictive signal "
                    f"and cause tree models to memorise training rows."
                ),
                "icon": "🔑",
            })

        return findings

    # ═══════════════════════════════════════════════════════════════════
    # DETECTOR 6 — Constant / Zero-Variance
    # ═══════════════════════════════════════════════════════════════════

    def _constant_column(self) -> List[Dict]:
        """Flag columns with only one unique non-null value."""
        findings: List[Dict] = []

        for col in self.df.columns:
            if col == self.target:
                continue

            n_uniq = self.df[col].nunique(dropna=True)
            if n_uniq > 1:
                continue

            findings.append({
                "column":   col,
                "category": "Constant / Zero-Variance",
                "severity": "medium",
                "evidence": (
                    f"'{col}' has only {n_uniq} unique value(s) — "
                    f"zero variance, carries no information"
                ),
                "fix": (
                    f"Drop '{col}' — constant features cannot help any model "
                    f"and add noise to distance-based algorithms."
                ),
                "icon": "🟠",
            })

        return findings