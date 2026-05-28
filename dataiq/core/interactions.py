"""
interactions.py — Feature Interaction Strength
================================================
Measures how strongly features interact with each other and with the
target variable using three complementary methods:

Methods
-------
  1. Mutual Information (MI)
       - Captures linear AND non-linear dependencies
       - Works for numeric↔numeric, numeric↔categorical
       - Normalized to [0, 1] so columns are comparable
       - Uses sklearn mutual_info_regression / mutual_info_classif

  2. Cramér's V
       - Association between two categorical columns
       - Normalized chi-square, range [0, 1]
       - 0 = independent, 1 = perfectly associated

  3. Correlation Ratio (η²)
       - Association between one categorical and one numeric column
       - Measures how much variance in numeric is explained by categorical
       - Range [0, 1]

Outputs
-------
  • Pairwise interaction matrix  (all methods combined)
  • Top-N strongest pairs        (sorted by score)
  • Feature-vs-target scores     (which features interact most with target)
  • Redundancy groups            (clusters of highly interacting features)
  • Heatmap-ready matrix dict    (for HTML report)

Strength Labels
---------------
  score ≥ 0.70  → Strong
  score ≥ 0.40  → Moderate
  score ≥ 0.15  → Weak
  score <  0.15 → Negligible
"""
from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── strength labels ────────────────────────────────────────────────────────────
_STRENGTH_LEVELS: List[Tuple[float, str, str]] = [
    (0.70, "Strong",     "#ef4444"),
    (0.40, "Moderate",   "#f59e0b"),
    (0.15, "Weak",       "#84cc16"),
    (0.00, "Negligible", "#94a3b8"),
]


def _strength(score: float) -> Tuple[str, str]:
    for threshold, label, color in _STRENGTH_LEVELS:
        if score >= threshold:
            return label, color
    return "Negligible", "#94a3b8"


# ── helpers ────────────────────────────────────────────────────────────────────

def _classify(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):           return "categorical"
    if pd.api.types.is_datetime64_any_dtype(s): return "datetime"
    if pd.api.types.is_numeric_dtype(s):        return "numeric"
    return "categorical"


def _safe_fillna(s: pd.Series, kind: str) -> pd.Series:
    """Fill missing so MI computation doesn't fail."""
    if kind == "numeric":
        return s.fillna(s.median())
    return s.fillna("__missing__")


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    """
    Cramér's V — association between two categorical columns.
    Returns 0.0 if computation fails.
    """
    try:
        from scipy.stats import chi2_contingency
        ct       = pd.crosstab(a.astype(str), b.astype(str))
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return 0.0
        chi2, _, _, _ = chi2_contingency(ct, correction=False)
        n        = ct.values.sum()
        min_dim  = min(ct.shape) - 1
        if min_dim == 0 or n == 0:
            return 0.0
        v = float(np.sqrt(chi2 / (n * min_dim)))
        return round(min(v, 1.0), 4)
    except Exception:
        return 0.0


def _correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    """
    Correlation ratio η² — how much variance in `num` is explained by `cat`.
    Returns 0.0 if computation fails.
    """
    try:
        combined = pd.DataFrame({"cat": cat, "num": num}).dropna()
        if len(combined) < 4:
            return 0.0
        overall_mean = combined["num"].mean()
        groups       = combined.groupby("cat")["num"]
        n_total      = len(combined)
        # Between-group sum of squares
        ss_between   = sum(
            len(g) * (g.mean() - overall_mean) ** 2
            for _, g in groups
        )
        # Total sum of squares
        ss_total = ((combined["num"] - overall_mean) ** 2).sum()
        if ss_total == 0:
            return 0.0
        eta2 = float(ss_between / ss_total)
        return round(min(eta2, 1.0), 4)
    except Exception:
        return 0.0


def _mi_numeric_pair(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalized Mutual Information between two numeric arrays.
    Normalizes by sqrt(MI(a,a) * MI(b,b)) so result is in [0,1].
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
        a2 = a.reshape(-1, 1)
        b2 = b.reshape(-1, 1)
        mi_ab  = float(mutual_info_regression(a2, b, random_state=42)[0])
        mi_aa  = float(mutual_info_regression(a2, a, random_state=42)[0])
        mi_bb  = float(mutual_info_regression(b2, b, random_state=42)[0])
        denom  = np.sqrt(max(mi_aa, 1e-9) * max(mi_bb, 1e-9))
        return round(min(mi_ab / denom, 1.0), 4)
    except Exception:
        return 0.0


def _mi_with_target(
    feature: np.ndarray,
    target: np.ndarray,
    is_classification: bool,
) -> float:
    """Mutual Information between a feature and the target (not normalized)."""
    try:
        if is_classification:
            from sklearn.feature_selection import mutual_info_classif
            mi = float(mutual_info_classif(
                feature.reshape(-1, 1), target, random_state=42
            )[0])
        else:
            from sklearn.feature_selection import mutual_info_regression
            mi = float(mutual_info_regression(
                feature.reshape(-1, 1), target, random_state=42
            )[0])
        return round(mi, 4)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════

class FeatureInteractions:
    """
    Compute pairwise feature interaction strengths.

    Parameters
    ----------
    df              : pd.DataFrame
    target          : str, optional — target column name
    max_cols        : int           — cap columns to avoid O(n²) explosion (default 25)
    top_n_pairs     : int           — how many top pairs to return (default 20)

    Example
    -------
    >>> fi = FeatureInteractions(df, target='churn')
    >>> report = fi.compute()
    >>> for pair in report['top_pairs'][:5]:
    ...     print(pair['col_a'], pair['col_b'], pair['score'], pair['strength'])
    """

    def __init__(
        self,
        df:           pd.DataFrame,
        target:       Optional[str] = None,
        max_cols:     int           = 25,
        top_n_pairs:  int           = 20,
    ):
        self.df          = df.copy()
        self.target      = target
        self.max_cols    = max_cols
        self.top_n_pairs = top_n_pairs
        self._kinds      = {c: _classify(df[c]) for c in df.columns}

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def compute(self) -> Dict[str, Any]:
        """
        Run all interaction computations.

        Returns
        -------
        {
            feature_cols        : List[str]   — columns analyzed
            pairs               : List[Dict]  — ALL pairwise scores
            top_pairs           : List[Dict]  — top_n strongest pairs
            target_scores       : List[Dict]  — each feature vs target
            matrix_cols         : List[str]   — column order for matrix
            matrix              : List[List]  — NxN score matrix (float)
            redundancy_groups   : List[List]  — groups of highly correlated features
            n_strong            : int
            n_moderate          : int
            summary             : str         — one-line human summary
        }
        """
        feature_cols = self._select_feature_cols()

        if len(feature_cols) < 2:
            return {
                "feature_cols":      feature_cols,
                "pairs":             [],
                "top_pairs":         [],
                "target_scores":     [],
                "matrix_cols":       feature_cols,
                "matrix":            [],
                "redundancy_groups": [],
                "n_strong":          0,
                "n_moderate":        0,
                "summary":           "Not enough columns to compute interactions.",
            }

        pairs          = self._compute_pairs(feature_cols)
        target_scores  = self._compute_target_scores(feature_cols)
        matrix_cols, matrix = self._build_matrix(feature_cols, pairs)
        redundancy_groups   = self._find_redundancy_groups(pairs)

        top_pairs = sorted(pairs, key=lambda x: -x["score"])[: self.top_n_pairs]

        n_strong   = sum(1 for p in pairs if p["strength"] == "Strong")
        n_moderate = sum(1 for p in pairs if p["strength"] == "Moderate")

        summary = self._summarize(n_strong, n_moderate, top_pairs, redundancy_groups)

        return {
            "feature_cols":      feature_cols,
            "pairs":             pairs,
            "top_pairs":         top_pairs,
            "target_scores":     target_scores,
            "matrix_cols":       matrix_cols,
            "matrix":            matrix,
            "redundancy_groups": redundancy_groups,
            "n_strong":          n_strong,
            "n_moderate":        n_moderate,
            "summary":           summary,
        }

    # ═══════════════════════════════════════════════════════════════════
    # COLUMN SELECTION
    # ═══════════════════════════════════════════════════════════════════

    def _select_feature_cols(self) -> List[str]:
        """
        Pick columns to analyze:
          - Exclude target, datetime, and free-text columns
          - Cap at max_cols (prefer numeric, then categorical by cardinality)
        """
        exclude = {self.target} if self.target else set()

        candidates: List[Tuple[int, str]] = []
        for col in self.df.columns:
            if col in exclude:
                continue
            kind = self._kinds[col]
            if kind == "datetime":
                continue
            # Heuristic: skip near-unique categoricals (likely ID or free text)
            if kind == "categorical":
                ratio = self.df[col].nunique() / max(len(self.df[col].dropna()), 1)
                if ratio > 0.85:
                    continue
            # Priority: numeric first (lower sort key = earlier)
            priority = 0 if kind == "numeric" else 1
            candidates.append((priority, col))

        candidates.sort(key=lambda x: x[0])
        return [col for _, col in candidates][: self.max_cols]

    # ═══════════════════════════════════════════════════════════════════
    # PAIRWISE INTERACTIONS
    # ═══════════════════════════════════════════════════════════════════

    def _compute_pairs(self, cols: List[str]) -> List[Dict[str, Any]]:
        """
        Compute interaction score for every pair of feature columns.
        Method chosen based on column types:
          numeric  × numeric     → Normalized MI
          category × category    → Cramér's V
          numeric  × category    → Correlation Ratio η²
        """
        pairs: List[Dict[str, Any]] = []

        for col_a, col_b in combinations(cols, 2):
            kind_a = self._kinds[col_a]
            kind_b = self._kinds[col_b]

            score, method = self._pair_score(col_a, kind_a, col_b, kind_b)
            strength, color = _strength(score)

            pairs.append({
                "col_a":    col_a,
                "col_b":    col_b,
                "score":    score,
                "method":   method,
                "strength": strength,
                "color":    color,
                "kind_a":   kind_a,
                "kind_b":   kind_b,
            })

        return sorted(pairs, key=lambda x: -x["score"])

    def _pair_score(
        self,
        col_a: str, kind_a: str,
        col_b: str, kind_b: str,
    ) -> Tuple[float, str]:
        """Return (score, method_name) for a column pair."""
        s_a = _safe_fillna(self.df[col_a], kind_a)
        s_b = _safe_fillna(self.df[col_b], kind_b)

        # Drop rows where either is missing after fill (shouldn't happen, but safe)
        mask = s_a.notna() & s_b.notna()
        s_a  = s_a[mask]
        s_b  = s_b[mask]

        if len(s_a) < 5:
            return 0.0, "insufficient_data"

        # numeric × numeric
        if kind_a == "numeric" and kind_b == "numeric":
            score = _mi_numeric_pair(s_a.values.astype(float), s_b.values.astype(float))
            return score, "Mutual Information"

        # categorical × categorical
        if kind_a == "categorical" and kind_b == "categorical":
            score = _cramers_v(s_a, s_b)
            return score, "Cramér's V"

        # mixed: numeric × categorical (either order)
        if kind_a == "numeric" and kind_b == "categorical":
            score = _correlation_ratio(s_b, s_a)
            return score, "Correlation Ratio η²"

        if kind_a == "categorical" and kind_b == "numeric":
            score = _correlation_ratio(s_a, s_b)
            return score, "Correlation Ratio η²"

        return 0.0, "unsupported"

    # ═══════════════════════════════════════════════════════════════════
    # FEATURE VS TARGET
    # ═══════════════════════════════════════════════════════════════════

    def _compute_target_scores(self, cols: List[str]) -> List[Dict[str, Any]]:
        """
        Score each feature against the target using MI.
        Determines classification vs regression from target dtype/cardinality.
        """
        if not self.target or self.target not in self.df.columns:
            return []

        tgt       = self.df[self.target].dropna()
        is_cls    = (
            not pd.api.types.is_numeric_dtype(tgt)
            or tgt.nunique() <= 15
        )
        tgt_kind  = _classify(self.df[self.target])

        if tgt_kind not in ("numeric", "categorical"):
            return []

        scores: List[Dict[str, Any]] = []

        for col in cols:
            kind = self._kinds[col]
            if kind != "numeric":
                # For categorical features: use correlation ratio against numeric target
                # or Cramér's V against categorical target
                if tgt_kind == "numeric":
                    s = _safe_fillna(self.df[col], kind)
                    t = self.df[self.target]
                    mask = s.notna() & t.notna()
                    score = _correlation_ratio(s[mask], t[mask])
                    method = "Correlation Ratio η²"
                else:
                    s = _safe_fillna(self.df[col], kind)
                    t = _safe_fillna(self.df[self.target], tgt_kind)
                    score = _cramers_v(s, t)
                    method = "Cramér's V"
            else:
                # numeric feature — use MI
                combined = pd.DataFrame({
                    "f": self.df[col],
                    "t": self.df[self.target],
                }).dropna()
                if len(combined) < 5:
                    continue
                score  = _mi_with_target(
                    combined["f"].values.astype(float),
                    combined["t"].values,
                    is_classification=is_cls,
                )
                method = "Mutual Information"

            strength, color = _strength(score)
            scores.append({
                "column":   col,
                "target":   self.target,
                "score":    score,
                "method":   method,
                "strength": strength,
                "color":    color,
                "kind":     kind,
            })

        return sorted(scores, key=lambda x: -x["score"])

    # ═══════════════════════════════════════════════════════════════════
    # INTERACTION MATRIX
    # ═══════════════════════════════════════════════════════════════════

    def _build_matrix(
        self,
        cols:  List[str],
        pairs: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Build a symmetric N×N interaction score matrix.
        Diagonal = 1.0 (self-interaction).
        """
        n   = len(cols)
        idx = {col: i for i, col in enumerate(cols)}
        mat = [[0.0] * n for _ in range(n)]

        # Set diagonal
        for i in range(n):
            mat[i][i] = 1.0

        # Fill symmetric pairs
        for p in pairs:
            i = idx.get(p["col_a"])
            j = idx.get(p["col_b"])
            if i is None or j is None:
                continue
            mat[i][j] = p["score"]
            mat[j][i] = p["score"]

        # Round for JSON cleanliness
        mat = [[round(v, 4) for v in row] for row in mat]

        return cols, mat

    # ═══════════════════════════════════════════════════════════════════
    # REDUNDANCY GROUPS
    # ═══════════════════════════════════════════════════════════════════

    def _find_redundancy_groups(
        self,
        pairs:     List[Dict[str, Any]],
        threshold: float = 0.70,
    ) -> List[List[str]]:
        """
        Group features that are strongly interacting (score ≥ threshold).
        Uses a simple union-find approach so transitive groups are merged.
        """
        strong_pairs = [(p["col_a"], p["col_b"])
                        for p in pairs if p["score"] >= threshold]

        if not strong_pairs:
            return []

        # Union-Find
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent.get(x, x), x)   # path compression
                x = parent.get(x, x)
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in strong_pairs:
            if a not in parent: parent[a] = a
            if b not in parent: parent[b] = b
            union(a, b)

        # Collect groups
        groups: Dict[str, List[str]] = {}
        for col in parent:
            root = find(col)
            groups.setdefault(root, []).append(col)

        # Only return groups with 2+ members, sorted largest first
        result = [sorted(g) for g in groups.values() if len(g) >= 2]
        return sorted(result, key=len, reverse=True)

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════

    def _summarize(
        self,
        n_strong:    int,
        n_moderate:  int,
        top_pairs:   List[Dict],
        red_groups:  List[List[str]],
    ) -> str:
        parts: List[str] = []

        if n_strong == 0 and n_moderate == 0:
            return "All feature pairs show negligible interaction — features are largely independent."

        if n_strong > 0:
            top = top_pairs[0]
            parts.append(
                f"{n_strong} strongly interacting pair(s) detected. "
                f"Strongest: '{top['col_a']}' ↔ '{top['col_b']}' "
                f"({top['method']}: {top['score']:.2f})."
            )

        if n_moderate > 0:
            parts.append(f"{n_moderate} moderately interacting pair(s) found.")

        if red_groups:
            largest = red_groups[0]
            parts.append(
                f"Redundancy group detected: {largest} — "
                f"consider dropping all but one or using PCA to collapse them."
            )

        return " ".join(parts)