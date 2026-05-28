"""
narrative.py — NarrativeEngine
================================
Converts raw analysis dicts into 2–4 sentence plain-English paragraphs.
One narrative per report section — what a data scientist would write.

Original methods (preserved, bugs fixed):
    overview, missing, duplicates, outliers,
    correlations, class_imbalance, pca, skewness, readiness

Four new methods added:
    leakage        — summarises LeakageDetector findings
    interactions   — summarises FeatureInteractions top pairs
    temporal       — summarises temporal drift findings
    problem_type   — describes the auto-detected ML task

Bug fixed in correlations(): was reading p['r'], key is now p['corr'].
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


class NarrativeEngine:
    """
    Generate plain-English narratives from a DataIQ analysis result dict.

    Parameters
    ----------
    analysis : dict   — full result from AdvancedAnalyzer.analyze()
    target   : str    — target column name (optional)

    Example
    -------
    >>> ne = NarrativeEngine(analysis, target="churn")
    >>> stories = ne.all_narratives()
    >>> print(stories["overview"])
    """

    def __init__(self, analysis: Dict[str, Any], target: Optional[str] = None):
        self.a      = analysis
        self.target = target

    def all_narratives(self) -> Dict[str, str]:
        return {
            "overview":      self.overview(),
            "missing":       self.missing(),
            "duplicates":    self.duplicates(),
            "outliers":      self.outliers(),
            "correlations":  self.correlations(),
            "class_imbalance": self.class_imbalance(),
            "pca":           self.pca(),
            "skewness":      self.skewness(),
            "readiness":     self.readiness(),
            # ── new ───────────────────────────────
            "leakage":       self.leakage(),
            "interactions":  self.interactions(),
            "temporal":      self.temporal(),
            "problem_type":  self.problem_type(),
        }

    # ══════════════════════════════════════════════════════════════════
    # ORIGINAL METHODS  (bugs fixed, logic unchanged)
    # ══════════════════════════════════════════════════════════════════

    def overview(self) -> str:
        ov     = self.a.get("overview", {})
        n      = ov.get("n_rows", 0)
        k      = ov.get("n_cols", 0)
        miss   = ov.get("missing_pct", 0)
        dup    = ov.get("duplicate_pct", 0)
        mem    = ov.get("memory_human", "?")
        kinds  = ov.get("kinds_count", {})
        n_num  = kinds.get("numeric", 0)
        n_cat  = kinds.get("categorical", 0)

        size_adj = "small" if n < 1_000 else "medium-sized" if n < 50_000 else "large"
        miss_comment = (
            "Missing values are pervasive — imputation strategy is critical."
            if miss > 20 else
            "Missing values are moderate — targeted imputation is recommended."
            if miss > 5 else
            "Missing values are negligible — the dataset is largely complete."
        )
        return (
            f"Your dataset contains {n:,} rows × {k} columns ({mem} in memory), "
            f"a {size_adj} dataset with {n_num} numeric and {n_cat} categorical features. "
            f"Overall, {miss:.1f}% of cells are missing and {dup:.1f}% of rows are exact duplicates. "
            f"{miss_comment}"
        )

    def missing(self) -> str:
        mv   = self.a.get("missing", {})
        bars = mv.get("bars", [])
        if not bars:
            return "No missing values detected. The dataset is complete — no imputation needed."
        hi  = [b for b in bars if b["severity"] == "high"]
        med = [b for b in bars if b["severity"] == "medium"]
        lo  = [b for b in bars if b["severity"] == "low"]
        parts: List[str] = []
        if hi:
            cols = ", ".join(f"'{b['column']}' ({b['pct']}%)" for b in hi[:3])
            parts.append(
                f"{len(hi)} column(s) exceed 40% missing: {cols}. "
                "These should be dropped unless domain knowledge justifies imputation."
            )
        if med:
            parts.append(
                f"{len(med)} column(s) have moderate missingness (5–40%) "
                "— median or KNN imputation is recommended."
            )
        if lo:
            parts.append(
                f"{len(lo)} column(s) have minor missingness (<5%) "
                "— simple mode/median imputation is sufficient."
            )
        return " ".join(parts)

    def duplicates(self) -> str:
        dups = self.a.get("duplicates", {})
        n    = dups.get("n_duplicates", 0)
        pct  = dups.get("pct", 0)
        if n == 0:
            return "No duplicate rows found. Each row represents a unique observation."
        severity = "significant" if pct > 10 else "moderate" if pct > 3 else "minor"
        return (
            f"{n:,} duplicate rows ({pct:.1f}%) were detected — a {severity} amount. "
            "Duplicates inflate training statistics and can cause data leakage if they "
            "span train and test splits. Drop them before splitting your data."
        )

    def outliers(self) -> str:
        ov    = self.a.get("outliers", {})
        items = ov.get("items", [])
        if not items:
            return "No significant outliers detected across numeric columns."
        hi_pct = [i for i in items if i["iqr_pct"] > 5]
        if not hi_pct:
            return f"{len(items)} column(s) have minor outliers — Winsorization is optional."
        col_names = ", ".join(f"'{i['column']}'" for i in hi_pct[:4])
        extra     = f" and {len(hi_pct) - 4} more" if len(hi_pct) > 4 else ""
        return (
            f"{len(hi_pct)} column(s) have significant outlier rates (>5% of values): "
            f"{col_names}{extra}. "
            "Extreme values will dominate linear models and distance-based algorithms "
            "(KNN, SVM, K-Means). "
            "Use IQR-based capping (Winsorization) or remove outlier rows before training."
        )

    def correlations(self) -> str:
        corr  = self.a.get("correlations", {})
        # key is 'strong_pairs' in analyzer; fall back to 'high_pairs' for safety
        pairs = corr.get("strong_pairs", corr.get("high_pairs", []))
        high  = [p for p in pairs if abs(p.get("corr", p.get("r", 0))) > 0.85]
        if not high:
            return (
                "No highly correlated feature pairs detected (|r| > 0.85). "
                "Multicollinearity is not a concern for this dataset."
            )
        pairs_str = ", ".join(
            f"({p['col1']} ↔ {p['col2']}: {p.get('corr', p.get('r', '?'))})"
            for p in high[:3]
        )
        return (
            f"{len(high)} feature pair(s) show high correlation (|r| > 0.85): {pairs_str}. "
            "Multicollinear features destabilize linear model coefficients and slow convergence. "
            "Consider dropping the weaker feature of each pair, or use PCA to collapse "
            "correlated groups."
        )

    def class_imbalance(self) -> str:
        ci = self.a.get("class_imbalance")
        if not ci:
            return "No target column specified — class imbalance check skipped."
        tgt   = ci.get("target", "target")
        ratio = ci.get("imbalance_ratio", 1.0)
        n_cls = ci.get("n_classes", 2)
        maj   = ci.get("majority", "?")
        mino  = ci.get("minority", "?")
        if not ci.get("is_imbalanced"):
            return (
                f"The target '{tgt}' is well-balanced across {n_cls} class(es) "
                "— no special handling needed."
            )
        severity = "severely" if ratio > 10 else "moderately"
        return (
            f"The target '{tgt}' is {severity} imbalanced with a {ratio:.1f}× ratio "
            f"(majority: '{maj}', minority: '{mino}'). "
            "Imbalance causes models to predict the majority class and ignore the minority. "
            "Use stratified splits, class_weight='balanced', SMOTE, or ADASYN to correct this."
        )

    def pca(self) -> str:
        pca = self.a.get("pca_summary", {})
        if not pca or "error" in pca:
            return "PCA analysis could not be performed (too few numeric columns or rows)."
        n95  = pca.get("n_for_95pct", "?")
        nc   = pca.get("n_components", "?")
        evr  = pca.get("explained_variance_ratio", [])
        top1 = round(evr[0] * 100, 1) if evr else 0
        compress = (
            "High dimensionality compression is possible — consider PCA or other "
            "dim-reduction before modeling."
            if isinstance(n95, int) and isinstance(nc, int) and n95 < nc * 0.6
            else "The features are relatively independent — dimensionality reduction "
                 "offers limited benefit."
        )
        return (
            f"PCA reveals that {n95} principal component(s) (out of {nc}) are needed to "
            f"explain 95% of variance. "
            f"The first component alone captures {top1:.1f}% of variance. "
            f"{compress}"
        )

    def skewness(self) -> str:
        sk     = self.a.get("skewness_kurtosis", {})
        items  = sk.get("items", [])
        severe = [i for i in items if abs(i["skewness"]) > 2]
        high   = [i for i in items if 1 < abs(i["skewness"]) <= 2]
        if not items:
            return "No numeric columns to assess skewness."
        if not severe and not high:
            return "All numeric features have acceptable skewness — no transformation required."
        parts: List[str] = []
        if severe:
            names = ", ".join(f"'{i['column']}' ({i['skewness']:.2f})" for i in severe[:3])
            parts.append(
                f"{len(severe)} column(s) are severely skewed: {names}. "
                "Apply log or Yeo-Johnson transform before training."
            )
        if high:
            names = ", ".join(f"'{i['column']}'" for i in high[:3])
            parts.append(
                f"{len(high)} column(s) have high skewness — "
                f"power transform recommended for {names}."
            )
        return " ".join(parts)

    def readiness(self) -> str:
        rs = self.a.get("readiness_score", {})
        if not rs:
            return ""
        score  = rs.get("dataset_score", 0)
        grade  = rs.get("dataset_grade", "?")
        issues = rs.get("top_issues", [])
        issue_str = (
            ", ".join(f"'{i['column']}' ({i['grade']})" for i in issues[:3])
            if issues else "none"
        )
        verdict = (
            "The dataset is ML-ready with minimal preprocessing needed."
            if score >= 90 else
            "The dataset is mostly ready with some preprocessing needed."
            if score >= 75 else
            "The dataset requires significant preparation before training."
            if score >= 60 else
            "The dataset has major quality issues — address them before any modeling."
        )
        return (
            f"Overall ML Readiness Score: {score}/100 (Grade {grade}). "
            f"{verdict} "
            f"Lowest-scoring columns: {issue_str}. "
            "Review the Readiness tab for per-column breakdowns and dimension-level scores."
        )

    # ══════════════════════════════════════════════════════════════════
    # NEW METHOD 1 — Leakage
    # ══════════════════════════════════════════════════════════════════

    def leakage(self) -> str:
        """Summarise LeakageDetector findings in plain English."""
        lk = self.a.get("leakage", {})
        if not lk:
            return "Leakage analysis was not run."

        n_total    = lk.get("n_total", 0)
        n_critical = lk.get("n_critical", 0)
        n_high     = lk.get("n_high", 0)
        risk_level = lk.get("risk_level", "CLEAN")
        findings   = lk.get("findings", [])

        if risk_level == "CLEAN" or n_total == 0:
            return (
                "No data leakage patterns detected. "
                "All feature columns appear safe to use for training."
            )

        # Name the worst offenders
        critical_cols = [f["column"] for f in findings if f["severity"] == "critical"]
        high_cols     = [f["column"] for f in findings if f["severity"] == "high"]

        parts: List[str] = []

        if critical_cols:
            cols_str = ", ".join(f"'{c}'" for c in critical_cols[:3])
            parts.append(
                f"Critical leakage detected in {len(critical_cols)} column(s): {cols_str}. "
                "These columns almost certainly encode the target answer and must be removed "
                "before training."
            )
        if high_cols:
            cols_str = ", ".join(f"'{c}'" for c in high_cols[:3])
            extra    = f" (+{len(high_cols) - 3} more)" if len(high_cols) > 3 else ""
            parts.append(
                f"High-risk leakage in {cols_str}{extra} — investigate before training."
            )

        parts.append(
            "Data leakage produces models that appear highly accurate in validation but "
            "fail completely in production."
        )
        return " ".join(parts)

    # ══════════════════════════════════════════════════════════════════
    # NEW METHOD 2 — Feature Interactions
    # ══════════════════════════════════════════════════════════════════

    def interactions(self) -> str:
        """Summarise FeatureInteractions results in plain English."""
        fi = self.a.get("feature_interactions", {})
        if not fi or fi.get("available") is False:
            return "Feature interaction analysis was not available."

        n_strong    = fi.get("n_strong", 0)
        n_moderate  = fi.get("n_moderate", 0)
        top_pairs   = fi.get("top_pairs", [])
        red_groups  = fi.get("redundancy_groups", [])
        tgt_scores  = fi.get("target_scores", [])

        if not top_pairs and n_strong == 0 and n_moderate == 0:
            return (
                "All feature pairs show negligible interaction strength. "
                "Features are largely independent — no redundancy to remove."
            )

        parts: List[str] = []

        if n_strong > 0 and top_pairs:
            top = top_pairs[0]
            parts.append(
                f"{n_strong} strongly interacting feature pair(s) detected "
                f"(score ≥ 0.70). "
                f"Strongest: '{top['col_a']}' ↔ '{top['col_b']}' "
                f"({top['method']}: {top['score']:.2f}). "
                "Redundant features waste model capacity and can harm interpretability."
            )
        elif n_moderate > 0:
            parts.append(
                f"{n_moderate} moderately interacting pair(s) detected "
                "— monitor for redundancy."
            )

        if red_groups:
            largest = red_groups[0]
            parts.append(
                f"Redundancy group found: {largest}. "
                "Consider keeping only one representative feature or applying PCA to "
                "collapse this group."
            )

        if tgt_scores:
            best = tgt_scores[0]
            parts.append(
                f"Strongest predictor of '{best['target']}': "
                f"'{best['column']}' ({best['method']}: {best['score']:.3f})."
            )

        return " ".join(parts)

    # ══════════════════════════════════════════════════════════════════
    # NEW METHOD 3 — Temporal Awareness
    # ══════════════════════════════════════════════════════════════════

    def temporal(self) -> str:
        """Summarise temporal trend and concept drift findings."""
        ta = self.a.get("temporal_awareness", {})
        if not ta or not ta.get("available"):
            return "No datetime columns found — temporal analysis was not performed."

        items = ta.get("items", [])
        if not items:
            return "Datetime columns were found but contained insufficient data for trend analysis."

        drifted_cols: List[str] = []
        stable_cols:  List[str] = []
        for item in items:
            if item.get("has_concept_drift"):
                drifted_cols.append(item["datetime_col"])
            else:
                stable_cols.append(item["datetime_col"])

        parts: List[str] = []

        # Per-column drift findings
        all_findings: List[Dict[str, Any]] = []
        for item in items:
            for f in item.get("drift_findings", []):
                if f.get("has_drift"):
                    all_findings.append({**f, "dt_col": item["datetime_col"]})

        if all_findings:
            feats = ", ".join(f"'{f['feature']}' ({f['drift_pct']:.0f}% shift)" for f in all_findings[:3])
            extra = f" and {len(all_findings) - 3} more" if len(all_findings) > 3 else ""
            parts.append(
                f"Concept drift detected: {feats}{extra} show >20% mean shift "
                "from the earliest to latest time period. "
                "This suggests the data distribution is changing over time, which can degrade "
                "model performance on recent data."
            )
        else:
            parts.append(
                "Feature distributions appear stable across time — "
                "no significant concept drift detected."
            )

        if items:
            date_ranges = [item["date_range"] for item in items if "date_range" in item]
            if date_ranges:
                parts.append(f"Date range covered: {date_ranges[0]}.")

        return " ".join(parts)

    # ══════════════════════════════════════════════════════════════════
    # NEW METHOD 4 — Problem Type
    # ══════════════════════════════════════════════════════════════════

    def problem_type(self) -> str:
        """Describe the auto-detected ML problem type."""
        pt = self.a.get("problem_type", {})
        if not pt or pt.get("type") == "unknown":
            return (
                "No target column specified — problem type could not be determined. "
                "Set target= when creating DataIQ for automatic detection."
            )

        ptype      = pt.get("type", "unknown")
        confidence = pt.get("confidence", "low")
        reason     = pt.get("reason", "")
        n_classes  = pt.get("n_classes")

        type_label = ptype.replace("_", " ").title()

        advice_map = {
            "binary_classification": (
                "Use AUC-ROC, F1, and Precision-Recall as evaluation metrics. "
                "Check class balance and consider stratified k-fold cross-validation."
            ),
            "multiclass_classification": (
                "Use macro/weighted F1 and confusion matrix. "
                "One-vs-Rest or One-vs-One strategies may be needed for some algorithms."
            ),
            "regression": (
                "Use RMSE, MAE, and R² as evaluation metrics. "
                "Check target distribution — high skew may benefit from log-transforming the target."
            ),
        }
        advice = advice_map.get(ptype, "Review the target column and choose an appropriate evaluation metric.")

        classes_note = f" ({n_classes} classes)" if n_classes else ""
        conf_note = "" if confidence == "high" else f" (confidence: {confidence})"

        return (
            f"Auto-detected problem type: {type_label}{classes_note}{conf_note}. "
            f"{reason} "
            f"{advice}"
        )