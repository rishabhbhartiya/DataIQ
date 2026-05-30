"""
code_gen.py — Pipeline Code Generator
=======================================
Reads the MLRadar analysis result and emits a complete, copy-pasteable
Python script with a production-grade sklearn Pipeline.

What it generates
-----------------
  • Correct ColumnTransformer per column type
  • Right imputer per column  (median / mode / KNN based on miss %)
  • Right encoder per column  (OHE / Ordinal / Frequency based on cardinality)
  • Right scaler              (Standard vs Robust based on outlier rate)
  • PowerTransformer stub     for skewed columns
  • SMOTE stub                if class imbalance detected
  • Train/test split          with stratify for classification
  • Cross-validation block    with right scoring metric
  • Test-set evaluation block with right metrics
  • SHAP stub                 for feature importance
  • joblib save/load stub

Design rules
------------
  - Every generated line is runnable as-is (no "fill in here" stubs for required parts)
  - Optional sections are clearly marked with "# optional — uncomment"
  - Column lists at the top so the user can adjust without touching pipeline code
  - Three model options offered, one uncommented, two commented
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════

class PipelineCodeGenerator:
    """
    Generate a complete sklearn Pipeline script from a MLRadar analysis dict.

    Parameters
    ----------
    analysis        : dict returned by AdvancedAnalyzer.analyze() (enriched)
    target          : target column name
    problem_type    : "classification" | "regression" | None  (auto-detect)

    Example
    -------
    >>> gen  = PipelineCodeGenerator(analysis, target="churn")
    >>> code = gen.generate()
    >>> print(code)
    >>> # or save directly:
    >>> Path("pipeline.py").write_text(code)
    """

    def __init__(
        self,
        analysis:     Dict[str, Any],
        target:       Optional[str] = None,
        problem_type: Optional[str] = None,
    ):
        self.analysis     = analysis
        self.target       = target
        self.problem_type = problem_type or self._detect_problem_type()
        self._parse()

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def generate(self) -> str:
        """Return the full pipeline script as a single string."""
        blocks = [
            self._header(),
            self._imports(),
            self._column_lists(),
            self._load_block(),
            self._numeric_pipeline(),
            self._categorical_pipeline(),
            self._column_transformer(),
            self._model_block(),
            self._full_pipeline(),
            self._split_block(),
            self._imbalance_block(),
            self._fit_block(),
            self._evaluate_block(),
            self._shap_block(),
            self._save_block(),
        ]
        return "\n".join(blocks)

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS PARSING
    # ═══════════════════════════════════════════════════════════════════

    def _detect_problem_type(self) -> str:
        pt = self.analysis.get("problem_type", {})
        detected = pt.get("type", "")
        if "classification" in detected:
            return "classification"
        if "regression" in detected:
            return "regression"
        return "classification"   # safe default

    def _parse(self) -> None:
        """
        Walk column_meta + analysis sections to build column lists
        that drive every generated pipeline step.
        """
        meta     = self.analysis.get("column_meta", [])
        target   = self.target

        # ── column buckets ────────────────────────────────────────────
        self.num_cols:      List[str] = []   # all numeric (excl. target)
        self.cat_ohe:       List[str] = []   # categorical, cardinality ≤ 15  → OHE
        self.cat_ord:       List[str] = []   # categorical, cardinality 16-100 → Ordinal
        self.cat_freq:      List[str] = []   # categorical, cardinality > 100  → Frequency
        self.drop_cols:     List[str] = []   # >40% missing / text / datetime

        # ── imputer choices per bucket ────────────────────────────────
        self.num_miss:      List[str] = []   # numeric with any missing
        self.cat_miss:      List[str] = []   # categorical with any missing

        # ── transform flags ───────────────────────────────────────────
        self.skewed_cols:   List[str] = []   # need PowerTransform
        self.outlier_cols:  List[str] = []   # inform scaler choice

        for m in meta:
            col  = m["name"]
            kind = m["kind"]
            miss = m["missing_pct"]

            if col == target:
                continue
            if kind in ("text", "datetime") or miss > 40:
                self.drop_cols.append(col)
                continue

            if kind == "numeric":
                self.num_cols.append(col)
                if miss > 0:
                    self.num_miss.append(col)
            elif kind in ("categorical", "boolean"):
                n_uniq = m["unique"]
                if n_uniq <= 15:
                    self.cat_ohe.append(col)
                elif n_uniq <= 100:
                    self.cat_ord.append(col)
                else:
                    self.cat_freq.append(col)
                if miss > 0:
                    self.cat_miss.append(col)

        # skewed numeric cols (|skew| > 1.5, positive values only)
        for item in self.analysis.get("skewness_kurtosis", {}).get("items", []):
            col = item["column"]
            if col in self.num_cols and abs(item["skewness"]) > 1.5:
                self.skewed_cols.append(col)

        # outlier-heavy numeric cols (IQR outlier % > 5)
        for item in self.analysis.get("outliers", {}).get("items", []):
            col = item["column"]
            if col in self.num_cols and item["iqr_pct"] > 5:
                self.outlier_cols.append(col)

        # scaler: use Robust if >30% of numeric cols have heavy outliers
        self.use_robust = (
            len(self.outlier_cols) > len(self.num_cols) * 0.3
            if self.num_cols else False
        )

        # imputer: KNN if ≤5 numeric cols have missing, else median
        self.use_knn = 0 < len(self.num_miss) <= 5

        # class imbalance
        ci = self.analysis.get("class_imbalance")
        self.is_imbalanced = (
            ci is not None and ci.get("is_imbalanced", False)
            and self.problem_type == "classification"
        )
        self.imbalance_ratio = ci.get("imbalance_ratio", 1.0) if ci else 1.0

        # n_classes for multiclass detection
        self.n_classes = ci.get("n_classes", 2) if ci else 2

    # ═══════════════════════════════════════════════════════════════════
    # CODE BLOCKS
    # ═══════════════════════════════════════════════════════════════════

    def _header(self) -> str:
        pt_label = self.problem_type.replace("_", " ").title()
        target   = self.target or "UNDEFINED"
        return textwrap.dedent(f"""\
            #!/usr/bin/env python3
            \"\"\"
            pipeline.py — Auto-generated sklearn Pipeline
            ==============================================
            Generated by  : MLRadar PipelineCodeGenerator
            Target column : {target}
            Problem type  : {pt_label}

            How to use
            ----------
            1. Set DATA_PATH to your CSV file path.
            2. Adjust TARGET if needed.
            3. Uncomment an alternative model if desired.
            4. Run:  python pipeline.py
            \"\"\"
        """)

    def _imports(self) -> str:
        base = textwrap.dedent("""\
            import warnings
            warnings.filterwarnings("ignore")

            import numpy as np
            import pandas as pd
            from pathlib import Path

            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import (
                StandardScaler, RobustScaler,
                OneHotEncoder, OrdinalEncoder, PowerTransformer,
            )
            from sklearn.impute import SimpleImputer, KNNImputer
            from sklearn.model_selection import (
                train_test_split, cross_val_score,
                StratifiedKFold, KFold,
            )
        """)

        if self.problem_type == "classification":
            base += textwrap.dedent("""\
                from sklearn.metrics import (
                    classification_report, roc_auc_score,
                    confusion_matrix, ConfusionMatrixDisplay,
                )
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from sklearn.linear_model import LogisticRegression
            """)
        else:
            base += textwrap.dedent("""\
                from sklearn.metrics import (
                    mean_squared_error, mean_absolute_error, r2_score,
                )
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.linear_model import Ridge, Lasso
            """)

        return base + "\n"

    def _column_lists(self) -> str:
        lines = [
            "# ── Column Lists (auto-detected by MLRadar) "
            + "─" * 34,
            f"TARGET       = {self.target!r}",
            "",
            f"# Numeric features  ({len(self.num_cols)} columns)",
            f"NUMERIC_COLS  = {self.num_cols!r}",
            "",
            f"# Low-cardinality categoricals → One-Hot  ({len(self.cat_ohe)} columns)",
            f"CAT_OHE_COLS  = {self.cat_ohe!r}",
            "",
            f"# Medium-cardinality categoricals → Ordinal  ({len(self.cat_ord)} columns)",
            f"CAT_ORD_COLS  = {self.cat_ord!r}",
            "",
            f"# High-cardinality categoricals → Frequency  ({len(self.cat_freq)} columns)",
            f"CAT_FREQ_COLS = {self.cat_freq!r}",
            "",
            f"# Dropped: >40% missing / free-text / datetime  ({len(self.drop_cols)} columns)",
            f"DROP_COLS     = {self.drop_cols!r}",
        ]
        if self.skewed_cols:
            lines += [
                "",
                f"# Skewed numerics (|skew| > 1.5) — will be power-transformed",
                f"SKEWED_COLS   = {self.skewed_cols!r}",
            ]
        if self.outlier_cols:
            lines += [
                "",
                f"# Outlier-heavy numerics (IQR outliers > 5%) — inform scaler choice",
                f"OUTLIER_COLS  = {self.outlier_cols!r}",
            ]
        return "\n".join(lines) + "\n\n"

    def _load_block(self) -> str:
        return textwrap.dedent("""\
            # ── Load Data ──────────────────────────────────────────────────────────
            DATA_PATH = "your_data.csv"   # ← set this

            df = pd.read_csv(DATA_PATH)
            df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

            X = df.drop(columns=[TARGET])
            y = df[TARGET]

            print(f"Dataset: {X.shape[0]:,} rows × {X.shape[1]} features")
            print(f"Target : {y.value_counts().to_dict() if y.nunique() <= 10 else y.describe().to_dict()}")

        """)

    def _numeric_pipeline(self) -> str:
        if not self.num_cols:
            return "# No numeric columns detected — numeric pipeline skipped\n\n"

        # Imputer
        if self.use_knn:
            imputer_line = "    ('imputer', KNNImputer(n_neighbors=5)),"
            imputer_note = "# KNN imputation (≤5 cols with missing values)"
        else:
            imputer_line = "    ('imputer', SimpleImputer(strategy='median')),"
            imputer_note = "# Median imputation"

        # Scaler
        if self.use_robust:
            scaler_line = "    ('scaler',  RobustScaler()),"
            scaler_note = "# Robust scaler chosen — outlier-heavy columns detected"
        else:
            scaler_line = "    ('scaler',  StandardScaler()),"
            scaler_note = "# Standard scaler"

        # Power transform for skewed cols
        if self.skewed_cols:
            power_line = (
                "    ('power',   PowerTransformer(method='yeo-johnson', standardize=False)),  "
                "# applied before scaling\n"
            )
        else:
            power_line = ""

        return (
            f"# ── Numeric Pipeline  {imputer_note}  {scaler_note}\n"
            f"numeric_pipeline = Pipeline([\n"
            f"{imputer_line}\n"
            f"{power_line}"
            f"{scaler_line}\n"
            f"])\n\n"
        )

    def _categorical_pipeline(self) -> str:
        blocks: List[str] = []

        # OHE pipeline
        if self.cat_ohe:
            blocks.append(textwrap.dedent("""\
                # Low-cardinality → One-Hot Encoding
                cat_ohe_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(
                        handle_unknown='ignore',
                        sparse_output=False,
                        drop='if_binary',       # drop one col for binary features
                    )),
                ])
            """))

        # Ordinal pipeline
        if self.cat_ord:
            blocks.append(textwrap.dedent("""\
                # Medium-cardinality → Ordinal Encoding
                # Tip: replace with sklearn.preprocessing.TargetEncoder (sklearn ≥ 1.3)
                #      for better performance on tree models
                cat_ord_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1,
                    )),
                ])
            """))

        # Frequency encoding (manual transformer)
        if self.cat_freq:
            blocks.append(textwrap.dedent("""\
                # High-cardinality → Frequency Encoding
                # Replaces each category with its relative frequency in the training set
                from sklearn.base import BaseEstimator, TransformerMixin

                class FrequencyEncoder(BaseEstimator, TransformerMixin):
                    def fit(self, X, y=None):
                        self.freq_maps_ = {}
                        for i in range(X.shape[1]):
                            col = pd.Series(X[:, i].astype(str))
                            self.freq_maps_[i] = col.value_counts(normalize=True).to_dict()
                        return self

                    def transform(self, X):
                        out = np.zeros(X.shape, dtype=float)
                        for i, freq_map in self.freq_maps_.items():
                            col = pd.Series(X[:, i].astype(str))
                            out[:, i] = col.map(freq_map).fillna(0).values
                        return out

                cat_freq_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', FrequencyEncoder()),
                ])
            """))

        return "\n".join(blocks) if blocks else "# No categorical columns detected\n\n"

    def _column_transformer(self) -> str:
        transformers: List[str] = []

        if self.num_cols:
            transformers.append(
                "        ('numeric', numeric_pipeline,   NUMERIC_COLS),"
            )
        if self.cat_ohe:
            transformers.append(
                "        ('cat_ohe', cat_ohe_pipeline,   CAT_OHE_COLS),"
            )
        if self.cat_ord:
            transformers.append(
                "        ('cat_ord', cat_ord_pipeline,   CAT_ORD_COLS),"
            )
        if self.cat_freq:
            transformers.append(
                "        ('cat_freq', cat_freq_pipeline, CAT_FREQ_COLS),"
            )

        if not transformers:
            return "# No transformers — all columns dropped\n\n"

        t_str = "\n".join(f"    {t.strip()}" for t in transformers)
        lines = [
            "# ── Column Transformer ─────────────────────────────────────────────────",
            "preprocessor = ColumnTransformer(",
            "    transformers=[",
            t_str,
            "    ],",
            "    remainder='drop',                    # drop unlisted columns",
            "    verbose_feature_names_out=False,     # cleaner feature names",
            ")",
            "",
        ]
        return "\n".join(lines) + "\n"

    def _model_block(self) -> str:
        is_cls   = self.problem_type == "classification"
        balanced = "class_weight='balanced', " if (is_cls and self.is_imbalanced) else ""

        if is_cls:
            m1 = f"RandomForestClassifier(n_estimators=300, {balanced}random_state=42, n_jobs=-1)"
            m2 = f"GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)"
            m3 = f"LogisticRegression(C=1.0, max_iter=1000, {balanced}solver='lbfgs')"
        else:
            m1 = "RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)"
            m2 = "GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)"
            m3 = "Ridge(alpha=1.0)"

        imbalance_note = ""
        if self.is_imbalanced:
            imbalance_note = (
                f"# ⚠  Class imbalance detected (ratio ≈ {self.imbalance_ratio:.1f}×)\n"
                f"#    class_weight='balanced' set on default model.\n"
                f"#    See SMOTE block below for oversampling alternative.\n"
            )

        lines = [
            "# ── Model Selection ────────────────────────────────────────────────────",
        ]
        if self.is_imbalanced:
            lines += [
                f"# ⚠  Class imbalance detected (ratio ≈ {self.imbalance_ratio:.1f}×)",
                "#    class_weight='balanced' set on default model.",
                "#    See SMOTE block below for oversampling alternative.",
            ]
        lines += [
            "",
            "# Uncomment the model you want to use:",
            f"model = {m1}",
            f"# model = {m2}",
            f"# model = {m3}",
            "",
        ]
        return "\n".join(lines) + "\n"

    def _full_pipeline(self) -> str:
        return textwrap.dedent("""\
            # ── Full Pipeline ───────────────────────────────────────────────────────
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model',        model),
            ])

        """)

    def _split_block(self) -> str:
        is_cls   = self.problem_type == "classification"
        stratify = "stratify=y, " if is_cls else ""

        return textwrap.dedent(f"""\
            # ── Train / Test Split ──────────────────────────────────────────────────
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.20,
                random_state=42,
                {stratify}shuffle=True,
            )
            print(f"Train: {{len(X_train):,}} rows  |  Test: {{len(X_test):,}} rows")

        """)

    def _imbalance_block(self) -> str:
        if not self.is_imbalanced:
            return ""
        return textwrap.dedent("""\
            # ── Class Imbalance — SMOTE (optional) ─────────────────────────────────
            # Uncomment to use SMOTE oversampling instead of class_weight.
            # Requires: pip install imbalanced-learn
            #
            # from imblearn.over_sampling import SMOTE
            # from imblearn.pipeline import Pipeline as ImbPipeline
            #
            # smote = SMOTE(random_state=42)
            # X_train_res, y_train_res = smote.fit_resample(
            #     preprocessor.fit_transform(X_train), y_train
            # )
            # Then fit model directly: model.fit(X_train_res, y_train_res)

        """)

    def _fit_block(self) -> str:
        is_cls = self.problem_type == "classification"

        if is_cls:
            cv_obj    = "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"
            cv_metric = "roc_auc" if self.n_classes == 2 else "roc_auc_ovr_weighted"
        else:
            cv_obj    = "KFold(n_splits=5, shuffle=True, random_state=42)"
            cv_metric = "neg_root_mean_squared_error"

        return textwrap.dedent(f"""\
            # ── Fit ─────────────────────────────────────────────────────────────────
            print("\\nTraining pipeline…")
            pipeline.fit(X_train, y_train)

            # Cross-validation on training set
            cv       = {cv_obj}
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv, scoring='{cv_metric}', n_jobs=-1,
            )
            print(f"CV {cv_metric}: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")

        """)

    def _evaluate_block(self) -> str:
        is_cls = self.problem_type == "classification"

        if is_cls:
            eval_code = textwrap.dedent("""\
                y_pred = pipeline.predict(X_test)

                print("\\n── Classification Report ───────────────────────────────")
                print(classification_report(y_test, y_pred))

                # AUC-ROC
                if hasattr(pipeline.named_steps['model'], 'predict_proba'):
                    y_prob = pipeline.predict_proba(X_test)
                    if y_prob.shape[1] == 2:
                        auc = roc_auc_score(y_test, y_prob[:, 1])
                        print(f"AUC-ROC : {auc:.4f}")
                    else:
                        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                        print(f"AUC-ROC (weighted OVR) : {auc:.4f}")

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                print("\\nConfusion Matrix:")
                print(cm)
            """)
        else:
            eval_code = textwrap.dedent("""\
                y_pred = pipeline.predict(X_test)

                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae  = mean_absolute_error(y_test, y_pred)
                r2   = r2_score(y_test, y_pred)

                print("\\n── Regression Metrics ──────────────────────────────────")
                print(f"RMSE : {rmse:.4f}")
                print(f"MAE  : {mae:.4f}")
                print(f"R²   : {r2:.4f}")
            """)

        return (
            "# ── Evaluate on Test Set ───────────────────────────────────────────────\n"
            + eval_code
            + "\n"
        )

    def _shap_block(self) -> str:
        return textwrap.dedent("""\
            # ── SHAP Feature Importance (optional) ─────────────────────────────────
            # Requires: pip install shap
            #
            # import shap
            #
            # X_train_t = pipeline.named_steps['preprocessor'].transform(X_train)
            # explainer  = shap.TreeExplainer(pipeline.named_steps['model'])
            # shap_vals  = explainer.shap_values(X_train_t)
            #
            # shap.summary_plot(shap_vals, X_train_t)
            # shap.waterfall_plot(explainer.expected_value, shap_vals[0])

        """)

    def _save_block(self) -> str:
        return textwrap.dedent("""\
            # ── Save / Load Pipeline (optional) ────────────────────────────────────
            # import joblib
            # joblib.dump(pipeline, 'pipeline.pkl')
            # loaded_pipeline = joblib.load('pipeline.pkl')
            # loaded_pipeline.predict(X_new)

            print("\\nDone ✓")
        """)