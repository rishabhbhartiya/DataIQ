"""
analyzer.py — AdvancedAnalyzer
================================
Full EDA engine for MLRadar.

Original sections (unchanged from mlprofiler2):
  overview, missing, duplicates, outliers,
  univariate_numeric, univariate_categorical,
  bivariate, correlations, normality_tests,
  skewness_kurtosis, datetime_features, text_features,
  variance, pca_summary, class_imbalance,
  feature_target, chi_square, recommendations, column_meta

Three new sections added:
  _detect_problem_type   — auto-infer regression / classification from target
  _temporal_awareness    — rolling trend + concept-drift windows over datetime cols
  _feature_interactions  — delegate to interactions.py (MI / Cramér V / eta²)
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, kstest, normaltest, shapiro
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

warnings.filterwarnings("ignore")

# ── Plotly dark theme ──────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(10,13,20,0)",
    plot_bgcolor="rgba(10,13,20,0)",
    font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
)


import os
import hashlib

# Directory where chart files are saved — set before calling AdvancedAnalyzer
# e.g. set_charts_dir("my_report_charts/")
_CHARTS_DIR: str = "MLRadar_charts"


def set_charts_dir(path: str) -> None:
    """Call this before running analysis to control where chart files are saved."""
    global _CHARTS_DIR
    _CHARTS_DIR = path
    os.makedirs(path, exist_ok=True)


def fig_to_html(fig) -> str:
    """
    Save a Plotly figure as a standalone .html file and return an <iframe> tag.
    This avoids inline JS rendering issues — each chart is a self-contained page.
    """
    global _CHARTS_DIR
    os.makedirs(_CHARTS_DIR, exist_ok=True)

    fig.update_layout(**PLOTLY_DARK)

    # Deterministic filename based on chart title
    title = fig.layout.title.text or "chart"
    safe  = "".join(c if c.isalnum() else "_" for c in title)[:40]
    uid   = hashlib.md5(safe.encode()).hexdigest()[:6]
    fname = f"{safe}_{uid}.html"
    fpath = os.path.join(_CHARTS_DIR, fname)

    pio.write_html(
        fig, file=fpath, full_html=True,
        include_plotlyjs="cdn",
        config={"displayModeBar": False, "responsive": True},
    )

    return (
        f'<iframe src="{fpath}" '
        f'style="width:100%;height:360px;border:none;border-radius:8px;'
        f'background:transparent" loading="lazy"></iframe>'
    )


# ══════════════════════════════════════════════════════════════════════════════

class AdvancedAnalyzer:
    def __init__(
        self,
        df:     pd.DataFrame,
        target: Optional[str] = None,
        label:  str           = "original",
    ):
        self.df     = df.copy()
        self.target = target
        self.label  = label
        self.n, self.k = df.shape
        self._col_kinds = self._classify()

    # ── Column classification ──────────────────────────────────────────────────
    def _classify(self) -> Dict[str, str]:
        kinds: Dict[str, str] = {}
        for c in self.df.columns:
            s = self.df[c]
            if pd.api.types.is_bool_dtype(s):
                kinds[c] = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(s):
                kinds[c] = "datetime"
            elif pd.api.types.is_numeric_dtype(s):
                kinds[c] = "numeric"
            elif s.dtype == object:
                avg = s.dropna().astype(str).str.len().mean() if s.notna().any() else 0
                kinds[c] = "text" if (avg > 60 or s.nunique() / max(len(s.dropna()), 1) > 0.85) else "categorical"
            else:
                kinds[c] = "categorical"
        return kinds

    def cols_of(self, *kinds: str) -> List[str]:
        return [c for c, k in self._col_kinds.items() if k in kinds]

    # ── Main entry ─────────────────────────────────────────────────────────────
    def analyze(self) -> Dict[str, Any]:
        num   = self.cols_of("numeric")
        cat   = self.cols_of("categorical")
        dt    = self.cols_of("datetime")
        bool_ = self.cols_of("boolean")
        txt   = self.cols_of("text")

        return {
            "label":                  self.label,
            # ── original ──────────────────────────────────────────────
            "overview":               self._overview(),
            "missing":                self._missing(),
            "duplicates":             self._duplicates(),
            "outliers":               self._outliers(num),
            "univariate_numeric":     self._univariate_numeric(num),
            "univariate_categorical": self._univariate_categorical(cat + bool_),
            "bivariate":              self._bivariate(num, cat),
            "correlations":           self._correlations(num),
            "normality":              self._normality_tests(num),
            "skewness_kurtosis":      self._skew_kurt(num),
            "datetime_features":      self._datetime_analysis(dt),
            "text_features":          self._text_analysis(txt),
            "variance":               self._variance_analysis(num),
            "pca_summary":            self._pca_summary(num),
            "class_imbalance":        self._class_imbalance(),
            "feature_target":         self._feature_target(num, cat),
            "chi_square":             self._chi_square(cat),
            "recommendations":        self._recommendations(num, cat, dt, txt),
            "column_meta":            self._column_meta(),
            # ── new ───────────────────────────────────────────────────
            "problem_type":           self._detect_problem_type(),
            "temporal_awareness":     self._temporal_awareness(dt, num),
            "feature_interactions":   self._feature_interactions(num, cat),
        }

    # ══════════════════════════════════════════════════════════════════
    # ORIGINAL METHODS (preserved exactly)
    # ══════════════════════════════════════════════════════════════════

    def _overview(self) -> Dict[str, Any]:
        total = self.n * self.k
        miss  = int(self.df.isnull().sum().sum())
        dup   = int(self.df.duplicated().sum())
        mem   = self.df.memory_usage(deep=True).sum()
        kinds_count: Dict[str, int] = {}
        for k in self._col_kinds.values():
            kinds_count[k] = kinds_count.get(k, 0) + 1
        return {
            "n_rows": self.n, "n_cols": self.k, "total_cells": total,
            "missing_cells": miss, "missing_pct": round(miss / total * 100, 2),
            "duplicate_rows": dup, "duplicate_pct": round(dup / self.n * 100, 2),
            "memory_bytes": int(mem),
            "memory_human": f"{mem/1024:.1f} KB" if mem < 1e6 else f"{mem/1e6:.2f} MB",
            "col_kinds": self._col_kinds,
            "kinds_count": kinds_count,
        }

    def _missing(self) -> Dict[str, Any]:
        miss = self.df.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        bars = []
        for col, cnt in miss.items():
            pct = round(cnt / self.n * 100, 2)
            bars.append({
                "column": col, "count": int(cnt), "pct": pct,
                "severity": "high" if pct > 40 else "medium" if pct > 10 else "low",
                "type": self._col_kinds.get(col, "?"),
                "mcar_likely": pct < 5,
            })
        chart = ""
        if bars:
            fig = go.Figure(go.Bar(
                x=[b["pct"] for b in bars], y=[b["column"] for b in bars],
                orientation="h",
                marker=dict(
                    color=[("#ef4444" if b["severity"] == "high" else
                            "#f59e0b" if b["severity"] == "medium" else "#22c55e") for b in bars],
                    line=dict(width=0),
                ),
                text=[f"{b['pct']}%" for b in bars], textposition="outside",
            ))
            fig.update_layout(title="Missing Values %", xaxis_title="Missing %",
                              yaxis=dict(autorange="reversed"), **PLOTLY_DARK)
            chart = fig_to_html(fig)
        return {"has_missing": len(bars) > 0, "n_cols_missing": len(bars), "bars": bars, "chart": chart}

    def _duplicates(self) -> Dict[str, Any]:
        n_dup = int(self.df.duplicated().sum())
        return {"n_duplicates": n_dup, "pct": round(n_dup / self.n * 100, 2), "has_duplicates": n_dup > 0}

    def _outliers(self, num_cols: List[str]) -> Dict[str, Any]:
        results = []
        for col in num_cols:
            s = self.df[col].dropna()
            if len(s) < 4: continue
            q1, q3  = s.quantile(0.25), s.quantile(0.75)
            iqr     = q3 - q1
            lo, hi  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_n   = int(((s < lo) | (s > hi)).sum())
            zs      = np.abs(stats.zscore(s))
            z_n     = int((zs > 3).sum())
            pct_5   = s.quantile(0.05)
            pct_95  = s.quantile(0.95)
            if iqr_n > 0 or z_n > 0:
                results.append({
                    "column": col, "iqr_count": iqr_n, "iqr_pct": round(iqr_n / len(s) * 100, 2),
                    "z_count": z_n, "z_pct": round(z_n / len(s) * 100, 2),
                    "lower_iqr": round(float(lo), 4), "upper_iqr": round(float(hi), 4),
                    "p5": round(float(pct_5), 4), "p95": round(float(pct_95), 4),
                    "min": round(float(s.min()), 4), "max": round(float(s.max()), 4),
                })
        charts: Dict[str, str] = {}
        for r in results[:6]:
            col = r["column"]
            fig = go.Figure()
            fig.add_trace(go.Box(y=self.df[col].dropna(), name=col,
                                 marker_color="#4f8ef7", line_color="#4f8ef7",
                                 boxmean="sd", fillcolor="rgba(79,142,247,0.15)"))
            fig.update_layout(title=f"Box Plot — {col}", showlegend=False, **PLOTLY_DARK)
            charts[col] = fig_to_html(fig)
        return {"items": results, "charts": charts}

    def _univariate_numeric(self, num_cols: List[str]) -> List[Dict[str, Any]]:
        items = []
        for col in num_cols:
            s = self.df[col].dropna()
            if len(s) == 0: continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Distribution", "Box Plot"])
            fig.add_trace(go.Histogram(x=s, nbinsx=40, name="Freq",
                                       marker_color="rgba(79,142,247,0.6)",
                                       marker_line=dict(width=0)), row=1, col=1)
            fig.add_trace(go.Box(y=s, name=col, marker_color="#7c3aed",
                                 fillcolor="rgba(124,58,237,0.15)", boxmean="sd"), row=1, col=2)
            fig.update_layout(title=f"Univariate — {col}", showlegend=False,
                              height=280, **PLOTLY_DARK)
            items.append({
                "column": col,
                "count": int(s.count()), "missing": int(self.df[col].isnull().sum()),
                "mean": round(float(s.mean()), 4), "median": round(float(s.median()), 4),
                "mode": round(float(s.mode()[0]), 4) if len(s.mode()) > 0 else None,
                "std": round(float(s.std()), 4), "var": round(float(s.var()), 4),
                "min": round(float(s.min()), 4), "max": round(float(s.max()), 4),
                "range": round(float(s.max() - s.min()), 4),
                "q1": round(float(q1), 4), "q3": round(float(q3), 4),
                "iqr": round(float(q3 - q1), 4),
                "p5": round(float(s.quantile(0.05)), 4),
                "p95": round(float(s.quantile(0.95)), 4),
                "skewness": round(float(s.skew()), 4),
                "kurtosis": round(float(s.kurt()), 4),
                "skew_label": ("Approx Normal" if abs(s.skew()) < 0.5 else
                               "Moderate Skew" if abs(s.skew()) < 1 else "High Skew"),
                "zeros": int((s == 0).sum()),
                "negatives": int((s < 0).sum()),
                "cv": round(float(s.std() / s.mean() * 100), 2) if s.mean() != 0 else None,
                "chart": fig_to_html(fig),
            })
        return items

    def _univariate_categorical(self, cat_cols: List[str]) -> List[Dict[str, Any]]:
        items = []
        for col in cat_cols:
            s = self.df[col].dropna()
            if len(s) == 0: continue
            vc  = s.value_counts()
            top = vc.head(15)
            fig = go.Figure(go.Bar(
                x=top.values.tolist(), y=top.index.astype(str).tolist(),
                orientation="h",
                marker=dict(color=px.colors.sequential.Plasma[::-1][:len(top)], line=dict(width=0)),
                text=[f"{v / len(s) * 100:.1f}%" for v in top.values],
                textposition="outside",
            ))
            fig.update_layout(title=f"Value Counts — {col}",
                              yaxis=dict(autorange="reversed"),
                              height=max(200, len(top) * 28), **PLOTLY_DARK)
            items.append({
                "column": col,
                "kind": self._col_kinds.get(col, "categorical"),
                "count": int(s.count()), "missing": int(self.df[col].isnull().sum()),
                "unique": int(s.nunique()),
                "unique_pct": round(s.nunique() / len(s) * 100, 2),
                "top_value": str(vc.index[0]) if len(vc) > 0 else "",
                "top_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "top_pct": round(vc.iloc[0] / len(s) * 100, 2) if len(vc) > 0 else 0,
                "entropy": round(float(stats.entropy(vc.values / len(s) + 1e-9)), 4),
                "is_high_cardinality": s.nunique() > 20,
                "top_values": [{"label": str(k), "count": int(v), "pct": round(v / len(s) * 100, 2)}
                               for k, v in vc.head(10).items()],
                "chart": fig_to_html(fig),
            })
        return items

    def _bivariate(self, num_cols: List[str], cat_cols: List[str]) -> Dict[str, str]:
        charts: Dict[str, str] = {}

        pairs = [(num_cols[i], num_cols[j])
                 for i in range(min(4, len(num_cols)))
                 for j in range(i + 1, min(4, len(num_cols)))]
        if pairs:
            fig = make_subplots(rows=1, cols=len(pairs),
                                subplot_titles=[f"{a} vs {b}" for a, b in pairs[:4]])
            for idx, (a, b) in enumerate(pairs[:4]):
                d = self.df[[a, b]].dropna()
                fig.add_trace(go.Scatter(x=d[a], y=d[b], mode="markers",
                                         marker=dict(color="#4f8ef7", size=4, opacity=0.5),
                                         name=f"{a}x{b}"), row=1, col=idx + 1)
            fig.update_layout(title="Bivariate Scatterplots (Numeric vs Numeric)",
                              height=320, showlegend=False, **PLOTLY_DARK)
            charts["scatter"] = fig_to_html(fig)

        if cat_cols and num_cols:
            cat, num = cat_cols[0], num_cols[0]
            d    = self.df[[cat, num]].dropna()
            cats = d[cat].value_counts().head(10).index.tolist()
            d    = d[d[cat].isin(cats)]
            fig  = go.Figure()
            for c in cats:
                fig.add_trace(go.Box(y=d[d[cat] == c][num], name=str(c),
                                     boxmean="sd", marker_size=3))
            fig.update_layout(title=f"{num} by {cat}", height=320,
                              showlegend=False, **PLOTLY_DARK)
            charts["cat_num_box"] = fig_to_html(fig)

        if self.target and self.target in self.df.columns:
            tgt           = self.df[self.target]
            is_num_target = pd.api.types.is_numeric_dtype(tgt)

            if is_num_target and num_cols:
                corrs = {c: round(float(self.df[c].corr(tgt)), 3)
                         for c in num_cols if c != self.target}
                corrs = dict(sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:15])
                fig = go.Figure(go.Bar(
                    x=list(corrs.values()), y=list(corrs.keys()), orientation="h",
                    marker=dict(color=["#22c55e" if v > 0 else "#ef4444" for v in corrs.values()],
                                line=dict(width=0)),
                ))
                fig.update_layout(title=f"Feature Correlation with Target: {self.target}",
                                  yaxis=dict(autorange="reversed"), height=350, **PLOTLY_DARK)
                charts["target_corr"] = fig_to_html(fig)

            elif not is_num_target and num_cols:
                col     = num_cols[0]
                classes = tgt.value_counts().head(6).index.tolist()
                fig     = go.Figure()
                colors  = px.colors.qualitative.Plotly
                for i, cls in enumerate(classes):
                    vals = self.df[self.df[self.target] == cls][col].dropna()
                    fig.add_trace(go.Violin(y=vals, name=str(cls),
                                           fillcolor=colors[i % len(colors)],
                                           line_color=colors[i % len(colors)],
                                           opacity=0.7, meanline_visible=True, box_visible=True))
                fig.update_layout(title=f"{col} Distribution by {self.target}",
                                  height=340, **PLOTLY_DARK)
                charts["target_violin"] = fig_to_html(fig)

        return charts

    def _correlations(self, num_cols: List[str]) -> Dict[str, Any]:
        if len(num_cols) < 2:
            return {"matrix": [], "columns": [], "strong_pairs": [], "chart": "", "cov_chart": ""}
        sub  = self.df[num_cols].dropna(how="all")
        corr = sub.corr()

        fig = go.Figure(go.Heatmap(
            z=corr.values.tolist(), x=num_cols, y=num_cols,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}", showscale=True,
            colorbar=dict(thickness=12, len=0.8),
        ))
        fig.update_layout(title="Pearson Correlation Matrix",
                          height=max(350, len(num_cols) * 40 + 100), **PLOTLY_DARK)

        strong = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                v = corr.iloc[i, j]
                if abs(v) > 0.5:
                    strong.append({
                        "col1": num_cols[i], "col2": num_cols[j],
                        "corr": round(float(v), 3),
                        "strength": ("Very Strong" if abs(v) > 0.9 else
                                     "Strong" if abs(v) > 0.7 else "Moderate"),
                    })
        strong.sort(key=lambda x: abs(x["corr"]), reverse=True)

        cov  = sub.cov()
        fig2 = go.Figure(go.Heatmap(
            z=cov.values.tolist(), x=num_cols, y=num_cols,
            colorscale="Viridis", showscale=True,
            colorbar=dict(thickness=12, len=0.8),
        ))
        fig2.update_layout(title="Covariance Matrix",
                           height=max(350, len(num_cols) * 40 + 100), **PLOTLY_DARK)
        return {
            "columns": num_cols,
            "matrix": [[round(float(v), 3) for v in row] for row in corr.values],
            "strong_pairs": strong[:20],
            "chart": fig_to_html(fig),
            "cov_chart": fig_to_html(fig2),
        }

    def _normality_tests(self, num_cols: List[str]) -> Dict[str, Any]:
        results = []
        for col in num_cols:
            s = self.df[col].dropna()
            if len(s) < 8: continue
            sample = s.sample(min(5000, len(s)), random_state=42)
            sw_stat = sw_p = nt_stat = nt_p = ks_stat = ks_p = None
            try: sw_stat, sw_p = shapiro(sample[:min(5000, len(sample))])
            except Exception: pass
            try: nt_stat, nt_p = normaltest(sample)
            except Exception: pass
            try: ks_stat, ks_p = kstest(stats.zscore(sample), "norm")
            except Exception: pass
            is_normal = (sw_p or 1) > 0.05 and (nt_p or 1) > 0.05
            results.append({
                "column": col,
                "shapiro_stat":    round(float(sw_stat), 4) if sw_stat is not None else None,
                "shapiro_p":       round(float(sw_p),    4) if sw_p    is not None else None,
                "normaltest_stat": round(float(nt_stat), 4) if nt_stat is not None else None,
                "normaltest_p":    round(float(nt_p),    4) if nt_p    is not None else None,
                "ks_stat":         round(float(ks_stat), 4) if ks_stat is not None else None,
                "ks_p":            round(float(ks_p),    4) if ks_p    is not None else None,
                "is_normal":  is_normal,
                "verdict":    "Normal" if is_normal else "Not Normal",
            })
        charts: Dict[str, str] = {}
        for r in results[:4]:
            col = r["column"]
            s   = self.df[col].dropna().sample(min(1000, len(self.df[col].dropna())), random_state=42)
            (osm, osr), (slope, intercept, _) = stats.probplot(s, dist="norm")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(osm), y=list(osr), mode="markers",
                                      marker=dict(color="#4f8ef7", size=4, opacity=0.7), name="Data"))
            line_x = [min(osm), max(osm)]
            line_y = [slope * x + intercept for x in line_x]
            fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines",
                                      line=dict(color="#ef4444", width=2), name="Normal"))
            fig.update_layout(title=f"Q-Q Plot — {col}", height=280, **PLOTLY_DARK)
            charts[col] = fig_to_html(fig)
        return {"tests": results, "charts": charts}

    def _skew_kurt(self, num_cols: List[str]) -> Dict[str, Any]:
        items = []
        for col in num_cols:
            s = self.df[col].dropna()
            if len(s) < 3: continue
            sk = float(s.skew())
            ku = float(s.kurt())
            items.append({
                "column": col, "skewness": round(sk, 4), "kurtosis": round(ku, 4),
                "skew_direction": "Right (+)" if sk > 0 else "Left (-)" if sk < 0 else "Symmetric",
                "skew_severity": ("Severe"   if abs(sk) > 2   else
                                  "High"     if abs(sk) > 1   else
                                  "Moderate" if abs(sk) > 0.5 else "Normal"),
                "kurt_type": ("Leptokurtic (heavy tails)" if ku > 1 else
                              "Platykurtic (light tails)" if ku < -1 else "Mesokurtic (normal)"),
                "needs_transform": abs(sk) > 1,
            })
        chart = ""
        if items:
            cols_sk = [i["column"] for i in items]
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Skewness", "Kurtosis"])
            fig.add_trace(go.Bar(x=cols_sk, y=[i["skewness"] for i in items],
                                  marker=dict(
                                      color=["#ef4444" if abs(i["skewness"]) > 1 else
                                             "#f59e0b" if abs(i["skewness"]) > 0.5 else
                                             "#22c55e" for i in items],
                                      line=dict(width=0))), row=1, col=1)
            fig.add_trace(go.Bar(x=cols_sk, y=[i["kurtosis"] for i in items],
                                  marker=dict(color="#7c3aed", line=dict(width=0))), row=1, col=2)
            fig.update_layout(title="Skewness & Kurtosis", height=300,
                              showlegend=False, **PLOTLY_DARK)
            chart = fig_to_html(fig)
        return {"items": items, "chart": chart}

    def _datetime_analysis(self, dt_cols: List[str]) -> List[Dict[str, Any]]:
        items = []
        for col in dt_cols:
            s = self.df[col].dropna()
            if len(s) == 0: continue
            items.append({
                "column": col,
                "min": str(s.min()), "max": str(s.max()),
                "range_days": (s.max() - s.min()).days,
                "extracted": {
                    "year":    s.dt.year.value_counts().head(5).to_dict(),
                    "month":   s.dt.month.value_counts().sort_index().to_dict(),
                    "weekday": s.dt.dayofweek.value_counts().sort_index().to_dict(),
                },
            })
        return items

    def _text_analysis(self, txt_cols: List[str]) -> List[Dict[str, Any]]:
        items = []
        for col in txt_cols:
            s       = self.df[col].dropna().astype(str)
            lengths = s.str.len()
            words   = s.str.split().str.len()
            items.append({
                "column":     col,
                "avg_length": round(float(lengths.mean()), 1),
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
                "avg_words":  round(float(words.mean()), 1),
                "unique_pct": round(s.nunique() / len(s) * 100, 2),
            })
        return items

    def _variance_analysis(self, num_cols: List[str]) -> List[Dict[str, Any]]:
        items = []
        for col in num_cols:
            s = self.df[col].dropna()
            if len(s) < 2: continue
            var = float(s.var())
            items.append({
                "column": col, "variance": round(var, 6),
                "std": round(float(s.std()), 4),
                "cv":  round(float(s.std() / abs(s.mean()) * 100), 2) if s.mean() != 0 else None,
                "low_variance": var < 0.01,
            })
        items.sort(key=lambda x: x["variance"])
        return items

    def _pca_summary(self, num_cols: List[str]) -> Dict[str, Any]:
        if len(num_cols) < 2: return {}
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            sub          = self.df[num_cols].dropna()
            if len(sub) < 4: return {}
            scaled       = StandardScaler().fit_transform(sub)
            n_components = min(len(num_cols), len(sub), 10)
            pca          = PCA(n_components=n_components)
            pca.fit(scaled)
            evr = pca.explained_variance_ratio_.tolist()
            cum = np.cumsum(evr).tolist()
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Explained Variance per PC", "Cumulative Variance"])
            fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(evr))],
                                  y=[round(v * 100, 2) for v in evr],
                                  marker=dict(color="#4f8ef7", line=dict(width=0)),
                                  name="Var %"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(cum))],
                                      y=[round(v * 100, 2) for v in cum],
                                      line=dict(color="#22c55e", width=2),
                                      mode="lines+markers", name="Cumul %"), row=1, col=2)
            fig.add_hline(y=95, line_dash="dash", line_color="#f59e0b",
                          annotation_text="95%", row=1, col=2)
            fig.update_layout(title="PCA — Explained Variance", height=300,
                              showlegend=False, **PLOTLY_DARK)
            n_for_95 = next((i + 1 for i, v in enumerate(cum) if v >= 0.95), len(cum))
            return {
                "n_components": n_components,
                "explained_variance_ratio": [round(v, 4) for v in evr],
                "cumulative": [round(v, 4) for v in cum],
                "n_for_95pct": n_for_95,
                "chart": fig_to_html(fig),
            }
        except Exception as e:
            return {"error": str(e)}

    def _class_imbalance(self) -> Optional[Dict[str, Any]]:
        if not self.target or self.target not in self.df.columns: return None
        s     = self.df[self.target].dropna()
        vc    = s.value_counts()
        total = len(s)
        classes = [{"label": str(k), "count": int(v), "pct": round(v / total * 100, 2)}
                   for k, v in vc.items()]
        ratio = float(vc.max() / max(vc.min(), 1))
        fig = go.Figure(go.Pie(
            labels=[c["label"] for c in classes],
            values=[c["count"]  for c in classes],
            hole=0.45,
            marker=dict(colors=px.colors.qualitative.Plotly,
                        line=dict(color="rgba(0,0,0,0)", width=0)),
        ))
        fig.update_layout(title=f"Class Distribution — {self.target}",
                          height=320, **PLOTLY_DARK)
        return {
            "target": self.target, "classes": classes,
            "n_classes": len(vc), "imbalance_ratio": round(ratio, 2),
            "is_imbalanced": ratio > 2,
            "majority": str(vc.index[0]), "minority": str(vc.index[-1]),
            "chart": fig_to_html(fig),
        }

    def _feature_target(self, num_cols: List[str], cat_cols: List[str]) -> Dict[str, str]:
        if not self.target or self.target not in self.df.columns: return {}
        charts: Dict[str, str] = {}
        tgt           = self.df[self.target]
        is_num_target = pd.api.types.is_numeric_dtype(tgt)

        if is_num_target and num_cols:
            corrs = {c: round(float(self.df[c].corr(tgt)), 3)
                     for c in num_cols if c != self.target}
            corrs = dict(sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:12])
            fig = go.Figure(go.Bar(
                x=list(corrs.values()), y=list(corrs.keys()), orientation="h",
                marker=dict(color=["#22c55e" if v > 0 else "#ef4444" for v in corrs.values()],
                            line=dict(width=0)),
                text=[str(v) for v in corrs.values()], textposition="outside",
            ))
            fig.update_layout(title=f"Pearson Correlation with {self.target}",
                              yaxis=dict(autorange="reversed"), height=380, **PLOTLY_DARK)
            charts["numeric_corr"] = fig_to_html(fig)

        if is_num_target and num_cols:
            top2 = [c for c in num_cols if c != self.target][:2]
            for feat in top2:
                d = self.df[[feat, self.target]].dropna()
                try:
                    fig = px.scatter(d, x=feat, y=self.target, trendline="ols",
                                     trendline_color_override="#ef4444",
                                     color_discrete_sequence=["rgba(79,142,247,0.5)"])
                    fig.update_layout(title=f"{feat} vs {self.target}", height=300, **PLOTLY_DARK)
                    charts[f"scatter_{feat}"] = fig_to_html(fig)
                except Exception:
                    pass

        return charts

    def _chi_square(self, cat_cols: List[str]) -> List[Dict[str, Any]]:
        results = []
        pairs = [(cat_cols[i], cat_cols[j])
                 for i in range(min(5, len(cat_cols)))
                 for j in range(i + 1, min(5, len(cat_cols)))]
        for a, b in pairs[:10]:
            try:
                ct = pd.crosstab(self.df[a], self.df[b])
                if ct.shape[0] < 2 or ct.shape[1] < 2: continue
                chi2, p, dof, _ = chi2_contingency(ct)
                results.append({
                    "col1": a, "col2": b,
                    "chi2": round(float(chi2), 3), "p_value": round(float(p), 4),
                    "dof": dof,
                    "significant": p < 0.05,
                    "verdict": "Dependent" if p < 0.05 else "Independent",
                })
            except Exception:
                pass
        return sorted(results, key=lambda x: x["p_value"])

    def _column_meta(self) -> List[Dict[str, Any]]:
        meta = []
        for col in self.df.columns:
            s = self.df[col]
            meta.append({
                "name": col, "dtype": str(s.dtype),
                "kind": self._col_kinds.get(col, "?"),
                "count": int(s.count()),
                "missing": int(s.isnull().sum()),
                "missing_pct": round(s.isnull().mean() * 100, 2),
                "unique": int(s.nunique()),
                "unique_pct": round(s.nunique() / max(len(s), 1) * 100, 2),
                "is_target": col == self.target,
            })
        return meta

    def _recommendations(self, num: List[str], cat: List[str],
                          dt: List[str], txt: List[str]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        df   = self.df
        miss = df.isnull().sum()

        hi      = miss[miss / self.n > 0.4]
        med_num = [c for c in num if 0.05 < miss.get(c, 0) / self.n <= 0.4]
        med_cat = [c for c in cat if 0.05 < miss.get(c, 0) / self.n <= 0.4]

        if len(hi):
            recs.append({"id": "drop_high_miss", "category": "Missing Values", "severity": "high", "icon": "⚠️",
                         "title": f"Drop {len(hi)} columns with >40% missing",
                         "description": f"Columns: {list(hi.index)}",
                         "affected_columns": list(hi.index),
                         "actions": [{"label": "Drop columns", "action_id": "drop_high_missing"},
                                     {"label": "Keep",         "action_id": "skip"}]})
        if med_num:
            recs.append({"id": "impute_num", "category": "Missing Values", "severity": "medium", "icon": "🔧",
                         "title": f"Impute {len(med_num)} numeric columns",
                         "description": str(med_num), "affected_columns": med_num,
                         "actions": [{"label": "Median", "action_id": "impute_median"},
                                     {"label": "Mean",   "action_id": "impute_mean"},
                                     {"label": "KNN",    "action_id": "impute_knn"}]})
        if med_cat:
            recs.append({"id": "impute_cat", "category": "Missing Values", "severity": "medium", "icon": "🔧",
                         "title": f"Impute {len(med_cat)} categorical columns",
                         "description": str(med_cat), "affected_columns": med_cat,
                         "actions": [{"label": "Mode",      "action_id": "impute_mode"},
                                     {"label": "'Unknown'", "action_id": "impute_unknown"}]})

        nd = int(df.duplicated().sum())
        if nd:
            recs.append({"id": "dups", "category": "Duplicates", "severity": "medium", "icon": "📋",
                         "title": f"Remove {nd} duplicates ({round(nd / self.n * 100, 1)}%)",
                         "description": "Exact duplicate rows found.", "affected_columns": [],
                         "actions": [{"label": "Drop duplicates", "action_id": "drop_duplicates"},
                                     {"label": "Keep first",       "action_id": "keep_first"}]})

        out_cols = []
        for c in num:
            s = df[c].dropna()
            if len(s) < 4: continue
            q1, q3 = s.quantile(.25), s.quantile(.75)
            iqr    = q3 - q1
            if iqr > 0 and ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum() / len(s) > 0.05:
                out_cols.append(c)
        if out_cols:
            recs.append({"id": "outliers", "category": "Outliers", "severity": "medium", "icon": "📊",
                         "title": f"Outliers in {len(out_cols)} columns",
                         "description": str(out_cols), "affected_columns": out_cols,
                         "actions": [{"label": "Cap (Winsorize)", "action_id": "cap_outliers"},
                                     {"label": "Remove rows",     "action_id": "remove_outliers"},
                                     {"label": "Keep",            "action_id": "skip"}]})

        skew_cols = [c for c in num if len(df[c].dropna()) > 2 and abs(df[c].skew()) > 1.5 and df[c].min() > 0]
        if skew_cols:
            recs.append({"id": "skew", "category": "Transformation", "severity": "low", "icon": "📈",
                         "title": f"High skewness in {len(skew_cols)} columns",
                         "description": str(skew_cols), "affected_columns": skew_cols,
                         "actions": [{"label": "Log Transform", "action_id": "log_transform"},
                                     {"label": "Sqrt",          "action_id": "sqrt_transform"},
                                     {"label": "Yeo-Johnson",   "action_id": "yeo_johnson"},
                                     {"label": "Skip",          "action_id": "skip"}]})

        if num:
            recs.append({"id": "scale", "category": "Scaling", "severity": "low", "icon": "⚖️",
                         "title": "Apply feature scaling",
                         "description": "Recommended for most ML models.",
                         "affected_columns": num,
                         "actions": [{"label": "StandardScaler", "action_id": "scale_standard"},
                                     {"label": "MinMax",         "action_id": "scale_minmax"},
                                     {"label": "Robust",         "action_id": "scale_robust"},
                                     {"label": "Skip",           "action_id": "skip"}]})

        lo_card = [c for c in cat if df[c].nunique() <= 15]
        hi_card = [c for c in cat if df[c].nunique() >  15]
        if lo_card:
            recs.append({"id": "enc_low", "category": "Encoding", "severity": "low", "icon": "🏷️",
                         "title": f"Encode {len(lo_card)} low-cardinality categoricals",
                         "description": str(lo_card), "affected_columns": lo_card,
                         "actions": [{"label": "One-Hot", "action_id": "encode_onehot"},
                                     {"label": "Label",   "action_id": "encode_label"},
                                     {"label": "Skip",    "action_id": "skip"}]})
        if hi_card:
            recs.append({"id": "enc_high", "category": "Encoding", "severity": "medium", "icon": "🔢",
                         "title": f"Handle {len(hi_card)} high-cardinality columns",
                         "description": str(hi_card), "affected_columns": hi_card,
                         "actions": [{"label": "Frequency", "action_id": "encode_frequency"},
                                     {"label": "Target",    "action_id": "encode_target"},
                                     {"label": "Label",     "action_id": "encode_label"},
                                     {"label": "Skip",      "action_id": "skip"}]})
        return recs

    # ══════════════════════════════════════════════════════════════════
    # NEW METHOD 1 — Auto Problem Type Detection
    # ══════════════════════════════════════════════════════════════════

    def _detect_problem_type(self) -> Dict[str, Any]:
        """
        Infer ML problem type from target column dtype + cardinality.

        Returns
        -------
        {
          type       : "binary_classification" | "multiclass_classification"
                       | "regression" | "unknown"
          confidence : "high" | "medium" | "low"
          reason     : str
          n_classes  : int | None
        }
        """
        if not self.target or self.target not in self.df.columns:
            return {"type": "unknown", "confidence": "low",
                    "reason": "No target column specified.", "n_classes": None}

        s      = self.df[self.target].dropna()
        n_uniq = s.nunique()
        is_num = pd.api.types.is_numeric_dtype(s)

        # Bool or 2 unique → binary
        if pd.api.types.is_bool_dtype(s) or n_uniq == 2:
            return {"type": "binary_classification", "confidence": "high",
                    "reason": "Target has exactly 2 unique values.", "n_classes": 2}

        # Non-numeric → multiclass
        if not is_num:
            return {"type": "multiclass_classification",
                    "confidence": "high" if n_uniq <= 20 else "medium",
                    "reason": f"Target is categorical with {n_uniq} classes.",
                    "n_classes": n_uniq}

        # Numeric integer with low cardinality → likely class labels
        try:
            all_int = bool((s == s.astype(int)).all())
        except Exception:
            all_int = False

        if all_int and n_uniq <= 20:
            return {"type": "multiclass_classification", "confidence": "medium",
                    "reason": f"Target is integer with {n_uniq} distinct values — likely class labels.",
                    "n_classes": n_uniq}

        # Continuous numeric → regression
        return {"type": "regression", "confidence": "high",
                "reason": f"Target is continuous numeric with {n_uniq} unique values.",
                "n_classes": None}

    # ══════════════════════════════════════════════════════════════════
    # NEW METHOD 2 — Temporal Awareness
    # ══════════════════════════════════════════════════════════════════

    def _temporal_awareness(
        self, dt_cols: List[str], num_cols: List[str]
    ) -> Dict[str, Any]:
        """
        For each datetime column (up to 2):
          1. Sort data by time.
          2. Split into 4 time-quartiles; measure mean shift Q1->Q4 per
             numeric feature. Shift >20% flagged as concept drift.
          3. Produce rolling-mean trend chart for first numeric col.

        Returns
        -------
        { available: bool, items: List[Dict] }
        """
        if not dt_cols or not num_cols:
            return {"available": False, "items": []}

        items = []
        for dt_col in dt_cols[:2]:
            try:
                num_subset = [c for c in num_cols[:4] if c != self.target]
                if not num_subset:
                    continue

                tmp = self.df[[dt_col] + num_subset].dropna(subset=[dt_col]).copy()
                tmp = tmp.sort_values(dt_col).reset_index(drop=True)

                # concept drift via quartile mean shift
                drift_findings: List[Dict[str, Any]] = []
                try:
                    tmp["_q"] = pd.qcut(
                        tmp[dt_col].astype(np.int64), q=4,
                        labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop",
                    )
                    for nc in num_subset:
                        grp    = tmp.dropna(subset=[nc, "_q"]).groupby("_q", observed=True)[nc]
                        qmeans = grp.mean()
                        if len(qmeans) < 2: continue
                        q1_mean   = float(qmeans.iloc[0])
                        q4_mean   = float(qmeans.iloc[-1])
                        drift_pct = abs(q4_mean - q1_mean) / max(abs(q1_mean), 1e-9) * 100
                        drift_findings.append({
                            "feature":   nc,
                            "drift_pct": round(drift_pct, 2),
                            "q1_mean":   round(q1_mean, 4),
                            "q4_mean":   round(q4_mean, 4),
                            "has_drift": drift_pct > 20,
                        })
                except Exception:
                    pass

                # rolling mean chart
                chart = ""
                nc    = num_subset[0]
                if nc in tmp.columns and len(tmp) > 10:
                    window  = max(5, len(tmp) // 20)
                    rolling = (
                        tmp.set_index(dt_col)[nc]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .dropna()
                    )
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=rolling.index.tolist(), y=rolling.values.tolist(),
                        mode="lines", name=f"Rolling mean: {nc}",
                        line=dict(color="#4f8ef7", width=1.5),
                    ))
                    fig.update_layout(
                        title=f"Temporal Trend — {nc} over {dt_col}",
                        xaxis_title=dt_col, yaxis_title=nc,
                        height=260, **PLOTLY_DARK,
                    )
                    chart = fig_to_html(fig)

                items.append({
                    "datetime_col":     dt_col,
                    "date_range":       f"{tmp[dt_col].min().date()} -> {tmp[dt_col].max().date()}",
                    "n_rows":           len(tmp),
                    "drift_findings":   drift_findings,
                    "has_concept_drift": any(d["has_drift"] for d in drift_findings),
                    "chart":            chart,
                })
            except Exception:
                continue

        return {"available": bool(items), "items": items}

    # ══════════════════════════════════════════════════════════════════
    # NEW METHOD 3 — Feature Interactions
    # ══════════════════════════════════════════════════════════════════

    def _feature_interactions(
        self, num_cols: List[str], cat_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Delegate to FeatureInteractions (interactions.py).
        MI / Cramer's V / eta-squared pairwise scores, top pairs,
        feature-vs-target scores, NxN matrix, redundancy groups.
        Returns a safe empty dict if the module fails for any reason.
        """
        try:
            from .interactions import FeatureInteractions
            fi = FeatureInteractions(self.df, target=self.target, max_cols=20, top_n_pairs=15)
            return fi.compute()
        except Exception as exc:
            return {
                "available":         False,
                "error":             str(exc),
                "pairs":             [],
                "top_pairs":         [],
                "target_scores":     [],
                "matrix_cols":       [],
                "matrix":            [],
                "redundancy_groups": [],
                "n_strong":          0,
                "n_moderate":        0,
                "summary":           "Feature interaction analysis unavailable.",
            }