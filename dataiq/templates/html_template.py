"""html_template.py — DataIQ HTML Report Template

Original template preserved exactly.
Five new nav sections + render functions added:
  • ML Readiness Score  (readiness)
  • Leakage Detective   (leakage)
  • Feature Interactions (interactions)
  • Temporal Awareness  (temporal)
  • Problem Type        (shown in overview, no separate tab)

Branding updated: MLProfiler → DataIQ
Placeholder added: __REPORT_MODE__  (used by report_builder, ignored by template)
"""

ML_KNOWLEDGE = """
<div class="kb-section">
  <div class="kb-title">1. Problem Framing</div>
  <ul>
    <li><b>Regression</b> — predict a continuous value (e.g. price, temperature)</li>
    <li><b>Classification</b> — Binary (yes/no), Multiclass (One-vs-One, One-vs-All)</li>
    <li><b>Clustering</b> — find natural groupings without labels (K-Means, DBSCAN)</li>
    <li><b>Recommendation / Ranking</b> — collaborative filtering, matrix factorization</li>
    <li><b>Metrics</b> — Accuracy, Precision, Recall, F1, AUC-ROC · RMSE, MAE, R²</li>
    <li><b>Fairness</b> — check for bias across demographic groups</li>
  </ul>
</div>
<div class="kb-section">
  <div class="kb-title">2. Data Collection</div>
  <ul>
    <li>SQL / NoSQL databases, CSV / Excel / JSON / XML</li>
    <li>REST APIs, Web scraping, IoT sensors, logs, streams</li>
    <li>Batch pipelines vs Real-time streaming pipelines</li>
  </ul>
</div>
<div class="kb-section">
  <div class="kb-title">3. Data Understanding</div>
  <ul>
    <li>Shape, dtypes, unique values, cardinality</li>
    <li>Missing values, duplicates</li>
    <li>Mean, median, mode, std, skewness, kurtosis</li>
    <li>Correlation matrix, covariance, pairplots</li>
    <li>Pandas .describe(), .info(), Sweetviz, Pandas Profiling</li>
  </ul>
</div>
<div class="kb-section">
  <div class="kb-title">4. Data Preprocessing</div>
  <ul>
    <li><b>Missing</b> — CCA, mean/median/mode imputation, KNN/MICE, MCAR/MAR/MNAR</li>
    <li><b>Outliers</b> — Z-score, IQR detection; Winsorization, trimming</li>
    <li><b>Categorical</b> — Label Encoding, One-Hot, high-cardinality encodings</li>
    <li><b>Numeric</b> — Standard/MinMax/Robust/MaxAbs scaling, Log/Box-Cox/Yeo-Johnson, binning</li>
    <li><b>Datetime</b> — year/month/day/weekday/quarter extraction, rolling averages</li>
    <li><b>Text</b> — tokenization, stopwords, TF-IDF, Word2Vec, GloVe, FastText</li>
  </ul>
</div>
<div class="kb-section">
  <div class="kb-title">5. EDA Techniques</div>
  <ul>
    <li><b>Univariate</b> — histogram, boxplot, countplot, skewness, kurtosis</li>
    <li><b>Bivariate</b> — scatterplot, correlation, boxplot by category, chi-square</li>
    <li><b>Multivariate</b> — PCA, clustering, pairplots, 3D scatter</li>
    <li><b>Patterns</b> — Feature-target relationships, anomaly detection</li>
  </ul>
</div>
<div class="kb-section">
  <div class="kb-title">6. Feature Engineering</div>
  <ul>
    <li><b>Construction</b> — ratios, interactions, polynomial features</li>
    <li><b>Extraction</b> — PCA, ICA, LDA, text/image embeddings</li>
    <li><b>Selection</b> — Variance threshold, correlation threshold, RFE, tree importance, Lasso</li>
    <li><b>Imbalance</b> — SMOTE, ADASYN, oversampling, undersampling, stratified split</li>
  </ul>
</div>
<div class="kb-section">
  <div class="kb-title">7. Modeling Prep</div>
  <ul>
    <li>Train/test split, stratified split</li>
    <li>K-Fold / Stratified K-Fold cross-validation</li>
    <li>Preventing data leakage</li>
    <li>sklearn Pipelines for preprocessing + modeling</li>
  </ul>
</div>
<div class="kb-section">
  <div class="kb-title">8. Model Training</div>
  <ul>
    <li><b>Regression</b> — Linear, Ridge, Lasso, ElasticNet, RF, XGBoost, LightGBM, CatBoost, NN</li>
    <li><b>Classification</b> — Logistic, DT, RF, GBM, SVM, KNN, Naive Bayes, NN</li>
    <li><b>Clustering</b> — K-Means, Hierarchical, DBSCAN</li>
    <li><b>Dim Reduction</b> — PCA, t-SNE, UMAP</li>
    <li><b>Tuning</b> — GridSearch, RandomSearch, Bayesian Opt, Optuna</li>
    <li><b>Regularization</b> — L1, L2, Dropout, Early stopping</li>
    <li><b>Ensembles</b> — Bagging, Boosting, Stacking</li>
  </ul>
</div>
<div class="kb-section">
  <div class="kb-title">9. Model Evaluation</div>
  <ul>
    <li>Regression — MSE, RMSE, MAE, R²</li>
    <li>Classification — Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC</li>
    <li>Learning curves, Bias-Variance tradeoff</li>
    <li>SHAP, LIME, Permutation Importance</li>
  </ul>
</div>
<div class="kb-section">
  <div class="kb-title">10–12. Optimization & Deployment</div>
  <ul>
    <li>Hyperparameter refinements, feature engineering iterations, stacking</li>
    <li>pickle / joblib / ONNX serialization</li>
    <li>FastAPI / Flask / Django APIs</li>
    <li>Docker, Kubernetes, CI/CD</li>
    <li>Model drift, data drift monitoring, retraining pipelines</li>
  </ul>
</div>
<div class="kb-section">
  <div class="kb-title">13. Advanced Topics</div>
  <ul>
    <li>Time Series — ARIMA, SARIMA, Prophet, LSTM</li>
    <li>NLP — Transformers, seq2seq, embeddings</li>
    <li>Computer Vision — CNNs, transfer learning, augmentation</li>
    <li>Graph Data — GNNs</li>
    <li>Probabilistic / Bayesian Modeling</li>
    <li>Reinforcement Learning</li>
    <li>Causal Inference</li>
  </ul>
</div>
"""


def get_template() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>DataIQ — __DATASET_NAME__</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
/* ══════════════════════════════════════════════
   ROOT DESIGN TOKENS
══════════════════════════════════════════════ */
:root{
  --bg:#05070f;
  --bg2:#080c18;
  --bg3:#0d1225;
  --glass:rgba(15,20,45,0.6);
  --glass2:rgba(20,28,60,0.5);
  --border:rgba(79,142,247,0.12);
  --border2:rgba(255,255,255,0.06);
  --accent:#4f8ef7;
  --accent2:#7c3aed;
  --accent3:#06b6d4;
  --green:#22c55e;
  --yellow:#f59e0b;
  --red:#ef4444;
  --pink:#ec4899;
  --text:#e2e8f0;
  --text2:#94a3b8;
  --text3:#475569;
  --text4:#1e293b;
  --mono:'JetBrains Mono',monospace;
  --sans:'Space Grotesk',sans-serif;
  --radius:14px;
  --radius-sm:8px;
  --sb-width:260px;
  --kb-width:240px;
  --glow:0 0 30px rgba(79,142,247,0.08);
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth;font-size:13px}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;overflow-x:hidden}

/* ── Background mesh ── */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(ellipse 60% 40% at 20% 10%,rgba(79,142,247,0.06) 0%,transparent 60%),
    radial-gradient(ellipse 50% 50% at 80% 80%,rgba(124,58,237,0.05) 0%,transparent 60%),
    radial-gradient(ellipse 40% 30% at 60% 30%,rgba(6,182,212,0.04) 0%,transparent 50%);
}

/* ══════════════════════════════════════════════
   LAYOUT
══════════════════════════════════════════════ */
.layout{display:flex;min-height:100vh;position:relative;z-index:1}
.sidebar-left{
  width:var(--sb-width);flex-shrink:0;
  background:rgba(8,12,24,0.9);
  backdrop-filter:blur(20px);
  border-right:1px solid var(--border);
  position:sticky;top:0;height:100vh;overflow-y:auto;
  display:flex;flex-direction:column;
}
.main-area{flex:1;min-width:0;padding:28px 32px;overflow:hidden}
.sidebar-right{
  width:var(--kb-width);flex-shrink:0;
  background:rgba(8,12,24,0.85);
  backdrop-filter:blur(20px);
  border-left:1px solid var(--border);
  position:sticky;top:0;height:100vh;overflow-y:auto;
}

/* ══════════════════════════════════════════════
   LEFT SIDEBAR — NAVIGATION
══════════════════════════════════════════════ */
.logo-area{padding:22px 18px 16px;border-bottom:1px solid var(--border)}
.logo-name{
  font-size:20px;font-weight:700;letter-spacing:-0.5px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.logo-ver{font-size:10px;font-family:var(--mono);color:var(--text3);margin-top:2px}
.dataset-info{padding:14px 18px;border-bottom:1px solid var(--border)}
.ds-name{font-weight:600;font-size:13px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ds-meta{font-size:10px;font-family:var(--mono);color:var(--text3);margin-top:4px}
.ds-badge{display:inline-flex;align-items:center;gap:4px;background:rgba(79,142,247,0.1);border:1px solid rgba(79,142,247,0.2);border-radius:5px;padding:1px 6px;font-size:10px;color:var(--accent);font-family:var(--mono);margin-top:6px}

.nav-group{padding:10px 0}
.nav-header{padding:5px 18px 3px;font-size:9.5px;font-weight:600;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase;font-family:var(--mono)}
.nav-item{
  display:flex;align-items:center;gap:9px;padding:8px 18px;
  cursor:pointer;color:var(--text2);font-size:12.5px;font-weight:500;
  transition:all .15s;border-left:2px solid transparent;position:relative;
}
.nav-item:hover{background:rgba(79,142,247,0.06);color:var(--text)}
.nav-item.active{
  background:linear-gradient(90deg,rgba(79,142,247,0.12),transparent);
  color:var(--accent);border-left-color:var(--accent);
}
.nav-item .ni{font-size:14px;width:18px;text-align:center;flex-shrink:0}
.nav-badge{
  margin-left:auto;background:var(--bg3);border:1px solid var(--border2);
  border-radius:999px;padding:0 6px;font-size:10px;
  font-family:var(--mono);color:var(--text3);min-width:20px;text-align:center;
}
.nav-badge.warn{background:rgba(245,158,11,.15);border-color:rgba(245,158,11,.3);color:var(--yellow)}
.nav-badge.bad{background:rgba(239,68,68,.15);border-color:rgba(239,68,68,.3);color:var(--red)}
.nav-badge.good{background:rgba(34,197,94,.15);border-color:rgba(34,197,94,.3);color:var(--green)}

.compare-toggle{margin:12px 18px;display:flex;background:var(--bg3);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.ct-btn{flex:1;padding:7px;text-align:center;font-size:11px;font-weight:600;cursor:pointer;transition:all .15s;color:var(--text3)}
.ct-btn.active{background:var(--accent);color:#fff}

/* ══════════════════════════════════════════════
   RIGHT SIDEBAR — ML KNOWLEDGE BASE
══════════════════════════════════════════════ */
.sidebar-right{
  width:var(--kb-width);flex-shrink:0;
  background:rgba(8,12,24,0.85);
  backdrop-filter:blur(20px);
  border-left:1px solid var(--border);
  position:sticky;top:0;height:100vh;
  display:flex;flex-direction:column;
  min-width:180px;max-width:520px;
  user-select:none;
}
.kb-resize-handle{
  position:absolute;left:0;top:0;bottom:0;width:5px;
  cursor:col-resize;z-index:10;
  background:transparent;transition:background .2s;
}
.kb-resize-handle:hover,.kb-resize-handle.dragging{background:var(--accent)}
.kb-header{
  padding:14px 16px 10px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:8px;flex-shrink:0;
}
.kb-title-main{font-weight:700;font-size:13px;color:var(--text);flex:1}
.kb-section-label{
  font-size:9px;font-family:var(--mono);font-weight:700;
  color:var(--accent);text-transform:uppercase;letter-spacing:1.2px;
  padding:6px 16px 4px;opacity:0.7;
}
.kb-scroll{flex:1;overflow-y:auto;overflow-x:hidden}
.kb-panel{display:none;padding:0 0 24px 0}
.kb-panel.active{display:block}
/* KB topic blocks */
.kb-block{padding:10px 16px;border-bottom:1px solid rgba(255,255,255,0.04)}
.kb-block-title{
  font-size:11px;font-weight:700;color:var(--accent);
  margin-bottom:6px;font-family:var(--mono);
  display:flex;align-items:center;gap:6px;cursor:pointer;
}
.kb-block-title::after{content:'▾';font-size:9px;margin-left:auto;color:var(--text3);transition:transform .2s}
.kb-block-title.collapsed::after{transform:rotate(-90deg)}
.kb-block-body{font-size:11px;color:var(--text2);line-height:1.6}
.kb-block-body.collapsed{display:none}
.kb-block-body p{margin-bottom:6px}
.kb-block-body ul{list-style:none;padding:0;margin:0}
.kb-block-body li{padding:2px 0;display:flex;gap:6px}
.kb-block-body li::before{content:'›';color:var(--accent);flex-shrink:0}
.kb-block-body b{color:var(--text)}
.kb-block-body code{
  font-family:var(--mono);font-size:10px;
  background:var(--bg3);border:1px solid var(--border2);
  border-radius:3px;padding:0 4px;color:var(--accent3);
}
.kb-tag{
  display:inline-block;background:rgba(79,142,247,.1);
  border:1px solid rgba(79,142,247,.2);border-radius:4px;
  padding:1px 6px;font-size:9px;font-family:var(--mono);
  color:var(--accent);margin:1px;
}
.kb-formula{
  font-family:var(--mono);font-size:10px;
  background:var(--bg3);border-left:2px solid var(--accent2);
  padding:5px 10px;margin:4px 0;border-radius:0 4px 4px 0;
  color:var(--text);line-height:1.6;
}


/* ══════════════════════════════════════════════
   MAIN AREA
══════════════════════════════════════════════ */
.page-header{margin-bottom:24px}
.page-title{font-size:26px;font-weight:700;letter-spacing:-0.5px;line-height:1.2}
.page-title span{background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.page-meta{color:var(--text3);font-size:11.5px;margin-top:6px;font-family:var(--mono)}
.page-meta b{color:var(--text2)}

/* ── Sections ── */
.section{display:none;animation:fadeUp .25s ease}
.section.active{display:block}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

/* ── Glass card ── */
.card{
  background:var(--glass);
  backdrop-filter:blur(16px);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:20px 22px;
  margin-bottom:16px;
  box-shadow:var(--glow);
  position:relative;
  overflow:hidden;
}
.card::before{
  content:'';position:absolute;inset:0;border-radius:var(--radius);pointer-events:none;
  background:linear-gradient(135deg,rgba(79,142,247,0.03) 0%,transparent 60%);
}
.card-sm{padding:14px 18px}
.card h2{font-size:15px;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:8px;color:var(--text)}
.card-grid{display:grid;gap:14px}
.card-grid-2{grid-template-columns:repeat(2,1fr)}
.card-grid-3{grid-template-columns:repeat(3,1fr)}

/* ── Stats strip ── */
.stats-strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-bottom:20px}
.stat-box{
  background:var(--glass2);backdrop-filter:blur(12px);
  border:1px solid var(--border2);border-radius:var(--radius-sm);
  padding:14px 16px;position:relative;overflow:hidden;
  transition:border-color .2s,transform .15s;
}
.stat-box:hover{border-color:var(--border);transform:translateY(-1px)}
.stat-box::after{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--accent-gradient,linear-gradient(90deg,var(--accent),var(--accent2)));
  opacity:0;transition:opacity .2s;
}
.stat-box:hover::after{opacity:1}
.sb-label{font-size:9.5px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:1px;font-family:var(--mono)}
.sb-value{font-size:22px;font-weight:700;margin-top:3px;line-height:1;font-family:var(--mono)}
.sb-value.accent{color:var(--accent)}
.sb-value.green{color:var(--green)}
.sb-value.red{color:var(--red)}
.sb-value.yellow{color:var(--yellow)}
.sb-value.cyan{color:var(--accent3)}
.sb-sub{font-size:10px;color:var(--text3);margin-top:3px;font-family:var(--mono)}

/* ── Comparison panels ── */
.compare-pane{display:none}
.compare-pane.active{display:block}
.compare-split{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.panel-before{border-top:2px solid rgba(239,68,68,0.5)}
.panel-after{border-top:2px solid rgba(34,197,94,0.5)}
.panel-label{font-size:10px;font-family:var(--mono);font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.panel-label.before{color:var(--red)}
.panel-label.after{color:var(--green)}

/* ── Badges / Chips ── */
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:999px;font-size:10px;font-family:var(--mono);font-weight:600;border:1px solid}
.badge-numeric{background:rgba(79,142,247,.12);color:var(--accent);border-color:rgba(79,142,247,.25)}
.badge-categorical{background:rgba(124,58,237,.12);color:#a78bfa;border-color:rgba(124,58,237,.25)}
.badge-datetime{background:rgba(6,182,212,.12);color:var(--accent3);border-color:rgba(6,182,212,.25)}
.badge-boolean{background:rgba(34,197,94,.12);color:var(--green);border-color:rgba(34,197,94,.25)}
.badge-text{background:rgba(236,72,153,.12);color:var(--pink);border-color:rgba(236,72,153,.25)}
.badge-high{background:rgba(239,68,68,.12);color:var(--red);border-color:rgba(239,68,68,.25)}
.badge-medium{background:rgba(245,158,11,.12);color:var(--yellow);border-color:rgba(245,158,11,.25)}
.badge-low{background:rgba(34,197,94,.12);color:var(--green);border-color:rgba(34,197,94,.25)}
.badge-critical{background:rgba(239,68,68,.2);color:#ff6b6b;border-color:rgba(239,68,68,.4)}

/* ── Progress bar ── */
.pbar-wrap{display:flex;align-items:center;gap:8px;min-width:80px}
.pbar-track{flex:1;height:4px;background:var(--bg3);border-radius:2px;overflow:hidden}
.pbar-fill{height:100%;border-radius:2px;transition:width .4s}
.pbar-label{font-size:10px;font-family:var(--mono);color:var(--text3);white-space:nowrap}

/* ── Table ── */
.tbl{width:100%;border-collapse:collapse;font-size:12px}
.tbl th{text-align:left;padding:8px 10px;font-size:10px;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid var(--border2);font-family:var(--mono)}
.tbl td{padding:9px 10px;border-bottom:1px solid rgba(255,255,255,0.03);color:var(--text2);vertical-align:middle}
.tbl tr:last-child td{border-bottom:none}
.tbl tr:hover td{background:rgba(79,142,247,0.03)}
.mono{font-family:var(--mono)!important;font-size:11px!important}

/* ── Recommendations ── */
.rec-card{background:var(--glass2);border:1px solid var(--border2);border-radius:var(--radius-sm);padding:16px;margin-bottom:10px;transition:border-color .2s}
.rec-card:hover{border-color:var(--border)}
.sev-high{border-left:3px solid var(--red)}
.sev-medium{border-left:3px solid var(--yellow)}
.sev-low{border-left:3px solid var(--green)}
.sev-critical{border-left:3px solid #ff4040}
.rec-header{display:flex;gap:12px;align-items:flex-start}
.rec-icon{font-size:20px;flex-shrink:0;margin-top:1px}
.rec-body{flex:1}
.rec-title{font-size:13.5px;font-weight:600;line-height:1.3}
.rec-desc{font-size:11.5px;color:var(--text2);margin-top:3px}
.rec-cols{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}
.col-chip{background:var(--bg3);border:1px solid var(--border2);border-radius:5px;padding:1px 7px;font-size:10px;font-family:var(--mono);color:var(--text2)}
.rec-actions{display:flex;flex-wrap:wrap;gap:7px;margin-top:12px}
.btn{display:inline-flex;align-items:center;gap:5px;padding:6px 13px;border-radius:7px;font-size:11.5px;font-weight:600;cursor:pointer;transition:all .15s;border:1px solid transparent;font-family:var(--sans)}
.btn-primary{background:linear-gradient(135deg,var(--accent),#3b7de9);color:#fff;box-shadow:0 2px 12px rgba(79,142,247,0.25)}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 4px 18px rgba(79,142,247,0.35)}
.btn-ghost{background:transparent;color:var(--text2);border-color:var(--border2)}
.btn-ghost:hover{background:var(--glass2);color:var(--text)}
.btn-danger{color:var(--red);border-color:rgba(239,68,68,.25)}
.btn-danger:hover{background:rgba(239,68,68,.08)}
.btn:disabled{opacity:.35;cursor:not-allowed;transform:none!important}

/* ── Column detail accordion ── */
.col-accordion{margin-bottom:8px}
.col-accordion-header{
  display:flex;align-items:center;gap:10px;padding:11px 16px;
  background:var(--glass2);border:1px solid var(--border2);border-radius:var(--radius-sm);
  cursor:pointer;transition:all .15s;
}
.col-accordion-header:hover{border-color:var(--border);background:rgba(79,142,247,0.05)}
.col-accordion-header.open{border-color:var(--border);border-bottom-left-radius:0;border-bottom-right-radius:0}
.col-accordion-body{
  background:rgba(8,12,24,0.5);border:1px solid var(--border2);
  border-top:none;border-radius:0 0 var(--radius-sm) var(--radius-sm);
  padding:16px;display:none;
}
.col-accordion-body.open{display:block}
.col-acc-name{font-family:var(--mono);font-size:12.5px;font-weight:600;color:var(--text)}
.col-acc-stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px;margin-bottom:12px}
.mini-stat{background:var(--bg3);border:1px solid var(--border2);border-radius:6px;padding:8px 10px}
.mini-stat-label{font-size:9px;color:var(--text3);font-family:var(--mono);text-transform:uppercase;letter-spacing:.8px}
.mini-stat-value{font-size:14px;font-weight:600;margin-top:2px;font-family:var(--mono);color:var(--text)}
.chevron{margin-left:auto;font-size:10px;color:var(--text3);transition:transform .2s}
.col-accordion-header.open .chevron{transform:rotate(180deg)}

/* ── Chart wrap ── */
.chart-wrap{border-radius:10px;overflow:hidden;background:rgba(5,7,15,0.4);margin-top:10px;min-height:280px}
.chart-wrap .js-plotly-plot{width:100%!important}
.chart-wrap .plotly-graph-div{width:100%!important}

/* ── Normality indicator ── */
.norm-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;display:inline-block}
.norm-yes{background:var(--green);box-shadow:0 0 6px var(--green)}
.norm-no{background:var(--red);box-shadow:0 0 6px var(--red)}

/* ── Section header ── */
.sec-header{margin-bottom:20px}
.sec-title{font-size:20px;font-weight:700;letter-spacing:-.3px}
.sec-desc{color:var(--text2);font-size:12.5px;margin-top:4px}

/* ── Tabs ── */
.tabs{display:flex;gap:3px;background:var(--bg3);border:1px solid var(--border2);border-radius:9px;padding:3px;margin-bottom:16px;flex-wrap:wrap}
.tab{padding:7px 14px;border-radius:7px;font-size:12px;font-weight:500;cursor:pointer;color:var(--text2);transition:all .15s;white-space:nowrap}
.tab.active{background:var(--accent);color:#fff;box-shadow:0 2px 10px rgba(79,142,247,0.3)}
.tab-panel{display:none}
.tab-panel.active{display:block}

/* ── Readiness score ring ── */
.rs-ring-wrap{display:flex;align-items:center;justify-content:center;padding:24px 0}
.rs-ring{position:relative;width:140px;height:140px}
.rs-ring svg{transform:rotate(-90deg)}
.rs-ring-label{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.rs-score{font-size:32px;font-weight:700;font-family:var(--mono)}
.rs-grade{font-size:13px;font-weight:600;margin-top:2px;font-family:var(--mono)}
.dim-bar{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.dim-name{font-size:11px;color:var(--text2);width:130px;flex-shrink:0;font-family:var(--mono)}
.dim-track{flex:1;height:6px;background:var(--bg3);border-radius:3px;overflow:hidden}
.dim-fill{height:100%;border-radius:3px;transition:width .6s ease}
.dim-val{font-size:11px;font-family:var(--mono);color:var(--text3);width:32px;text-align:right}

/* ── Toast ── */
.toast{position:fixed;bottom:22px;right:22px;background:var(--glass);backdrop-filter:blur(20px);border:1px solid var(--border);border-radius:10px;padding:11px 16px;font-size:12.5px;color:var(--text);z-index:9999;transform:translateY(60px);opacity:0;transition:all .25s;max-width:300px}
.toast.show{transform:translateY(0);opacity:1}
.toast.ok{border-color:rgba(34,197,94,.3)}
.toast.err{border-color:rgba(239,68,68,.3);color:var(--red)}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

/* ── Responsive ── */
@media(max-width:1200px){.sidebar-right{display:none}}
@media(max-width:900px){.sidebar-left{display:none}.main-area{padding:16px}}
@media(max-width:600px){.stats-strip{grid-template-columns:repeat(2,1fr)}.compare-split{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="layout">

<!-- ═══════════ LEFT SIDEBAR ═══════════ -->
<nav class="sidebar-left">
  <div class="logo-area">
    <div class="logo-name">DataIQ</div>
    <div class="logo-ver">v1.0 · ML Data Intelligence Suite</div>
  </div>
  <div class="dataset-info">
    <div class="ds-name" id="sb-name">—</div>
    <div class="ds-meta" id="sb-meta">—</div>
    <div class="ds-badge" id="sb-badge">● Loading...</div>
  </div>

  <!-- Compare toggle -->
  <div style="padding:10px 14px 0">
    <div class="compare-toggle" id="compare-toggle" style="display:none">
      <div class="ct-btn active" onclick="switchView('before')">Before</div>
      <div class="ct-btn" onclick="switchView('after')">After</div>
      <div class="ct-btn" onclick="switchView('diff')">Diff</div>
    </div>
  </div>

  <!-- Nav -->
  <div class="nav-group">
    <div class="nav-header">Intelligence</div>
    <div class="nav-item active" onclick="nav('overview',this)"><span class="ni">◈</span>Overview</div>
    <div class="nav-item" onclick="nav('readiness',this)"><span class="ni">🎯</span>ML Readiness<span class="nav-badge" id="nb-mrs">—</span></div>
    <div class="nav-item" onclick="nav('leakage',this)"><span class="ni">🔍</span>Leakage<span class="nav-badge" id="nb-leak">—</span></div>
    <div class="nav-item" onclick="nav('interactions',this)"><span class="ni">⟡</span>Interactions</div>
    <div class="nav-item" onclick="nav('temporal',this)"><span class="ni">📈</span>Temporal</div>
  </div>
  <div class="nav-group">
    <div class="nav-header">EDA</div>
    <div class="nav-item" onclick="nav('columns',this)"><span class="ni">⊞</span>Columns<span class="nav-badge" id="nb-cols">—</span></div>
    <div class="nav-item" onclick="nav('missing',this)"><span class="ni">◌</span>Missing Values<span class="nav-badge" id="nb-miss">—</span></div>
    <div class="nav-item" onclick="nav('outliers',this)"><span class="ni">◉</span>Outliers<span class="nav-badge" id="nb-out">—</span></div>
    <div class="nav-item" onclick="nav('skewness',this)"><span class="ni">〜</span>Skewness &amp; Kurtosis</div>
    <div class="nav-item" onclick="nav('correlations',this)"><span class="ni">⊗</span>Correlations</div>
    <div class="nav-item" onclick="nav('univariate',this)"><span class="ni">▦</span>Univariate Analysis</div>
    <div class="nav-item" onclick="nav('bivariate',this)"><span class="ni">⋈</span>Bivariate Analysis</div>
    <div class="nav-item" onclick="nav('normality',this)"><span class="ni">∿</span>Normality Tests</div>
    <div class="nav-item" onclick="nav('pca',this)"><span class="ni">⟳</span>PCA Analysis</div>
    <div class="nav-item" onclick="nav('variance',this)"><span class="ni">σ</span>Variance</div>
  </div>
  <div class="nav-group">
    <div class="nav-header">Target &amp; Features</div>
    <div class="nav-item" onclick="nav('imbalance',this)"><span class="ni">⊜</span>Class Imbalance<span class="nav-badge" id="nb-imb">—</span></div>
    <div class="nav-item" onclick="nav('feature_target',this)"><span class="ni">⟶</span>Feature-Target</div>
    <div class="nav-item" onclick="nav('chisquare',this)"><span class="ni">χ</span>Chi-Square</div>
  </div>
  <div class="nav-group">
    <div class="nav-header">Actions</div>
    <div class="nav-item" onclick="nav('recommendations',this)"><span class="ni">✦</span>Recommendations<span class="nav-badge" id="nb-recs">—</span></div>
    <div class="nav-item" onclick="nav('history',this)"><span class="ni">↺</span>History<span class="nav-badge" id="nb-hist">0</span></div>
  </div>

  <div style="padding:12px 14px;border-top:1px solid var(--border);margin-top:auto;font-size:10px;font-family:var(--mono);color:var(--text3)">
    Generated __GENERATED_AT__
  </div>
</nav>

<!-- ═══════════ MAIN AREA ═══════════ -->
<main class="main-area">
  <div class="page-header">
    <div class="page-title">__DATASET_NAME__ <span>— DataIQ Report</span></div>
    <div class="page-meta" id="page-meta">Loading...</div>
  </div>

  <!-- ── Overview ── -->
  <div class="section active" id="sec-overview">
    <div class="sec-header"><div class="sec-title">Dataset Overview</div><div class="sec-desc">Shape, quality metrics, memory, column type distribution, and auto-detected problem type.</div></div>
    <div id="ov-strip"></div>
    <div id="ov-problem-type"></div>
    <div id="ov-types-card"></div>
  </div>

  <!-- ── ML Readiness Score (NEW) ── -->
  <div class="section" id="sec-readiness">
    <div class="sec-header"><div class="sec-title">ML Readiness Score</div><div class="sec-desc">0–100 score per column and dataset-wide across 7 dimensions. Grade A–F.</div></div>
    <div id="readiness-content"></div>
  </div>

  <!-- ── Leakage Detective (NEW) ── -->
  <div class="section" id="sec-leakage">
    <div class="sec-header"><div class="sec-title">Leakage Detective</div><div class="sec-desc">Auto-detects 6 categories of data leakage that silently destroy model performance.</div></div>
    <div id="leakage-content"></div>
  </div>

  <!-- ── Feature Interactions (NEW) ── -->
  <div class="section" id="sec-interactions">
    <div class="sec-header"><div class="sec-title">Feature Interactions</div><div class="sec-desc">Mutual Information, Cramér's V, and Correlation Ratio between all feature pairs.</div></div>
    <div id="interactions-content"></div>
  </div>

  <!-- ── Temporal Awareness (NEW) ── -->
  <div class="section" id="sec-temporal">
    <div class="sec-header"><div class="sec-title">Temporal Awareness</div><div class="sec-desc">Rolling trends and concept drift windows over datetime columns.</div></div>
    <div id="temporal-content"></div>
  </div>

  <!-- ── Columns ── -->
  <div class="section" id="sec-columns">
    <div class="sec-header"><div class="sec-title">Column Analysis</div><div class="sec-desc">Click any column to expand its full statistics.</div></div>
    <div id="col-search-wrap" style="margin-bottom:12px">
      <input id="col-search" placeholder="🔍  Filter columns..." oninput="filterCols()" style="width:100%;padding:9px 14px;background:var(--glass2);border:1px solid var(--border2);border-radius:8px;color:var(--text);font-family:var(--sans);font-size:13px;outline:none"/>
    </div>
    <div id="columns-content"></div>
  </div>

  <!-- ── Missing ── -->
  <div class="section" id="sec-missing">
    <div class="sec-header"><div class="sec-title">Missing Values</div><div class="sec-desc">Severity and distribution of missing data. MCAR/MAR/MNAR classification hints.</div></div>
    <div id="missing-content"></div>
  </div>

  <!-- ── Outliers ── -->
  <div class="section" id="sec-outliers">
    <div class="sec-header"><div class="sec-title">Outlier Analysis</div><div class="sec-desc">IQR &amp; Z-score detection with box plots.</div></div>
    <div id="outliers-content"></div>
  </div>

  <!-- ── Skewness & Kurtosis ── -->
  <div class="section" id="sec-skewness">
    <div class="sec-header"><div class="sec-title">Skewness &amp; Kurtosis</div><div class="sec-desc">Distribution shape analysis. High skew (|s|&gt;1) may need transformations.</div></div>
    <div id="skewness-content"></div>
  </div>

  <!-- ── Correlations ── -->
  <div class="section" id="sec-correlations">
    <div class="sec-header"><div class="sec-title">Correlation Analysis</div><div class="sec-desc">Pearson correlation heatmap &amp; covariance matrix. Strong pairs (|r|&gt;0.5) listed.</div></div>
    <div id="correlations-content"></div>
  </div>

  <!-- ── Univariate ── -->
  <div class="section" id="sec-univariate">
    <div class="sec-header"><div class="sec-title">Univariate Analysis</div><div class="sec-desc">Per-column histograms, box plots, and detailed statistics.</div></div>
    <div id="univariate-content"></div>
  </div>

  <!-- ── Bivariate ── -->
  <div class="section" id="sec-bivariate">
    <div class="sec-header"><div class="sec-title">Bivariate Analysis</div><div class="sec-desc">Scatterplots, category distributions, feature-target scatter with trend lines.</div></div>
    <div id="bivariate-content"></div>
  </div>

  <!-- ── Normality ── -->
  <div class="section" id="sec-normality">
    <div class="sec-header"><div class="sec-title">Normality Tests</div><div class="sec-desc">Shapiro-Wilk, D'Agostino, Kolmogorov-Smirnov tests + Q-Q plots.</div></div>
    <div id="normality-content"></div>
  </div>

  <!-- ── PCA ── -->
  <div class="section" id="sec-pca">
    <div class="sec-header"><div class="sec-title">PCA — Dimensionality Reduction</div><div class="sec-desc">Explained variance per principal component. How many PCs to retain 95% variance?</div></div>
    <div id="pca-content"></div>
  </div>

  <!-- ── Variance ── -->
  <div class="section" id="sec-variance">
    <div class="sec-header"><div class="sec-title">Variance Analysis</div><div class="sec-desc">Feature variance, coefficient of variation, low-variance flags.</div></div>
    <div id="variance-content"></div>
  </div>

  <!-- ── Class Imbalance ── -->
  <div class="section" id="sec-imbalance">
    <div class="sec-header"><div class="sec-title">Class Imbalance</div><div class="sec-desc">Target variable distribution. Imbalance ratio &gt;2 may require SMOTE/resampling.</div></div>
    <div id="imbalance-content"></div>
  </div>

  <!-- ── Feature-Target ── -->
  <div class="section" id="sec-feature_target">
    <div class="sec-header"><div class="sec-title">Feature–Target Relationships</div><div class="sec-desc">Correlation with target, scatter plots with OLS trend lines.</div></div>
    <div id="ft-content"></div>
  </div>

  <!-- ── Chi-Square ── -->
  <div class="section" id="sec-chisquare">
    <div class="sec-header"><div class="sec-title">Chi-Square Tests</div><div class="sec-desc">Statistical independence between categorical feature pairs (p &lt; 0.05 = dependent).</div></div>
    <div id="chi-content"></div>
  </div>

  <!-- ── Recommendations ── -->
  <div class="section" id="sec-recommendations">
    <div class="sec-header"><div class="sec-title">Recommendations</div><div class="sec-desc">Actionable data quality fixes. Click an action to apply and track it.</div></div>
    <div id="rec-content"></div>
  </div>

  <!-- ── History ── -->
  <div class="section" id="sec-history">
    <div class="sec-header"><div class="sec-title">Transformation History</div><div class="sec-desc">Log of all applied transformations.</div></div>
    <div id="history-content"></div>
  </div>
</main>

<!-- ═══════════ RIGHT SIDEBAR — KNOWLEDGE BASE ═══════════ -->
<aside class="sidebar-right" id="kb-sidebar">
  <div class="kb-resize-handle" id="kb-drag"></div>
  <div class="kb-header">
    <span style="font-size:16px">📚</span>
    <div class="kb-title-main">ML Knowledge Base</div>
  </div>
  <div class="kb-scroll" id="kb-scroll">
    <div id="kb-content">
      <!-- populated by JS based on active nav section -->
    </div>
  </div>
</aside>

</div><!-- /layout -->

<div class="toast" id="toast"></div>

<script>
// ═══════════════════════════════════════════════
//  DATA
// ═══════════════════════════════════════════════
const DATA = __ANALYSIS_DATA__;
const HAS_AFTER = DATA.after !== undefined;
let currentView = 'before';
const appHistory = [];
let appliedRecs = {};

// ═══════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  if (HAS_AFTER) document.getElementById('compare-toggle').style.display = 'flex';
  renderAll(DATA.before || DATA);
  updateNavBadges(DATA.before || DATA);
  // Force Plotly to resize all charts after layout is painted
  setTimeout(() => {
    document.querySelectorAll('.js-plotly-plot').forEach(el => {
      try { Plotly.relayout(el, {}); } catch(e) {}
    });
  }, 300);
});

function getD() {
  if (!HAS_AFTER) return DATA;
  return currentView === 'after' ? DATA.after : DATA.before;
}

function switchView(v) {
  currentView = v;
  document.querySelectorAll('.ct-btn').forEach((b,i) => {
    b.classList.toggle('active', ['before','after','diff'][i] === v);
  });
  renderAll(getD());
  toast(v === 'diff' ? 'Showing diff view' : `Switched to ${v} dataset`, 'ok');
}

// ═══════════════════════════════════════════════
//  NAVIGATION
// ═══════════════════════════════════════════════
function nav(id, el) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('sec-' + id).classList.add('active');
  el.classList.add('active');
  // Sync KB with active section
  renderKB(SECTION_TO_KB[id] || 'overview');
  // Resize Plotly charts in newly visible section
  setTimeout(() => {
    document.querySelectorAll('#sec-' + id + ' .js-plotly-plot').forEach(el => {
      try { Plotly.relayout(el, {}); } catch(e) {}
    });
  }, 50);
}

// ═══════════════════════════════════════════════
//  RENDER ALL
// ═══════════════════════════════════════════════
function renderAll(d) {
  const ov = d.overview;
  document.getElementById('sb-name').textContent = '__DATASET_NAME__';
  document.getElementById('sb-meta').textContent = `${ov.n_rows.toLocaleString()} rows × ${ov.n_cols} cols`;
  document.getElementById('sb-badge').innerHTML = `● ${d.label || 'dataset'}`;
  document.getElementById('page-meta').innerHTML =
    `Dataset: <b>__DATASET_NAME__</b> &nbsp;·&nbsp; <b>${ov.n_rows.toLocaleString()}</b> rows × <b>${ov.n_cols}</b> cols &nbsp;·&nbsp; ${ov.memory_human} &nbsp;·&nbsp; Generated __GENERATED_AT__`;

  renderOverview(d);
  renderReadiness(d);
  renderLeakage(d);
  renderInteractions(d);
  renderTemporal(d);
  renderColumns(d);
  renderMissing(d);
  renderOutliers(d);
  renderSkewness(d);
  renderCorrelations(d);
  renderUnivariate(d);
  renderBivariate(d);
  renderNormality(d);
  renderPCA(d);
  renderVariance(d);
  renderImbalance(d);
  renderFeatureTarget(d);
  renderChiSquare(d);
  renderRecommendations(d);
  renderHistory();
}

function updateNavBadges(d) {
  const ov = d.overview;
  document.getElementById('nb-cols').textContent = ov.n_cols;
  const nm = d.missing.n_cols_missing;
  const nbm = document.getElementById('nb-miss');
  nbm.textContent = nm; nbm.className = 'nav-badge' + (nm>3?' bad':nm>0?' warn':' good');
  const no = (d.outliers.items||[]).length;
  const nbo = document.getElementById('nb-out');
  nbo.textContent = no; nbo.className = 'nav-badge' + (no>3?' bad':no>0?' warn':' good');
  document.getElementById('nb-recs').textContent = (d.recommendations||[]).length;
  if (d.class_imbalance) {
    const ci = d.class_imbalance;
    const nb = document.getElementById('nb-imb');
    nb.textContent = ci.is_imbalanced ? '⚠' : '✓';
    nb.className = 'nav-badge ' + (ci.is_imbalanced ? 'bad' : 'good');
  }
  // Readiness badge
  const rs = d.readiness_score;
  if (rs) {
    const nb = document.getElementById('nb-mrs');
    nb.textContent = rs.dataset_grade;
    nb.className = 'nav-badge ' + (['A','B'].includes(rs.dataset_grade)?'good':rs.dataset_grade==='C'?'warn':'bad');
  }
  // Leakage badge
  const lk = d.leakage;
  if (lk) {
    const nb = document.getElementById('nb-leak');
    nb.textContent = lk.n_total;
    nb.className = 'nav-badge ' + (lk.n_critical>0?'bad':lk.n_high>0?'warn':'good');
  }
}

// ═══════════════════════════════════════════════
//  OVERVIEW
// ═══════════════════════════════════════════════
function renderOverview(d) {
  const ov = d.overview;
  const mp = ov.missing_pct, dp = ov.duplicate_pct;
  const stats = [
    {l:'Rows', v:ov.n_rows.toLocaleString(), cls:'accent', s:'observations'},
    {l:'Columns', v:ov.n_cols, cls:'accent', s:'features'},
    {l:'Missing', v:ov.missing_cells.toLocaleString(), cls:mp>10?'red':mp>2?'yellow':'green', s:mp+'% of data'},
    {l:'Duplicates', v:ov.duplicate_rows.toLocaleString(), cls:dp>5?'red':dp>0?'yellow':'green', s:dp+'% of rows'},
    {l:'Memory', v:ov.memory_human, cls:'cyan', s:'in-memory'},
    {l:'Numeric', v:(ov.kinds_count.numeric||0), cls:'', s:'numeric cols'},
    {l:'Categorical', v:(ov.kinds_count.categorical||0), cls:'', s:'cat. cols'},
    {l:'Datetime', v:(ov.kinds_count.datetime||0), cls:'', s:'date cols'},
    {l:'Boolean', v:(ov.kinds_count.boolean||0), cls:'', s:'bool cols'},
    {l:'Text', v:(ov.kinds_count.text||0), cls:'', s:'text cols'},
  ];
  document.getElementById('ov-strip').innerHTML =
    `<div class="stats-strip">${stats.map(s=>`
      <div class="stat-box">
        <div class="sb-label">${s.l}</div>
        <div class="sb-value ${s.cls}">${s.v}</div>
        <div class="sb-sub">${s.s}</div>
      </div>`).join('')}</div>`;

  // Problem type card
  const pt = d.problem_type;
  if (pt && pt.type !== 'unknown') {
    const typeLabel = pt.type.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
    const confColor = pt.confidence==='high'?'var(--green)':pt.confidence==='medium'?'var(--yellow)':'var(--red)';
    document.getElementById('ov-problem-type').innerHTML = `
      <div class="card card-sm" style="margin-bottom:16px;border-left:3px solid var(--accent2)">
        <div style="display:flex;align-items:center;gap:14px">
          <div style="font-size:28px">🤖</div>
          <div>
            <div style="font-size:11px;color:var(--text3);font-family:var(--mono);text-transform:uppercase;letter-spacing:1px">Auto-detected Problem Type</div>
            <div style="font-size:17px;font-weight:700;color:var(--text);margin-top:3px">${typeLabel}</div>
            <div style="font-size:11px;color:var(--text2);margin-top:2px">${pt.reason}</div>
          </div>
          <div style="margin-left:auto;text-align:right">
            <div style="font-size:10px;color:var(--text3);font-family:var(--mono)">Confidence</div>
            <div style="font-size:14px;font-weight:700;color:${confColor};font-family:var(--mono)">${pt.confidence.toUpperCase()}</div>
            ${pt.n_classes?`<div style="font-size:10px;color:var(--text3);font-family:var(--mono)">${pt.n_classes} classes</div>`:''}
          </div>
        </div>
      </div>`;
  }

  const kc = ov.kinds_count;
  const kinds = [{k:'numeric',label:'Numeric'},{k:'categorical',label:'Categorical'},
    {k:'datetime',label:'Datetime'},{k:'boolean',label:'Boolean'},{k:'text',label:'Text'}]
    .filter(r=>(kc[r.k]||0)>0);
  const cols_by_kind = {};
  for (const [c,k] of Object.entries(ov.col_kinds)) { (cols_by_kind[k]||(cols_by_kind[k]=[])).push(c); }

  document.getElementById('ov-types-card').innerHTML = `<div class="card">
    <h2>Column Type Breakdown</h2>
    <table class="tbl">
      <thead><tr><th>Type</th><th>Count</th><th>% of Cols</th><th>Columns</th></tr></thead>
      <tbody>${kinds.map(r=>`<tr>
        <td><span class="badge badge-${r.k}">${r.label}</span></td>
        <td class="mono">${kc[r.k]}</td>
        <td><div class="pbar-wrap"><div class="pbar-track"><div class="pbar-fill" style="width:${kc[r.k]/ov.n_cols*100}%;background:var(--accent)"></div></div><div class="pbar-label">${Math.round(kc[r.k]/ov.n_cols*100)}%</div></div></td>
        <td style="color:var(--text2);font-size:11px">${(cols_by_kind[r.k]||[]).slice(0,6).join(', ')}${(cols_by_kind[r.k]||[]).length>6?` +${(cols_by_kind[r.k]||[]).length-6}`:''}</td>
      </tr>`).join('')}</tbody>
    </table>
  </div>`;
}

// ═══════════════════════════════════════════════
//  ML READINESS SCORE  (NEW)
// ═══════════════════════════════════════════════
function renderReadiness(d) {
  const rs = d.readiness_score;
  const el = document.getElementById('readiness-content');
  if (!rs) { el.innerHTML='<div class="card">Readiness score not available.</div>'; return; }

  const score = rs.dataset_score;
  const grade = rs.dataset_grade;
  const color = rs.dataset_color;
  const label = rs.dataset_label;
  const r = 54, cx = 70, cy = 70;
  const circ = 2 * Math.PI * r;
  const fill = (score / 100) * circ;

  const dimColors = {completeness:'#22c55e',leakage_risk:'#ef4444',outlier_severity:'#f97316',
    distribution:'#4f8ef7',cardinality:'#7c3aed',type_fitness:'#06b6d4',consistency:'#f59e0b'};

  const gradeDistHtml = ['A','B','C','D','F'].map(g=>`
    <div class="stat-box">
      <div class="sb-label">Grade ${g}</div>
      <div class="sb-value" style="color:${['A','B'].includes(g)?'var(--green)':g==='C'?'var(--yellow)':g==='D'?'var(--red)':'#ff4040'}">${rs.grade_dist[g]||0}</div>
      <div class="sb-sub">columns</div>
    </div>`).join('');

  const dimsHtml = Object.entries(rs.dimension_avgs||{}).map(([dim,val])=>{
    const color2 = dimColors[dim]||'var(--accent)';
    const dimLabel = dim.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
    return `<div class="dim-bar">
      <div class="dim-name">${dimLabel}</div>
      <div class="dim-track"><div class="dim-fill" style="width:${val}%;background:${color2}"></div></div>
      <div class="dim-val">${val}</div>
    </div>`;
  }).join('');

  const issuesHtml = (rs.top_issues||[]).map(c=>`
    <tr>
      <td class="mono">${c.column}</td>
      <td><span class="badge" style="background:${c.color}22;color:${c.color};border-color:${c.color}44">${c.grade}</span></td>
      <td class="mono">${c.total}</td>
      <td style="font-size:11px;color:var(--text2)">${c.kind}</td>
    </tr>`).join('');

  const colsHtml = (rs.columns||[]).map(c=>`
    <tr>
      <td class="mono">${c.column}${c.is_target?' <span class="badge badge-high" style="font-size:9px">TARGET</span>':''}</td>
      <td><span class="badge" style="background:${c.color}22;color:${c.color};border-color:${c.color}44">${c.grade}</span></td>
      <td class="mono">${c.total}</td>
      <td style="font-family:var(--mono);font-size:10px">
        ${Object.entries(c.dimensions||{}).map(([dim,val])=>{
          const bg = val>=75?'rgba(34,197,94,.15)':val>=50?'rgba(245,158,11,.15)':'rgba(239,68,68,.15)';
          const tc = val>=75?'var(--green)':val>=50?'var(--yellow)':'var(--red)';
          return `<span style="display:inline-block;background:${bg};color:${tc};border-radius:4px;padding:1px 5px;margin:1px;font-size:9px">${dim.split('_')[0]}:${Math.round(val)}</span>`;
        }).join('')}
      </td>
    </tr>`).join('');

  el.innerHTML = `
    <div class="stats-strip" style="margin-bottom:16px">
      <div class="card card-sm" style="grid-column:span 2;display:flex;align-items:center;gap:24px;padding:20px 24px">
        <div class="rs-ring">
          <svg width="140" height="140" viewBox="0 0 140 140">
            <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--bg3)" stroke-width="10"/>
            <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${color}" stroke-width="10"
              stroke-dasharray="${fill} ${circ}" stroke-linecap="round"/>
          </svg>
          <div class="rs-ring-label">
            <div class="rs-score" style="color:${color}">${score}</div>
            <div class="rs-grade" style="color:${color}">${grade}</div>
          </div>
        </div>
        <div style="flex:1">
          <div style="font-size:18px;font-weight:700;color:${color};margin-bottom:4px">${label}</div>
          <div style="font-size:12px;color:var(--text2);margin-bottom:16px">Dataset ML Readiness Score</div>
          ${dimsHtml}
        </div>
      </div>
      ${gradeDistHtml}
    </div>
    ${issuesHtml?`<div class="card"><h2>⚠ Lowest-Scoring Columns</h2>
      <table class="tbl"><thead><tr><th>Column</th><th>Grade</th><th>Score</th><th>Type</th></tr></thead>
      <tbody>${issuesHtml}</tbody></table></div>`:''}
    <div class="card"><h2>All Columns</h2>
      <table class="tbl"><thead><tr><th>Column</th><th>Grade</th><th>Score</th><th>Dimension Scores</th></tr></thead>
      <tbody>${colsHtml}</tbody></table></div>`;
}

// ═══════════════════════════════════════════════
//  LEAKAGE DETECTIVE  (NEW)
// ═══════════════════════════════════════════════
function renderLeakage(d) {
  const lk = d.leakage;
  const el = document.getElementById('leakage-content');
  if (!lk) { el.innerHTML='<div class="card">Leakage analysis not available.</div>'; return; }

  const riskColors = {CRITICAL:'#ef4444',HIGH:'#f97316',MODERATE:'#f59e0b',CLEAN:'#22c55e'};
  const riskColor  = riskColors[lk.risk_level]||'var(--text2)';

  const findingsHtml = (lk.findings||[]).map(f=>{
    const sevColor = f.severity==='critical'?'var(--red)':f.severity==='high'?'var(--red)':f.severity==='medium'?'var(--yellow)':'var(--green)';
    return `<div class="rec-card sev-${f.severity==='critical'?'high':f.severity}" style="margin-bottom:10px">
      <div class="rec-header">
        <div class="rec-icon">${f.icon}</div>
        <div class="rec-body">
          <div class="rec-title">
            <span class="mono" style="color:var(--accent)">${f.column}</span>
            &nbsp;<span class="badge" style="background:${sevColor}22;color:${sevColor};border-color:${sevColor}44">${f.severity.toUpperCase()}</span>
          </div>
          <div style="font-size:11px;color:var(--text3);font-family:var(--mono);margin-top:2px">${f.category}</div>
          <div class="rec-desc" style="margin-top:6px">${f.evidence}</div>
          <div style="margin-top:8px;padding:8px 10px;background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.15);border-radius:6px;font-size:11px;color:var(--yellow)">
            💡 ${f.fix}
          </div>
        </div>
      </div>
    </div>`;
  }).join('');

  el.innerHTML = `
    <div class="card card-sm" style="margin-bottom:16px;border-left:3px solid ${riskColor}">
      <div style="display:flex;align-items:center;gap:16px">
        <div style="font-size:32px">${lk.risk_level==='CLEAN'?'✅':'⚠️'}</div>
        <div style="flex:1">
          <div style="font-size:17px;font-weight:700;color:${riskColor}">${lk.risk_level}</div>
          <div style="font-size:12px;color:var(--text2);margin-top:2px">${lk.verdict}</div>
        </div>
        <div style="display:flex;gap:12px;font-family:var(--mono);font-size:11px">
          <div style="text-align:center"><div style="color:var(--red);font-size:20px;font-weight:700">${lk.n_critical}</div><div style="color:var(--text3)">Critical</div></div>
          <div style="text-align:center"><div style="color:var(--red);font-size:20px;font-weight:700">${lk.n_high}</div><div style="color:var(--text3)">High</div></div>
          <div style="text-align:center"><div style="color:var(--yellow);font-size:20px;font-weight:700">${lk.n_medium}</div><div style="color:var(--text3)">Medium</div></div>
          <div style="text-align:center"><div style="color:var(--green);font-size:20px;font-weight:700">${lk.n_low}</div><div style="color:var(--text3)">Low</div></div>
        </div>
      </div>
    </div>
    ${findingsHtml||`<div class="card" style="text-align:center;padding:40px"><div style="font-size:48px;margin-bottom:12px">✅</div><div style="font-size:18px;font-weight:600;color:var(--green)">No leakage detected!</div></div>`}
    ${lk.safe_cols&&lk.safe_cols.length?`<div class="card card-sm"><div style="font-size:12px;color:var(--text2);margin-bottom:8px">✓ Safe columns (no leakage found):</div>
      <div style="display:flex;flex-wrap:wrap;gap:5px">${lk.safe_cols.map(c=>`<span class="col-chip">${c}</span>`).join('')}</div></div>`:''}`;
}

// ═══════════════════════════════════════════════
//  FEATURE INTERACTIONS  (NEW)
// ═══════════════════════════════════════════════
function renderInteractions(d) {
  const fi = d.feature_interactions;
  const el = document.getElementById('interactions-content');
  if (!fi || fi.available===false) {
    el.innerHTML=`<div class="card" style="text-align:center;padding:40px"><div style="color:var(--text2)">${fi?.summary||'Feature interactions not available.'}</div></div>`;
    return;
  }

  const methodBadge = m => {
    const colors = {'Mutual Information':'var(--accent)','Cramér\'s V':'#a78bfa','Correlation Ratio η²':'var(--accent3)'};
    const c = colors[m]||'var(--text2)';
    return `<span style="font-size:10px;font-family:var(--mono);color:${c}">${m}</span>`;
  };

  const pairsHtml = (fi.top_pairs||[]).map(p=>{
    const w = Math.round(p.score*100);
    const c = p.strength==='Strong'?'var(--red)':p.strength==='Moderate'?'var(--yellow)':'var(--green)';
    return `<tr>
      <td class="mono">${p.col_a}</td><td class="mono">${p.col_b}</td>
      <td><div class="pbar-wrap"><div class="pbar-track"><div class="pbar-fill" style="width:${w}%;background:${c}"></div></div><div class="pbar-label">${p.score}</div></div></td>
      <td><span class="badge badge-${p.strength==='Strong'?'high':p.strength==='Moderate'?'medium':'low'}">${p.strength}</span></td>
      <td>${methodBadge(p.method)}</td>
    </tr>`;
  }).join('');

  const tgtHtml = (fi.target_scores||[]).slice(0,10).map(t=>{
    const w = Math.round(t.score*100);
    const c = t.strength==='Strong'?'var(--red)':t.strength==='Moderate'?'var(--yellow)':'var(--green)';
    return `<tr>
      <td class="mono">${t.column}</td>
      <td><div class="pbar-wrap" style="min-width:120px"><div class="pbar-track"><div class="pbar-fill" style="width:${w}%;background:${c}"></div></div><div class="pbar-label">${t.score}</div></div></td>
      <td><span class="badge badge-${t.strength==='Strong'?'high':t.strength==='Moderate'?'medium':'low'}">${t.strength}</span></td>
      <td>${methodBadge(t.method)}</td>
    </tr>`;
  }).join('');

  const redHtml = (fi.redundancy_groups||[]).map((g,i)=>`
    <div style="margin-bottom:8px">
      <div style="font-size:11px;color:var(--text3);font-family:var(--mono);margin-bottom:4px">Group ${i+1}</div>
      <div style="display:flex;flex-wrap:wrap;gap:5px">${g.map(c=>`<span class="col-chip" style="border-color:rgba(239,68,68,.3);color:var(--red)">${c}</span>`).join('')}</div>
    </div>`).join('');

  el.innerHTML = `
    <div class="stats-strip">
      <div class="stat-box"><div class="sb-label">Pairs Analyzed</div><div class="sb-value accent">${(fi.pairs||[]).length}</div></div>
      <div class="stat-box"><div class="sb-label">Strong</div><div class="sb-value red">${fi.n_strong}</div><div class="sb-sub">score ≥ 0.70</div></div>
      <div class="stat-box"><div class="sb-label">Moderate</div><div class="sb-value yellow">${fi.n_moderate}</div><div class="sb-sub">score ≥ 0.40</div></div>
      <div class="stat-box"><div class="sb-label">Redundancy Groups</div><div class="sb-value ${(fi.redundancy_groups||[]).length?'red':'green'}">${(fi.redundancy_groups||[]).length}</div></div>
    </div>
    ${fi.summary?`<div class="card card-sm" style="margin-bottom:16px;border-left:3px solid var(--accent2)"><div style="font-size:13px;color:var(--text2)">${fi.summary}</div></div>`:''}
    ${pairsHtml?`<div class="card"><h2>Top Feature Pairs</h2>
      <table class="tbl"><thead><tr><th>Feature A</th><th>Feature B</th><th>Score</th><th>Strength</th><th>Method</th></tr></thead>
      <tbody>${pairsHtml}</tbody></table></div>`:''}
    ${tgtHtml?`<div class="card"><h2>Feature vs Target</h2>
      <table class="tbl"><thead><tr><th>Feature</th><th>Score</th><th>Strength</th><th>Method</th></tr></thead>
      <tbody>${tgtHtml}</tbody></table></div>`:''}
    ${redHtml?`<div class="card"><h2>⚠ Redundancy Groups</h2>
      <div style="font-size:12px;color:var(--text2);margin-bottom:12px">Features in the same group are strongly interacting — consider keeping only one or using PCA.</div>
      ${redHtml}</div>`:''}`;
}

// ═══════════════════════════════════════════════
//  TEMPORAL AWARENESS  (NEW)
// ═══════════════════════════════════════════════
function renderTemporal(d) {
  const ta = d.temporal_awareness;
  const el = document.getElementById('temporal-content');
  if (!ta || !ta.available) {
    el.innerHTML='<div class="card" style="text-align:center;padding:40px"><div style="color:var(--text2)">No datetime columns found — temporal analysis not available.</div></div>';
    return;
  }

  el.innerHTML = (ta.items||[]).map(item=>{
    const driftRows = (item.drift_findings||[]).map(f=>`<tr>
      <td class="mono">${f.feature}</td>
      <td class="mono">${f.q1_mean}</td>
      <td class="mono">${f.q4_mean}</td>
      <td><div class="pbar-wrap"><div class="pbar-track"><div class="pbar-fill" style="width:${Math.min(f.drift_pct,100)}%;background:${f.has_drift?'var(--red)':'var(--green)'}"></div></div><div class="pbar-label">${f.drift_pct}%</div></div></td>
      <td>${f.has_drift?'<span class="badge badge-high">⚠ Drift</span>':'<span class="badge badge-low">Stable</span>'}</td>
    </tr>`).join('');

    return `<div class="card" style="margin-bottom:16px">
      <h2>📅 ${item.datetime_col}
        <span style="font-size:11px;font-family:var(--mono);color:var(--text3);font-weight:400;margin-left:8px">${item.date_range}</span>
        ${item.has_concept_drift?'<span class="badge badge-high" style="margin-left:8px">⚠ Concept Drift</span>':'<span class="badge badge-low" style="margin-left:8px">✓ Stable</span>'}
      </h2>
      ${item.chart?`<div class="chart-wrap">${item.chart}</div>`:''}
      ${driftRows?`<div style="margin-top:16px"><h2 style="margin-bottom:10px">Mean Shift Q1 → Q4</h2>
        <table class="tbl"><thead><tr><th>Feature</th><th>Q1 Mean</th><th>Q4 Mean</th><th>Drift %</th><th>Status</th></tr></thead>
        <tbody>${driftRows}</tbody></table></div>`:''}
    </div>`;
  }).join('');
}

// ═══════════════════════════════════════════════
//  COLUMNS
// ═══════════════════════════════════════════════
function renderColumns(d) {
  const meta = d.column_meta || [];
  const kindBadge = k => `<span class="badge badge-${k}">${k}</span>`;
  const allHtml = meta.map((c, idx) => {
    const numData = (d.univariate_numeric||[]).find(u=>u.column===c.name);
    const catData = (d.univariate_categorical||[]).find(u=>u.column===c.name);
    const mp = c.missing_pct;
    const mpColor = mp>40?'var(--red)':mp>10?'var(--yellow)':'var(--green)';
    let statsHtml = '';
    if (numData) {
      statsHtml = `<div class="col-acc-stats">
        ${[['Mean',numData.mean],['Median',numData.median],['Std',numData.std],
           ['Min',numData.min],['Max',numData.max],['Skew',numData.skewness],
           ['Kurt',numData.kurtosis],['IQR',numData.iqr]].map(([l,v])=>
          `<div class="mini-stat"><div class="mini-stat-label">${l}</div><div class="mini-stat-value">${v}</div></div>`
        ).join('')}
      </div>
      ${numData.chart ? `<div class="chart-wrap">${numData.chart}</div>` : ''}`;
    } else if (catData) {
      statsHtml = `<div class="col-acc-stats">
        ${[['Unique',catData.unique],['Top',catData.top_value],['Top%',catData.top_pct+'%'],['Entropy',catData.entropy]].map(([l,v])=>
          `<div class="mini-stat"><div class="mini-stat-label">${l}</div><div class="mini-stat-value" style="font-size:12px">${v}</div></div>`
        ).join('')}
      </div>
      ${catData.chart ? `<div class="chart-wrap">${catData.chart}</div>` : ''}`;
    }
    return `<div class="col-accordion" id="col-acc-${idx}">
      <div class="col-accordion-header" onclick="toggleAcc(${idx})">
        <span class="col-acc-name">${c.name}</span>
        ${kindBadge(c.kind)}
        ${c.is_target?'<span class="badge badge-high">TARGET</span>':''}
        <span style="margin-left:8px;font-family:var(--mono);font-size:10px;color:var(--text3)">${c.dtype}</span>
        <div style="margin-left:auto;display:flex;align-items:center;gap:12px">
          <div class="pbar-wrap" style="width:90px">
            <div class="pbar-track"><div class="pbar-fill" style="width:${mp}%;background:${mpColor}"></div></div>
            <div class="pbar-label" style="color:${mpColor}">${mp}%</div>
          </div>
          <span style="font-family:var(--mono);font-size:10px;color:var(--text3)">${c.unique} uniq</span>
          <span class="chevron">▼</span>
        </div>
      </div>
      <div class="col-accordion-body" id="col-body-${idx}">${statsHtml||'<div style="color:var(--text3);font-size:12px">No detailed stats available.</div>'}</div>
    </div>`;
  });
  document.getElementById('columns-content').innerHTML = allHtml.join('');
}

function toggleAcc(idx) {
  const h = document.querySelector(`#col-acc-${idx} .col-accordion-header`);
  const b = document.getElementById(`col-body-${idx}`);
  h.classList.toggle('open'); b.classList.toggle('open');
}

function filterCols() {
  const q = document.getElementById('col-search').value.toLowerCase();
  document.querySelectorAll('.col-accordion').forEach(el => {
    el.style.display = el.querySelector('.col-acc-name').textContent.toLowerCase().includes(q) ? '' : 'none';
  });
}

// ═══════════════════════════════════════════════
//  MISSING
// ═══════════════════════════════════════════════
function renderMissing(d) {
  const m = d.missing;
  const el = document.getElementById('missing-content');
  document.getElementById('nb-miss').textContent = m.n_cols_missing;
  if (!m.has_missing) {
    el.innerHTML=`<div class="card" style="text-align:center;padding:48px"><div style="font-size:48px;margin-bottom:12px">✅</div><div style="font-size:18px;font-weight:600;color:var(--green)">No missing values!</div><div style="color:var(--text2);margin-top:4px">Your dataset is complete.</div></div>`; return;
  }
  el.innerHTML=`
    ${m.chart?`<div class="card"><h2>Missing Values Distribution</h2><div class="chart-wrap">${m.chart}</div></div>`:''}
    <div class="card"><h2>Missing Values Detail</h2>
    <table class="tbl"><thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>%</th><th>Severity</th><th>MCAR Hint</th></tr></thead>
    <tbody>${m.bars.map(b=>`<tr>
      <td class="mono">${b.column}</td>
      <td><span class="badge badge-${b.type}">${b.type}</span></td>
      <td class="mono">${b.count.toLocaleString()}</td>
      <td><div class="pbar-wrap"><div class="pbar-track"><div class="pbar-fill" style="width:${b.pct}%;background:${b.severity==='high'?'var(--red)':b.severity==='medium'?'var(--yellow)':'var(--green)'}"></div></div><div class="pbar-label">${b.pct}%</div></div></td>
      <td><span class="badge badge-${b.severity}">${b.severity.toUpperCase()}</span></td>
      <td style="color:var(--text2);font-size:11px">${b.mcar_likely?'Likely MCAR':'May be MAR/MNAR'}</td>
    </tr>`).join('')}</tbody></table></div>`;
}

// ═══════════════════════════════════════════════
//  OUTLIERS
// ═══════════════════════════════════════════════
function renderOutliers(d) {
  const outs = d.outliers;
  const el = document.getElementById('outliers-content');
  if (!outs.items.length) {
    el.innerHTML=`<div class="card" style="text-align:center;padding:48px"><div style="font-size:48px;margin-bottom:12px">✅</div><div style="font-size:18px;font-weight:600;color:var(--green)">No significant outliers</div></div>`; return;
  }
  el.innerHTML=`<div class="card"><h2>Outlier Summary (${outs.items.length} columns)</h2>
    <table class="tbl"><thead><tr><th>Column</th><th>IQR Outliers</th><th>IQR %</th><th>Z-Score Outliers</th><th>Z %</th><th>IQR Bounds</th></tr></thead>
    <tbody>${outs.items.map(o=>`<tr>
      <td class="mono">${o.column}</td>
      <td class="mono" style="color:var(--red)">${o.iqr_count}</td>
      <td><div class="pbar-wrap"><div class="pbar-track"><div class="pbar-fill" style="width:${Math.min(o.iqr_pct,100)}%;background:var(--red)"></div></div><div class="pbar-label">${o.iqr_pct}%</div></div></td>
      <td class="mono" style="color:var(--yellow)">${o.z_count}</td>
      <td class="mono">${o.z_pct}%</td>
      <td style="font-family:var(--mono);font-size:10px;color:var(--text2)">[${o.lower_iqr}, ${o.upper_iqr}]</td>
    </tr>`).join('')}</tbody></table></div>
  ${Object.entries(outs.charts||{}).map(([col,html])=>`<div class="card card-sm"><h2>${col}</h2><div class="chart-wrap">${html}</div></div>`).join('')}`;
}

// ═══════════════════════════════════════════════
//  SKEWNESS & KURTOSIS
// ═══════════════════════════════════════════════
function renderSkewness(d) {
  const sk = d.skewness_kurtosis;
  const el = document.getElementById('skewness-content');
  if (!sk || !sk.items.length) { el.innerHTML='<div class="card">No numeric columns.</div>'; return; }
  el.innerHTML=`
    ${sk.chart?`<div class="card"><div class="chart-wrap">${sk.chart}</div></div>`:''}
    <div class="card"><h2>Skewness &amp; Kurtosis Table</h2>
    <table class="tbl"><thead><tr><th>Column</th><th>Skewness</th><th>Direction</th><th>Severity</th><th>Kurtosis</th><th>Kurt Type</th><th>Transform?</th></tr></thead>
    <tbody>${sk.items.map(i=>`<tr>
      <td class="mono">${i.column}</td>
      <td class="mono" style="color:${Math.abs(i.skewness)>1?'var(--red)':Math.abs(i.skewness)>0.5?'var(--yellow)':'var(--green)'}">${i.skewness}</td>
      <td style="font-size:11px;color:var(--text2)">${i.skew_direction}</td>
      <td><span class="badge badge-${i.skew_severity==='Severe'?'high':i.skew_severity==='High'?'medium':'low'}">${i.skew_severity}</span></td>
      <td class="mono">${i.kurtosis}</td>
      <td style="font-size:11px;color:var(--text2)">${i.kurt_type}</td>
      <td>${i.needs_transform?'<span class="badge badge-medium">Recommended</span>':'<span class="badge badge-low">Not needed</span>'}</td>
    </tr>`).join('')}</tbody></table></div>`;
}

// ═══════════════════════════════════════════════
//  CORRELATIONS
// ═══════════════════════════════════════════════
function renderCorrelations(d) {
  const corr = d.correlations;
  const el = document.getElementById('correlations-content');
  if (!corr || !corr.columns.length) { el.innerHTML='<div class="card">Not enough numeric columns.</div>'; return; }
  el.innerHTML=`
    <div class="card"><h2>Pearson Correlation Heatmap</h2><div class="chart-wrap">${corr.chart}</div></div>
    <div class="card"><h2>Covariance Matrix</h2><div class="chart-wrap">${corr.cov_chart}</div></div>
    ${corr.strong_pairs.length?`<div class="card"><h2>Strong Correlations</h2>
    <table class="tbl"><thead><tr><th>Feature A</th><th>Feature B</th><th>Correlation</th><th>Strength</th></tr></thead>
    <tbody>${corr.strong_pairs.map(p=>`<tr>
      <td class="mono">${p.col1}</td><td class="mono">${p.col2}</td>
      <td class="mono" style="color:${p.corr>0?'var(--accent)':'var(--red)'}">${p.corr>0?'+':''}${p.corr}</td>
      <td><span class="badge badge-${p.strength==='Very Strong'?'high':p.strength==='Strong'?'medium':'low'}">${p.strength}</span></td>
    </tr>`).join('')}</tbody></table></div>`:''}`;
}

// ═══════════════════════════════════════════════
//  UNIVARIATE
// ═══════════════════════════════════════════════
function renderUnivariate(d) {
  const num = d.univariate_numeric || [];
  const cat = d.univariate_categorical || [];
  const el  = document.getElementById('univariate-content');
  const tabs = `<div class="tabs">
    <div class="tab active" onclick="switchTab('uni','numeric',this)">Numeric (${num.length})</div>
    <div class="tab" onclick="switchTab('uni','categorical',this)">Categorical (${cat.length})</div>
  </div>`;
  const numHtml = num.map(c=>`<div class="card card-sm" style="margin-bottom:12px">
    <h2>${c.column} <span class="badge badge-numeric">numeric</span></h2>
    <div class="col-acc-stats">
      ${[['Mean',c.mean],['Median',c.median],['Mode',c.mode],['Std',c.std],
         ['Min',c.min],['Max',c.max],['Q1',c.q1],['Q3',c.q3],
         ['IQR',c.iqr],['Skew',c.skewness],['Kurt',c.kurtosis],['CV%',c.cv]
        ].map(([l,v])=>`<div class="mini-stat"><div class="mini-stat-label">${l}</div><div class="mini-stat-value">${v??'—'}</div></div>`).join('')}
    </div>
    ${c.chart?`<div class="chart-wrap">${c.chart}</div>`:''}
  </div>`).join('');
  const catHtml = cat.map(c=>`<div class="card card-sm" style="margin-bottom:12px">
    <h2>${c.column} <span class="badge badge-${c.kind}">${c.kind}</span></h2>
    <div class="col-acc-stats">
      ${[['Unique',c.unique],['Top',c.top_value],['Top%',c.top_pct+'%'],['Entropy',c.entropy]].map(([l,v])=>
        `<div class="mini-stat"><div class="mini-stat-label">${l}</div><div class="mini-stat-value" style="font-size:12px">${v}</div></div>`).join('')}
    </div>
    ${c.chart?`<div class="chart-wrap">${c.chart}</div>`:''}
  </div>`).join('');
  el.innerHTML=`${tabs}
    <div class="tab-panel active" id="uni-numeric">${numHtml||'<div class="card">No numeric columns.</div>'}</div>
    <div class="tab-panel" id="uni-categorical">${catHtml||'<div class="card">No categorical columns.</div>'}</div>`;
}

// ═══════════════════════════════════════════════
//  BIVARIATE
// ═══════════════════════════════════════════════
function renderBivariate(d) {
  const biv = d.bivariate || {};
  const el  = document.getElementById('bivariate-content');
  if (!Object.keys(biv).length) { el.innerHTML='<div class="card">Insufficient data for bivariate analysis.</div>'; return; }
  el.innerHTML = Object.entries(biv).map(([k,html])=>`<div class="card card-sm" style="margin-bottom:12px"><div class="chart-wrap">${html}</div></div>`).join('');
}

// ═══════════════════════════════════════════════
//  NORMALITY
// ═══════════════════════════════════════════════
function renderNormality(d) {
  const norm = d.normality || {};
  const el   = document.getElementById('normality-content');
  const tests = norm.tests || [];
  if (!tests.length) { el.innerHTML='<div class="card">No numeric data for normality tests.</div>'; return; }
  const qqHtml = Object.entries(norm.charts||{}).map(([col,html])=>`<div class="card card-sm"><h2>Q-Q Plot — ${col}</h2><div class="chart-wrap">${html}</div></div>`).join('');
  el.innerHTML=`<div class="card"><h2>Normality Test Results</h2>
    <table class="tbl"><thead><tr><th>Column</th><th>Shapiro-Wilk p</th><th>D'Agostino p</th><th>KS p</th><th>Verdict</th></tr></thead>
    <tbody>${tests.map(t=>`<tr>
      <td class="mono">${t.column}</td>
      <td class="mono" style="color:${(t.shapiro_p||0)>0.05?'var(--green)':'var(--red)'}">${t.shapiro_p??'—'}</td>
      <td class="mono" style="color:${(t.normaltest_p||0)>0.05?'var(--green)':'var(--red)'}">${t.normaltest_p??'—'}</td>
      <td class="mono" style="color:${(t.ks_p||0)>0.05?'var(--green)':'var(--red)'}">${t.ks_p??'—'}</td>
      <td><span class="norm-dot ${t.is_normal?'norm-yes':'norm-no'}"></span> ${t.verdict}</td>
    </tr>`).join('')}</tbody></table></div>${qqHtml}`;
}

// ═══════════════════════════════════════════════
//  PCA
// ═══════════════════════════════════════════════
function renderPCA(d) {
  const pca = d.pca_summary || {};
  const el  = document.getElementById('pca-content');
  if (pca.error || !pca.n_components) { el.innerHTML='<div class="card">PCA not available (need ≥2 numeric cols or scikit-learn).</div>'; return; }
  el.innerHTML=`<div class="card"><h2>PCA Summary</h2>
    <div class="stats-strip">
      <div class="stat-box"><div class="sb-label">Components</div><div class="sb-value accent">${pca.n_components}</div><div class="sb-sub">computed</div></div>
      <div class="stat-box"><div class="sb-label">PCs for 95%</div><div class="sb-value yellow">${pca.n_for_95pct}</div><div class="sb-sub">retain 95% variance</div></div>
    </div>
    ${pca.chart?`<div class="chart-wrap">${pca.chart}</div>`:''}
  </div>`;
}

// ═══════════════════════════════════════════════
//  VARIANCE
// ═══════════════════════════════════════════════
function renderVariance(d) {
  const items = d.variance || [];
  const el    = document.getElementById('variance-content');
  if (!items.length) { el.innerHTML='<div class="card">No numeric columns.</div>'; return; }
  el.innerHTML=`<div class="card"><h2>Feature Variance Analysis</h2>
    <table class="tbl"><thead><tr><th>Column</th><th>Variance</th><th>Std Dev</th><th>CV %</th><th>Low Variance?</th></tr></thead>
    <tbody>${items.map(i=>`<tr>
      <td class="mono">${i.column}</td>
      <td class="mono">${i.variance}</td>
      <td class="mono">${i.std}</td>
      <td class="mono">${i.cv??'—'}</td>
      <td>${i.low_variance?'<span class="badge badge-high">Low Variance</span>':'<span class="badge badge-low">OK</span>'}</td>
    </tr>`).join('')}</tbody></table></div>`;
}

// ═══════════════════════════════════════════════
//  CLASS IMBALANCE
// ═══════════════════════════════════════════════
function renderImbalance(d) {
  const ci = d.class_imbalance;
  const el = document.getElementById('imbalance-content');
  if (!ci) { el.innerHTML='<div class="card" style="text-align:center;padding:32px"><div style="color:var(--text2)">No target column specified.</div></div>'; return; }
  el.innerHTML=`
    <div class="stats-strip">
      <div class="stat-box"><div class="sb-label">Target</div><div class="sb-value accent" style="font-size:16px">${ci.target}</div></div>
      <div class="stat-box"><div class="sb-label">Classes</div><div class="sb-value accent">${ci.n_classes}</div></div>
      <div class="stat-box"><div class="sb-label">Imbalance Ratio</div><div class="sb-value ${ci.is_imbalanced?'red':'green'}">${ci.imbalance_ratio}×</div><div class="sb-sub">${ci.is_imbalanced?'⚠ Imbalanced':'✓ Balanced'}</div></div>
    </div>
    ${ci.chart?`<div class="card"><h2>Class Distribution</h2><div class="chart-wrap">${ci.chart}</div></div>`:''}
    <div class="card"><h2>Class Counts</h2>
    <table class="tbl"><thead><tr><th>Class</th><th>Count</th><th>% of Total</th></tr></thead>
    <tbody>${ci.classes.map(c=>`<tr>
      <td class="mono">${c.label}</td><td class="mono">${c.count.toLocaleString()}</td>
      <td><div class="pbar-wrap"><div class="pbar-track"><div class="pbar-fill" style="width:${c.pct}%;background:var(--accent)"></div></div><div class="pbar-label">${c.pct}%</div></div></td>
    </tr>`).join('')}</tbody></table>
    ${ci.is_imbalanced?`<div style="margin-top:14px;padding:12px 14px;background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.2);border-radius:8px;font-size:12px;color:var(--yellow)">
      ⚠ Imbalance ratio ${ci.imbalance_ratio}× — Consider SMOTE, ADASYN, class_weight='balanced', or stratified sampling.</div>`:''}</div>`;
}

// ═══════════════════════════════════════════════
//  FEATURE-TARGET
// ═══════════════════════════════════════════════
function renderFeatureTarget(d) {
  const charts = d.feature_target || {};
  const el     = document.getElementById('ft-content');
  if (!Object.keys(charts).length) { el.innerHTML='<div class="card" style="text-align:center;padding:32px"><div style="color:var(--text2)">No target column specified.</div></div>'; return; }
  el.innerHTML = Object.entries(charts).map(([k,html])=>`<div class="card card-sm" style="margin-bottom:12px"><div class="chart-wrap">${html}</div></div>`).join('');
}

// ═══════════════════════════════════════════════
//  CHI-SQUARE
// ═══════════════════════════════════════════════
function renderChiSquare(d) {
  const tests = d.chi_square || [];
  const el    = document.getElementById('chi-content');
  if (!tests.length) { el.innerHTML='<div class="card">No categorical pairs for chi-square test.</div>'; return; }
  el.innerHTML=`<div class="card"><h2>Chi-Square Independence Tests</h2>
    <table class="tbl"><thead><tr><th>Column A</th><th>Column B</th><th>χ² Stat</th><th>p-value</th><th>DoF</th><th>Verdict</th></tr></thead>
    <tbody>${tests.map(t=>`<tr>
      <td class="mono">${t.col1}</td><td class="mono">${t.col2}</td>
      <td class="mono">${t.chi2}</td>
      <td class="mono" style="color:${t.significant?'var(--red)':'var(--green)'}">${t.p_value}</td>
      <td class="mono">${t.dof}</td>
      <td><span class="badge badge-${t.significant?'high':'low'}">${t.verdict}</span></td>
    </tr>`).join('')}</tbody></table></div>`;
}

// ═══════════════════════════════════════════════
//  RECOMMENDATIONS
// ═══════════════════════════════════════════════
function renderRecommendations(d) {
  const recs = d.recommendations || [];
  const el   = document.getElementById('rec-content');
  document.getElementById('nb-recs').textContent = recs.length;
  if (!recs.length) {
    el.innerHTML=`<div class="card" style="text-align:center;padding:48px"><div style="font-size:48px;margin-bottom:12px">🎉</div><div style="font-size:18px;font-weight:600;color:var(--green)">No critical issues found!</div></div>`;
    return;
  }
  el.innerHTML = recs.map(rec=>{
    const applied  = appliedRecs[rec.id];
    const colTags  = (rec.affected_columns||[]).slice(0,8).map(c=>`<span class="col-chip">${c}</span>`).join('');
    const moreTag  = (rec.affected_columns||[]).length>8?`<span class="col-chip">+${rec.affected_columns.length-8}</span>`:'';
    const btns     = (rec.actions||[]).map((a,i)=>{
      if (a.action_id==='skip') return `<button class="btn btn-ghost" onclick="applyRec('${rec.id}','${a.action_id}',${JSON.stringify(rec.affected_columns||[])})">${a.label}</button>`;
      return `<button class="btn ${i===0?'btn-primary':'btn-ghost'}" onclick="applyRec('${rec.id}','${a.action_id}',${JSON.stringify(rec.affected_columns||[])})" ${applied?'disabled':''}>${a.label}</button>`;
    }).join('');
    return `<div class="rec-card sev-${rec.severity}" id="rec-${rec.id}">
      <div class="rec-header">
        <div class="rec-icon">${rec.icon}</div>
        <div class="rec-body">
          <div class="rec-title">${rec.title} <span class="badge badge-${rec.severity}">${rec.severity.toUpperCase()}</span>${applied?` <span class="badge badge-low">✓ Applied</span>`:''}</div>
          <div class="rec-desc">${rec.description}</div>
        </div>
      </div>
      ${colTags?`<div class="rec-cols">${colTags}${moreTag}</div>`:''}
      <div class="rec-actions">${btns}</div>
    </div>`;
  }).join('');
}

function applyRec(recId, actionId, cols) {
  if (actionId==='skip') { toast('Skipped ✓','ok'); return; }
  const d   = getD();
  const rec = (d.recommendations||[]).find(r=>r.id===recId);
  const label = rec?(rec.actions||[]).find(a=>a.action_id===actionId)?.label||actionId:actionId;
  appHistory.push({rec_id:recId,action_id:actionId,label,cols,rec_title:rec?.title||recId,time:new Date().toLocaleTimeString()});
  appliedRecs[recId] = true;
  document.getElementById('nb-hist').textContent = appHistory.length;
  document.querySelectorAll(`#rec-${recId} .btn`).forEach(b=>b.disabled=true);
  document.getElementById(`rec-${recId}`).querySelector('.rec-title').innerHTML += ` <span class="badge badge-low">✓ Applied</span>`;
  toast(`✓ Applied: ${label}`,'ok');
  renderHistory();
}

// ═══════════════════════════════════════════════
//  HISTORY
// ═══════════════════════════════════════════════
function renderHistory() {
  const el = document.getElementById('history-content');
  if (!appHistory.length) {
    el.innerHTML=`<div class="card" style="text-align:center;padding:40px"><div style="color:var(--text2)">No transformations applied yet.</div></div>`; return;
  }
  el.innerHTML = appHistory.map((h,i)=>`<div class="card card-sm" style="margin-bottom:10px;animation:fadeUp .2s ease">
    <div style="display:flex;align-items:center;gap:12px">
      <div style="width:26px;height:26px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#fff;flex-shrink:0">${i+1}</div>
      <div style="flex:1">
        <div style="font-weight:600;font-size:13px">${h.label}</div>
        <div style="font-size:11px;color:var(--text2)">${h.rec_title}</div>
      </div>
      <div style="font-family:var(--mono);font-size:10px;color:var(--text3)">${h.time}</div>
    </div>
    ${h.cols?.length?`<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:8px">${h.cols.slice(0,6).map(c=>`<span class="col-chip">${c}</span>`).join('')}${h.cols.length>6?`<span class="col-chip">+${h.cols.length-6}</span>`:''}</div>`:''}</div>`).join('');
}

// ═══════════════════════════════════════════════
//  TABS
// ═══════════════════════════════════════════════
function switchTab(prefix, name, el) {
  document.querySelectorAll(`#sec-${prefix==='uni'?'univariate':'bivariate'} .tab`).forEach(t=>t.classList.remove('active'));
  document.querySelectorAll(`#sec-${prefix==='uni'?'univariate':'bivariate'} .tab-panel`).forEach(p=>p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById(`${prefix}-${name}`)?.classList.add('active');
}

// ═══════════════════════════════════════════════
//  TOAST
// ═══════════════════════════════════════════════
function toast(msg, type='ok') {
  const t = document.getElementById('toast');
  t.textContent=msg; t.className=`toast ${type} show`;
  setTimeout(()=>t.classList.remove('show'), 2800);
}

// ═══════════════════════════════════════════════
//  ML KNOWLEDGE BASE — full content map
// ═══════════════════════════════════════════════
const KB = {
  overview: {
    label: 'Dataset Overview', blocks: [
      { title: 'What to look for first', body: `<ul>
        <li>Check <b>shape</b> — rows &lt; 1,000 is small; &gt;100K is large. Small datasets overfit easily.</li>
        <li><b>Missing % &gt; 40</b> on a column → drop it. 5–40% → impute carefully.</li>
        <li><b>Duplicate rows</b> inflate metrics and cause train/test leakage if they span the split.</li>
        <li><b>Memory usage</b> — if &gt;1 GB, consider chunked processing or columnar formats (Parquet).</li>
        <li>Check <b>column types</b> — numerics stored as <code>object</code> are silent bugs.</li>
      </ul>` },
      { title: 'Column type guide', body: `<ul>
        <li><b>Numeric</b> — continuous values. Check skewness, outliers, scale.</li>
        <li><b>Categorical</b> — finite set of labels. Check cardinality before encoding.</li>
        <li><b>Boolean</b> — binary flag. Encode as 0/1 or leave as-is for tree models.</li>
        <li><b>Datetime</b> — extract year/month/day/weekday/quarter as numeric features.</li>
        <li><b>Text (free)</b> — needs NLP pipeline: TF-IDF, embeddings, or transformers.</li>
      </ul>` },
      { title: 'Auto-detected problem type', body: `<ul>
        <li><b>Binary classification</b> — target has 2 unique values. Use AUC-ROC, F1.</li>
        <li><b>Multiclass</b> — target is categorical or integer with ≤20 classes. Use macro-F1.</li>
        <li><b>Regression</b> — target is continuous float. Use RMSE, MAE, R².</li>
        <li>Override detection by passing <code>problem_type=</code> to <code>export_pipeline_code()</code>.</li>
      </ul>` },
    ]
  },
  readiness: {
    label: 'ML Readiness Score', blocks: [
      { title: 'Score dimensions explained', body: `<ul>
        <li><b>Completeness (weight 2.5×)</b> — missing value rate. 0% = 100pts, &gt;50% = 5pts. Most critical dimension.</li>
        <li><b>Leakage Risk (2.0×)</b> — name proximity + correlation with target. High score = LOW risk.</li>
        <li><b>Outlier Severity (1.5×)</b> — IQR outlier rate. &gt;15% = 10pts. Outliers destroy linear models.</li>
        <li><b>Distribution (1.0×)</b> — skewness (70pts) + kurtosis (30pts) combined.</li>
        <li><b>Cardinality (1.0×)</b> — encoding complexity. Binary=100, &gt;200 unique=10.</li>
        <li><b>Type Fitness (0.8×)</b> — how well dtype suits ML. Boolean=100, free-text=32.</li>
        <li><b>Consistency (0.2×)</b> — constant columns, mixed types, all-zeros.</li>
      </ul>` },
      { title: 'Grade interpretation', body: `<ul>
        <li><b>A (≥90)</b> — production-ready. Train immediately.</li>
        <li><b>B (≥75)</b> — minor prep: maybe impute 1–2 cols, cap outliers.</li>
        <li><b>C (≥60)</b> — moderate prep: imputation, encoding, possibly transform.</li>
        <li><b>D (≥45)</b> — significant issues: high missing, heavy skew, leakage risk.</li>
        <li><b>F (&lt;45)</b> — do not train yet. Fix fundamental data quality issues first.</li>
      </ul>` },
      { title: 'How to improve your score', body: `<ul>
        <li>Drop columns with &gt;40% missing → big completeness gain.</li>
        <li>Cap IQR outliers → outlier severity jumps immediately.</li>
        <li>Log-transform skewed positives → distribution score improves.</li>
        <li>Remove ID columns → eliminates leakage risk and cardinality penalties.</li>
        <li>Cast numeric strings to <code>float64</code> → type fitness fix.</li>
      </ul>` },
    ]
  },
  leakage: {
    label: 'Leakage Detective', blocks: [
      { title: 'What is data leakage?', body: `<p>Leakage is when information from <b>outside the training window</b> contaminates your features. The model learns a pattern that doesn't exist in production, producing unrealistically high validation scores that collapse at deployment.</p>
      <p>A model with AUC 0.99 in validation but 0.51 in production is almost always a leakage problem.</p>` },
      { title: '6 leakage categories', body: `<ul>
        <li><b>Target Correlation</b> — feature with |r| &gt; 0.85 vs target. Drop it.</li>
        <li><b>Name Proximity</b> — column name shares tokens with target name (e.g. <code>churn_flag</code> when target is <code>churn</code>).</li>
        <li><b>Derived Feature</b> — column that was computed using the target post-observation (e.g. <code>churn_score</code>, <code>churn_date</code>).</li>
        <li><b>Future Data</b> — datetime column contains values after the prediction cutoff.</li>
        <li><b>ID / Primary Key</b> — unique identifier that memorises training rows in tree models.</li>
        <li><b>Constant / Zero-Variance</b> — single-value column. Zero information but may corrupt distance metrics.</li>
      </ul>` },
      { title: 'Prevention checklist', body: `<ul>
        <li>Always split <b>before</b> any feature engineering or imputation.</li>
        <li>Use sklearn <code>Pipeline</code> — it fits only on training data.</li>
        <li>For time-series: use <b>TimeSeriesSplit</b>, never random K-Fold.</li>
        <li>Review feature importance — suspiciously high importance on one feature is a leakage signal.</li>
        <li>Validate on a <b>temporal holdout</b> (most recent N months), not random split.</li>
      </ul>` },
    ]
  },
  interactions: {
    label: 'Feature Interactions', blocks: [
      { title: 'Three methods used', body: `<ul>
        <li><b>Mutual Information</b> — captures linear AND non-linear dependencies. Numeric × Numeric. Normalized to [0,1]. Pearson only captures linear; MI is always better.</li>
        <li><b>Cramér's V</b> — categorical × categorical association. Normalized chi-square. 0=independent, 1=perfectly associated.</li>
        <li><b>Correlation Ratio η²</b> — how much variance in a numeric column is explained by a categorical grouping. Like ANOVA R².</li>
      </ul>` },
      { title: 'Strength thresholds', body: `<ul>
        <li><b>Strong (≥0.70)</b> — redundant features. Keep one or use PCA to collapse them.</li>
        <li><b>Moderate (0.40–0.70)</b> — related but complementary. Both may be useful.</li>
        <li><b>Weak (0.15–0.40)</b> — slight relationship. Usually safe to keep both.</li>
        <li><b>Negligible (&lt;0.15)</b> — independent. No interaction to worry about.</li>
      </ul>` },
      { title: 'Redundancy groups', body: `<p>When A↔B and B↔C are both strong, they form a redundancy group {A,B,C}. Options:</p>
      <ul>
        <li>Drop all but the most predictive (use feature importance to choose).</li>
        <li>Apply <b>PCA</b> to collapse the group into 1–2 components.</li>
        <li>Apply <b>Lasso (L1)</b> — it automatically zeroes redundant coefficients.</li>
        <li>Use <b>VIF (Variance Inflation Factor)</b> to quantify multicollinearity for linear models.</li>
      </ul>` },
    ]
  },
  temporal: {
    label: 'Temporal Awareness', blocks: [
      { title: 'Concept drift explained', body: `<p><b>Concept drift</b> means the relationship between features and the target changes over time. Your model learns the past distribution; if the future is different, predictions degrade.</p>
      <p>DataIQ flags features with &gt;20% mean shift from earliest to latest time quartile as potential drift indicators.</p>` },
      { title: 'Types of drift', body: `<ul>
        <li><b>Sudden drift</b> — abrupt distribution change (e.g. COVID lockdown effect on spending data).</li>
        <li><b>Gradual drift</b> — slow shift over months (e.g. inflation gradually shifting income distributions).</li>
        <li><b>Seasonal drift</b> — predictable periodic patterns (not real drift — extract as features instead).</li>
        <li><b>Recurring drift</b> — pattern returns periodically.</li>
      </ul>` },
      { title: 'How to handle it', body: `<ul>
        <li>Use <b>time-based train/test split</b> — train on older data, test on recent.</li>
        <li><b>Rolling window retraining</b> — retrain every N weeks on the most recent M months.</li>
        <li><b>Sample weighting</b> — give higher weight to recent observations during training.</li>
        <li>Extract <b>time features</b> (month, weekday, quarter) so model learns seasonality.</li>
        <li>Monitor PSI monthly in production — set alert at PSI &gt; 0.20.</li>
      </ul>` },
    ]
  },
  columns: {
    label: 'Column Analysis', blocks: [
      { title: 'Key statistics guide', body: `<ul>
        <li><b>Mean vs Median</b> — large gap signals skewness or outliers. Use median for skewed data.</li>
        <li><b>CV (Coefficient of Variation)</b> — std/mean × 100. Useful for comparing variability across columns with different scales.</li>
        <li><b>IQR (Q3−Q1)</b> — robust spread measure. Less sensitive to outliers than std.</li>
        <li><b>Entropy</b> (categorical) — higher = more uniform distribution. 0 = constant column.</li>
        <li><b>Unique %</b> &gt; 85% — likely an ID/free-text column. Drop or encode carefully.</li>
      </ul>` },
      { title: 'Dtype best practices', body: `<ul>
        <li>Cast low-cardinality strings to <code>pd.Categorical</code> — saves 5–10× memory.</li>
        <li>Use <code>int8/int16</code> for small integers — reduces memory further.</li>
        <li>Parse date strings to <code>datetime64</code> — enables time-based feature extraction.</li>
        <li>Store binary flags as <code>bool</code> — cleaner than 0/1 integers.</li>
      </ul>` },
    ]
  },
  missing: {
    label: 'Missing Values', blocks: [
      { title: 'MCAR / MAR / MNAR', body: `<ul>
        <li><b>MCAR</b> (Missing Completely At Random) — no pattern. Simple imputation is safe. Test: Little's MCAR test.</li>
        <li><b>MAR</b> (Missing At Random) — missingness depends on observed data (e.g. older patients skip certain questions). Multiple imputation or KNN works.</li>
        <li><b>MNAR</b> (Missing Not At Random) — missingness depends on the missing value itself (e.g. high earners skip income question). Hardest — add a <b>missing indicator feature</b>.</li>
      </ul>` },
      { title: 'Imputation methods', body: `<ul>
        <li><b>Mean imputation</b> — distorts distribution, reduces variance. Use only for MCAR + symmetric data.</li>
        <li><b>Median imputation</b> — robust to outliers. Best default for skewed numeric columns.</li>
        <li><b>Mode imputation</b> — for categoricals. Inflates the most frequent class.</li>
        <li><b>KNN imputation</b> — finds k nearest complete rows and averages. Better than simple but slow on large data.</li>
        <li><b>MICE / Iterative Imputer</b> — imputes each column using all others as predictors. Best quality, most expensive.</li>
        <li><b>Missing indicator</b> — add a binary <code>was_missing</code> column. Tells model that missingness itself is informative.</li>
      </ul>` },
      { title: 'When to drop vs impute', body: `<ul>
        <li><b>&gt;40% missing</b> — drop the column unless domain knowledge says otherwise.</li>
        <li><b>5–40% missing</b> — impute. Add a missing indicator if MNAR suspected.</li>
        <li><b>&lt;5% missing</b> — simple median/mode imputation is fine.</li>
        <li>Never impute the <b>target column</b> — drop those rows instead.</li>
      </ul>` },
    ]
  },
  outliers: {
    label: 'Outlier Analysis', blocks: [
      { title: 'Detection methods', body: `<ul>
        <li><b>IQR method</b> — flag values outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR]. Robust, non-parametric. Best general choice.</li>
        <li><b>Z-score</b> — flag |z| &gt; 3. Assumes normality. Unreliable for skewed distributions.</li>
        <li><b>Percentile</b> — cap at p1/p99 or p5/p95. Simple, interpretable.</li>
        <li><b>Isolation Forest</b> — tree-based anomaly detection. Best for multivariate outliers.</li>
        <li><b>DBSCAN</b> — density-based. Points in low-density regions are outliers.</li>
      </ul>` },
      { title: 'Treatment options', body: `<ul>
        <li><b>Winsorization (capping)</b> — replace outliers with the fence value. Preserves row count. Best default.</li>
        <li><b>Removal (trimming)</b> — delete outlier rows. Only if you're sure they're data errors.</li>
        <li><b>Log transform</b> — compresses the right tail naturally. Only for positive columns.</li>
        <li><b>Robust scaler</b> — scales using median + IQR instead of mean + std. Outliers don't affect scaling.</li>
        <li><b>Keep</b> — tree-based models (RF, GBM) are naturally robust to outliers.</li>
      </ul>` },
      { title: 'Model impact by algorithm', body: `<ul>
        <li><b>Linear / Logistic Regression</b> — very sensitive. Always winsorize first.</li>
        <li><b>SVM</b> — very sensitive. Scale and remove outliers before training.</li>
        <li><b>KNN / K-Means</b> — sensitive (distance-based). Remove or cap.</li>
        <li><b>Random Forest / GBM</b> — robust. Outliers have minimal impact.</li>
        <li><b>Neural Networks</b> — sensitive to scale. Standardize; remove extreme outliers.</li>
      </ul>` },
    ]
  },
  skewness: {
    label: 'Skewness & Kurtosis', blocks: [
      { title: 'Skewness thresholds', body: `<ul>
        <li><b>|skew| &lt; 0.5</b> — approximately normal. No transformation needed.</li>
        <li><b>0.5–1.0</b> — moderate skew. Usually acceptable for tree models.</li>
        <li><b>1.0–2.0</b> — high skew. Transform before using linear models.</li>
        <li><b>&gt; 2.0</b> — severe skew. Must transform for any parametric model.</li>
      </ul>` },
      { title: 'Transformation guide', body: `<ul>
        <li><b>Log (ln(x+1))</b> — best for right-skewed positive columns. Simple and interpretable.</li>
        <li><b>Square root (√x)</b> — milder than log. Good for count data.</li>
        <li><b>Box-Cox</b> — parametric, finds optimal λ. Only for strictly positive values.</li>
        <li><b>Yeo-Johnson</b> — like Box-Cox but works for zero and negative values. Best general choice.</li>
        <li><b>Reciprocal (1/x)</b> — very strong compression. Use for extreme right skew.</li>
      </ul>
      <div class="kb-formula">Yeo-Johnson: if x≥0: ((x+1)^λ − 1) / λ<br>if x&lt;0: −((−x+1)^(2−λ) − 1) / (2−λ)</div>` },
      { title: 'Kurtosis interpretation', body: `<ul>
        <li><b>Mesokurtic (kurt ≈ 0)</b> — normal-like tails. No special treatment.</li>
        <li><b>Leptokurtic (kurt &gt; 1)</b> — heavy tails, more extreme values than normal. Common in financial data. Robust scalers help.</li>
        <li><b>Platykurtic (kurt &lt; −1)</b> — lighter tails, more uniform distribution. Rare in practice.</li>
      </ul>` },
    ]
  },
  correlations: {
    label: 'Correlation Analysis', blocks: [
      { title: 'Pearson vs alternatives', body: `<ul>
        <li><b>Pearson r</b> — linear relationships only. Sensitive to outliers. Range [−1, 1].</li>
        <li><b>Spearman ρ</b> — monotonic (not just linear) relationships. Robust to outliers. Use for skewed data.</li>
        <li><b>Kendall τ</b> — rank-based, better for small samples or ties.</li>
        <li><b>Mutual Information</b> — captures non-linear dependencies. Always ≥ 0. See Interactions tab.</li>
        <li><b>Point-Biserial</b> — Pearson between a continuous and a binary variable.</li>
      </ul>` },
      { title: 'Multicollinearity thresholds', body: `<ul>
        <li><b>|r| &gt; 0.95</b> — critical. Drop one column immediately. Near-perfect collinearity.</li>
        <li><b>|r| &gt; 0.85</b> — high. Drop the weaker predictor (lower correlation with target).</li>
        <li><b>|r| &gt; 0.70</b> — moderate. Monitor. Use regularization (Ridge/Lasso) or PCA.</li>
        <li><b>VIF &gt; 10</b> — strong collinearity. VIF = 1/(1−R²) for each feature.</li>
      </ul>
      <div class="kb-formula">VIF = 1 / (1 − R²_i)<br>R²_i = R² from regressing feature i on all other features</div>` },
      { title: 'What to do about it', body: `<ul>
        <li><b>Drop one</b> — keep the feature with higher correlation to target.</li>
        <li><b>PCA</b> — replace correlated group with uncorrelated principal components.</li>
        <li><b>Ridge regression</b> — L2 regularization handles multicollinearity gracefully.</li>
        <li><b>Lasso</b> — L1 zeroes out redundant features automatically.</li>
        <li>Tree models are <b>unaffected</b> by multicollinearity — split on whichever is most useful.</li>
      </ul>` },
    ]
  },
  univariate: {
    label: 'Univariate Analysis', blocks: [
      { title: 'Numeric checklist', body: `<ul>
        <li>Check <b>min/max</b> for impossible values (age=−1, income=0 for non-volunteers).</li>
        <li>Check <b>zero count</b> — are zeros meaningful or encoded missing?</li>
        <li>Compare <b>mean vs median</b> — gap &gt; 10% signals skewness.</li>
        <li>Check <b>CV% (coefficient of variation)</b> — &lt;15% is low variance, may not be useful.</li>
        <li>Look at p5 and p95 — are there extreme values outside the expected range?</li>
      </ul>` },
      { title: 'Categorical checklist', body: `<ul>
        <li><b>Entropy near 0</b> — near-constant column, low predictive value.</li>
        <li><b>Top value &gt; 90%</b> — dominant class. Model may ignore this feature.</li>
        <li><b>High cardinality</b> — many unique values relative to row count. Check for typos/case variations.</li>
        <li><b>Rare categories (&lt;1%)</b> — group them into an "Other" bucket before encoding.</li>
      </ul>` },
    ]
  },
  bivariate: {
    label: 'Bivariate Analysis', blocks: [
      { title: 'Num × Num (scatterplot)', body: `<ul>
        <li>Look for <b>linear trend</b> — correlation gives direction and strength.</li>
        <li>Look for <b>non-linear patterns</b> (U-shape, exponential) — Pearson misses these; use MI.</li>
        <li><b>Heteroscedasticity</b> (variance changes along x-axis) — log-transform one or both variables.</li>
        <li><b>Cluster structures</b> — suggest latent subgroups worth investigating.</li>
      </ul>` },
      { title: 'Cat × Num (box/violin)', body: `<ul>
        <li>Compare <b>medians and IQR</b> across groups — large differences = strong categorical predictor.</li>
        <li>Significant overlap between groups means the categorical feature has low discriminative power.</li>
        <li>Run <b>ANOVA F-test</b> to formally test if group means differ significantly.</li>
        <li><b>Kruskal-Wallis</b> is the non-parametric alternative when normality is violated.</li>
      </ul>` },
      { title: 'Feature-Target relationship', body: `<ul>
        <li>High |r| with target → strong linear predictor. But check for leakage first.</li>
        <li>For classification: <b>violin plots</b> show how the feature distributes across classes.</li>
        <li><b>OLS trendline</b> slope and intercept are useful for linear model baseline expectations.</li>
        <li>Low correlation doesn't mean useless — the feature may interact with others.</li>
      </ul>` },
    ]
  },
  normality: {
    label: 'Normality Tests', blocks: [
      { title: 'Test guide', body: `<ul>
        <li><b>Shapiro-Wilk</b> — best for n &lt; 5,000. Most powerful normality test. p &gt; 0.05 = can't reject normality.</li>
        <li><b>D'Agostino-Pearson</b> — uses skewness + kurtosis combined. Works for any sample size.</li>
        <li><b>Kolmogorov-Smirnov</b> — compares empirical CDF to theoretical normal. Less powerful than Shapiro-Wilk.</li>
        <li><b>Q-Q plot</b> — points along the diagonal = normal. S-curve = heavy tails. Banana curve = skewed.</li>
      </ul>` },
      { title: 'Why normality matters', body: `<ul>
        <li><b>Linear/Logistic Regression</b> — residuals should be normal, not necessarily features.</li>
        <li><b>LDA</b> — assumes multivariate normality.</li>
        <li><b>t-tests, ANOVA</b> — assume normally distributed groups.</li>
        <li><b>Tree models (RF, GBM)</b> — don't care about normality at all.</li>
        <li><b>Neural Networks</b> — benefit from standardized inputs, not necessarily normal ones.</li>
      </ul>` },
    ]
  },
  pca: {
    label: 'PCA Analysis', blocks: [
      { title: 'How PCA works', body: `<p>PCA finds the directions (principal components) of maximum variance in your data. PC1 captures the most variance, PC2 the second most, and so on — all orthogonal (uncorrelated).</p>
      <div class="kb-formula">X_pca = X_centered · V<br>V = eigenvectors of covariance matrix<br>Explained variance ratio = λ_i / Σλ</div>` },
      { title: 'n_components decision rule', body: `<ul>
        <li><b>Retain enough components to explain 95%</b> of variance — standard rule of thumb.</li>
        <li>If 2 components explain &gt;80%, plot the 2D projection — useful for visualization.</li>
        <li>Use the <b>elbow in the scree plot</b> — find where explained variance drops off sharply.</li>
        <li>For classification: components that separate classes matter more than total variance explained.</li>
      </ul>` },
      { title: 'When to use PCA', body: `<ul>
        <li>You have <b>highly correlated features</b> (multicollinearity) → PCA removes it.</li>
        <li>You have <b>&gt;50 features</b> and want to reduce dimensionality.</li>
        <li>You need <b>visualization</b> — reduce to 2D or 3D.</li>
        <li><b>Don't use PCA</b> when interpretability matters — components are not individual features.</li>
        <li><b>Always standardize</b> before PCA — otherwise high-variance columns dominate.</li>
      </ul>` },
    ]
  },
  variance: {
    label: 'Variance Analysis', blocks: [
      { title: 'Low variance features', body: `<ul>
        <li><b>Zero variance</b> — constant column. Drop immediately. Carries no information.</li>
        <li><b>Near-zero variance</b> — one value in &gt;95% of rows. Consider dropping.</li>
        <li>sklearn <code>VarianceThreshold(threshold=0.01)</code> removes low-variance features automatically.</li>
        <li>For tree models, low-variance features are rarely selected — safe to keep if unsure.</li>
      </ul>` },
      { title: 'CV% interpretation', body: `<ul>
        <li><b>CV &lt; 15%</b> — low relative variability. Feature may not discriminate well.</li>
        <li><b>CV 15–50%</b> — moderate. Usually useful.</li>
        <li><b>CV &gt; 100%</b> — highly variable. Check for outliers driving the std up.</li>
        <li>CV is undefined/misleading when the mean is near zero.</li>
      </ul>` },
    ]
  },
  imbalance: {
    label: 'Class Imbalance', blocks: [
      { title: 'Why it matters', body: `<p>A classifier trained on 95/5 split learns to predict the majority class almost always, achieving 95% accuracy while completely failing on the minority. Accuracy is a misleading metric here — use <b>AUC-ROC</b>, <b>F1</b>, or <b>Precision-Recall AUC</b> instead.</p>` },
      { title: 'Resampling strategies', body: `<ul>
        <li><b>Random oversampling</b> — duplicate minority rows. Simple but risks overfitting.</li>
        <li><b>SMOTE</b> — generate synthetic minority samples by interpolating between real ones. Better than duplication.</li>
        <li><b>ADASYN</b> — like SMOTE but focuses on hard-to-classify boundary samples.</li>
        <li><b>Random undersampling</b> — drop majority rows. Fast but wastes data.</li>
        <li><b>Tomek Links</b> — remove majority samples near the decision boundary.</li>
      </ul>` },
      { title: 'Algorithm-level fixes', body: `<ul>
        <li><code>class_weight='balanced'</code> — sklearn automatically adjusts loss function weights. Works for LR, SVM, RF, GBM.</li>
        <li><b>Threshold tuning</b> — lower the classification threshold (e.g. 0.3 instead of 0.5) to catch more minority cases.</li>
        <li><b>Anomaly detection</b> — if ratio &gt;100:1, treat as anomaly detection problem (Isolation Forest, One-Class SVM).</li>
        <li>Always use <b>stratified splits</b> to ensure both train and test contain minority samples.</li>
      </ul>
      <div class="kb-formula">F1 = 2 × (Precision × Recall) / (Precision + Recall)<br>AUC-ROC: area under TPR vs FPR curve<br>PR-AUC: area under Precision vs Recall curve (better for imbalance)</div>` },
    ]
  },
  feature_target: {
    label: 'Feature–Target', blocks: [
      { title: 'Feature selection methods', body: `<ul>
        <li><b>Pearson correlation</b> — fast, linear only. Good for first pass.</li>
        <li><b>Mutual Information</b> — captures non-linear. Use <code>mutual_info_classif</code> or <code>mutual_info_regression</code>.</li>
        <li><b>ANOVA F-test</b> — for categorical features vs numeric target.</li>
        <li><b>Chi-square</b> — for categorical features vs categorical target.</li>
        <li><b>RFE (Recursive Feature Elimination)</b> — train model, drop weakest features, repeat.</li>
        <li><b>Tree feature importance</b> — fast proxy. Use permutation importance for a more reliable estimate.</li>
        <li><b>SHAP values</b> — most reliable. Shows per-sample contribution of each feature.</li>
      </ul>` },
      { title: 'OLS trendline interpretation', body: `<ul>
        <li><b>Slope</b> — unit increase in feature → slope change in target.</li>
        <li><b>R² of OLS line</b> — proportion of target variance explained by this single feature.</li>
        <li>Low individual R² is fine — features often become useful in combination with others.</li>
        <li>Heteroscedastic residuals → try log-transforming the feature.</li>
      </ul>` },
    ]
  },
  chisquare: {
    label: 'Chi-Square Tests', blocks: [
      { title: 'What it tests', body: `<p>Chi-square tests whether two categorical variables are <b>independent</b>. H₀: the variables are independent. Low p-value (&lt;0.05) means they are <b>dependent</b> (associated).</p>
      <div class="kb-formula">χ² = Σ (O−E)² / E<br>O = observed count, E = expected count<br>df = (rows−1)(cols−1)</div>` },
      { title: 'Interpreting results', body: `<ul>
        <li><b>p &lt; 0.05</b> → Dependent. The variables are statistically associated. One may predict the other.</li>
        <li><b>p ≥ 0.05</b> → Independent. No strong association detected.</li>
        <li><b>χ² statistic</b> — larger = stronger association but depends on sample size. Use <b>Cramér's V</b> for effect size.</li>
        <li><b>DoF</b> — degrees of freedom. Depends on number of categories.</li>
      </ul>` },
      { title: "Cram\u00e9r's V (effect size)", body: `<ul>
        <li><b>V = 0.0–0.1</b> — negligible association.</li>
        <li><b>V = 0.1–0.3</b> — small association.</li>
        <li><b>V = 0.3–0.5</b> — moderate association.</li>
        <li><b>V &gt; 0.5</b> — strong association. Consider encoding one using the other (target encoding).</li>
      </ul>
      <div class="kb-formula">V = √(χ² / (n × min(r−1, c−1)))<br>n = sample size, r/c = number of rows/cols in contingency table</div>` },
    ]
  },
  recommendations: {
    label: 'Recommendations', blocks: [
      { title: 'Severity levels', body: `<ul>
        <li><b>High</b> — fix before training. Will materially hurt model performance if ignored.</li>
        <li><b>Medium</b> — fix before production. May cause subtle issues in model or deployment.</li>
        <li><b>Low</b> — best practice. Won't break the model but improves quality and performance.</li>
      </ul>` },
      { title: 'Correct order of operations', body: `<ul>
        <li>1. <b>Drop duplicates</b> — before any analysis or split.</li>
        <li>2. <b>Drop high-missing columns</b> (&gt;40%) — before imputation.</li>
        <li>3. <b>Train/test split</b> — before imputation and encoding.</li>
        <li>4. <b>Impute</b> — fit imputer on train only, transform both.</li>
        <li>5. <b>Cap outliers</b> — using train distribution bounds.</li>
        <li>6. <b>Encode categoricals</b> — fit encoder on train only.</li>
        <li>7. <b>Transform/scale</b> — fit scaler on train only.</li>
        <li>8. <b>Feature selection</b> — using train data only.</li>
      </ul>` },
      { title: 'Using sklearn Pipelines', body: `<p>Wrapping all preprocessing + the model in a single <code>Pipeline</code> guarantees that every fit step sees only training data, preventing leakage automatically.</p>
      <div class="kb-formula">pipeline = Pipeline([<br>  ('imputer', SimpleImputer()),<br>  ('scaler', StandardScaler()),<br>  ('model', RandomForestClassifier())<br>])<br>pipeline.fit(X_train, y_train)</div>` },
    ]
  },
  history: {
    label: 'Transformation History', blocks: [
      { title: 'Pipeline reproducibility', body: `<ul>
        <li>Document every transformation applied — this history is your preprocessing spec.</li>
        <li>Recreate the same pipeline in sklearn for production deployment.</li>
        <li>Use <code>export_pipeline_code()</code> to generate a matching sklearn script automatically.</li>
        <li>Save the fitted pipeline with <code>joblib.dump(pipeline, 'pipeline.pkl')</code>.</li>
      </ul>` },
    ]
  },
};

// Map each section to its KB key
const SECTION_TO_KB = {
  overview:'overview', readiness:'readiness', leakage:'leakage',
  interactions:'interactions', temporal:'temporal', columns:'columns',
  missing:'missing', outliers:'outliers', skewness:'skewness',
  correlations:'correlations', univariate:'univariate', bivariate:'bivariate',
  normality:'normality', pca:'pca', variance:'variance', imbalance:'imbalance',
  feature_target:'feature_target', chisquare:'chisquare',
  recommendations:'recommendations', history:'history',
};

let _activeKB = 'overview';

function renderKB(key) {
  const data = KB[key] || KB['overview'];
  _activeKB = key;
  const el = document.getElementById('kb-content');
  if (!el) return;

  el.innerHTML = `
    <div class="kb-section-label">${data.label}</div>
    ${data.blocks.map((b, i) => `
      <div class="kb-block">
        <div class="kb-block-title" onclick="toggleKBBlock(this)">${b.title}</div>
        <div class="kb-block-body">${b.body}</div>
      </div>`).join('')}`;
}

function toggleKBBlock(titleEl) {
  titleEl.classList.toggle('collapsed');
  titleEl.nextElementSibling.classList.toggle('collapsed');
}

// ── KB resize drag handle ──────────────────────────────────────────
(function initKBResize() {
  const handle  = document.getElementById('kb-drag');
  const sidebar = document.getElementById('kb-sidebar');
  if (!handle || !sidebar) return;
  let startX = 0, startW = 0;

  handle.addEventListener('mousedown', e => {
    startX = e.clientX;
    startW = sidebar.offsetWidth;
    handle.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    function onMove(e) {
      const delta = startX - e.clientX;    // dragging left = wider
      const newW  = Math.min(Math.max(startW + delta, 180), 520);
      sidebar.style.width = newW + 'px';
    }
    function onUp() {
      handle.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
})();

// ── Sync KB with active section on DOMContentLoaded ───────────────
document.addEventListener('DOMContentLoaded', () => {
  renderKB('overview');
});
</script>
</body>
</html>"""