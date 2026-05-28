#!/usr/bin/env python3
"""
mlprofiler_train.py — Interactive Model Training CLI
=====================================================
Run: python mlprofiler_train.py --data your_data.csv --target price

Walks you through:
  1. Problem type (regression / classification / clustering)
  2. Model selection (with parameters)
  3. Preprocessing choices
  4. Cross-validation
  5. Evaluation metrics + plots
  6. SHAP / feature importance
  7. Hyperparameter tuning (optional)
  8. Export HTML report + model pkl
"""

import os, sys, json, pickle, warnings, argparse, textwrap
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

try:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ─────────────────────────────────────────────────────────────────────
COLORS = {
    "cyan": "\033[96m", "green": "\033[92m", "yellow": "\033[93m",
    "red": "\033[91m", "bold": "\033[1m", "dim": "\033[2m",
    "blue": "\033[94m", "magenta": "\033[95m", "reset": "\033[0m",
}

def c(text, *clrs):
    if not sys.stdout.isatty(): return str(text)
    codes = "".join(COLORS.get(x,"") for x in clrs)
    return f"{codes}{text}{COLORS['reset']}"

def header(msg):
    w = 66
    print()
    print(c("┌" + "─"*(w-2) + "┐", "cyan"))
    print(c("│ " + msg.center(w-4) + " │", "cyan", "bold"))
    print(c("└" + "─"*(w-2) + "┘", "cyan"))
    print()

def section(msg):
    print()
    print(c(f"  ━━━  {msg}  ━━━", "blue", "bold"))
    print()

def ask(prompt, default=None, choices=None):
    """Prompt user for input with optional default and choices."""
    if choices:
        opts = "  ".join([f"[{c(i+1,'cyan','bold')}] {ch}" for i,ch in enumerate(choices)])
        print(f"  {opts}")
        hint = f" (1-{len(choices)}, default={default or 1}): "
        raw = input(c(f"  ❯ {prompt}{hint}", "yellow")).strip()
        if not raw:
            idx = (default or 1) - 1
        else:
            try:
                idx = int(raw) - 1
                if not (0 <= idx < len(choices)):
                    print(c("  Invalid choice, using default.", "red")); idx = (default or 1) - 1
            except:
                # Try to match by text
                matches = [i for i,ch in enumerate(choices) if raw.lower() in ch.lower()]
                idx = matches[0] if matches else (default or 1) - 1
        return choices[idx], idx + 1
    else:
        raw = input(c(f"  ❯ {prompt}" + (f" [{default}]" if default is not None else "") + ": ", "yellow")).strip()
        return raw if raw else (str(default) if default is not None else "")

def ask_multi(prompt, choices, default_all=False):
    """Multi-select from choices."""
    print(f"  (comma-separated numbers, e.g. 1,3 — or 'all')")
    for i,ch in enumerate(choices):
        print(f"    {c(i+1,'cyan')}. {ch}")
    raw = input(c(f"  ❯ {prompt}: ", "yellow")).strip()
    if not raw or raw.lower()=="all" or default_all:
        return choices
    try:
        idxs = [int(x.strip())-1 for x in raw.split(",")]
        return [choices[i] for i in idxs if 0<=i<len(choices)]
    except:
        return choices

def confirm(msg, default=True):
    hint = " [Y/n]" if default else " [y/N]"
    raw = input(c(f"  ❯ {msg}{hint}: ", "yellow")).strip().lower()
    if not raw: return default
    return raw in ("y","yes","1","true")

# ─────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────
REGRESSION_MODELS = {
    "Linear Regression": {
        "class": "sklearn.linear_model.LinearRegression",
        "params": {},
    },
    "Ridge Regression": {
        "class": "sklearn.linear_model.Ridge",
        "params": {"alpha": ("float", 1.0, "Regularization strength")},
    },
    "Lasso Regression": {
        "class": "sklearn.linear_model.Lasso",
        "params": {"alpha": ("float", 1.0, "Regularization strength")},
    },
    "ElasticNet": {
        "class": "sklearn.linear_model.ElasticNet",
        "params": {
            "alpha": ("float", 1.0, "Overall regularization"),
            "l1_ratio": ("float", 0.5, "L1 vs L2 mix (0–1)"),
        },
    },
    "Decision Tree": {
        "class": "sklearn.tree.DecisionTreeRegressor",
        "params": {
            "max_depth": ("int", 5, "Max tree depth (None=unlimited)"),
            "min_samples_split": ("int", 2, "Min samples to split"),
        },
    },
    "Random Forest": {
        "class": "sklearn.ensemble.RandomForestRegressor",
        "params": {
            "n_estimators": ("int", 100, "Number of trees"),
            "max_depth": ("int_none", None, "Max depth (blank=None)"),
            "min_samples_split": ("int", 2, "Min samples to split"),
        },
    },
    "Gradient Boosting": {
        "class": "sklearn.ensemble.GradientBoostingRegressor",
        "params": {
            "n_estimators": ("int", 100, "Boosting stages"),
            "learning_rate": ("float", 0.1, "Learning rate"),
            "max_depth": ("int", 3, "Max depth"),
        },
    },
    "XGBoost": {
        "class": "xgboost.XGBRegressor",
        "params": {
            "n_estimators": ("int", 100, "Boosting rounds"),
            "learning_rate": ("float", 0.1, "Step size"),
            "max_depth": ("int", 6, "Tree depth"),
        },
    },
    "LightGBM": {
        "class": "lightgbm.LGBMRegressor",
        "params": {
            "n_estimators": ("int", 100, "Boosting rounds"),
            "learning_rate": ("float", 0.1, "Step size"),
            "num_leaves": ("int", 31, "Max leaves"),
        },
    },
    "SVR": {
        "class": "sklearn.svm.SVR",
        "params": {
            "C": ("float", 1.0, "Regularization"),
            "kernel": ("choice", "rbf", "rbf/linear/poly"),
        },
    },
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": {
        "class": "sklearn.linear_model.LogisticRegression",
        "params": {
            "C": ("float", 1.0, "Inverse regularization"),
            "max_iter": ("int", 1000, "Max iterations"),
        },
    },
    "Decision Tree": {
        "class": "sklearn.tree.DecisionTreeClassifier",
        "params": {
            "max_depth": ("int", 5, "Max depth"),
            "min_samples_split": ("int", 2, "Min samples split"),
        },
    },
    "Random Forest": {
        "class": "sklearn.ensemble.RandomForestClassifier",
        "params": {
            "n_estimators": ("int", 100, "Number of trees"),
            "max_depth": ("int_none", None, "Max depth"),
            "class_weight": ("choice", "balanced", "balanced/None"),
        },
    },
    "Gradient Boosting": {
        "class": "sklearn.ensemble.GradientBoostingClassifier",
        "params": {
            "n_estimators": ("int", 100, "Boosting stages"),
            "learning_rate": ("float", 0.1, "Learning rate"),
            "max_depth": ("int", 3, "Max depth"),
        },
    },
    "XGBoost": {
        "class": "xgboost.XGBClassifier",
        "params": {
            "n_estimators": ("int", 100, "Rounds"),
            "learning_rate": ("float", 0.1, "Step size"),
            "use_label_encoder": ("bool", False, ""),
        },
    },
    "LightGBM": {
        "class": "lightgbm.LGBMClassifier",
        "params": {
            "n_estimators": ("int", 100, "Rounds"),
            "learning_rate": ("float", 0.1, "Step size"),
            "num_leaves": ("int", 31, "Max leaves"),
            "class_weight": ("choice", "balanced", "balanced/None"),
        },
    },
    "SVM": {
        "class": "sklearn.svm.SVC",
        "params": {
            "C": ("float", 1.0, "Regularization"),
            "kernel": ("choice", "rbf", "rbf/linear"),
            "probability": ("bool", True, "Enable proba"),
        },
    },
    "K-Nearest Neighbors": {
        "class": "sklearn.neighbors.KNeighborsClassifier",
        "params": {
            "n_neighbors": ("int", 5, "Number of neighbors"),
            "weights": ("choice", "uniform", "uniform/distance"),
        },
    },
    "Naive Bayes": {
        "class": "sklearn.naive_bayes.GaussianNB",
        "params": {},
    },
}

CLUSTERING_MODELS = {
    "K-Means": {
        "class": "sklearn.cluster.KMeans",
        "params": {
            "n_clusters": ("int", 3, "Number of clusters"),
            "n_init": ("int", 10, "Number of initializations"),
        },
    },
    "DBSCAN": {
        "class": "sklearn.cluster.DBSCAN",
        "params": {
            "eps": ("float", 0.5, "Max neighborhood distance"),
            "min_samples": ("int", 5, "Min samples in neighborhood"),
        },
    },
    "Agglomerative": {
        "class": "sklearn.cluster.AgglomerativeClustering",
        "params": {
            "n_clusters": ("int", 3, "Number of clusters"),
            "linkage": ("choice", "ward", "ward/complete/average"),
        },
    },
}

# ─────────────────────────────────────────────────────────────────────
def load_class(dotted_path: str):
    parts = dotted_path.rsplit(".", 1)
    try:
        import importlib
        mod = importlib.import_module(parts[0])
        return getattr(mod, parts[1])
    except Exception as e:
        return None

def collect_params(model_name: str, param_spec: Dict) -> Dict:
    """Interactively collect model parameters."""
    if not param_spec:
        return {}
    section(f"Parameters — {model_name}")
    print(c("  Press Enter to use defaults\n", "dim"))
    params = {}
    for pname, (ptype, default, desc) in param_spec.items():
        print(c(f"  {pname}", "cyan") + c(f" — {desc}", "dim"))
        raw = input(c(f"    [{default}]: ", "yellow")).strip()
        if not raw:
            if ptype != "bool": params[pname] = default
            continue
        try:
            if ptype == "int": params[pname] = int(raw)
            elif ptype == "int_none": params[pname] = None if raw.lower()=="none" else int(raw)
            elif ptype == "float": params[pname] = float(raw)
            elif ptype == "bool": params[pname] = raw.lower() in ("true","1","yes")
            elif ptype == "choice": params[pname] = None if raw.lower()=="none" else raw
            else: params[pname] = raw
        except:
            params[pname] = default
    return params

# ─────────────────────────────────────────────────────────────────────
def run_regression(X_train, X_test, y_train, y_test, model, model_name, cv, output_dir):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
    cv_rmse = -cv_scores.mean()

    section("Regression Metrics")
    print(f"  RMSE        : {c(f'{rmse:.4f}','green','bold')}")
    print(f"  MAE         : {c(f'{mae:.4f}','green')}")
    print(f"  R²          : {c(f'{r2:.4f}','green')}")
    print(f"  CV RMSE     : {c(f'{cv_rmse:.4f}±{(-cv_scores).std():.4f}','cyan')}")

    charts = {}
    if HAS_PLOTLY:
        # Actual vs Predicted
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                  marker=dict(color="rgba(79,142,247,0.5)",size=5),name="Predictions"))
        mn,mx = min(y_test.min(),y_pred.min()), max(y_test.max(),y_pred.max())
        fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",
                                  line=dict(color="#ef4444",width=2,dash="dash"),name="Perfect"))
        fig.update_layout(title="Actual vs Predicted",xaxis_title="Actual",yaxis_title="Predicted",
                          template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",height=400)
        charts["actual_vs_pred"] = pio.to_html(fig, full_html=False, include_plotlyjs=False,
                                                config={"displayModeBar":False})

        # Residuals
        resid = y_test - y_pred
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=y_pred,y=resid,mode="markers",
                                   marker=dict(color="rgba(124,58,237,0.5)",size=5),name="Residuals"))
        fig2.add_hline(y=0,line_dash="dash",line_color="#ef4444")
        fig2.update_layout(title="Residual Plot",xaxis_title="Predicted",yaxis_title="Residual",
                           template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",height=400)
        charts["residuals"] = pio.to_html(fig2, full_html=False, include_plotlyjs=False,
                                           config={"displayModeBar":False})

    return {
        "task": "regression", "model": model_name,
        "rmse": round(rmse,4), "mae": round(mae,4), "r2": round(r2,4),
        "cv_rmse": round(cv_rmse,4), "charts": charts,
    }

def run_classification(X_train, X_test, y_train, y_test, model, model_name, cv, output_dir, is_binary):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    try: y_prob = model.predict_proba(X_test)
    except: y_prob = None

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        if is_binary and y_prob is not None:
            auc = roc_auc_score(y_test, y_prob[:,1])
        elif y_prob is not None:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        else:
            auc = None
    except: auc = None

    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                                scoring="f1_weighted", n_jobs=-1)

    section("Classification Metrics")
    print(f"  Accuracy    : {c(f'{acc:.4f}','green','bold')}")
    print(f"  Precision   : {c(f'{prec:.4f}','green')}")
    print(f"  Recall      : {c(f'{rec:.4f}','green')}")
    print(f"  F1 (weight) : {c(f'{f1:.4f}','green')}")
    if auc: print(f"  ROC-AUC     : {c(f'{auc:.4f}','cyan')}")
    print(f"  CV F1       : {c(f'{cv_scores.mean():.4f}±{cv_scores.std():.4f}','cyan')}")
    print()
    print(c("  Classification Report:", "bold"))
    print(textwrap.indent(classification_report(y_test, y_pred), "    "))

    charts = {}
    if HAS_PLOTLY:
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(set(y_test.tolist()))
        fig = go.Figure(go.Heatmap(z=cm,x=[str(l) for l in labels],y=[str(l) for l in labels],
                                    colorscale="Blues",text=cm,texttemplate="%{text}",
                                    showscale=True))
        fig.update_layout(title="Confusion Matrix",xaxis_title="Predicted",yaxis_title="Actual",
                          template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",height=400)
        charts["confusion_matrix"] = pio.to_html(fig,full_html=False,include_plotlyjs=False,
                                                   config={"displayModeBar":False})

        # ROC (binary)
        if is_binary and y_prob is not None:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_prob[:,1])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(fpr),y=list(tpr),name=f"AUC={auc:.3f}",
                                       line=dict(color="#4f8ef7",width=2)))
            fig2.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(dash="dash",color="#475569"),name="Random"))
            fig2.update_layout(title="ROC Curve",xaxis_title="FPR",yaxis_title="TPR",
                               template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",height=400)
            charts["roc"] = pio.to_html(fig2,full_html=False,include_plotlyjs=False,
                                         config={"displayModeBar":False})

    return {
        "task": "classification", "model": model_name,
        "accuracy": round(acc,4), "precision": round(prec,4),
        "recall": round(rec,4), "f1": round(f1,4),
        "auc": round(auc,4) if auc else None,
        "cv_f1": round(cv_scores.mean(),4), "charts": charts,
    }

def run_clustering(X, model, model_name):
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int((labels==-1).sum())

    section("Clustering Results")
    print(f"  Clusters    : {c(n_clusters,'green','bold')}")
    print(f"  Noise pts   : {c(noise,'yellow')}")

    try:
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        valid = labels != -1
        if valid.sum() > 1 and n_clusters > 1:
            sil = silhouette_score(X[valid], labels[valid])
            db  = davies_bouldin_score(X[valid], labels[valid])
            print(f"  Silhouette  : {c(f'{sil:.4f}','green')} (higher=better, max=1)")
            print(f"  Davies-Bouldin: {c(f'{db:.4f}','green')} (lower=better)")
        else:
            sil = db = None
    except: sil = db = None

    return {"task":"clustering","model":model_name,"n_clusters":n_clusters,
            "noise_points":noise,"silhouette":round(sil,4) if sil else None,
            "davies_bouldin":round(db,4) if db else None}

def try_shap(model, X_train, X_test, feature_names, output_dir):
    try:
        import shap
        section("SHAP Feature Importance")
        print(c("  Computing SHAP values...", "dim"))
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test[:min(200,len(X_test))])
            if isinstance(shap_vals, list): shap_vals = shap_vals[1]
        except:
            try:
                explainer = shap.LinearExplainer(model, X_train[:min(500,len(X_train))])
                shap_vals = explainer.shap_values(X_test[:min(200,len(X_test))])
            except:
                explainer = shap.KernelExplainer(model.predict,
                                                  shap.sample(X_train, 50))
                shap_vals = explainer.shap_values(X_test[:min(50,len(X_test))])

        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_n = min(15, len(feature_names))
        top_idx = np.argsort(mean_abs)[-top_n:][::-1]
        print(c("\n  Top SHAP features:", "bold"))
        for i in top_idx:
            bar = "█" * int(mean_abs[i]/mean_abs.max()*20)
            print(f"    {feature_names[i]:<25} {c(bar,'cyan')} {mean_abs[i]:.4f}")

        # Save SHAP chart
        if HAS_PLOTLY:
            fig = go.Figure(go.Bar(
                x=mean_abs[top_idx[::-1]].tolist(),
                y=[feature_names[i] for i in top_idx[::-1]],
                orientation="h",marker=dict(color="#4f8ef7",line=dict(width=0))
            ))
            fig.update_layout(title="SHAP Feature Importance",
                              template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",height=max(300,top_n*28+80))
            out = os.path.join(output_dir, "shap_importance.html")
            pio.write_html(fig, out, include_plotlyjs=True, config={"displayModeBar":False})
            print(c(f"\n  SHAP chart saved → {out}", "green"))

        return {i: float(mean_abs[i]) for i in top_idx}
    except ImportError:
        print(c("  Install shap for SHAP analysis: pip install shap", "yellow"))
        return None
    except Exception as e:
        print(c(f"  SHAP failed: {e}", "red")); return None

def try_tree_importance(model, feature_names, output_dir):
    if not hasattr(model, "feature_importances_"): return
    imp = model.feature_importances_
    top_n = min(20, len(feature_names))
    top_idx = np.argsort(imp)[-top_n:][::-1]
    section("Tree Feature Importance")
    for i in top_idx:
        bar = "█" * int(imp[i]/imp.max()*20)
        print(f"    {feature_names[i]:<25} {c(bar,'magenta')} {imp[i]:.4f}")

    if HAS_PLOTLY:
        fig = go.Figure(go.Bar(
            x=imp[top_idx[::-1]].tolist(),
            y=[feature_names[i] for i in top_idx[::-1]],
            orientation="h",marker=dict(color="#7c3aed",line=dict(width=0))
        ))
        fig.update_layout(title="Feature Importance (Tree)",
                          template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",height=max(300,top_n*28+80))
        out = os.path.join(output_dir, "feature_importance.html")
        pio.write_html(fig, out, include_plotlyjs=True, config={"displayModeBar":False})
        print(c(f"\n  Feature importance chart → {out}", "green"))

def try_tuning(model_cls, params_space, X_train, y_train, task, cv):
    section("Hyperparameter Tuning")
    print(c("  Choose tuning strategy:", "bold"))
    choice, _ = ask("Strategy", default=1,
                    choices=["GridSearchCV","RandomizedSearchCV","Optuna (Bayesian)","Skip"])
    if choice == "Skip": return None

    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    scoring = "neg_root_mean_squared_error" if task=="regression" else "f1_weighted"

    if choice == "GridSearchCV":
        grid = {}
        for p, (typ, default, desc) in params_space.items():
            raw = input(c(f"  Values for {p} (comma-sep, e.g. 0.01,0.1,1.0) [{default}]: ","yellow")).strip()
            if not raw: continue
            try:
                vals = [float(x.strip()) if typ=="float" else int(x.strip())
                        for x in raw.split(",")]
                grid[p] = vals
            except: pass
        if not grid: print(c("  No grid defined, skipping.", "yellow")); return None
        print(c("  Running GridSearchCV...", "dim"))
        gs = GridSearchCV(model_cls(), grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        print(c(f"\n  Best params: {gs.best_params_}", "green"))
        print(c(f"  Best score : {gs.best_score_:.4f}", "green"))
        return gs.best_estimator_

    elif choice == "RandomizedSearchCV":
        from scipy.stats import uniform, randint
        dist = {}
        n_iter = int(ask("n_iter (trials)", default=20) or 20)
        for p, (typ, default, desc) in params_space.items():
            if typ == "float":
                raw_lo = float(ask(f"  {p} min", default=default/10 or 0.001) or default/10 or 0.001)
                raw_hi = float(ask(f"  {p} max", default=default*10) or default*10)
                dist[p] = uniform(raw_lo, raw_hi-raw_lo)
            elif typ == "int":
                raw_lo = int(ask(f"  {p} min", default=max(1,default//2)) or max(1,default//2))
                raw_hi = int(ask(f"  {p} max", default=default*3) or default*3)
                dist[p] = randint(raw_lo, raw_hi)
        if not dist: print(c("  No distributions defined, skipping.", "yellow")); return None
        print(c("  Running RandomizedSearchCV...", "dim"))
        rs = RandomizedSearchCV(model_cls(), dist, n_iter=n_iter, cv=cv,
                                scoring=scoring, n_jobs=-1, random_state=42)
        rs.fit(X_train, y_train)
        print(c(f"\n  Best params: {rs.best_params_}", "green"))
        print(c(f"  Best score : {rs.best_score_:.4f}", "green"))
        return rs.best_estimator_

    elif "Optuna" in choice:
        try:
            import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
            n_trials = int(ask("n_trials", default=50) or 50)
            def objective(trial):
                kw = {}
                for p,(typ,default,desc) in params_space.items():
                    if typ=="float": kw[p]=trial.suggest_float(p,default/20 or 0.001,default*20 or 100)
                    elif typ=="int": kw[p]=trial.suggest_int(p,max(1,default//3),default*5)
                mod = model_cls(**kw)
                scores = cross_val_score(mod,X_train,y_train,cv=cv,scoring=scoring,n_jobs=-1)
                return scores.mean()
            direction = "maximize" if task!="regression" else "maximize"  # neg RMSE is higher=better
            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            print(c(f"\n  Best params: {study.best_params}", "green"))
            print(c(f"  Best value : {study.best_value:.4f}", "green"))
            best_model = model_cls(**study.best_params)
            best_model.fit(X_train, y_train)
            return best_model
        except ImportError:
            print(c("  Install optuna: pip install optuna", "yellow")); return None

def build_model_report(results: Dict, transform_log: List, output_dir: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    charts_html = "".join([
        f'<div class="card"><h2>{k.replace("_"," ").title()}</h2><div class="chart-wrap">{v}</div></div>'
        for k,v in results.get("charts",{}).items()
    ])
    metrics_rows = "".join([
        f'<tr><td class="mono">{k}</td><td class="mono" style="color:var(--green)">{v}</td></tr>'
        for k,v in results.items() if k not in ("task","model","charts")
    ])
    steps_html = "".join([
        f'<li style="font-size:12px;color:var(--text2);padding:3px 0">{s}</li>'
        for s in transform_log
    ]) or "<li style='color:var(--text3)'>No transforms logged</li>"

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"/>
<title>Model Report — {results['model']}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{{--bg:#05070f;--panel:rgba(15,20,45,0.7);--border:rgba(79,142,247,0.12);--accent:#4f8ef7;--green:#22c55e;--text:#e2e8f0;--text2:#94a3b8;--text3:#475569;--mono:'JetBrains Mono',monospace;--sans:'Space Grotesk',sans-serif;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);padding:32px;min-height:100vh}}
body::before{{content:'';position:fixed;inset:0;pointer-events:none;background:radial-gradient(ellipse 60% 40% at 20% 10%,rgba(79,142,247,0.06),transparent 60%),radial-gradient(ellipse 50% 50% at 80% 80%,rgba(124,58,237,0.04),transparent 60%);z-index:0}}
.wrap{{max-width:1100px;margin:0 auto;position:relative;z-index:1}}
h1{{font-size:28px;font-weight:700;letter-spacing:-.5px;margin-bottom:6px}}
h1 span{{background:linear-gradient(135deg,#4f8ef7,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.meta{{color:var(--text3);font-size:11px;font-family:var(--mono);margin-bottom:28px}}
.card{{background:var(--panel);backdrop-filter:blur(16px);border:1px solid var(--border);border-radius:14px;padding:20px 22px;margin-bottom:16px}}
.card h2{{font-size:15px;font-weight:600;margin-bottom:14px}}
.tbl{{width:100%;border-collapse:collapse}}
.tbl td{{padding:9px 12px;border-bottom:1px solid rgba(255,255,255,0.04);font-size:13px}}
.mono{{font-family:var(--mono)}}
.chart-wrap{{border-radius:10px;overflow:hidden;background:rgba(5,7,15,0.4)}}
ul{{padding-left:18px}}
</style>
</head>
<body><div class="wrap">
<h1>Model Report — <span>{results['model']}</span></h1>
<div class="meta">Task: {results['task']} &nbsp;·&nbsp; Generated {ts}</div>
<div class="card"><h2>Evaluation Metrics</h2>
<table class="tbl"><tbody>{metrics_rows}</tbody></table></div>
{charts_html}
<div class="card"><h2>Preprocessing Steps Applied</h2><ul>{steps_html}</ul></div>
</div></body></html>"""

    out_path = os.path.join(output_dir, f"model_report_{results['model'].lower().replace(' ','_')}.html")
    with open(out_path, "w") as f: f.write(html)
    return out_path

# ─────────────────────────────────────────────────────────────────────
# MAIN CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MLProfiler Interactive Model Trainer")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--target", default=None, help="Target column name")
    parser.add_argument("--output", default="ml_output", help="Output directory")
    args = parser.parse_args()

    if not HAS_SKLEARN:
        print(c("✗ scikit-learn not installed. Run: pip install scikit-learn", "red")); sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    header("MLProfiler — Interactive Model Trainer v2")

    # ── 1. Load data ──────────────────────────────────────────────────
    section("Loading Data")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from mlprofiler2.core.dataset import Dataset
        ds = Dataset(args.data)
        df = ds.df
        print(c(f"  ✓ Loaded: {ds.name}", "green"))
    except Exception as e:
        print(c(f"  ✗ Failed to load: {e}", "red")); sys.exit(1)

    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Columns: {', '.join(df.columns[:8])}{'...' if len(df.columns)>8 else ''}")

    # ── 2. Target column ─────────────────────────────────────────────
    target = args.target
    if not target:
        section("Target Column")
        for i,c_ in enumerate(df.columns): print(f"  {i+1:>3}. {c_}  [{df[c_].dtype}]")
        raw = input(c("  ❯ Enter target column name or number: ","yellow")).strip()
        try: target = df.columns[int(raw)-1]
        except: target = raw
    if target not in df.columns:
        print(c(f"  ✗ '{target}' not in columns","red")); sys.exit(1)
    print(c(f"  Target: {target}","green"))

    # ── 3. Problem type ───────────────────────────────────────────────
    section("Problem Type")
    n_classes = df[target].nunique()
    is_num_target = pd.api.types.is_numeric_dtype(df[target])
    auto_task = "regression" if is_num_target and n_classes>10 else "classification" if n_classes<=20 else "regression"
    print(f"  Target unique values: {n_classes}  |  dtype: {df[target].dtype}")
    print(c(f"  Auto-detected: {auto_task.upper()}", "cyan"))
    task, _ = ask("Problem type", default=1,
                  choices=["Regression","Classification","Clustering"])
    task = task.lower()

    # ── 4. Preprocessing ─────────────────────────────────────────────
    section("Preprocessing")
    print(c("  Checking data quality...", "dim"))
    miss = df.isnull().sum().sum()
    dups = df.duplicated().sum()
    print(f"  Missing cells: {c(miss,'yellow' if miss else 'green')}  |  Duplicates: {c(dups,'yellow' if dups else 'green')}")

    transform_log = []
    if dups > 0 and confirm(f"Drop {dups} duplicates?"):
        df = df.drop_duplicates(); transform_log.append(f"Dropped {dups} duplicates")

    num_cols_with_miss = [c_ for c_ in df.select_dtypes(include=[np.number]).columns if df[c_].isnull().any() and c_ != target]
    cat_cols_with_miss = [c_ for c_ in df.select_dtypes(include=["object"]).columns if df[c_].isnull().any() and c_ != target]

    if num_cols_with_miss:
        strat, _ = ask(f"Impute {len(num_cols_with_miss)} numeric cols with missing", default=1,
                       choices=["Median","Mean","KNN","Drop rows","Skip"])
        if strat != "Skip":
            if strat == "Median": df[num_cols_with_miss] = df[num_cols_with_miss].fillna(df[num_cols_with_miss].median())
            elif strat == "Mean": df[num_cols_with_miss] = df[num_cols_with_miss].fillna(df[num_cols_with_miss].mean())
            elif strat == "KNN":
                from sklearn.impute import KNNImputer
                df[num_cols_with_miss] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols_with_miss])
            elif strat == "Drop rows": df = df.dropna(subset=num_cols_with_miss)
            transform_log.append(f"Imputed {len(num_cols_with_miss)} numeric cols with {strat}")

    if cat_cols_with_miss:
        df[cat_cols_with_miss] = df[cat_cols_with_miss].fillna("Unknown")
        transform_log.append(f"Filled {len(cat_cols_with_miss)} categorical cols with 'Unknown'")

    # ── 5. Feature prep ───────────────────────────────────────────────
    section("Feature Preparation")
    feature_cols = [c_ for c_ in df.columns if c_ != target]
    print(f"  Available features: {len(feature_cols)}")
    if confirm("Select specific features? (No = use all)", default=False):
        feature_cols = ask_multi("Select feature columns", feature_cols, default_all=True)

    X = df[feature_cols].copy()
    y = df[target].copy()
    y_clean = y.dropna()
    X = X.loc[y_clean.index]
    y = y_clean

    # Encode categoricals
    cat_feats = X.select_dtypes(include=["object","category"]).columns.tolist()
    if cat_feats:
        enc_choice, _ = ask(f"Encode {len(cat_feats)} categorical features", default=1,
                            choices=["Label Encoding","One-Hot Encoding","Drop categoricals"])
        if enc_choice == "Label Encoding":
            for c_ in cat_feats: X[c_] = LabelEncoder().fit_transform(X[c_].astype(str))
            transform_log.append(f"Label encoded: {cat_feats}")
        elif enc_choice == "One-Hot Encoding":
            X = pd.get_dummies(X, columns=cat_feats, drop_first=True)
            transform_log.append(f"One-hot encoded: {cat_feats}")
        else:
            X = X.drop(columns=cat_feats)
            transform_log.append(f"Dropped categoricals: {cat_feats}")

    # Encode target for classification
    le_target = None
    if task == "classification" and not pd.api.types.is_numeric_dtype(y):
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y.astype(str)), index=y.index)
        transform_log.append("Label encoded target")

    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    feature_names = list(X.columns)
    X_arr = X.values.astype(float)

    # Scaling
    scale_choice, _ = ask("Feature scaling", default=2,
                          choices=["StandardScaler","No scaling","MinMaxScaler","RobustScaler"])
    scaler = None
    if scale_choice != "No scaling":
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        scaler_cls = {"StandardScaler":StandardScaler,"MinMaxScaler":MinMaxScaler,"RobustScaler":RobustScaler}[scale_choice]
        scaler = scaler_cls()
        X_arr = scaler.fit_transform(X_arr)
        transform_log.append(f"Applied {scale_choice}")

    # ── 6. Train/Test split ───────────────────────────────────────────
    section("Train / Test Split")
    test_size_str = ask("Test set size", default="0.2")
    test_size = float(test_size_str)
    cv_folds = int(ask("CV folds", default="5"))

    if task == "clustering":
        X_train, X_test = X_arr, X_arr
        y_train, y_test = y.values, y.values
        print(c("  (Clustering uses full dataset)", "dim"))
    else:
        stratify = y.values if task=="classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y.values, test_size=test_size, random_state=42, stratify=stratify)
        print(c(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}", "green"))

    # ── 7. Model selection ───────────────────────────────────────────
    section("Model Selection")
    if task == "regression": registry = REGRESSION_MODELS
    elif task == "classification": registry = CLASSIFICATION_MODELS
    else: registry = CLUSTERING_MODELS

    model_names = list(registry.keys())
    model_name, _ = ask("Choose model", default=1, choices=model_names)
    model_info = registry[model_name]
    params = collect_params(model_name, model_info["params"])

    model_cls = load_class(model_info["class"])
    if model_cls is None:
        pkg = model_info["class"].split(".")[0]
        print(c(f"  ✗ {pkg} not installed. Run: pip install {pkg}", "red")); sys.exit(1)
    model = model_cls(**params)

    # ── 8. Train + Evaluate ───────────────────────────────────────────
    section(f"Training — {model_name}")
    print(c("  Fitting model...", "dim"))

    results = {}
    if task == "regression":
        results = run_regression(X_train, X_test, y_train, y_test, model, model_name, cv_folds, args.output)
    elif task == "classification":
        is_binary = len(set(y_train)) == 2
        results = run_classification(X_train, X_test, y_train, y_test, model, model_name, cv_folds, args.output, is_binary)
    else:
        results = run_clustering(X_arr, model, model_name)

    # ── 9. Feature importance / SHAP ─────────────────────────────────
    if task != "clustering":
        try_tree_importance(model, feature_names, args.output)
        if confirm("\nRun SHAP analysis?", default=True):
            try_shap(model, X_train, X_test, feature_names, args.output)

    # ── 10. Hyperparameter tuning ─────────────────────────────────────
    if task != "clustering" and confirm("\nRun hyperparameter tuning?", default=False):
        tuned = try_tuning(model_cls, model_info["params"], X_train, y_train, task, cv_folds)
        if tuned is not None:
            model = tuned
            print(c("  Re-evaluating with tuned model...", "dim"))
            if task == "regression":
                results = run_regression(X_train, X_test, y_train, y_test, model, model_name+" (tuned)", cv_folds, args.output)
            else:
                is_binary = len(set(y_train)) == 2
                results = run_classification(X_train, X_test, y_train, y_test, model, model_name+" (tuned)", cv_folds, args.output, is_binary)

    # ── 11. Save model ────────────────────────────────────────────────
    section("Saving Outputs")
    model_path = os.path.join(args.output, f"model_{model_name.lower().replace(' ','_')}.pkl")
    with open(model_path, "wb") as f: pickle.dump(model, f)
    print(c(f"  Model saved → {model_path}", "green"))

    report_path = build_model_report(results, transform_log, args.output)
    print(c(f"  HTML report → {report_path}", "green"))

    if HAS_PLOTLY:
        try: webbrowser.open(f"file://{os.path.abspath(report_path)}")
        except: pass

    header("Training Complete!")
    print(f"  Model:   {c(model_name, 'cyan', 'bold')}")
    print(f"  Task:    {c(task.title(), 'cyan')}")
    print(f"  Output:  {c(args.output, 'dim')}")
    print()

if __name__ == "__main__":
    main()