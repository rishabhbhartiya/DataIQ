"""Transformer — applies preprocessing transformations."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional

warnings.filterwarnings("ignore")


class Transformer:
    ACTIONS = {
        "drop_high_missing","drop_duplicates","keep_first",
        "impute_mean","impute_median","impute_mode","impute_unknown","impute_knn",
        "cap_outliers","remove_outliers",
        "log_transform","sqrt_transform","yeo_johnson",
        "scale_standard","scale_minmax","scale_robust",
        "encode_onehot","encode_label","encode_target","encode_frequency",
        "skip",
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def apply(self, action_id: str, cols: Optional[List[str]] = None, target: Optional[str] = None) -> pd.DataFrame:
        if action_id not in self.ACTIONS:
            raise ValueError(f"Unknown action: {action_id}")
        if action_id == "skip": return self.df.copy()
        handler = getattr(self, f"_{action_id}")
        return handler(self.df.copy(), cols, target)

    def _drop_high_missing(self, df, cols, tgt):
        if cols: return df.drop(columns=[c for c in cols if c in df.columns])
        thresh = int(len(df)*0.4)
        return df.dropna(thresh=thresh, axis=1)

    def _drop_duplicates(self, df, cols, tgt): return df.drop_duplicates()
    def _keep_first(self, df, cols, tgt): return df.drop_duplicates(keep="first")

    def _impute_mean(self, df, cols, tgt):
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        for c in cols:
            if c in df: df[c] = df[c].fillna(df[c].mean())
        return df

    def _impute_median(self, df, cols, tgt):
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        for c in cols:
            if c in df: df[c] = df[c].fillna(df[c].median())
        return df

    def _impute_mode(self, df, cols, tgt):
        cols = cols or df.columns.tolist()
        for c in cols:
            if c in df:
                m = df[c].mode()
                if len(m): df[c] = df[c].fillna(m[0])
        return df

    def _impute_unknown(self, df, cols, tgt):
        cols = cols or df.select_dtypes(include=["object","category"]).columns.tolist()
        for c in cols:
            if c in df: df[c] = df[c].fillna("Unknown")
        return df

    def _impute_knn(self, df, cols, tgt):
        try:
            from sklearn.impute import KNNImputer
            cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
            vc = [c for c in cols if c in df]
            df[vc] = KNNImputer(n_neighbors=5).fit_transform(df[vc])
        except: return self._impute_median(df, cols, tgt)
        return df

    def _cap_outliers(self, df, cols, tgt):
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        for c in cols:
            if c not in df: continue
            q1,q3 = df[c].quantile(.25), df[c].quantile(.75)
            df[c] = df[c].clip(q1-1.5*(q3-q1), q3+1.5*(q3-q1))
        return df

    def _remove_outliers(self, df, cols, tgt):
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        mask = pd.Series([True]*len(df), index=df.index)
        for c in cols:
            if c not in df: continue
            q1,q3 = df[c].quantile(.25), df[c].quantile(.75); iqr=q3-q1
            mask &= (df[c]>=q1-1.5*iqr)&(df[c]<=q3+1.5*iqr)
        return df[mask]

    def _log_transform(self, df, cols, tgt):
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        for c in cols:
            if c in df: df[c] = np.log1p(df[c].clip(lower=0))
        return df

    def _sqrt_transform(self, df, cols, tgt):
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        for c in cols:
            if c in df: df[c] = np.sqrt(df[c].clip(lower=0))
        return df

    def _yeo_johnson(self, df, cols, tgt):
        try:
            from sklearn.preprocessing import PowerTransformer
            cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
            vc = [c for c in cols if c in df and df[c].notna().sum()>1]
            if vc:
                filled = df[vc].fillna(df[vc].median())
                df[vc] = PowerTransformer(method="yeo-johnson").fit_transform(filled)
        except: return self._log_transform(df, cols, tgt)
        return df

    def _scale_standard(self, df, cols, tgt):
        try:
            from sklearn.preprocessing import StandardScaler
            cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
            vc = [c for c in cols if c in df]
            df[vc] = StandardScaler().fit_transform(df[vc].fillna(0))
        except: pass
        return df

    def _scale_minmax(self, df, cols, tgt):
        try:
            from sklearn.preprocessing import MinMaxScaler
            cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
            vc = [c for c in cols if c in df]
            df[vc] = MinMaxScaler().fit_transform(df[vc].fillna(0))
        except: pass
        return df

    def _scale_robust(self, df, cols, tgt):
        try:
            from sklearn.preprocessing import RobustScaler
            cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
            vc = [c for c in cols if c in df]
            df[vc] = RobustScaler().fit_transform(df[vc].fillna(df[vc].median()))
        except: pass
        return df

    def _encode_onehot(self, df, cols, tgt):
        cols = cols or df.select_dtypes(include=["object","category"]).columns.tolist()
        vc = [c for c in cols if c in df]
        return pd.get_dummies(df, columns=vc, drop_first=False) if vc else df

    def _encode_label(self, df, cols, tgt):
        cols = cols or df.select_dtypes(include=["object","category"]).columns.tolist()
        for c in cols:
            if c in df: df[c] = pd.factorize(df[c])[0]
        return df

    def _encode_frequency(self, df, cols, tgt):
        cols = cols or df.select_dtypes(include=["object","category"]).columns.tolist()
        for c in cols:
            if c in df:
                freq = df[c].value_counts(normalize=True)
                df[c] = df[c].map(freq)
        return df

    def _encode_target(self, df, cols, tgt):
        if tgt and tgt in df.columns:
            for c in (cols or []):
                if c in df:
                    means = df.groupby(c)[tgt].mean()
                    df[c] = df[c].map(means)
        else:
            return self._encode_frequency(df, cols, tgt)
        return df