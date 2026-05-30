"""Dataset — Universal data loader."""
import json
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class Dataset:
    SUPPORTED = {
        ".csv": "_csv", ".tsv": "_tsv", ".json": "_json", ".jsonl": "_jsonl",
        ".xlsx": "_excel", ".xls": "_excel", ".parquet": "_parquet",
        ".feather": "_feather", ".xml": "_xml", ".pkl": "_pickle",
        ".pickle": "_pickle", ".h5": "_hdf", ".hdf5": "_hdf",
    }

    def __init__(self, data, name=None, target=None, **kw):
        self._kw = kw
        self.target = target
        self.source_path = None
        self.df = self._load(data)
        self.name = name or self._infer_name(data)

    def _load(self, data):
        if isinstance(data, pd.DataFrame): return data.copy()
        if isinstance(data, np.ndarray): return pd.DataFrame(data)
        if isinstance(data, (dict, list)): return pd.DataFrame(data)
        p = Path(data)
        if not p.exists(): raise FileNotFoundError(f"Not found: {data}")
        self.source_path = str(p.resolve())
        ext = p.suffix.lower()
        if ext not in self.SUPPORTED: raise ValueError(f"Unsupported: {ext}")
        return getattr(self, self.SUPPORTED[ext])(p)

    def _infer_name(self, data):
        return Path(data).stem if isinstance(data, (str, Path)) else "dataset"

    def _csv(self, p): return pd.read_csv(p, **self._kw)
    def _tsv(self, p): return pd.read_csv(p, sep="\t", **self._kw)
    def _excel(self, p): return pd.read_excel(p, sheet_name=self._kw.pop("sheet_name",0), **self._kw)
    def _parquet(self, p): return pd.read_parquet(p, **self._kw)
    def _feather(self, p): return pd.read_feather(p, **self._kw)
    def _pickle(self, p): return pd.read_pickle(p)
    def _hdf(self, p):
        store = pd.HDFStore(str(p), mode="r"); keys = store.keys(); store.close()
        return pd.read_hdf(p, key=keys[0])
    def _json(self, p):
        try: return pd.read_json(p, **self._kw)
        except:
            with open(p) as f: raw = json.load(f)
            if isinstance(raw, list): return pd.json_normalize(raw)
            for k in ["data","records","rows","results","items"]:
                if k in raw and isinstance(raw[k], list): return pd.json_normalize(raw[k])
            return pd.json_normalize([raw])
    def _jsonl(self, p):
        return pd.json_normalize([json.loads(l) for l in open(p) if l.strip()])
    def _xml(self, p):
        try: return pd.read_xml(p)
        except:
            import xml.etree.ElementTree as ET
            root = ET.parse(p).getroot()
            return pd.DataFrame([{s.tag: s.text for s in c} for c in root])