from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.util import hash_pandas_object


@dataclass(frozen=True)
class BaselineEDAConfig:
    split_dir: Path
    tables_out: Path
    visuals_out: Path
    text_col: str = "REAC_pt_symptom_v2"
    id_col: str = "primaryid"


def _infer_y_cols(df: pd.DataFrame) -> List[str]:
    y_cols = sorted([c for c in df.columns if c.startswith("y_") and c != "y_labels"])
    if not y_cols:
        raise ValueError("y_* label ustunlari topilmadi.")
    return y_cols


def _hash_text(s: pd.Series) -> np.ndarray:
    return hash_pandas_object(s.fillna("").astype(str), index=False).to_numpy()


def load_splits(cfg: BaselineEDAConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_csv = cfg.split_dir / "train.csv"
    val_csv   = cfg.split_dir / "val.csv"
    test_csv  = cfg.split_dir / "test.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv topilmadi: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"val.csv topilmadi: {val_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv topilmadi: {test_csv}")

    df_train = pd.read_csv(train_csv, low_memory=False)
    df_val   = pd.read_csv(val_csv, low_memory=False)
    df_test  = pd.read_csv(test_csv, low_memory=False)
    return df_train, df_val, df_test


def validate_splits(cfg: BaselineEDAConfig, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> Dict:
    # columns
    if cfg.text_col not in df_train.columns:
        raise ValueError(f"text_col topilmadi: {cfg.text_col}")

    y_cols = _infer_y_cols(df_train)

    # empty text
    def empty_text(df: pd.DataFrame) -> int:
        return int((df[cfg.text_col].fillna("").astype(str).str.strip() == "").sum())

    # 0-label
    def zero_label(df: pd.DataFrame) -> int:
        return int((df[y_cols].sum(axis=1) == 0).sum())

    # y binary check
    vals = set(pd.unique(pd.concat([df_train[y_cols], df_val[y_cols], df_test[y_cols]], axis=0).to_numpy().ravel()))
    if not vals.issubset({0, 1}):
        bad = sorted([v for v in vals if v not in {0, 1}])
        raise ValueError(f"Y ichida 0/1 dan boshqa qiymat bor: {bad}")

    # leakage by exact text hash
    htr = set(_hash_text(df_train[cfg.text_col]))
    hva = set(_hash_text(df_val[cfg.text_col]))
    hte = set(_hash_text(df_test[cfg.text_col]))

    leak_tv = int(len(htr & hva))
    leak_tt = int(len(htr & hte))
    leak_vt = int(len(hva & hte))

    # primaryid overlap (if exists)
    id_overlap = {"train_val": None, "train_test": None, "val_test": None}
    if cfg.id_col in df_train.columns and cfg.id_col in df_val.columns and cfg.id_col in df_test.columns:
        it = set(df_train[cfg.id_col].values)
        iv = set(df_val[cfg.id_col].values)
        ie = set(df_test[cfg.id_col].values)
        id_overlap = {
            "train_val": int(len(it & iv)),
            "train_test": int(len(it & ie)),
            "val_test": int(len(iv & ie)),
        }

    meta = {
        "text_col": cfg.text_col,
        "num_labels": int(len(y_cols)),
        "y_cols": y_cols,
        "rows_train": int(len(df_train)),
        "rows_val": int(len(df_val)),
        "rows_test": int(len(df_test)),
        "empty_text_train": empty_text(df_train),
        "empty_text_val": empty_text(df_val),
        "empty_text_test": empty_text(df_test),
        "zero_label_train": zero_label(df_train),
        "zero_label_val": zero_label(df_val),
        "zero_label_test": zero_label(df_test),
        "exact_text_leak_train_val": leak_tv,
        "exact_text_leak_train_test": leak_tt,
        "exact_text_leak_val_test": leak_vt,
        "primaryid_overlap": id_overlap,
    }
    return meta


def prevalence_table(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, y_cols: List[str]) -> pd.DataFrame:
    def prev(df: pd.DataFrame) -> pd.Series:
        return (df[y_cols].mean() * 100).round(3)

    pt = pd.DataFrame({
        "label": y_cols,
        "train_%": prev(df_train).values,
        "val_%": prev(df_val).values,
        "test_%": prev(df_test).values,
        "overall_%": prev(pd.concat([df_train, df_val, df_test], axis=0)).values,
    })
    pt["label"] = pt["label"].str.replace("^y_", "", regex=True)
    pt = pt.sort_values("overall_%", ascending=False).reset_index(drop=True)
    return pt


def save_prevalence_plot(prev_table: pd.DataFrame, out_path: Path, top_n: int = 16) -> None:
    top_n = min(top_n, len(prev_table))
    plt.figure(figsize=(10, 4))
    plt.bar(prev_table["label"].head(top_n), prev_table["overall_%"].head(top_n))
    plt.xticks(rotation=45, ha="right")
    plt.title("Label prevalence (overall, top)")
    plt.ylabel("% of reports")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def run_baseline_eda_report(cfg: BaselineEDAConfig) -> Dict[str, Path]:
    cfg.tables_out.mkdir(parents=True, exist_ok=True)
    cfg.visuals_out.mkdir(parents=True, exist_ok=True)

    df_train, df_val, df_test = load_splits(cfg)
    y_cols = _infer_y_cols(df_train)

    # 1) validation meta
    meta = validate_splits(cfg, df_train, df_val, df_test)

    # 2) prevalence table
    prev_tbl = prevalence_table(df_train, df_val, df_test, y_cols)

    # save
    p_meta = cfg.tables_out / "baseline_eda_meta.json"
    p_prev = cfg.tables_out / "label_prevalence_by_split.csv"
    p_fig  = cfg.visuals_out / "label_prevalence_overall_top.png"

    p_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    prev_tbl.to_csv(p_prev, index=False, encoding="utf-8-sig")
    save_prevalence_plot(prev_tbl, p_fig)

    return {"meta": p_meta, "prevalence_csv": p_prev, "prevalence_plot": p_fig}