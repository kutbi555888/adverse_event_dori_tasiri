from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Text normalize helpers
# -----------------------------
_SPLIT_SEMI = re.compile(r"\s*;\s*")
_MULTI_SPACE = re.compile(r"\s+")


def normalize_symptom_text(x: str) -> str:
    """
    - delimiter: faqat ';'
    - comma(,) bo‘yicha split YO‘Q
    - trim + multi-space -> 1 space
    - bo‘sh tokenlar tashlanadi
    - dedup (case-insensitive), order saqlanadi
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""

    parts = _SPLIT_SEMI.split(s)
    out = []
    seen = set()
    for t in parts:
        t = _MULTI_SPACE.sub(" ", t.strip())
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return "; ".join(out)


def split_terms(clean_text: str) -> Sequence[str]:
    if clean_text is None:
        return []
    s = str(clean_text).strip()
    if not s:
        return []
    return [t.strip() for t in _SPLIT_SEMI.split(s) if t.strip()]


# -----------------------------
# Robust root + CSV finder
# -----------------------------
RAW_FOLDER_CANDIDATES = ("Raw_data", "RaW_data", "raw_data", "RAW_DATA")
DATA_FOLDER_CANDIDATES = ("Data", "data", "DATA")


def find_csv_in_parents(filename: str, start: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Notebooks/Data tuzog‘iga tushmaslik uchun:
    PROJECT_ROOT ni 'Data bor' bilan emas, aynan
    Data/Raw_data/<filename> mavjud bo‘lgan parentdan topadi.

    Returns:
      (project_root, csv_path)
    """
    if start is None:
        start = Path.cwd()

    checked = []
    for p in [start] + list(start.parents):
        for data_name in DATA_FOLDER_CANDIDATES:
            for raw_name in RAW_FOLDER_CANDIDATES:
                cand = p / data_name / raw_name / filename
                checked.append(cand)
                if cand.exists():
                    return p, cand

    msg = (
        f"CSV topilmadi: {filename}\n"
        f"Start: {start.resolve()}\n"
        f"Oxirgi 10 tekshirilgan yo‘l:\n" + "\n".join(str(x) for x in checked[-10:])
    )
    raise FileNotFoundError(msg)


# -----------------------------
# y_* helpers
# -----------------------------
def get_y_cols(df: pd.DataFrame) -> list[str]:
    """
    y_ bilan boshlanadigan ustunlar:
    - y_labels emas
    - y_sum / label_sum emas (pipeline buzilmasin)
    """
    y_cols = [
        c for c in df.columns
        if c.startswith("y_") and c not in ("y_labels", "y_sum", "label_sum")
    ]
    return sorted(y_cols)


def safe_read_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, low_memory=False, encoding="latin1")


def enforce_binary_y(df: pd.DataFrame, y_cols: Sequence[str]) -> None:
    for c in y_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[list(y_cols)] = df[list(y_cols)].fillna(0).astype(int)


def compute_label_sum(df: pd.DataFrame, y_cols: Sequence[str]) -> pd.Series:
    return df[list(y_cols)].sum(axis=1).astype(int)


def rebuild_y_labels(df: pd.DataFrame, y_cols: Sequence[str]) -> pd.Series:
    label_names = [c.replace("y_", "", 1) for c in y_cols]
    arr = df[list(y_cols)].to_numpy(dtype=np.int8)
    sums = arr.sum(axis=1)

    out = np.empty(len(df), dtype=object)
    out[:] = np.nan

    nz = np.where(sums > 0)[0]
    for i in nz:
        idxs = np.flatnonzero(arr[i])
        out[i] = "; ".join(label_names[j] for j in idxs)

    return pd.Series(out, index=df.index, name="y_labels")


def zero_label_top_terms(
    df: pd.DataFrame,
    clean_col: str,
    label_sum_col: str = "label_sum",
    top_n: int = 2000,
) -> pd.DataFrame:
    mask0 = (df[label_sum_col] == 0)
    counter = Counter()

    for text in df.loc[mask0, clean_col].astype(str).tolist():
        for term in split_terms(text):
            t = term.strip().lower()
            if t:
                counter[t] += 1

    rows = counter.most_common(top_n)
    return pd.DataFrame(rows, columns=["term_lc", "freq"])


# -----------------------------
# Main runner
# -----------------------------
@dataclass
class ImprovementConfig:
    dataset_filename: str = "faers_25Q4_targets_multilabel_v2.csv"
    text_col: str = "REAC_pt_symptom_v2"
    clean_col: str = "REAC_pt_symptom_v2_clean"
    label_sum_col: str = "label_sum"

    out_subdir: str = "improvement_preprocess_v2"
    out_filename: str = "faers_25Q4_targets_multilabel_v2_textclean_v2.csv"

    top_terms_n: int = 2000


def run_improvement_preprocess(
    input_csv: Optional[str] = None,
    cfg: Optional[ImprovementConfig] = None,
) -> Tuple[Path, Path]:
    """
    08 uchun:
    - text clean
    - y_labels rebuild (y_* dan)
    - label_sum (y_sum emas!)
    - zero-label top terms audit

    Returns:
      (saved_dataset_path, saved_zero_terms_path)
    """
    cfg = cfg or ImprovementConfig()

    # 1) project_root + IN_CSV
    if input_csv:
        in_csv = Path(input_csv)
        if not in_csv.exists():
            raise FileNotFoundError(f"input_csv topilmadi: {in_csv}")
        # outputni input_csv yoniga emas, project ichiga saqlash uchun:
        # input_csv Data/Raw_data ichida bo‘lsa, project_root topiladi
        try:
            project_root, _ = find_csv_in_parents(in_csv.name, start=Path.cwd())
        except FileNotFoundError:
            project_root = Path.cwd()
    else:
        project_root, in_csv = find_csv_in_parents(cfg.dataset_filename, start=Path.cwd())

    # 2) out dir
    out_dir = project_root / "Data" / "Processed" / cfg.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / cfg.out_filename
    out_zero_terms = out_dir / "zero_label_top_terms_v2.csv"

    print("PROJECT_ROOT:", project_root.resolve())
    print("IN_CSV:", in_csv.resolve())
    print("OUT_DIR:", out_dir.resolve())

    # 3) Load
    df = safe_read_csv(in_csv)
    print("Loaded:", df.shape)

    if cfg.text_col not in df.columns:
        raise ValueError(f"TEXT_COL topilmadi: {cfg.text_col}")

    # 4) y cols
    y_cols = get_y_cols(df)
    if not y_cols:
        raise ValueError("y_* ustunlari topilmadi.")
    print("Num labels:", len(y_cols))

    # 5) text clean
    df[cfg.text_col] = df[cfg.text_col].fillna("").astype(str).str.strip()
    df[cfg.clean_col] = df[cfg.text_col].apply(normalize_symptom_text)

    # 6) y enforce + label_sum
    enforce_binary_y(df, y_cols)
    df[cfg.label_sum_col] = compute_label_sum(df, y_cols)

    # 7) y_labels rebuild
    df["y_labels"] = rebuild_y_labels(df, y_cols)

    bad1 = int(((df[cfg.label_sum_col] == 0) & (df["y_labels"].notna())).sum())
    bad2 = int(((df[cfg.label_sum_col] > 0) & (df["y_labels"].isna())).sum())
    print("Sanity bad (0 but has labels):", bad1)
    print("Sanity bad (>0 but NaN labels):", bad2)

    # 8) zero-label top terms
    top_terms = zero_label_top_terms(df, cfg.clean_col, cfg.label_sum_col, cfg.top_terms_n)
    top_terms.to_csv(out_zero_terms, index=False)
    print("Saved:", out_zero_terms.resolve())

    # 9) Save dataset
    df.to_csv(out_csv, index=False)
    print("Saved dataset:", out_csv.resolve())

    return out_csv, out_zero_terms