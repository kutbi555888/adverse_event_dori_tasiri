from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class BaselinePreprocessConfig:
    split_dir: Path
    out_dir: Path

    text_col: str = "REAC_pt_symptom_v2"
    id_col: str = "primaryid"

    # TF-IDF params (BEST baseline)
    char_ngram_range: Tuple[int, int] = (3, 5)
    char_min_df: int = 3
    char_max_df: float = 0.95
    char_max_features: int = 180_000

    word_ngram_range: Tuple[int, int] = (1, 2)
    word_min_df: int = 5
    word_max_df: float = 0.95
    word_max_features: int = 90_000
    word_token_pattern: str = r"(?u)\b[a-z0-9][a-z0-9\-\_]+\b"

    sublinear_tf: bool = True
    dtype: str = "float32"  # store sparse as float32 for memory


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[,\t\r\n]+")


def normalize_text(s: str) -> str:
    """
    Baseline text normalization:
    - lowercase
    - strip
    - ';' -> space (FAERS PT listni user matniga yaqinlashtiradi)
    - collapse spaces
    """
    s = "" if s is None else str(s)
    s = s.lower().strip()
    s = s.replace(";", " ")
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s)
    return s


def infer_y_cols(df: pd.DataFrame) -> List[str]:
    """
    y_* ustunlari: y_labels emas.
    """
    y_cols = [c for c in df.columns if c.startswith("y_") and c != "y_labels"]
    y_cols = sorted(y_cols)
    if not y_cols:
        raise ValueError("y_* label ustunlari topilmadi.")
    return y_cols


def load_splits(cfg: BaselinePreprocessConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def build_vectorizers(cfg: BaselinePreprocessConfig) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    dtype = np.float32 if cfg.dtype == "float32" else np.float64

    tfidf_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.char_ngram_range,
        min_df=cfg.char_min_df,
        max_df=cfg.char_max_df,
        sublinear_tf=cfg.sublinear_tf,
        dtype=dtype,
        max_features=cfg.char_max_features,
    )

    tfidf_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=cfg.word_ngram_range,
        min_df=cfg.word_min_df,
        max_df=cfg.word_max_df,
        sublinear_tf=cfg.sublinear_tf,
        dtype=dtype,
        max_features=cfg.word_max_features,
        token_pattern=cfg.word_token_pattern,
    )

    return tfidf_char, tfidf_word


def to_xy(df: pd.DataFrame, text_col: str, y_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if text_col not in df.columns:
        raise ValueError(f"text_col topilmadi: {text_col}")

    X_text = df[text_col].map(normalize_text).values
    Y = df[y_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.int8).values

    # minimal sanity
    if (pd.Series(X_text).astype(str).str.strip() == "").any():
        raise ValueError("Bo'sh text bor. Splitlar toza bo‘lishi kerak (empty_text=0).")
    if (Y.sum(axis=1) == 0).any():
        raise ValueError("0-label row bor. Splitlar toza bo‘lishi kerak (zero_label=0).")

    return X_text, Y


def fit_transform_all(
    cfg: BaselinePreprocessConfig,
    Xtr_text: np.ndarray,
    Xva_text: np.ndarray,
    Xte_text: np.ndarray,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, Dict]:
    tfidf_char, tfidf_word = build_vectorizers(cfg)

    # Fit on TRAIN only
    Xtr_c = tfidf_char.fit_transform(Xtr_text)
    Xva_c = tfidf_char.transform(Xva_text)
    Xte_c = tfidf_char.transform(Xte_text)

    Xtr_w = tfidf_word.fit_transform(Xtr_text)
    Xva_w = tfidf_word.transform(Xva_text)
    Xte_w = tfidf_word.transform(Xte_text)

    Xtr = hstack([Xtr_c, Xtr_w], format="csr")
    Xva = hstack([Xva_c, Xva_w], format="csr")
    Xte = hstack([Xte_c, Xte_w], format="csr")

    bundle = {
        "text_col": cfg.text_col,
        "tfidf_char": tfidf_char,
        "tfidf_word": tfidf_word,
        "version": "baseline_preprocess_v2_noleak",
    }
    return Xtr, Xva, Xte, bundle


def save_artifacts(
    cfg: BaselinePreprocessConfig,
    Xtr: sparse.csr_matrix,
    Xva: sparse.csr_matrix,
    Xte: sparse.csr_matrix,
    Ytr: np.ndarray,
    Yva: np.ndarray,
    Yte: np.ndarray,
    y_cols: List[str],
    bundle: Dict,
    ids: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Path]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "X_train": cfg.out_dir / "X_train.npz",
        "X_val":   cfg.out_dir / "X_val.npz",
        "X_test":  cfg.out_dir / "X_test.npz",
        "Y_train": cfg.out_dir / "Y_train.npy",
        "Y_val":   cfg.out_dir / "Y_val.npy",
        "Y_test":  cfg.out_dir / "Y_test.npy",
        "bundle":  cfg.out_dir / "tfidf_bundle.joblib",
        "meta":    cfg.out_dir / "preprocess_meta.json",
    }

    sparse.save_npz(paths["X_train"], Xtr)
    sparse.save_npz(paths["X_val"],   Xva)
    sparse.save_npz(paths["X_test"],  Xte)

    np.save(paths["Y_train"], Ytr)
    np.save(paths["Y_val"],   Yva)
    np.save(paths["Y_test"],  Yte)

    if ids:
        for k, arr in ids.items():
            p = cfg.out_dir / f"{k}.npy"
            np.save(p, arr)
            paths[k] = p

    # add labels into bundle (important for 06 stage)
    bundle = dict(bundle)
    bundle["y_cols"] = y_cols
    joblib.dump(bundle, paths["bundle"])

    meta = {
        "text_col": cfg.text_col,
        "num_labels": int(len(y_cols)),
        "y_cols": y_cols,
        "rows_train": int(Xtr.shape[0]),
        "rows_val": int(Xva.shape[0]),
        "rows_test": int(Xte.shape[0]),
        "num_features_total": int(Xtr.shape[1]),
        "num_features_char": int(len(bundle["tfidf_char"].get_feature_names_out())),
        "num_features_word": int(len(bundle["tfidf_word"].get_feature_names_out())),
        "params": {
            "char": {
                "analyzer": "char_wb",
                "ngram_range": list(cfg.char_ngram_range),
                "min_df": cfg.char_min_df,
                "max_df": cfg.char_max_df,
                "max_features": cfg.char_max_features,
                "sublinear_tf": cfg.sublinear_tf,
                "dtype": cfg.dtype,
            },
            "word": {
                "analyzer": "word",
                "ngram_range": list(cfg.word_ngram_range),
                "min_df": cfg.word_min_df,
                "max_df": cfg.word_max_df,
                "max_features": cfg.word_max_features,
                "token_pattern": cfg.word_token_pattern,
                "sublinear_tf": cfg.sublinear_tf,
                "dtype": cfg.dtype,
            },
        },
    }
    paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return paths


def run_baseline_preprocess(cfg: BaselinePreprocessConfig) -> Dict[str, Path]:
    df_train, df_val, df_test = load_splits(cfg)

    y_cols = infer_y_cols(df_train)

    # X/Y
    Xtr_text, Ytr = to_xy(df_train, cfg.text_col, y_cols)
    Xva_text, Yva = to_xy(df_val,   cfg.text_col, y_cols)
    Xte_text, Yte = to_xy(df_test,  cfg.text_col, y_cols)

    # Features
    Xtr, Xva, Xte, bundle = fit_transform_all(cfg, Xtr_text, Xva_text, Xte_text)

    # IDs (optional)
    ids = None
    if cfg.id_col in df_train.columns:
        ids = {
            "id_train": df_train[cfg.id_col].values,
            "id_val":   df_val[cfg.id_col].values,
            "id_test":  df_test[cfg.id_col].values,
        }

    paths = save_artifacts(cfg, Xtr, Xva, Xte, Ytr, Yva, Yte, y_cols, bundle, ids=ids)
    return paths