from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer


# ---------- helpers ----------
_TERM_SPLIT = re.compile(r"\s*;\s*")

EXCLUDE_Y_COLS_DEFAULT = {"y_labels", "y_sum", "label_sum"}


def find_project_root_and_split(
    start: Optional[Path] = None,
    split_candidates: Sequence[str] = ("splits_multilabel_noleakage",),
) -> Tuple[Path, Path]:
    """
    PROJECT_ROOT ni "Data bor" bilan emas,
    aynan Data/Processed/<split>/train|val|test.csv borligi bilan topadi.
    """
    start = start or Path.cwd()
    checked = []

    for p in [start] + list(start.parents):
        processed = p / "Data" / "Processed"
        if not processed.exists():
            continue

        for name in split_candidates:
            d = processed / name
            checked.append(d)
            if (d / "train.csv").exists() and (d / "val.csv").exists() and (d / "test.csv").exists():
                return p, d

    raise FileNotFoundError(
        "Split papka topilmadi (parents bo‘ylab qidirildi).\n"
        f"Start: {start.resolve()}\n"
        "Tekshirilgan processed/split papkalar (oxirgi 10):\n"
        + "\n".join(str(x) for x in checked[-10:])
    )


def get_y_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    y_cols = [c for c in df.columns if c.startswith("y_") and c not in exclude]
    return sorted(y_cols)


def meta_features(texts: Sequence[str]) -> sparse.csr_matrix:
    """
    3 ta meta feature:
      - log1p_len
      - n_terms
      - n_uniq_terms
    """
    lens = []
    n_terms = []
    n_uniq_terms = []

    for s in texts:
        s = (s or "").strip()
        lens.append(len(s))

        if not s:
            n_terms.append(0)
            n_uniq_terms.append(0)
            continue

        terms = [t.strip().lower() for t in _TERM_SPLIT.split(s) if t.strip()]
        n_terms.append(len(terms))
        n_uniq_terms.append(len(set(terms)))

    lens = np.array(lens, dtype=np.float32).reshape(-1, 1)
    n_terms = np.array(n_terms, dtype=np.float32).reshape(-1, 1)
    n_uniq_terms = np.array(n_uniq_terms, dtype=np.float32).reshape(-1, 1)

    feats = np.hstack([np.log1p(lens), n_terms, n_uniq_terms]).astype(np.float32)
    return sparse.csr_matrix(feats)


def make_featurizer() -> FeatureUnion:
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.98,
        sublinear_tf=True,
        norm="l2",
    )

    meta_transformer = FunctionTransformer(
        lambda x: meta_features(list(x)),
        validate=False,
    )

    return FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_tfidf),
            ("char_tfidf", char_tfidf),
            ("meta", meta_transformer),
        ],
        n_jobs=1,  # Windows/OneDrive uchun xavfsiz
    )


# ---------- config ----------
@dataclass
class FeatureEngineeringConfig:
    split_candidates: tuple[str, ...] = ("splits_multilabel_noleakage",)
    text_col: str = "REAC_pt_symptom_v2"
    fe_version: str = "fe_v1"
    exclude_y_cols: set[str] = None  # type: ignore
    save_ids: bool = True
    id_candidates: tuple[str, ...] = ("primaryid", "caseid", "CASEID", "PRIMARYID", "safetyreportid")

    def __post_init__(self):
        if self.exclude_y_cols is None:
            self.exclude_y_cols = set(EXCLUDE_Y_COLS_DEFAULT)


# ---------- main ----------
def run_feature_engineering(
    cfg: FeatureEngineeringConfig,
) -> dict:
    # 1) find root + split
    project_root, split_dir = find_project_root_and_split(
        start=Path.cwd(),
        split_candidates=cfg.split_candidates,
    )
    print("PROJECT_ROOT:", project_root.resolve())
    print("SPLIT_DIR:", split_dir.resolve())

    # 2) load splits
    train_df = pd.read_csv(split_dir / "train.csv", low_memory=False)
    val_df = pd.read_csv(split_dir / "val.csv", low_memory=False)
    test_df = pd.read_csv(split_dir / "test.csv", low_memory=False)

    if cfg.text_col not in train_df.columns:
        raise ValueError(f"TEXT_COL topilmadi: {cfg.text_col}")

    for d in (train_df, val_df, test_df):
        d[cfg.text_col] = d[cfg.text_col].fillna("").astype(str).str.strip()

    # 3) y_cols
    y_cols = get_y_cols(train_df, cfg.exclude_y_cols)
    if not y_cols:
        raise ValueError("y_* ustunlari topilmadi (train.csv).")

    # check val/test
    for name, d in [("val", val_df), ("test", test_df)]:
        miss = [c for c in y_cols if c not in d.columns]
        if miss:
            raise ValueError(f"y_cols mismatch in {name}: {miss[:10]}")

    print("Num labels:", len(y_cols))

    # 4) featurizer
    featurizer = make_featurizer()

    # 5) fit/transform
    X_train = featurizer.fit_transform(train_df[cfg.text_col].tolist())
    X_val = featurizer.transform(val_df[cfg.text_col].tolist())
    X_test = featurizer.transform(test_df[cfg.text_col].tolist())

    print("X_train:", X_train.shape, "nnz:", X_train.nnz)
    print("X_val  :", X_val.shape, "nnz:", X_val.nnz)
    print("X_test :", X_test.shape, "nnz:", X_test.nnz)

    # 6) Y arrays
    Y_train = train_df[y_cols].astype(np.int8).to_numpy()
    Y_val = val_df[y_cols].astype(np.int8).to_numpy()
    Y_test = test_df[y_cols].astype(np.int8).to_numpy()

    # 7) save artifacts
    art_dir = project_root / "artifacts" / "feature_engineering" / cfg.fe_version
    art_dir.mkdir(parents=True, exist_ok=True)

    featurizer_path = art_dir / "featurizer_union_word_char_meta.joblib"
    joblib.dump(featurizer, featurizer_path)

    # 8) save engineered data
    engineered_dir = project_root / "Data" / "Engineered_data" / cfg.fe_version
    engineered_dir.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(engineered_dir / "X_train.npz", X_train)
    sparse.save_npz(engineered_dir / "X_val.npz", X_val)
    sparse.save_npz(engineered_dir / "X_test.npz", X_test)

    np.save(engineered_dir / "Y_train.npy", Y_train)
    np.save(engineered_dir / "Y_val.npy", Y_val)
    np.save(engineered_dir / "Y_test.npy", Y_test)

    # ids
    saved_ids = None
    if cfg.save_ids:
        id_cols = [c for c in cfg.id_candidates if c in train_df.columns]
        if id_cols:
            train_df[id_cols].to_csv(engineered_dir / "ids_train.csv", index=False)
            val_df[id_cols].to_csv(engineered_dir / "ids_val.csv", index=False)
            test_df[id_cols].to_csv(engineered_dir / "ids_test.csv", index=False)
            saved_ids = id_cols
        else:
            np.save(engineered_dir / "idx_train.npy", train_df.index.to_numpy())
            np.save(engineered_dir / "idx_val.npy", val_df.index.to_numpy())
            np.save(engineered_dir / "idx_test.npy", test_df.index.to_numpy())

    # meta json
    meta = {
        "fe_version": cfg.fe_version,
        "text_col": cfg.text_col,
        "split_dir": str(split_dir),
        "y_cols": y_cols,
        "X_shapes": {
            "train": [int(X_train.shape[0]), int(X_train.shape[1])],
            "val": [int(X_val.shape[0]), int(X_val.shape[1])],
            "test": [int(X_test.shape[0]), int(X_test.shape[1])],
        },
        "Y_shapes": {
            "train": [int(Y_train.shape[0]), int(Y_train.shape[1])],
            "val": [int(Y_val.shape[0]), int(Y_val.shape[1])],
            "test": [int(Y_test.shape[0]), int(Y_test.shape[1])],
        },
        "saved_ids": saved_ids,
    }
    with open(engineered_dir / "engineered_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # artifact meta (featurizer params)
    with open(art_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "fe_version": cfg.fe_version,
                "text_col": cfg.text_col,
                "split_dir": str(split_dir),
                "y_cols_count": len(y_cols),
                "meta_features": ["log1p_len", "n_terms", "n_uniq_terms"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved featurizer:", featurizer_path.resolve())
    print("Saved engineered data:", engineered_dir.resolve())

    return {
        "project_root": str(project_root),
        "split_dir": str(split_dir),
        "featurizer_path": str(featurizer_path),
        "engineered_dir": str(engineered_dir),
        "n_labels": len(y_cols),
        "n_features": int(X_train.shape[1]),
    }