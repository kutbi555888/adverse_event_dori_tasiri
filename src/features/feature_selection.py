from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_selection import chi2


# -----------------------------
# Robust finder: Engineered_data/FE_VERSION ni topadi
# -----------------------------
def find_project_root_by_engineered(
    fe_version: str,
    start: Optional[Path] = None,
    engineered_dirname: str = "Engineered_data",
) -> Tuple[Path, Path]:
    """
    PROJECT_ROOT ni Data/Engineered_data/<fe_version>/X_train.npz bor joydan topadi.
    Returns: (project_root, engineered_dir)
    """
    start = start or Path.cwd()
    checked = []
    for p in [start] + list(start.parents):
        ed = p / "Data" / engineered_dirname / fe_version
        checked.append(ed)
        if (ed / "X_train.npz").exists() and (ed / "Y_train.npy").exists() and (ed / "engineered_meta.json").exists():
            return p, ed

    raise FileNotFoundError(
        "Engineered data topilmadi (parents bo‘ylab qidirildi).\n"
        f"FE_VERSION: {fe_version}\n"
        f"Start: {start.resolve()}\n"
        "Oxirgi 10 tekshirilgan yo‘l:\n" + "\n".join(str(x) for x in checked[-10:])
    )


# -----------------------------
# Config
# -----------------------------
@dataclass
class FeatureSelectionConfig:
    fe_version_in: str = "fe_v1"
    fs_version_out: str = "fe_v1_fs_chi2_v1"

    topk_per_label: int = 3000
    max_total_features: int = 250000
    keep_meta_last_n: int = 3

    engineered_dirname: str = "Engineered_data"
    feature_selected_dirname: str = "Feature_Selected"

    # IDs/idx copy
    copy_id_files: tuple[str, ...] = (
        "ids_train.csv", "ids_val.csv", "ids_test.csv",
        "idx_train.npy", "idx_val.npy", "idx_test.npy",
    )


# -----------------------------
# Core selection
# -----------------------------
def chi2_topk_union(
    X: sparse.csr_matrix,
    Y: np.ndarray,
    topk_per_label: int,
    max_total_features: int,
    keep_meta_last_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      selected_idx (sorted)
      mask (bool array length n_features)
    """
    n_samples, n_features = X.shape
    n_labels = Y.shape[1]

    selected = set()

    # keep meta last N
    if keep_meta_last_n and keep_meta_last_n > 0:
        meta_idx = list(range(n_features - keep_meta_last_n, n_features))
        for i in meta_idx:
            if 0 <= i < n_features:
                selected.add(i)

    # per label chi2
    for j in range(n_labels):
        y = Y[:, j].astype(np.int8)

        # if no positives -> skip
        if int(y.sum()) == 0:
            continue

        scores, _ = chi2(X, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        k = min(topk_per_label, n_features)
        top_idx = np.argpartition(scores, -k)[-k:]
        for idx in top_idx:
            selected.add(int(idx))

        if len(selected) > max_total_features:
            # stop early if too many
            break

    selected_idx = np.array(sorted(selected), dtype=np.int32)
    mask = np.zeros(n_features, dtype=bool)
    mask[selected_idx] = True

    return selected_idx, mask


# -----------------------------
# Runner
# -----------------------------
def run_feature_selection(cfg: FeatureSelectionConfig) -> dict:
    # 1) find project root + engineered input
    project_root, engineered_in = find_project_root_by_engineered(
        fe_version=cfg.fe_version_in,
        start=Path.cwd(),
        engineered_dirname=cfg.engineered_dirname,
    )
    print("PROJECT_ROOT:", project_root.resolve())
    print("ENGINEERED_IN:", engineered_in.resolve())

    # 2) load meta + X/Y
    with open(engineered_in / "engineered_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    y_cols = meta["y_cols"]

    X_train = sparse.load_npz(engineered_in / "X_train.npz").tocsr()
    X_val = sparse.load_npz(engineered_in / "X_val.npz").tocsr()
    X_test = sparse.load_npz(engineered_in / "X_test.npz").tocsr()

    Y_train = np.load(engineered_in / "Y_train.npy")
    Y_val = np.load(engineered_in / "Y_val.npy")
    Y_test = np.load(engineered_in / "Y_test.npy")

    print("X:", X_train.shape, X_val.shape, X_test.shape)
    print("Y:", Y_train.shape, Y_val.shape, Y_test.shape)

    # 3) select idx/mask
    selected_idx, mask = chi2_topk_union(
        X=X_train,
        Y=Y_train,
        topk_per_label=cfg.topk_per_label,
        max_total_features=cfg.max_total_features,
        keep_meta_last_n=cfg.keep_meta_last_n,
    )
    print("Selected features:", selected_idx.size, "/", X_train.shape[1])

    # 4) apply selection
    X_train_fs = X_train[:, mask]
    X_val_fs = X_val[:, mask]
    X_test_fs = X_test[:, mask]
    print("X_fs:", X_train_fs.shape, X_val_fs.shape, X_test_fs.shape)

    # 5) save to Data/Feature_Selected/<fs_version_out>/
    out_dir = project_root / "Data" / cfg.feature_selected_dirname / cfg.fs_version_out
    out_dir.mkdir(parents=True, exist_ok=True)
    print("OUT_DIR:", out_dir.resolve())

    sparse.save_npz(out_dir / "X_train.npz", X_train_fs)
    sparse.save_npz(out_dir / "X_val.npz", X_val_fs)
    sparse.save_npz(out_dir / "X_test.npz", X_test_fs)

    np.save(out_dir / "Y_train.npy", Y_train.astype(np.int8))
    np.save(out_dir / "Y_val.npy", Y_val.astype(np.int8))
    np.save(out_dir / "Y_test.npy", Y_test.astype(np.int8))

    np.save(out_dir / "feature_mask.npy", mask)
    np.save(out_dir / "selected_idx.npy", selected_idx)

    # copy ids/idx if exist
    for fn in cfg.copy_id_files:
        src = engineered_in / fn
        if src.exists():
            (out_dir / fn).write_bytes(src.read_bytes())

    # meta out
    out_meta = dict(meta)
    out_meta.update({
        "fe_version_in": cfg.fe_version_in,
        "fs_version_out": cfg.fs_version_out,
        "method": "chi2_topk_per_label_union",
        "topk_per_label": cfg.topk_per_label,
        "max_total_features": cfg.max_total_features,
        "keep_meta_last_n": cfg.keep_meta_last_n,
        "selected_features": int(selected_idx.size),
        "original_features": int(X_train.shape[1]),
        "output_dir": str(out_dir),
        "X_shapes": {
            "train": [int(X_train_fs.shape[0]), int(X_train_fs.shape[1])],
            "val": [int(X_val_fs.shape[0]), int(X_val_fs.shape[1])],
            "test": [int(X_test_fs.shape[0]), int(X_test_fs.shape[1])],
        },
        "Y_shapes": {
            "train": [int(Y_train.shape[0]), int(Y_train.shape[1])],
            "val": [int(Y_val.shape[0]), int(Y_val.shape[1])],
            "test": [int(Y_test.shape[0]), int(Y_test.shape[1])],
        },
        "y_cols": y_cols,
    })

    with open(out_dir / "engineered_meta.json", "w", encoding="utf-8") as f:
        json.dump(out_meta, f, ensure_ascii=False, indent=2)

    print("Saved Feature_Selected:", out_dir.resolve())

    return {
        "project_root": str(project_root),
        "engineered_in": str(engineered_in),
        "out_dir": str(out_dir),
        "selected_features": int(selected_idx.size),
        "original_features": int(X_train.shape[1]),
        "n_labels": len(y_cols),
    }