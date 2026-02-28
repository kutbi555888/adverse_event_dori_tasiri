from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from scipy import sparse

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass(frozen=True)
class TrainLogRegConfig:
    pp_dir: Path                         # Data/Processed/baseline_preprocess_v2
    model_dir: Path                      # Models/baseline_models
    tables_dir: Path                     # results/tables/baseline_train
    reports_dir: Path                    # results/reports/baseline_train

    # LogisticRegression params
    solver: str = "liblinear"
    max_iter: int = 200
    class_weight: str = "balanced"

    # Threshold tuning
    thr_min: float = 0.05
    thr_max: float = 0.95
    thr_steps: int = 19                  # 0.05..0.95, 19 step

    # Windows/joblib stability
    ovr_n_jobs: int = 1                  # MUST be 1 on Windows for stability


def _load_preprocess(pp_dir: Path) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, Dict]:
    Xtr = sparse.load_npz(pp_dir / "X_train.npz")
    Xva = sparse.load_npz(pp_dir / "X_val.npz")
    Xte = sparse.load_npz(pp_dir / "X_test.npz")

    Ytr = np.load(pp_dir / "Y_train.npy")
    Yva = np.load(pp_dir / "Y_val.npy")
    Yte = np.load(pp_dir / "Y_test.npy")

    bundle = joblib.load(pp_dir / "tfidf_bundle.joblib")
    return Xtr, Xva, Xte, Ytr, Yva, Yte, bundle


def _ensure_writeable_csr(X: sparse.spmatrix) -> sparse.csr_matrix:
    # Windows + joblib parallel issues -> make sure CSR + writeable
    Xc = sparse.csr_matrix(X).copy()
    return Xc


def tune_thresholds_f1(Y_true: np.ndarray, P: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_labels = Y_true.shape[1]
    best_t = np.zeros(n_labels, dtype=float)
    best_f = np.zeros(n_labels, dtype=float)

    for j in range(n_labels):
        yj = Y_true[:, j]
        pj = P[:, j]

        if yj.sum() == 0:
            best_t[j] = 0.5
            best_f[j] = 0.0
            continue

        best_score = -1.0
        best_thr = 0.5
        for t in thresholds:
            pred = (pj >= t).astype(int)
            f = f1_score(yj, pred, zero_division=0)
            if f > best_score:
                best_score = f
                best_thr = float(t)

        best_t[j] = best_thr
        best_f[j] = float(best_score)

    return best_t, best_f


def apply_thresholds(P: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return (P >= thr.reshape(1, -1)).astype(int)


def evaluate_micro_macro(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict[str, float]:
    micro_f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    micro_p = precision_score(Y_true, Y_pred, average="micro", zero_division=0)
    micro_r = recall_score(Y_true, Y_pred, average="micro", zero_division=0)
    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
    }


def per_label_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, thr: np.ndarray, y_cols: List[str]) -> pd.DataFrame:
    rows = []
    for j, col in enumerate(y_cols):
        name = col.replace("y_", "", 1)  # IMPORTANT: only prefix
        yj = Y_true[:, j]
        pj = Y_pred[:, j]
        rows.append({
            "label": name,
            "support": int(yj.sum()),
            "precision": float(precision_score(yj, pj, zero_division=0)),
            "recall": float(recall_score(yj, pj, zero_division=0)),
            "f1": float(f1_score(yj, pj, zero_division=0)),
            "threshold": float(thr[j]),
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def run_train_logreg(cfg: TrainLogRegConfig) -> Dict[str, Path]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    Xtr, Xva, Xte, Ytr, Yva, Yte, bundle = _load_preprocess(cfg.pp_dir)

    # Make X writeable CSR
    Xtr = _ensure_writeable_csr(Xtr)
    Xva = _ensure_writeable_csr(Xva)
    Xte = _ensure_writeable_csr(Xte)

    y_cols = bundle.get("y_cols")
    if not y_cols:
        raise ValueError("tfidf_bundle.joblib ichida y_cols topilmadi. 05_preprocess to'g'ri saqlanganini tekshiring.")
    y_cols = list(y_cols)

    # Train
    base_lr = LogisticRegression(
        solver=cfg.solver,
        max_iter=cfg.max_iter,
        class_weight=cfg.class_weight,
    )
    model = OneVsRestClassifier(base_lr, n_jobs=cfg.ovr_n_jobs)

    t0 = time.time()
    model.fit(Xtr, Ytr)
    train_minutes = (time.time() - t0) / 60.0

    # Predict proba
    Pva = model.predict_proba(Xva)
    Pte = model.predict_proba(Xte)

    # Threshold tuning on VAL
    thr_grid = np.linspace(cfg.thr_min, cfg.thr_max, cfg.thr_steps)
    best_thr, best_f1 = tune_thresholds_f1(Yva, Pva, thresholds=thr_grid)

    thr_df = pd.DataFrame({
        "label": [c.replace("y_", "", 1) for c in y_cols],
        "best_threshold": np.round(best_thr, 3),
        "val_f1_at_best": np.round(best_f1, 4),
    }).sort_values("val_f1_at_best", ascending=False).reset_index(drop=True)

    # Evaluate TEST
    Yte_pred = apply_thresholds(Pte, best_thr)
    metrics = evaluate_micro_macro(Yte, Yte_pred)
    per_label_df = per_label_metrics(Yte, Yte_pred, best_thr, y_cols)

    # Save artifacts
    run_name = "baseline_ovr_logreg"

    model_path = cfg.model_dir / f"{run_name}.joblib"
    thr_path = cfg.model_dir / f"{run_name}_thresholds.npy"
    thr_csv = cfg.tables_dir / f"{run_name}_thresholds_val.csv"
    per_csv = cfg.tables_dir / f"{run_name}_per_label_test.csv"
    summary_path = cfg.reports_dir / f"{run_name}_summary.json"

    joblib.dump(model, model_path)
    np.save(thr_path, best_thr)

    thr_df.to_csv(thr_csv, index=False, encoding="utf-8-sig")
    per_label_df.to_csv(per_csv, index=False, encoding="utf-8-sig")

    summary = {
        "run": run_name,
        "model_type": "OneVsRest(LogisticRegression)",
        "train_minutes": float(train_minutes),
        "text_col": bundle.get("text_col"),
        "num_labels": int(len(y_cols)),
        "labels": [c.replace("y_", "", 1) for c in y_cols],
        "metrics_test": metrics,
        "artifacts": {
            "model": str(model_path),
            "thresholds": str(thr_path),
            "tfidf_bundle": str(cfg.pp_dir / "tfidf_bundle.joblib"),
            "thresholds_table": str(thr_csv),
            "per_label_table": str(per_csv),
        },
        "config": {
            "solver": cfg.solver,
            "max_iter": cfg.max_iter,
            "class_weight": cfg.class_weight,
            "ovr_n_jobs": cfg.ovr_n_jobs,
            "thr_grid": {"min": cfg.thr_min, "max": cfg.thr_max, "steps": cfg.thr_steps},
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "model": model_path,
        "thresholds": thr_path,
        "thresholds_table": thr_csv,
        "per_label_table": per_csv,
        "summary": summary_path,
    }