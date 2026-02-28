from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB


# =========================
# FINDERS
# =========================
def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Notebooks tuzog‘iga tushmaslik uchun:
    PROJECT_ROOT mezoni: Data/Raw_data yoki Data/Processed ichida REAL content bo‘lsin.
    """
    start = start or Path.cwd()
    for p in [start] + list(start.parents):
        data = p / "Data"
        raw = data / "Raw_data"
        processed = data / "Processed"

        ok = False
        if raw.exists() and any(raw.iterdir()):
            ok = True
        if processed.exists() and any(processed.iterdir()):
            ok = True

        if ok:
            return p
    return start


def find_dataset_dir(
    project_root: Path,
    version: str,
    prefer_feature_selected: bool = True,
) -> Path:
    """
    Prefer: Data/Feature_Selected/<version>/
    Fallback: Data/Engineered_data/<version>/
    """
    fs = project_root / "Data" / "Feature_Selected" / version
    eng = project_root / "Data" / "Engineered_data" / version

    if prefer_feature_selected and fs.exists() and (fs / "X_train.npz").exists():
        return fs
    if eng.exists() and (eng / "X_train.npz").exists():
        return eng
    if fs.exists() and (fs / "X_train.npz").exists():
        return fs

    raise FileNotFoundError(
        f"Dataset dir topilmadi.\n"
        f"Qidirildi:\n- {fs}\n- {eng}\n"
        "Ye chim: 09_feature_engineering yoki 09b_feature_selection ni run qiling."
    )


# =========================
# METRICS + THRESHOLDS
# =========================
def prf_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def multilabel_micro_macro(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    tp = int(((Y_true == 1) & (Y_pred == 1)).sum())
    fp = int(((Y_true == 0) & (Y_pred == 1)).sum())
    fn = int(((Y_true == 1) & (Y_pred == 0)).sum())
    micro_p, micro_r, micro_f1 = prf_from_counts(tp, fp, fn)

    f1s, ps, rs = [], [], []
    for j in range(Y_true.shape[1]):
        y = Y_true[:, j]
        p = Y_pred[:, j]
        tpj = int(((y == 1) & (p == 1)).sum())
        fpj = int(((y == 0) & (p == 1)).sum())
        fnj = int(((y == 1) & (p == 0)).sum())
        pj_, rj_, f1j_ = prf_from_counts(tpj, fpj, fnj)
        ps.append(pj_)
        rs.append(rj_)
        f1s.append(f1j_)
    return {
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(np.mean(ps)),
        "macro_recall": float(np.mean(rs)),
        "macro_f1": float(np.mean(f1s)),
    }


def per_label_report(y_cols: list[str], Y_true: np.ndarray, Y_pred: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, lab in enumerate(y_cols):
        y = Y_true[:, j]
        p = Y_pred[:, j]
        tp = int(((y == 1) & (p == 1)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        tn = int(((y == 0) & (p == 0)).sum())
        prec, rec, f1 = prf_from_counts(tp, fp, fn)
        rows.append({
            "label": lab,
            "support_pos": int(y.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "threshold": float(thresholds[j]),
        })
    return pd.DataFrame(rows)


def score_matrix(model, X) -> tuple[np.ndarray, str]:
    if hasattr(model, "predict_proba"):
        P = model.predict_proba(X)
        return np.asarray(P), "proba"
    if hasattr(model, "decision_function"):
        S = model.decision_function(X)
        return np.asarray(S), "score"
    Yp = model.predict(X)
    return np.asarray(Yp), "binary"


def tune_thresholds_per_label(Y_true: np.ndarray, scores: np.ndarray, mode: str, n_thr: int = 61) -> np.ndarray:
    n_labels = Y_true.shape[1]
    thr_out = np.zeros(n_labels, dtype=np.float32)
    q = np.linspace(0.01, 0.99, n_thr)

    for j in range(n_labels):
        y = Y_true[:, j].astype(np.int8)
        s = scores[:, j].astype(np.float32)

        if int(y.sum()) == 0:
            thr_out[j] = 1.0 if mode == "proba" else float(np.max(s) + 1.0)
            continue

        thr_grid = np.unique(np.quantile(s, q))
        if thr_grid.size < 10:
            mn, mx = float(np.min(s)), float(np.max(s))
            if mn == mx:
                thr_grid = np.array([mn], dtype=np.float32)
            else:
                thr_grid = np.linspace(mn, mx, num=31, dtype=np.float32)

        best_f1 = -1.0
        best_thr = float(thr_grid[len(thr_grid)//2])

        for thr in thr_grid:
            pred = (s >= thr).astype(np.int8)
            tp = int(((y == 1) & (pred == 1)).sum())
            fp = int(((y == 0) & (pred == 1)).sum())
            fn = int(((y == 1) & (pred == 0)).sum())
            _, _, f1 = prf_from_counts(tp, fp, fn)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        thr_out[j] = best_thr

    return thr_out


def apply_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (scores >= thresholds.reshape(1, -1)).astype(np.int8)


# =========================
# MODEL ZOO (single pick)
# =========================
def build_model(model_name: str, n_jobs: int = 1, random_state: int = 42):
    if model_name == "ovr_logreg_bal_C1":
        return OneVsRestClassifier(
            LogisticRegression(solver="liblinear", max_iter=3000, C=1.0, class_weight="balanced"),
            n_jobs=n_jobs,
        )
    if model_name == "ovr_logreg_bal_C2":
        return OneVsRestClassifier(
            LogisticRegression(solver="liblinear", max_iter=3000, C=2.0, class_weight="balanced"),
            n_jobs=n_jobs,
        )
    if model_name == "ovr_linearsvc_C1":
        return OneVsRestClassifier(
            LinearSVC(C=1.0, random_state=random_state),
            n_jobs=n_jobs,
        )
    if model_name == "ovr_sgd_logloss":
        return OneVsRestClassifier(
            SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, tol=1e-3, random_state=random_state),
            n_jobs=n_jobs,
        )
    if model_name == "ovr_sgd_hinge":
        return OneVsRestClassifier(
            SGDClassifier(loss="hinge", alpha=1e-5, max_iter=2000, tol=1e-3, random_state=random_state),
            n_jobs=n_jobs,
        )
    if model_name == "ovr_complementnb_a05":
        return OneVsRestClassifier(
            ComplementNB(alpha=0.5),
            n_jobs=n_jobs,
        )

    raise ValueError(
        f"Unknown model_name: {model_name}\n"
        "Allowed: ovr_logreg_bal_C1, ovr_logreg_bal_C2, ovr_linearsvc_C1, "
        "ovr_sgd_logloss, ovr_sgd_hinge, ovr_complementnb_a05"
    )


@dataclass
class TrainImprovementConfig:
    version: str = "fe_v1"          # Engineered_data yoki Feature_Selected version
    prefer_feature_selected: bool = True

    model_name: str = "ovr_logreg_bal_C1"
    n_thr: int = 61
    n_jobs: int = 1
    random_state: int = 42

    # Save locations
    models_dir_rel: Path = Path("Models") / "improvement_models" / "Improvement_Models"
    results_dir_rel: Path = Path("results") / "improvement"


def run_train_improvement(cfg: TrainImprovementConfig) -> dict:
    project_root = find_project_root()
    data_dir = find_dataset_dir(project_root, cfg.version, prefer_feature_selected=cfg.prefer_feature_selected)

    print("PROJECT_ROOT:", project_root.resolve())
    print("DATA_DIR:", data_dir.resolve())

    # load meta
    with open(data_dir / "engineered_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    y_cols = meta["y_cols"]

    # load X/Y
    X_train = sparse.load_npz(data_dir / "X_train.npz").tocsr()
    X_val = sparse.load_npz(data_dir / "X_val.npz").tocsr()
    X_test = sparse.load_npz(data_dir / "X_test.npz").tocsr()

    Y_train = np.load(data_dir / "Y_train.npy")
    Y_val = np.load(data_dir / "Y_val.npy")
    Y_test = np.load(data_dir / "Y_test.npy")

    print("X:", X_train.shape, X_val.shape, X_test.shape)
    print("Y:", Y_train.shape, Y_val.shape, Y_test.shape)

    # build model
    clf = build_model(cfg.model_name, n_jobs=cfg.n_jobs, random_state=cfg.random_state)
    print("MODEL:", cfg.model_name)

    # fit
    clf.fit(X_train, Y_train)

    # val scoring + thresholds
    S_val, mode = score_matrix(clf, X_val)
    thr = tune_thresholds_per_label(Y_val, S_val, mode=mode, n_thr=cfg.n_thr)
    Y_val_pred = apply_thresholds(S_val, thr)

    val_overall = multilabel_micro_macro(Y_val, Y_val_pred)
    val_per_label = per_label_report(y_cols, Y_val, Y_val_pred, thr)

    # test eval (using val thresholds)
    S_test, _ = score_matrix(clf, X_test)
    Y_test_pred = apply_thresholds(S_test, thr)
    test_overall = multilabel_micro_macro(Y_test, Y_test_pred)
    test_per_label = per_label_report(y_cols, Y_test, Y_test_pred, thr)

    # save
    run_id = datetime.now().strftime("impr_%Y%m%d_%H%M%S")
    results_dir = project_root / cfg.results_dir_rel / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    models_dir = project_root / cfg.models_dir_rel
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{cfg.model_name}.joblib"
    thr_path = models_dir / f"{cfg.model_name}_thresholds.json"

    joblib.dump(clf, model_path)

    # thresholds dict: label name (without y_) -> thr
    thr_dict = {lab.replace("y_", "", 1): float(t) for lab, t in zip(y_cols, thr)}
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(thr_dict, f, ensure_ascii=False, indent=2)

    # metrics
    val_per_label.to_csv(results_dir / f"{cfg.model_name}_val_per_label_metrics.csv", index=False)
    test_per_label.to_csv(results_dir / f"{cfg.model_name}_test_per_label_metrics.csv", index=False)

    with open(results_dir / f"{cfg.model_name}_val_summary.json", "w", encoding="utf-8") as f:
        json.dump({"model": cfg.model_name, "mode": mode, **val_overall}, f, ensure_ascii=False, indent=2)
    with open(results_dir / f"{cfg.model_name}_test_summary.json", "w", encoding="utf-8") as f:
        json.dump({"model": cfg.model_name, **test_overall}, f, ensure_ascii=False, indent=2)

    print("Saved model:", model_path.resolve())
    print("Saved thresholds:", thr_path.resolve())
    print("Saved results:", results_dir.resolve())

    return {
        "project_root": str(project_root),
        "data_dir": str(data_dir),
        "model_name": cfg.model_name,
        "model_path": str(model_path),
        "thresholds_path": str(thr_path),
        "results_dir": str(results_dir),
        "val": val_overall,
        "test": test_overall,
    }