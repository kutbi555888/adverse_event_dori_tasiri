# Optuna tuning (4 ta algoritm) → eng yaxshi parametrlarni topadi, model+threshold+summary’larni saqlaydi

# Best model final train → 4 ta tuned model ichidan eng yaxshisini tanlaydi, final train qiladi

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import sparse
import joblib

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

try:
    import optuna
except Exception as e:
    optuna = None
    _OPTUNA_IMPORT_ERROR = e


# ----------------------------
# Paths / finders (root adashmaydi)
# ----------------------------
def find_project_root(start: Optional[Path] = None) -> Path:
    start = start or Path.cwd()
    for p in [start] + list(start.parents):
        d = p / "Data"
        if not d.exists():
            continue
        for sub in ["Feature_Selected", "Engineered_data", "Processed", "Raw_data"]:
            x = d / sub
            if x.exists() and any(x.iterdir()):
                return p
    return start


def find_data_dir(project_root: Path, version: str, prefer_fs: bool = True) -> Tuple[Path, str]:
    fs = project_root / "Data" / "Feature_Selected" / version
    eng = project_root / "Data" / "Engineered_data" / version

    if prefer_fs and (fs / "X_train.npz").exists():
        return fs, "Feature_Selected"
    if (eng / "X_train.npz").exists():
        return eng, "Engineered_data"
    if (fs / "X_train.npz").exists():
        return fs, "Feature_Selected"

    raise FileNotFoundError(f"X_train.npz topilmadi: {fs} yoki {eng}")


# ----------------------------
# Threshold + scoring helpers
# ----------------------------
def prf_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def score_matrix(model, X) -> tuple[np.ndarray, str]:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)), "proba"
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)), "score"
    return np.asarray(model.predict(X)), "binary"


def tune_thresholds_per_label(Y_true: np.ndarray, scores: np.ndarray, n_thr: int = 61) -> np.ndarray:
    n_labels = Y_true.shape[1]
    thr_out = np.zeros(n_labels, dtype=np.float32)
    q = np.linspace(0.01, 0.99, n_thr)

    for j in range(n_labels):
        y = Y_true[:, j].astype(np.int8)
        s = scores[:, j].astype(np.float32)

        if int(y.sum()) == 0:
            thr_out[j] = float(np.max(s) + 1.0)
            continue

        thr_grid = np.unique(np.quantile(s, q))
        if thr_grid.size < 10:
            mn, mx = float(np.min(s)), float(np.max(s))
            thr_grid = np.array([mn], dtype=np.float32) if mn == mx else np.linspace(mn, mx, num=31, dtype=np.float32)

        best_f1 = -1.0
        best_thr = float(thr_grid[len(thr_grid) // 2])

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


def apply_thresholds(scores: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return (scores >= thr.reshape(1, -1)).astype(np.int8)


def micro_macro_f1(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    return {
        "micro_f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
    }


# ----------------------------
# Save helpers
# ----------------------------
def save_best_bundle(
    model,
    tag: str,
    y_cols: List[str],
    best_params: dict,
    thr: np.ndarray,
    val_summary: dict,
    test_summary: dict,
    model_save_dir: Path,
    results_dir: Path,
) -> None:
    model_save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_save_dir / f"{tag}.joblib"
    joblib.dump(model, model_path)

    thr_dict = {c.replace("y_", "", 1): float(t) for c, t in zip(y_cols, thr)}
    with open(model_save_dir / f"{tag}_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thr_dict, f, ensure_ascii=False, indent=2)

    with open(model_save_dir / f"{tag}_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    with open(results_dir / f"{tag}_val_summary.json", "w", encoding="utf-8") as f:
        json.dump({"model": tag, **val_summary}, f, ensure_ascii=False, indent=2)

    with open(results_dir / f"{tag}_test_summary.json", "w", encoding="utf-8") as f:
        json.dump({"model": tag, **test_summary}, f, ensure_ascii=False, indent=2)


@dataclass
class OptunaTuningConfig:
    version: str = "fe_v1_fs_chi2_v1"
    prefer_feature_selected: bool = True

    n_trials: int = 15
    timeout_sec: Optional[int] = 1800

    n_thr_fast: int = 31
    n_thr_final: int = 61
    train_subsample: Optional[int] = 60000

    n_jobs: int = 1
    random_state: int = 42

    # output base
    models_base_rel: Path = Path("Models") / "improvement_models" / "Improvement_Models"
    results_base_rel: Path = Path("results") / "optuna_tuning"


def run_optuna_tuning(cfg: OptunaTuningConfig) -> dict:
    if optuna is None:
        raise ImportError(
            "optuna o‘rnatilmagan. O‘rnating:\n"
            "python -m pip install optuna\n"
            f"Original error: {_OPTUNA_IMPORT_ERROR}"
        )

    project_root = find_project_root()
    data_dir, data_source = find_data_dir(project_root, cfg.version, cfg.prefer_feature_selected)

    with open(data_dir / "engineered_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    y_cols = meta["y_cols"]

    X_train = sparse.load_npz(data_dir / "X_train.npz").tocsr()
    X_val = sparse.load_npz(data_dir / "X_val.npz").tocsr()
    X_test = sparse.load_npz(data_dir / "X_test.npz").tocsr()

    Y_train = np.load(data_dir / "Y_train.npy")
    Y_val = np.load(data_dir / "Y_val.npy")
    Y_test = np.load(data_dir / "Y_test.npy")

    run_id = datetime.now().strftime("optuna_%Y%m%d_%H%M%S")
    model_save_dir = project_root / cfg.models_base_rel / f"Optuna_Tuned_{run_id}"
    results_dir = project_root / cfg.results_base_rel / run_id
    model_save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # subsample for trials
    if cfg.train_subsample is not None and X_train.shape[0] > cfg.train_subsample:
        rng = np.random.RandomState(cfg.random_state)
        idx = rng.choice(X_train.shape[0], size=cfg.train_subsample, replace=False)
        X_tr = X_train[idx]
        Y_tr = Y_train[idx]
    else:
        X_tr = X_train
        Y_tr = Y_train

    # optuna sampler/pruner
    sampler = optuna.samplers.TPESampler(seed=cfg.random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    def _final_eval_and_save(tag: str, best_params: dict, best_model) -> dict:
        # final thresholds (VAL) + test
        S_val, _ = score_matrix(best_model, X_val)
        thr = tune_thresholds_per_label(Y_val, S_val, n_thr=cfg.n_thr_final)
        Y_val_pred = apply_thresholds(S_val, thr)
        val_sum = micro_macro_f1(Y_val, Y_val_pred)

        S_test, _ = score_matrix(best_model, X_test)
        Y_test_pred = apply_thresholds(S_test, thr)
        test_sum = micro_macro_f1(Y_test, Y_test_pred)

        save_best_bundle(
            best_model, tag, y_cols, best_params, thr,
            val_sum, test_sum, model_save_dir, results_dir
        )
        return {"tag": tag, "best_params": best_params, "val": val_sum, "test": test_sum}

    # ----------------------------
    # 1) LogReg
    # ----------------------------
    def objective_logreg(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 0.25, 8.0, log=True)
        cw = trial.suggest_categorical("class_weight", [None, "balanced"])
        base = LogisticRegression(solver="liblinear", max_iter=2500, C=C, class_weight=cw)
        clf = OneVsRestClassifier(base, n_jobs=cfg.n_jobs)
        clf.fit(X_tr, Y_tr)
        S_val, _ = score_matrix(clf, X_val)
        thr = tune_thresholds_per_label(Y_val, S_val, n_thr=cfg.n_thr_fast)
        Yp = apply_thresholds(S_val, thr)
        m = micro_macro_f1(Y_val, Yp)
        trial.set_user_attr("macro_f1", m["macro_f1"])
        return m["micro_f1"]

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=f"logreg_{run_id}")
    study.optimize(objective_logreg, n_trials=cfg.n_trials, timeout=cfg.timeout_sec)
    pd.DataFrame(study.trials_dataframe()).to_csv(results_dir / "logreg_trials.csv", index=False)

    bp = study.best_params
    base = LogisticRegression(solver="liblinear", max_iter=6000, C=bp["C"], class_weight=bp["class_weight"])
    best_model = OneVsRestClassifier(base, n_jobs=cfg.n_jobs)
    best_model.fit(X_train, Y_train)
    out_logreg = _final_eval_and_save("optuna_logreg_best", bp, best_model)

    # ----------------------------
    # 2) LinearSVC
    # ----------------------------
    def objective_linearsvc(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 0.25, 8.0, log=True)
        cw = trial.suggest_categorical("class_weight", [None, "balanced"])
        base = LinearSVC(C=C, class_weight=cw, random_state=cfg.random_state)
        clf = OneVsRestClassifier(base, n_jobs=cfg.n_jobs)
        clf.fit(X_tr, Y_tr)
        S_val, _ = score_matrix(clf, X_val)
        thr = tune_thresholds_per_label(Y_val, S_val, n_thr=cfg.n_thr_fast)
        Yp = apply_thresholds(S_val, thr)
        m = micro_macro_f1(Y_val, Yp)
        trial.set_user_attr("macro_f1", m["macro_f1"])
        return m["micro_f1"]

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=f"linearsvc_{run_id}")
    study.optimize(objective_linearsvc, n_trials=cfg.n_trials, timeout=cfg.timeout_sec)
    pd.DataFrame(study.trials_dataframe()).to_csv(results_dir / "linearsvc_trials.csv", index=False)

    bp = study.best_params
    base = LinearSVC(C=bp["C"], class_weight=bp["class_weight"], random_state=cfg.random_state)
    best_model = OneVsRestClassifier(base, n_jobs=cfg.n_jobs)
    best_model.fit(X_train, Y_train)
    out_svc = _final_eval_and_save("optuna_linearsvc_best", bp, best_model)

    # ----------------------------
    # 3) SGD log_loss
    # ----------------------------
    def objective_sgd_logloss(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-6, 1e-4, log=True)
        penalty = trial.suggest_categorical("penalty", ["l2", "elasticnet"])
        l1_ratio = None
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)

        base = SGDClassifier(
            loss="log_loss", penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            max_iter=2000, tol=1e-3, random_state=cfg.random_state
        )
        clf = OneVsRestClassifier(base, n_jobs=cfg.n_jobs)
        clf.fit(X_tr, Y_tr)
        S_val, _ = score_matrix(clf, X_val)
        thr = tune_thresholds_per_label(Y_val, S_val, n_thr=cfg.n_thr_fast)
        Yp = apply_thresholds(S_val, thr)
        m = micro_macro_f1(Y_val, Yp)
        trial.set_user_attr("macro_f1", m["macro_f1"])
        return m["micro_f1"]

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=f"sgd_logloss_{run_id}")
    study.optimize(objective_sgd_logloss, n_trials=cfg.n_trials, timeout=cfg.timeout_sec)
    pd.DataFrame(study.trials_dataframe()).to_csv(results_dir / "sgd_logloss_trials.csv", index=False)

    bp = study.best_params
    base = SGDClassifier(
        loss="log_loss", penalty=bp["penalty"], alpha=bp["alpha"],
        l1_ratio=bp.get("l1_ratio", None),
        max_iter=5000, tol=1e-3, random_state=cfg.random_state
    )
    best_model = OneVsRestClassifier(base, n_jobs=cfg.n_jobs)
    best_model.fit(X_train, Y_train)
    out_sgdll = _final_eval_and_save("optuna_sgd_logloss_best", bp, best_model)

    # ----------------------------
    # 4) SGD hinge
    # ----------------------------
    def objective_sgd_hinge(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-6, 1e-4, log=True)
        cw = trial.suggest_categorical("class_weight", [None, "balanced"])

        base = SGDClassifier(
            loss="hinge", penalty="l2", alpha=alpha, class_weight=cw,
            max_iter=2000, tol=1e-3, random_state=cfg.random_state
        )
        clf = OneVsRestClassifier(base, n_jobs=cfg.n_jobs)
        clf.fit(X_tr, Y_tr)
        S_val, _ = score_matrix(clf, X_val)
        thr = tune_thresholds_per_label(Y_val, S_val, n_thr=cfg.n_thr_fast)
        Yp = apply_thresholds(S_val, thr)
        m = micro_macro_f1(Y_val, Yp)
        trial.set_user_attr("macro_f1", m["macro_f1"])
        return m["micro_f1"]

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=f"sgd_hinge_{run_id}")
    study.optimize(objective_sgd_hinge, n_trials=cfg.n_trials, timeout=cfg.timeout_sec)
    pd.DataFrame(study.trials_dataframe()).to_csv(results_dir / "sgd_hinge_trials.csv", index=False)

    bp = study.best_params
    base = SGDClassifier(
        loss="hinge", penalty="l2", alpha=bp["alpha"], class_weight=bp["class_weight"],
        max_iter=5000, tol=1e-3, random_state=cfg.random_state
    )
    best_model = OneVsRestClassifier(base, n_jobs=cfg.n_jobs)
    best_model.fit(X_train, Y_train)
    out_sgdh = _final_eval_and_save("optuna_sgd_hinge_best", bp, best_model)

    # leaderboard
    lb = pd.DataFrame([
        {"tag": out_logreg["tag"], **out_logreg["val"], "test_micro_f1": out_logreg["test"]["micro_f1"], "test_macro_f1": out_logreg["test"]["macro_f1"]},
        {"tag": out_svc["tag"], **out_svc["val"], "test_micro_f1": out_svc["test"]["micro_f1"], "test_macro_f1": out_svc["test"]["macro_f1"]},
        {"tag": out_sgdll["tag"], **out_sgdll["val"], "test_micro_f1": out_sgdll["test"]["micro_f1"], "test_macro_f1": out_sgdll["test"]["macro_f1"]},
        {"tag": out_sgdh["tag"], **out_sgdh["val"], "test_micro_f1": out_sgdh["test"]["micro_f1"], "test_macro_f1": out_sgdh["test"]["macro_f1"]},
    ]).sort_values(["micro_f1", "macro_f1"], ascending=False)

    lb.to_csv(results_dir / "leaderboard.csv", index=False)

    return {
        "run_id": run_id,
        "project_root": str(project_root),
        "data_source": data_source,
        "data_dir": str(data_dir),
        "model_save_dir": str(model_save_dir),
        "results_dir": str(results_dir),
        "leaderboard_path": str(results_dir / "leaderboard.csv"),
        "leaderboard": lb.to_dict(orient="records"),
    }