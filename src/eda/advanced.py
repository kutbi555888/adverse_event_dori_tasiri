from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
import joblib
from pandas.util import hash_pandas_object
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass(frozen=True)
class AdvancedEDAConfig:
    split_dir: Path
    pp_dir: Path
    model_dir: Path

    tables_out: Path
    visuals_out: Path

    text_col: str = "REAC_pt_symptom_v2"
    id_col: str = "primaryid"

    # outputs detail
    worst_k_labels: int = 5
    top_errors_per_label: int = 25
    reliability_topk: int = 6
    reliability_bins: int = 10


def _infer_y_cols(df: pd.DataFrame) -> List[str]:
    y_cols = sorted([c for c in df.columns if c.startswith("y_") and c != "y_labels"])
    if not y_cols:
        raise ValueError("y_* label ustunlari topilmadi (split csv).")
    return y_cols


def _labname(col: str) -> str:
    # IMPORTANT: only remove prefix, not internal 'y_'
    return col.replace("y_", "", 1)


def _load_thresholds(model_dir: Path) -> np.ndarray:
    # support multiple names
    cand = [
        model_dir / "baseline_ovr_logreg_thresholds.npy",
        model_dir / "thresholds.npy",
        model_dir / "baseline_thresholds.npy",
    ]
    for p in cand:
        if p.exists():
            return np.load(p)
    # last resort: any *_thresholds.npy
    any_thr = list(model_dir.glob("*thresholds*.npy"))
    if any_thr:
        return np.load(any_thr[0])
    raise FileNotFoundError("Threshold .npy topilmadi (MODEL_DIR ichida).")


def _load_model(model_dir: Path):
    cand = [
        model_dir / "baseline_ovr_logreg.joblib",
        model_dir / "baseline_ovr_logreg.pkl",
    ]
    for p in cand:
        if p.exists():
            return joblib.load(p), p
    any_model = list(model_dir.glob("*.joblib"))
    if any_model:
        return joblib.load(any_model[0]), any_model[0]
    raise FileNotFoundError("Model fayli topilmadi (MODEL_DIR ichida *.joblib).")


def _hash_text(s: pd.Series) -> np.ndarray:
    return hash_pandas_object(s.fillna("").astype(str), index=False).to_numpy()


def _micro_macro(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "micro_f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(Y_true, Y_pred, average="micro", zero_division=0)),
    }


def _per_label_table(Y_true: np.ndarray, Y_pred: np.ndarray, P: np.ndarray, thr: np.ndarray, y_cols: List[str]) -> pd.DataFrame:
    rows = []
    for j, col in enumerate(y_cols):
        name = _labname(col)
        yj = Y_true[:, j]
        pj = Y_pred[:, j]
        rows.append({
            "label": name,
            "support": int(yj.sum()),
            "precision": float(precision_score(yj, pj, zero_division=0)),
            "recall": float(recall_score(yj, pj, zero_division=0)),
            "f1": float(f1_score(yj, pj, zero_division=0)),
            "threshold": float(thr[j]),
            "pred_pos_rate_%": float(pj.mean() * 100),
            "prob_mean": float(P[:, j].mean()),
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=True).reset_index(drop=True)


def _top_errors_for_label(
    df_test: pd.DataFrame,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    P: np.ndarray,
    y_cols: List[str],
    label_idx: int,
    text_col: str,
    top_k: int,
) -> pd.DataFrame:
    col = y_cols[label_idx]
    lab = _labname(col)

    y_true = Y_true[:, label_idx]
    y_hat = Y_pred[:, label_idx]
    p = P[:, label_idx]

    # FP: pred=1 true=0, high prob
    fp_idx = np.where((y_hat == 1) & (y_true == 0))[0]
    fp_sorted = fp_idx[np.argsort(-p[fp_idx])] if len(fp_idx) else np.array([], dtype=int)

    # FN: pred=0 true=1, low prob
    fn_idx = np.where((y_hat == 0) & (y_true == 1))[0]
    fn_sorted = fn_idx[np.argsort(p[fn_idx])] if len(fn_idx) else np.array([], dtype=int)

    def pack(idxs: np.ndarray, kind: str) -> pd.DataFrame:
        take = idxs[:top_k]
        if len(take) == 0:
            return pd.DataFrame(columns=["label","error_type","row_idx","prob","text"])
        return pd.DataFrame({
            "label": lab,
            "error_type": kind,
            "row_idx": take,
            "prob": p[take],
            "text": df_test.iloc[take][text_col].astype(str).values,
        })

    fp_df = pack(fp_sorted, "FP_high_prob")
    fn_df = pack(fn_sorted, "FN_low_prob")
    return pd.concat([fp_df, fn_df], axis=0, ignore_index=True)


def _slice_by_token_len(df_test: pd.DataFrame, Y_true: np.ndarray, Y_pred: np.ndarray, text_col: str) -> pd.DataFrame:
    tok_len = df_test[text_col].fillna("").astype(str).str.split().apply(len)
    bins = pd.cut(tok_len, bins=[0,1,2,3,5,8,12,20,999], right=True)

    rows = []
    for b in bins.unique().sort_values():
        mask = (bins == b).values
        if mask.sum() < 200:
            continue
        Yt = Y_true[mask]
        Yp = Y_pred[mask]
        rows.append({
            "token_bin": str(b),
            "rows": int(mask.sum()),
            "micro_f1": float(f1_score(Yt, Yp, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(Yt, Yp, average="macro", zero_division=0)),
            "avg_labels_true": float(Yt.sum(axis=1).mean()),
            "avg_labels_pred": float(Yp.sum(axis=1).mean()),
        })
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def _reliability_table(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(p_pred, bins) - 1
    ids = np.clip(ids, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        m = (ids == b)
        if m.sum() == 0:
            continue
        rows.append({
            "bin": int(b),
            "count": int(m.sum()),
            "p_mean": float(p_pred[m].mean()),
            "y_rate": float(y_true[m].mean()),
        })
    return pd.DataFrame(rows)


def _save_reliability_plot(rt: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1])
    plt.scatter(rt["p_mean"], rt["y_rate"])
    plt.title(title)
    plt.xlabel("mean predicted prob")
    plt.ylabel("empirical positive rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def _top_word_features_topk(
    model,
    bundle: Dict,
    y_cols: List[str],
    labels: List[str],
    topk: int = 25,
    which_labels: List[int] | None = None,
) -> pd.DataFrame:
    # Works for OvR LogisticRegression: model.estimators_[j].coef_
    tfidf_char = bundle["tfidf_char"]
    tfidf_word = bundle["tfidf_word"]

    feat_word = tfidf_word.get_feature_names_out()
    n_char = len(tfidf_char.get_feature_names_out())

    if which_labels is None:
        which_labels = list(range(len(y_cols)))

    rows = []
    for j in which_labels:
        est = model.estimators_[j]
        w = est.coef_.ravel()
        w_word = w[n_char:]

        top_pos = np.argsort(w_word)[::-1][:topk]
        top_neg = np.argsort(w_word)[:topk]

        lab = labels[j]
        for rank, idx in enumerate(top_pos, start=1):
            rows.append({"label": lab, "direction": "pos", "rank": rank, "term": feat_word[idx], "coef": float(w_word[idx])})
        for rank, idx in enumerate(top_neg, start=1):
            rows.append({"label": lab, "direction": "neg", "rank": rank, "term": feat_word[idx], "coef": float(w_word[idx])})

    return pd.DataFrame(rows)


def run_advanced_eda(cfg: AdvancedEDAConfig) -> Dict[str, Path]:
    cfg.tables_out.mkdir(parents=True, exist_ok=True)
    cfg.visuals_out.mkdir(parents=True, exist_ok=True)

    # Load raw splits (text)
    df_test = pd.read_csv(cfg.split_dir / "test.csv", low_memory=False)
    df_val = pd.read_csv(cfg.split_dir / "val.csv", low_memory=False)

    if cfg.text_col not in df_test.columns:
        raise ValueError(f"text_col topilmadi test.csv ichida: {cfg.text_col}")

    y_cols = _infer_y_cols(df_test)
    labels = [_labname(c) for c in y_cols]

    # Load X/Y from preprocess
    Xte = sparse.load_npz(cfg.pp_dir / "X_test.npz")
    Yte = np.load(cfg.pp_dir / "Y_test.npy")

    Xva = sparse.load_npz(cfg.pp_dir / "X_val.npz")
    Yva = np.load(cfg.pp_dir / "Y_val.npy")

    bundle = joblib.load(cfg.pp_dir / "tfidf_bundle.joblib")

    # Load model & thresholds
    model, model_path = _load_model(cfg.model_dir)
    thr = _load_thresholds(cfg.model_dir)

    # Predict probs
    Pte = model.predict_proba(Xte)
    Pva = model.predict_proba(Xva)

    Ypred = (Pte >= thr.reshape(1, -1)).astype(int)

    # Overall metrics
    overall = _micro_macro(Yte, Ypred)
    (cfg.tables_out / "test_overall_metrics.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")

    # Per-label
    per_label_df = _per_label_table(Yte, Ypred, Pte, thr, y_cols)
    per_label_path = cfg.tables_out / "test_per_label_metrics.csv"
    per_label_df.to_csv(per_label_path, index=False, encoding="utf-8-sig")

    # Worst-k labels errors
    worst_df = per_label_df.head(cfg.worst_k_labels)
    worst_labels = worst_df["label"].tolist()
    worst_idxs = [labels.index(x) for x in worst_labels]

    err_frames = []
    for idx in worst_idxs:
        err_frames.append(_top_errors_for_label(df_test, Yte, Ypred, Pte, y_cols, idx, cfg.text_col, cfg.top_errors_per_label))
    err_df = pd.concat(err_frames, axis=0, ignore_index=True)
    err_path = cfg.tables_out / "worst5_labels_fp_fn_examples.csv"
    err_df.to_csv(err_path, index=False, encoding="utf-8-sig")

    # Slice by token length
    slice_df = _slice_by_token_len(df_test, Yte, Ypred, cfg.text_col)
    slice_path = cfg.tables_out / "slice_by_token_length.csv"
    slice_df.to_csv(slice_path, index=False, encoding="utf-8-sig")

    # Slice plot
    if not slice_df.empty:
        plt.figure(figsize=(10, 4))
        plt.plot(slice_df["token_bin"], slice_df["micro_f1"], marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title("Micro-F1 by token-length bin (TEST)")
        plt.ylabel("micro_f1")
        plt.tight_layout()
        p = cfg.visuals_out / "slice_micro_f1_by_token_bin.png"
        plt.savefig(p, dpi=220, bbox_inches="tight")
        plt.close()
    else:
        p = cfg.visuals_out / "slice_micro_f1_by_token_bin.png"

    # Reliability for top support labels
    top_support = per_label_df.sort_values("support", ascending=False).head(cfg.reliability_topk)
    rel_paths = []
    for lab in top_support["label"].tolist():
        idx = labels.index(lab)
        y = Yte[:, idx]
        pprob = Pte[:, idx]
        rt = _reliability_table(y, pprob, n_bins=cfg.reliability_bins)
        rt_csv = cfg.tables_out / f"reliability_{lab}.csv"
        rt.to_csv(rt_csv, index=False, encoding="utf-8-sig")
        rt_png = cfg.visuals_out / f"reliability_{lab}.png"
        _save_reliability_plot(rt, rt_png, title=f"Reliability (TEST): {lab}")
        rel_paths.append(rt_png)

    # Interpretability: top word features for top support labels
    feat_df = _top_word_features_topk(
        model=model,
        bundle=bundle,
        y_cols=y_cols,
        labels=labels,
        topk=25,
        which_labels=[labels.index(x) for x in top_support["label"].tolist()],
    )
    feat_path = cfg.tables_out / "top_word_features_top_support_labels.csv"
    feat_df.to_csv(feat_path, index=False, encoding="utf-8-sig")

    # Density summary
    true_cnt = Yte.sum(axis=1)
    pred_cnt = Ypred.sum(axis=1)
    density = pd.DataFrame({
        "true_labels_per_row_mean": [float(true_cnt.mean())],
        "pred_labels_per_row_mean": [float(pred_cnt.mean())],
        "true_labels_per_row_p95": [float(np.quantile(true_cnt, 0.95))],
        "pred_labels_per_row_p95": [float(np.quantile(pred_cnt, 0.95))],
    })
    density_path = cfg.tables_out / "label_density_summary.csv"
    density.to_csv(density_path, index=False, encoding="utf-8-sig")

    # Density plot
    plt.figure(figsize=(7, 4))
    plt.hist(true_cnt, bins=20, alpha=0.7, label="true")
    plt.hist(pred_cnt, bins=20, alpha=0.7, label="pred")
    plt.title("Labels per row distribution (TEST)")
    plt.xlabel("#labels")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    density_png = cfg.visuals_out / "labels_per_row_true_vs_pred.png"
    plt.savefig(density_png, dpi=220, bbox_inches="tight")
    plt.close()

    # Final summary
    final = {
        "model_path": str(model_path),
        "text_col": cfg.text_col,
        "num_labels": int(len(y_cols)),
        "test_metrics": overall,
        "worst_labels_by_f1": worst_df[["label", "support", "precision", "recall", "f1", "threshold"]].to_dict(orient="records"),
        "best_labels_by_f1": per_label_df.tail(cfg.worst_k_labels)[["label", "support", "precision", "recall", "f1", "threshold"]].to_dict(orient="records"),
        "outputs": {
            "tables_out": str(cfg.tables_out),
            "visuals_out": str(cfg.visuals_out),
            "per_label_csv": str(per_label_path),
            "worst_fp_fn_csv": str(err_path),
            "slice_csv": str(slice_path),
            "features_csv": str(feat_path),
            "density_csv": str(density_path),
        },
        "notes": [
            "Worst FP/FN examples are useful for updating LABEL_PATTERNS (improvement preprocessing).",
            "Reliability plots show probability calibration quality for top-support labels.",
        ],
    }
    summary_path = cfg.tables_out / "advanced_eda_summary.json"
    summary_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "overall_metrics": cfg.tables_out / "test_overall_metrics.json",
        "per_label_metrics": per_label_path,
        "worst_fp_fn_examples": err_path,
        "slice_by_token_len": slice_path,
        "top_word_features": feat_path,
        "density_summary": density_path,
        "advanced_summary": summary_path,
        "slice_plot": p,
        "density_plot": density_png,
    }