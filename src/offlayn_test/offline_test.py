from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import json
import re
import numpy as np
import pandas as pd
from scipy import sparse
import joblib
from tabulate import tabulate
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import brier_score_loss
import heapq
import __main__


# =========================================================
# ROOT helpers
# =========================================================

def find_project_root(start: Optional[Path] = None) -> Path:
    start = start or Path.cwd()
    for p in [start] + list(start.parents):
        if (p / "Data").exists() and (p / "Models").exists():
            return p
    for p in [start] + list(start.parents):
        if (p / "Data").exists():
            return p
    return start


# =========================================================
# Custom meta funcs (joblib load uchun __main__ ichiga inject qilamiz)
# =========================================================

_term_split = re.compile(r"\s*;\s*")


def meta_features(texts: List[str]) -> np.ndarray:
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

        terms = [t.strip().lower() for t in _term_split.split(s) if t.strip()]
        n_terms.append(len(terms))
        n_uniq_terms.append(len(set(terms)))

    lens = np.array(lens, dtype=np.float32).reshape(-1, 1)
    n_terms = np.array(n_terms, dtype=np.float32).reshape(-1, 1)
    n_uniq_terms = np.array(n_uniq_terms, dtype=np.float32).reshape(-1, 1)

    feats = np.hstack([np.log1p(lens), n_terms, n_uniq_terms]).astype(np.float32)
    return feats


def meta_to_sparse(texts):
    feats = meta_features(list(texts))
    return sparse.csr_matrix(feats)


def ensure_main_has_meta_functions() -> None:
    # joblib pickle __main__.meta_to_sparse qidirgani uchun shu yerga inject qilamiz
    setattr(__main__, "_term_split", _term_split)
    setattr(__main__, "meta_features", meta_features)
    setattr(__main__, "meta_to_sparse", meta_to_sparse)
    setattr(__main__, "FunctionTransformer", FunctionTransformer)
    setattr(__main__, "sparse", sparse)


# =========================================================
# Loaders
# =========================================================

@dataclass
class OfflineConfig:
    fe_version: str = "fe_v1"
    fs_version: str = "fe_v1_fs_chi2_v1"
    model_name: str = "optuna_logreg_best"
    text_col: str = "REAC_pt_symptom_v2"
    csv_path: str = "Data/Raw_data/faers_25Q4_targets_multilabel_v2.csv"
    topk: int = 10


@dataclass
class OfflineAssets:
    project_root: Path
    featurizer: Any
    mask: np.ndarray
    model: Any
    label_names: List[str]
    thr: np.ndarray
    csv_path: Path


def load_assets(cfg: OfflineConfig, project_root: Optional[Path] = None) -> OfflineAssets:
    project_root = project_root or find_project_root()

    # paths
    vect_path = project_root / "Data" / "Engineered_data" / cfg.fe_version / "tfidf_vectorizer.joblib"
    sel_path = project_root / "Data" / "Feature_Selected" / cfg.fs_version / "feature_selector.joblib"

    model_dir = project_root / "Models" / "best_model" / cfg.model_name
    model_path = model_dir / f"{cfg.model_name}.joblib"
    thr_path = model_dir / f"{cfg.model_name}_thresholds.json"

    meta_candidates = [
        project_root / "Data" / "Feature_Selected" / cfg.fs_version / "engineered_meta.json",
        project_root / "Data" / "Engineered_data" / cfg.fe_version / "meta.json",
    ]

    csv_path = project_root / cfg.csv_path

    # checks
    if not vect_path.exists():
        raise FileNotFoundError(f"Topilmadi: {vect_path}")
    if not sel_path.exists():
        raise FileNotFoundError(f"Topilmadi: {sel_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Topilmadi: {model_path}")
    if not thr_path.exists():
        raise FileNotFoundError(f"Topilmadi: {thr_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Topilmadi: {csv_path}")

    meta_path = next((p for p in meta_candidates if p.exists()), None)
    if meta_path is None:
        raise FileNotFoundError(f"Meta topilmadi: {meta_candidates}")

    # load meta -> label order
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    y_cols = meta["y_cols"]
    label_names = [c.replace("y_", "", 1) for c in y_cols]

    # load thresholds
    with open(thr_path, "r", encoding="utf-8") as f:
        thr_dict = json.load(f)
    thr = np.array([float(thr_dict.get(name, 0.5)) for name in label_names], dtype=float)

    # load selector (mask payload)
    selector_payload = joblib.load(sel_path)
    if not (isinstance(selector_payload, dict) and "mask" in selector_payload):
        raise ValueError("feature_selector.joblib ichida 'mask' yo‘q. Payload dict bo‘lishi kerak.")
    mask = np.array(selector_payload["mask"], dtype=bool)

    # load featurizer (NECESSARY: inject meta_to_sparse into __main__)
    ensure_main_has_meta_functions()
    featurizer = joblib.load(vect_path)

    # load model
    model = joblib.load(model_path)

    return OfflineAssets(
        project_root=project_root,
        featurizer=featurizer,
        mask=mask,
        model=model,
        label_names=label_names,
        thr=thr,
        csv_path=csv_path,
    )


# =========================================================
# Prediction utilities
# =========================================================

def predict_text(
    assets: OfflineAssets,
    text: str,
    topk: int = 10,
) -> Dict[str, Any]:
    X_full = assets.featurizer.transform(pd.Series([str(text)]))
    X_fs = X_full[:, assets.mask]

    if hasattr(assets.model, "predict_proba"):
        scores = np.asarray(assets.model.predict_proba(X_fs)).reshape(1, -1)[0]
    else:
        scores = np.asarray(assets.model.decision_function(X_fs)).reshape(1, -1)[0]

    pred = (scores >= assets.thr).astype(int)
    pred_labels = [assets.label_names[i] for i, v in enumerate(pred) if v == 1]

    top_idx = np.argsort(scores)[::-1][:topk]
    top = [(assets.label_names[i], float(scores[i]), float(assets.thr[i])) for i in top_idx]

    return {
        "pred_labels": pred_labels,
        "top": top,
        "confidence_max": float(np.max(scores)),
        "scores": scores,
    }


def load_df_raw_min(csv_path: Path, text_col: str, keep_y: bool = True) -> pd.DataFrame:
    usecols = ["primaryid", text_col]
    if keep_y:
        usecols.append("y_labels")
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["primaryid"]).copy()
    df["primaryid"] = df["primaryid"].astype(np.int64)
    df = df.set_index("primaryid")
    return df


def predict_by_primaryid(
    assets: OfflineAssets,
    df_raw_indexed: pd.DataFrame,
    primaryid: int,
    topk: int = 10,
) -> Dict[str, Any]:
    if primaryid not in df_raw_indexed.index:
        return {"status": "NOT_FOUND", "primaryid": primaryid}

    text = str(df_raw_indexed.loc[primaryid, assets_config_text_col(df_raw_indexed, assets)])
    true_labels = None
    if "y_labels" in df_raw_indexed.columns:
        true_labels = df_raw_indexed.loc[primaryid, "y_labels"]

    pred = predict_text(assets, text=text, topk=topk)
    return {
        "status": "OK",
        "primaryid": primaryid,
        "text": text,
        "true_y_labels": true_labels,
        **pred,
    }


def assets_config_text_col(df_raw_indexed: pd.DataFrame, assets: OfflineAssets) -> str:
    # df_raw_indexed columns ichida REAC_pt_symptom_v2 bor bo‘lsa shuni olamiz, bo‘lmasa fallback
    if "REAC_pt_symptom_v2" in df_raw_indexed.columns:
        return "REAC_pt_symptom_v2"
    if "REAC_pt_symptom" in df_raw_indexed.columns:
        return "REAC_pt_symptom"
    # last resort: first col
    return df_raw_indexed.columns[0]


# =========================================================
# Test reliability (agar X_test/Y_test bor bo‘lsa)
# =========================================================

def eval_on_test_if_exists(assets: OfflineAssets, fs_version: str) -> Optional[Dict[str, float]]:
    x_path = assets.project_root / "Data" / "Feature_Selected" / fs_version / "X_test.npz"
    y_path = assets.project_root / "Data" / "Feature_Selected" / fs_version / "Y_test.npy"
    if not (x_path.exists() and y_path.exists()):
        return None

    X_test = sparse.load_npz(x_path).tocsr()
    Y_test = np.load(y_path)

    if hasattr(assets.model, "predict_proba"):
        s = np.asarray(assets.model.predict_proba(X_test))
    else:
        s = np.asarray(assets.model.decision_function(X_test))
    s = s.reshape(X_test.shape[0], -1)

    Y_pred = (s >= assets.thr.reshape(1, -1)).astype(int)

    tp = int(((Y_test == 1) & (Y_pred == 1)).sum())
    fp = int(((Y_test == 0) & (Y_pred == 1)).sum())
    fn = int(((Y_test == 1) & (Y_pred == 0)).sum())

    micro_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

    f1s = []
    for j in range(Y_test.shape[1]):
        y = Y_test[:, j]
        p = Y_pred[:, j]
        tpj = int(((y == 1) & (p == 1)).sum())
        fpj = int(((y == 0) & (p == 1)).sum())
        fnj = int(((y == 1) & (p == 0)).sum())
        precj = tpj / (tpj + fpj) if (tpj + fpj) > 0 else 0.0
        recj = tpj / (tpj + fnj) if (tpj + fnj) > 0 else 0.0
        f1j = (2 * precj * recj / (precj + recj)) if (precj + recj) > 0 else 0.0
        f1s.append(f1j)
    macro_f1 = float(np.mean(f1s))

    briers = [brier_score_loss(Y_test[:, j], s[:, j]) for j in range(Y_test.shape[1])]
    mean_brier = float(np.mean(briers))

    return {
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "mean_brier": float(mean_brier),
    }


# =========================================================
# Hard-case miner (top N hardest, pick K)
# =========================================================

def split_labels(s: Any) -> List[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    return [t.strip() for t in str(s).split(";") if t.strip()]


def mine_hard_cases(
    assets: OfflineAssets,
    csv_path: Path,
    text_col: str,
    top: int = 30,
    pick: int = 8,
    chunksize: int = 2000,
    max_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    label2idx = {name: i for i, name in enumerate(assets.label_names)}

    heap: List[Tuple[float, Dict[str, Any]]] = []  # store (-margin, rec)

    def push(rec: Dict[str, Any]):
        key = -float(rec["overall_margin"])
        if len(heap) < top:
            heapq.heappush(heap, (key, rec))
        else:
            if key > heap[0][0]:
                heapq.heapreplace(heap, (key, rec))

    usecols = ["primaryid", text_col, "y_labels"]
    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False)

    for ci, chunk in enumerate(reader, start=1):
        if max_chunks is not None and ci > max_chunks:
            break

        chunk["primaryid"] = pd.to_numeric(chunk["primaryid"], errors="coerce")
        chunk = chunk.dropna(subset=["primaryid", text_col]).copy()
        if len(chunk) == 0:
            continue
        chunk["primaryid"] = chunk["primaryid"].astype(np.int64)

        texts = chunk[text_col].astype(str)

        X_full = assets.featurizer.transform(texts)
        X_fs = X_full[:, assets.mask]

        if hasattr(assets.model, "predict_proba"):
            scores = np.asarray(assets.model.predict_proba(X_fs))
        else:
            scores = np.asarray(assets.model.decision_function(X_fs))

        if scores.ndim == 1:
            scores = scores.reshape(-1, len(assets.label_names))

        deltas = scores - assets.thr.reshape(1, -1)

        for i in range(len(chunk)):
            pid = int(chunk.iloc[i]["primaryid"])
            text = str(chunk.iloc[i][text_col])
            y_true_str = chunk.iloc[i]["y_labels"]
            true_names = split_labels(y_true_str)

            true_idx = [label2idx[n] for n in true_names if n in label2idx]
            if len(true_idx) == 0:
                continue

            d = deltas[i]
            pos_margin = float(np.min(d[true_idx]))

            mask_false = np.ones(d.shape[0], dtype=bool)
            mask_false[true_idx] = False
            max_false_delta = float(np.max(d[mask_false])) if mask_false.any() else -np.inf
            neg_margin = float(-max_false_delta)

            overall_margin = float(min(pos_margin, neg_margin))

            pred_idx = np.where(scores[i] >= assets.thr)[0]
            pred_names = [assets.label_names[j] for j in pred_idx]

            top5_idx = np.argsort(scores[i])[::-1][:5]
            top5 = "; ".join([f"{assets.label_names[j]}:{scores[i, j]:.3f}(thr={assets.thr[j]:.3f})" for j in top5_idx])

            rec = {
                "primaryid": pid,
                "overall_margin": overall_margin,
                "pos_margin": pos_margin,
                "neg_margin": neg_margin,
                "true_n": len(true_idx),
                "pred_n": len(pred_names),
                "true_labels": "; ".join(true_names),
                "pred_labels": "; ".join(pred_names),
                "top5": top5,
                "text": text,
                "text_len": len(text),
            }
            push(rec)

    hard = [r for _, r in sorted(heap, key=lambda x: x[0], reverse=True)]
    df_hard = pd.DataFrame(hard).sort_values("overall_margin", ascending=True).reset_index(drop=True)

    # pick K by diverse true_n
    picked: List[int] = []
    used = set()
    for target_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
        sub = df_hard[df_hard["true_n"] == target_n]
        if len(sub) == 0:
            continue
        pid = int(sub.iloc[0]["primaryid"])
        if pid in used:
            continue
        used.add(pid)
        picked.append(pid)
        if len(picked) >= pick:
            break

    if len(picked) < pick:
        for pid in df_hard["primaryid"].tolist():
            pid = int(pid)
            if pid in used:
                continue
            used.add(pid)
            picked.append(pid)
            if len(picked) >= pick:
                break

    return {"df_hard": df_hard, "picked": picked}


def format_hard_table(df_hard: pd.DataFrame) -> str:
    cols = ["primaryid", "overall_margin", "pos_margin", "neg_margin", "true_n", "pred_n", "true_labels", "pred_labels", "top5", "text_len"]
    return tabulate(df_hard[cols], headers="keys", tablefmt="github", showindex=False, floatfmt=".6f")


def detailed_true_pred_for_pids(
    assets: OfflineAssets,
    df_raw_indexed: pd.DataFrame,
    pids: List[int],
    topk: int = 10,
    text_col: str = "REAC_pt_symptom_v2",
) -> str:
    out_lines: List[str] = []
    for pid in pids:
        out_lines.append("\n" + "=" * 110)
        out_lines.append(f"PRIMARYID: {pid}")
        out_lines.append("=" * 110)

        if pid not in df_raw_indexed.index:
            out_lines.append("❌ NOT FOUND in df_raw")
            continue

        text = str(df_raw_indexed.loc[pid, text_col])
        true_str = df_raw_indexed.loc[pid, "y_labels"] if "y_labels" in df_raw_indexed.columns else None
        true_set = set(split_labels(true_str))

        pred = predict_text(assets, text=text, topk=topk)
        pred_set = set(pred["pred_labels"])

        missing = sorted(list(true_set - pred_set))
        extra = sorted(list(pred_set - true_set))

        out_lines.append("\nTEXT:")
        out_lines.append(text)
        out_lines.append("\nTRUE y_labels:")
        out_lines.append(str(true_str))
        out_lines.append("\nPRED labels:")
        out_lines.append("; ".join(pred["pred_labels"]) if pred["pred_labels"] else "(none)")
        out_lines.append("\nDIFF:")
        out_lines.append(" - missing_true: " + ("; ".join(missing) if missing else "(none)"))
        out_lines.append(" - extra_pred  : " + ("; ".join(extra) if extra else "(none)"))

        out_lines.append("\nTop-10 scores (score vs thr):")
        out_lines.append(tabulate(pred["top"], headers=["label", "score", "thr"], tablefmt="github", floatfmt=".4f"))

    return "\n".join(out_lines)