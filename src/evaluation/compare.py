from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from tabulate import tabulate


# =========================================================
# DEFAULT (hard-coded) metrics — siz notebookdan olgan raqamlar
# =========================================================

def get_default_results() -> Dict[str, List[Dict[str, Any]]]:
    baseline_rows = [
        {"model": "baseline_ovr_linearsvc",            "test_micro_f1": 0.999500, "test_macro_f1": 0.998800, "test_micro_precision": 0.999200, "test_micro_recall": 0.999800},
        {"model": "baseline_ovr_calibrated_linearsvc", "test_micro_f1": 0.999200, "test_macro_f1": 0.998400, "test_micro_precision": 0.998900, "test_micro_recall": 0.999500},
        {"model": "baseline_ovr_logreg",               "test_micro_f1": 0.997300, "test_macro_f1": 0.995800, "test_micro_precision": 0.996000, "test_micro_recall": 0.998600},
        {"model": "baseline_ovr_sgd_logloss",          "test_micro_f1": 0.995000, "test_macro_f1": 0.992000, "test_micro_precision": 0.993400, "test_micro_recall": 0.996600},
    ]

    improvement_rows = [
        {"model": "ovr_logreg_bal_C2",     "val_micro_f1": 0.980376, "val_macro_f1": 0.965836},
        {"model": "ovr_linearsvc_C1",      "val_micro_f1": 0.980182, "val_macro_f1": 0.964417},
        {"model": "ovr_logreg_bal_C1",     "val_micro_f1": 0.978403, "val_macro_f1": 0.944185},
        {"model": "ovr_sgd_logloss",       "val_micro_f1": 0.974033, "val_macro_f1": 0.952951},
        {"model": "ovr_sgd_hinge",         "val_micro_f1": 0.957535, "val_macro_f1": 0.936905},
        {"model": "ovr_complementnb_a05",  "val_micro_f1": 0.913360, "val_macro_f1": 0.882115},
    ]

    tuning_rows = [
        {"model": "optuna_logreg",      "val_micro_f1": 0.962941, "val_macro_f1": 0.930969, "test_micro_f1": 0.957416, "test_macro_f1": 0.916767},
        {"model": "optuna_linearsvc",   "val_micro_f1": 0.961265, "val_macro_f1": 0.908998, "test_micro_f1": 0.957016, "test_macro_f1": 0.912317},
        {"model": "optuna_sgd_logloss", "val_micro_f1": 0.950963, "val_macro_f1": 0.916381, "test_micro_f1": 0.943415, "test_macro_f1": 0.903986},
        {"model": "optuna_sgd_hinge",   "val_micro_f1": 0.948021, "val_macro_f1": 0.888889, "test_micro_f1": 0.938087, "test_macro_f1": 0.895398},
    ]

    best_model_rows = [
        {"model": "optuna_logreg_best_final", "val_micro_f1": 0.984831, "val_macro_f1": 0.973785, "test_micro_f1": 0.978199, "test_macro_f1": 0.966022},
    ]

    return {
        "baseline": baseline_rows,
        "improvement": improvement_rows,
        "tuning": tuning_rows,
        "best_model": best_model_rows,
    }


# =========================================================
# Formatting helpers (🟩/🟥) + printing/saving
# =========================================================

def mark_max_min(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        if s.isna().all():
            continue
        vmax, vmin = s.max(), s.min()

        def fmt(v):
            if pd.isna(v):
                return "—"
            v = float(v)
            txt = f"{v:.6f}"
            if v == vmax and v == vmin:
                return txt + "🟩🟥"
            if v == vmax:
                return txt + "🟩"
            if v == vmin:
                return txt + "🟥"
            return txt

        out[c] = s.map(fmt)
    return out


def to_markdown_table(df: pd.DataFrame) -> str:
    return tabulate(df, headers="keys", tablefmt="github", showindex=False)


def show_stage(title: str, rows: List[Dict[str, Any]], mark_cols: List[str], sort_by: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame(rows)
    if df.empty:
        md = f"\n{'='*90}\n{title}\n{'='*90}\n(Natija yo‘q)\n"
        print(md)
        return df, md

    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)

    df_disp = mark_max_min(df, mark_cols)
    md = f"\n{'='*90}\n{title}\n{'='*90}\n{to_markdown_table(df_disp)}\n"
    print(md)
    return df, md


def best_of_stage_summary(
    df_base: pd.DataFrame,
    df_impr: pd.DataFrame,
    df_tune: pd.DataFrame,
    df_best: pd.DataFrame,
) -> pd.DataFrame:
    def pick_best(df: pd.DataFrame, prefer_test: bool) -> Optional[Dict[str, Any]]:
        if df.empty:
            return None
        if prefer_test and "test_micro_f1" in df.columns:
            key = "test_micro_f1"
        elif "val_micro_f1" in df.columns:
            key = "val_micro_f1"
        else:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            key = num_cols[0] if num_cols else None
        if key is None:
            return None
        return df.sort_values(key, ascending=False).iloc[0].to_dict()

    rows = []
    b = pick_best(df_base, prefer_test=True)
    if b:
        b["stage"] = "baseline"
        rows.append(b)

    i = pick_best(df_impr, prefer_test=False)
    if i:
        i["stage"] = "improvement"
        rows.append(i)

    t = pick_best(df_tune, prefer_test=True)
    if t:
        t["stage"] = "tuning"
        rows.append(t)

    bm = pick_best(df_best, prefer_test=True)
    if bm:
        bm["stage"] = "best_model (final)"
        rows.append(bm)

    summary = pd.DataFrame(rows)
    wanted = ["stage", "model", "val_micro_f1", "val_macro_f1", "test_micro_f1", "test_macro_f1", "test_micro_precision", "test_micro_recall"]
    summary = summary[[c for c in wanted if c in summary.columns]]
    return summary


def run_compare(
    out_dir: Path,
    write_files: bool = True,
    data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    data = data or get_default_results()

    all_md_parts: List[str] = []

    df_base, md = show_stage(
        "BASELINE — TEST metrics",
        data["baseline"],
        mark_cols=["test_micro_f1", "test_macro_f1", "test_micro_precision", "test_micro_recall"],
        sort_by="test_micro_f1",
    )
    all_md_parts.append(md)

    df_impr, md = show_stage(
        "IMPROVEMENT — VAL metrics",
        data["improvement"],
        mark_cols=["val_micro_f1", "val_macro_f1"],
        sort_by="val_micro_f1",
    )
    all_md_parts.append(md)

    df_tune, md = show_stage(
        "TUNING (NO_OVERSAMPLING) — VAL/TEST metrics",
        data["tuning"],
        mark_cols=["val_micro_f1", "val_macro_f1", "test_micro_f1", "test_macro_f1"],
        sort_by="test_micro_f1",
    )
    all_md_parts.append(md)

    df_best, md = show_stage(
        "BEST MODEL (FINAL TRAIN) — VAL/TEST metrics",
        data["best_model"],
        mark_cols=["val_micro_f1", "val_macro_f1", "test_micro_f1", "test_macro_f1"],
        sort_by="test_micro_f1",
    )
    all_md_parts.append(md)

    # summary
    summary = best_of_stage_summary(df_base, df_impr, df_tune, df_best)
    sum_disp = mark_max_min(summary, [c for c in summary.columns if c not in ("stage", "model")])
    md_sum = f"\n{'='*90}\nBEST-OF-STAGE — yonma-yon taqqoslash\n{'='*90}\n{to_markdown_table(sum_disp)}\n"
    print(md_sum)
    all_md_parts.append(md_sum)

    if write_files:
        (out_dir / "compare_results.md").write_text("".join(all_md_parts), encoding="utf-8")
        summary.to_csv(out_dir / "best_of_stage.csv", index=False, encoding="utf-8")

    return {
        "baseline": df_base,
        "improvement": df_impr,
        "tuning": df_tune,
        "best_model": df_best,
        "best_of_stage": summary,
        "markdown": "".join(all_md_parts),
    }