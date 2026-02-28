from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.offline_testing import (
    OfflineConfig,
    load_assets,
    load_df_raw_min,
    predict_text,
    eval_on_test_if_exists,
    mine_hard_cases,
    format_hard_table,
    detailed_true_pred_for_pids,
)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--fe_version", type=str, default="fe_v1")
    ap.add_argument("--fs_version", type=str, default="fe_v1_fs_chi2_v1")
    ap.add_argument("--model_name", type=str, default="optuna_logreg_best")
    ap.add_argument("--text_col", type=str, default="REAC_pt_symptom_v2")
    ap.add_argument("--csv_path", type=str, default="Data/Raw_data/faers_25Q4_targets_multilabel_v2.csv")

    ap.add_argument("--primaryid", type=int, default=None, help="CSV ichidan primaryid bo‘yicha predict")
    ap.add_argument("--text", type=str, default=None, help="To‘g‘ridan-to‘g‘ri text berib predict")

    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--eval_test", action="store_true", help="Agar X_test/Y_test bo‘lsa test metrikani chiqaradi")

    ap.add_argument("--hard_cases", action="store_true", help="Hard-case topish")
    ap.add_argument("--hard_top", type=int, default=30)
    ap.add_argument("--hard_pick", type=int, default=8)
    ap.add_argument("--chunksize", type=int, default=2000)
    ap.add_argument("--max_chunks", type=int, default=None)

    ap.add_argument("--write_md", action="store_true", help="results/offline/ ichiga .md yozadi")
    args = ap.parse_args()

    cfg = OfflineConfig(
        fe_version=args.fe_version,
        fs_version=args.fs_version,
        model_name=args.model_name,
        text_col=args.text_col,
        csv_path=args.csv_path,
        topk=args.topk,
    )

    assets = load_assets(cfg)

    print("Loaded:")
    print(" - project_root:", assets.project_root.resolve())
    print(" - csv_path    :", assets.csv_path.resolve())
    print(" - labels      :", len(assets.label_names))

    # optional test eval
    if args.eval_test:
        rep = eval_on_test_if_exists(assets, fs_version=args.fs_version)
        if rep is None:
            print("\n(TEST eval) X_test/Y_test topilmadi -> skip")
        else:
            print("\n--- Model TEST Reliability ---")
            for k, v in rep.items():
                print(f"{k}: {v:.6f}")

    # load df_raw (indexed)
    df_raw = load_df_raw_min(assets.csv_path, text_col=args.text_col, keep_y=True)

    # predict by primaryid
    if args.primaryid is not None:
        pid = int(args.primaryid)
        if pid not in df_raw.index:
            print("\n❌ primaryid topilmadi:", pid)
        else:
            text = str(df_raw.loc[pid, args.text_col])
            true = df_raw.loc[pid, "y_labels"] if "y_labels" in df_raw.columns else None

            pred = predict_text(assets, text=text, topk=args.topk)
            print("\n" + "=" * 110)
            print("PRIMARYID:", pid)
            print("=" * 110)
            print("\nTEXT:\n", text)
            print("\nTRUE y_labels:\n", true)
            print("\nPRED labels:\n", "; ".join(pred["pred_labels"]) if pred["pred_labels"] else "(none)")
            from tabulate import tabulate
            print("\nTop scores (score vs thr):")
            print(tabulate(pred["top"], headers=["label", "score", "thr"], tablefmt="github", floatfmt=".4f"))

    # predict by raw text
    if args.text is not None:
        pred = predict_text(assets, text=args.text, topk=args.topk)
        print("\n" + "=" * 110)
        print("RAW TEXT PREDICTION")
        print("=" * 110)
        print("\nTEXT:\n", args.text)
        print("\nPRED labels:\n", "; ".join(pred["pred_labels"]) if pred["pred_labels"] else "(none)")
        from tabulate import tabulate
        print("\nTop scores (score vs thr):")
        print(tabulate(pred["top"], headers=["label", "score", "thr"], tablefmt="github", floatfmt=".4f"))

    # hard cases
    if args.hard_cases:
        res = mine_hard_cases(
            assets=assets,
            csv_path=assets.csv_path,
            text_col=args.text_col,
            top=args.hard_top,
            pick=args.hard_pick,
            chunksize=args.chunksize,
            max_chunks=args.max_chunks,
        )
        df_hard = res["df_hard"]
        picked = res["picked"]

        print("\n" + "=" * 110)
        print(f"TOP {args.hard_top} HARDEST (margin eng kichik)")
        print("=" * 110)
        print(format_hard_table(df_hard))

        print("\n" + "=" * 110)
        print("PICKED:", picked)
        print("=" * 110)

        detail = detailed_true_pred_for_pids(
            assets=assets,
            df_raw_indexed=df_raw,
            pids=picked,
            topk=args.topk,
            text_col=args.text_col,
        )
        print(detail)

        if args.write_md:
            out_dir = assets.project_root / "results" / "offline"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "hard_top.md").write_text(format_hard_table(df_hard), encoding="utf-8")
            (out_dir / "hard_picked_detail.md").write_text(detail, encoding="utf-8")
            print("\n✅ Saved:", (out_dir / "hard_top.md").resolve())
            print("✅ Saved:", (out_dir / "hard_picked_detail.md").resolve())


if __name__ == "__main__":
    main()