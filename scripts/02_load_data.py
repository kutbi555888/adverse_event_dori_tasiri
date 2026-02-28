from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.paths import find_project_root
from src.data.split_multilabel import (
    SplitConfig,
    load_minimal_dataset,
    sanitize_xy,
    make_train_val_test_splits,
    save_splits,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="03_load_data: multilabel dataset split (train/val/test)")
    p.add_argument("--in_csv", type=str, default="", help="Input CSV path (default: Data/Raw_data/faers_25Q4_targets_multilabel.csv)")
    p.add_argument("--out_dir", type=str, default="", help="Output dir (default: Data/Processed/splits_multilabel)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.10)
    p.add_argument("--val_size", type=float, default=0.10)
    p.add_argument("--text_col", type=str, default="REAC_pt_symptom")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = find_project_root()
    default_in = root / "Data" / "Raw_data" / "faers_25Q4_targets_multilabel.csv"
    default_out = root / "Data" / "Processed" / "splits_multilabel"

    in_csv = Path(args.in_csv).expanduser().resolve() if args.in_csv else default_in
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out

    cfg = SplitConfig(
        text_col=args.text_col,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    print(f"[03] PROJECT_ROOT: {root}")
    print(f"[03] IN_CSV: {in_csv}")
    print(f"[03] OUT_DIR: {out_dir}")

    df, y_cols = load_minimal_dataset(in_csv, cfg)
    print(f"[03] Loaded shape: {df.shape}")
    print(f"[03] Num labels: {len(y_cols)}")

    df, Y = sanitize_xy(df, y_cols, cfg)
    print(f"[03] After sanitize: {df.shape}")

    df_train, df_val, df_test, meta = make_train_val_test_splits(df, Y, cfg)
    print(f"[03] Split sizes -> train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,}")
    print(f"[03] Methods -> {meta['split1_method']} / {meta['split2_method']}")

    paths = save_splits(df_train, df_val, df_test, y_cols, out_dir, cfg, meta=meta)

    print("[03] Saved:")
    for k, v in paths.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
    
    
    
# python scripts/03_load_data.py    