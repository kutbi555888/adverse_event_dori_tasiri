from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.paths import find_project_root
from src.features.baseline_preprocess import BaselinePreprocessConfig, run_baseline_preprocess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="05_baseline_preprocess: TF-IDF (char+word) for multilabel baseline")
    p.add_argument("--split_dir", type=str, default="", help="Split folder (default: Data/Processed/splits_multilabel_noleakage2)")
    p.add_argument("--out_dir", type=str, default="", help="Output folder (default: Data/Processed/baseline_preprocess_v2)")
    p.add_argument("--text_col", type=str, default="REAC_pt_symptom_v2", help="Text column name")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = find_project_root()

    split_dir = Path(args.split_dir).expanduser().resolve() if args.split_dir else (root / "Data" / "Processed" / "splits_multilabel_noleakage2")
    out_dir   = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / "Data" / "Processed" / "baseline_preprocess_v2")

    cfg = BaselinePreprocessConfig(
        split_dir=split_dir,
        out_dir=out_dir,
        text_col=args.text_col,
    )

    print("[05] PROJECT_ROOT:", root)
    print("[05] SPLIT_DIR:", cfg.split_dir)
    print("[05] OUT_DIR:", cfg.out_dir)
    print("[05] TEXT_COL:", cfg.text_col)

    paths = run_baseline_preprocess(cfg)

    print("[05] SAVED:")
    for k, v in paths.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()