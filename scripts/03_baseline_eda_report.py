from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.paths import find_project_root
from src.eda.baseline_eda_report import BaselineEDAConfig, run_baseline_eda_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="04_baseline_eda_report: validation + baseline EDA summary outputs")
    p.add_argument("--split_dir", type=str, default="", help="Split folder (default: Data/Processed/splits_multilabel_noleakage2)")
    p.add_argument("--text_col", type=str, default="REAC_pt_symptom_v2", help="Text column name")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = find_project_root()

    split_dir = Path(args.split_dir).expanduser().resolve() if args.split_dir else (root / "Data" / "Processed" / "splits_multilabel_noleakage2")

    tables_out = root / "results" / "tables" / "eda_outputs"
    visuals_out = root / "visuals" / "baseline_EDA"

    cfg = BaselineEDAConfig(
        split_dir=split_dir,
        tables_out=tables_out,
        visuals_out=visuals_out,
        text_col=args.text_col,
    )

    print("[04] PROJECT_ROOT:", root)
    print("[04] SPLIT_DIR:", cfg.split_dir)
    print("[04] TABLES_OUT:", cfg.tables_out)
    print("[04] VISUALS_OUT:", cfg.visuals_out)
    print("[04] TEXT_COL:", cfg.text_col)

    paths = run_baseline_eda_report(cfg)

    print("[04] SAVED:")
    for k, v in paths.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()