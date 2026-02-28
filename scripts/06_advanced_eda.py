from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.paths import find_project_root
from src.eda.advanced_eda_report import AdvancedEDAConfig, run_advanced_eda


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="06_advanced_eda_report: advanced error analysis + calibration + interpretability")
    p.add_argument("--split_dir", type=str, default="", help="Split folder (default: Data/Processed/splits_multilabel_noleakage2)")
    p.add_argument("--pp_dir", type=str, default="", help="Preprocess folder (default: Data/Processed/baseline_preprocess_v2)")
    p.add_argument("--model_dir", type=str, default="", help="Model folder (default: Models/baseline_models)")
    p.add_argument("--text_col", type=str, default="REAC_pt_symptom_v2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = find_project_root()

    split_dir = Path(args.split_dir).expanduser().resolve() if args.split_dir else (root / "Data" / "Processed" / "splits_multilabel_noleakage2")
    pp_dir = Path(args.pp_dir).expanduser().resolve() if args.pp_dir else (root / "Data" / "Processed" / "baseline_preprocess_v2")
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else (root / "Models" / "baseline_models")

    tables_out = root / "results" / "tables" / "advanced_eda_outputs"
    visuals_out = root / "visuals" / "advanced_EDA"

    cfg = AdvancedEDAConfig(
        split_dir=split_dir,
        pp_dir=pp_dir,
        model_dir=model_dir,
        tables_out=tables_out,
        visuals_out=visuals_out,
        text_col=args.text_col,
    )

    print("[06_ADV] PROJECT_ROOT:", root)
    print("[06_ADV] SPLIT_DIR:", cfg.split_dir)
    print("[06_ADV] PP_DIR:", cfg.pp_dir)
    print("[06_ADV] MODEL_DIR:", cfg.model_dir)
    print("[06_ADV] TABLES_OUT:", cfg.tables_out)
    print("[06_ADV] VIS_OUT:", cfg.visuals_out)
    print("[06_ADV] TEXT_COL:", cfg.text_col)

    paths = run_advanced_eda(cfg)

    print("[06_ADV] SAVED:")
    for k, v in paths.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()