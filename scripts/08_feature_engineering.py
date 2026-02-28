from __future__ import annotations

import argparse

from src.features.feature_engineering import FeatureEngineeringConfig, run_feature_engineering


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="09 Feature Engineering (FAERS multilabel).")
    p.add_argument("--split_dir_name", type=str, default="splits_multilabel_noleakage",
                   help="Data/Processed ichidagi split papka nomi.")
    p.add_argument("--text_col", type=str, default="REAC_pt_symptom_v2",
                   help="Text column nomi.")
    p.add_argument("--fe_version", type=str, default="fe_v1",
                   help="Engineered_data/ va artifacts/ ichidagi versiya nomi.")
    p.add_argument("--no_ids", action="store_true",
                   help="ID/idx saqlashni o‘chiradi.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FeatureEngineeringConfig(
        split_candidates=(args.split_dir_name,),
        text_col=args.text_col,
        fe_version=args.fe_version,
        save_ids=(not args.no_ids),
    )
    run_feature_engineering(cfg)


if __name__ == "__main__":
    main()