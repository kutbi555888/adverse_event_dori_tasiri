from __future__ import annotations

import argparse

from src.features.feature_selection import FeatureSelectionConfig, run_feature_selection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="09B Feature Selection (chi2 top-k per label union).")
    p.add_argument("--fe_in", type=str, default="fe_v1", help="Engineered_data version (input).")
    p.add_argument("--fs_out", type=str, default="fe_v1_fs_chi2_v1", help="Feature_Selected version (output).")
    p.add_argument("--topk_per_label", type=int, default=3000, help="Top-k per label (chi2).")
    p.add_argument("--max_total_features", type=int, default=250000, help="Union cap.")
    p.add_argument("--keep_meta_last_n", type=int, default=3, help="Keep last N meta features.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FeatureSelectionConfig(
        fe_version_in=args.fe_in,
        fs_version_out=args.fs_out,
        topk_per_label=args.topk_per_label,
        max_total_features=args.max_total_features,
        keep_meta_last_n=args.keep_meta_last_n,
    )
    run_feature_selection(cfg)


if __name__ == "__main__":
    main()