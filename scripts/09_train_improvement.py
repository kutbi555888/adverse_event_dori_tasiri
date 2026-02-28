from __future__ import annotations

import argparse

from src.models.improvement_train import TrainImprovementConfig, run_train_improvement


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="09 Train Improvement (single model) + threshold tuning + save.")
    p.add_argument("--version", type=str, default="fe_v1", help="Input version: Feature_Selected yoki Engineered_data.")
    p.add_argument("--prefer_feature_selected", action="store_true", help="Avval Data/Feature_Selected dan qidirsin.")
    p.add_argument("--model_name", type=str, default="ovr_logreg_bal_C1",
                   help="Model name: ovr_logreg_bal_C1 | ovr_logreg_bal_C2 | ovr_linearsvc_C1 | "
                        "ovr_sgd_logloss | ovr_sgd_hinge | ovr_complementnb_a05")
    p.add_argument("--n_thr", type=int, default=61, help="Threshold tuning grid size.")
    p.add_argument("--n_jobs", type=int, default=1, help="n_jobs")
    p.add_argument("--random_state", type=int, default=42, help="random_state")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainImprovementConfig(
        version=args.version,
        prefer_feature_selected=args.prefer_feature_selected,
        model_name=args.model_name,
        n_thr=args.n_thr,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    run_train_improvement(cfg)


if __name__ == "__main__":
    main()