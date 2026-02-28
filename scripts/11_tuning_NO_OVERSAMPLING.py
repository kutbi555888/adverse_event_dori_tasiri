from __future__ import annotations

import argparse

from src.tuning.optuna_tuning import OptunaTuningConfig, run_optuna_tuning


def parse_args():
    p = argparse.ArgumentParser(description="Optuna tuning (LogReg, LinearSVC, SGD log_loss, SGD hinge) + save.")
    p.add_argument("--version", type=str, default="fe_v1_fs_chi2_v1")
    p.add_argument("--prefer_fs", action="store_true")

    p.add_argument("--n_trials", type=int, default=15)
    p.add_argument("--timeout_sec", type=int, default=1800)

    p.add_argument("--n_thr_fast", type=int, default=31)
    p.add_argument("--n_thr_final", type=int, default=61)
    p.add_argument("--train_subsample", type=int, default=60000)

    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OptunaTuningConfig(
        version=args.version,
        prefer_feature_selected=args.prefer_fs,
        n_trials=args.n_trials,
        timeout_sec=args.timeout_sec,
        n_thr_fast=args.n_thr_fast,
        n_thr_final=args.n_thr_final,
        train_subsample=(None if args.train_subsample <= 0 else args.train_subsample),
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    out = run_optuna_tuning(cfg)
    print("\nDONE. leaderboard:", out["leaderboard_path"])


if __name__ == "__main__":
    main()