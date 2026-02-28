from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.paths import find_project_root
from src.models.train_baseline_logreg import TrainLogRegConfig, run_train_logreg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="06_train_baseline_logreg: OvR LogisticRegression baseline training")
    p.add_argument("--pp_dir", type=str, default="", help="Preprocess dir (default: Data/Processed/baseline_preprocess_v2)")
    p.add_argument("--solver", type=str, default="liblinear")
    p.add_argument("--max_iter", type=int, default=200)
    p.add_argument("--thr_min", type=float, default=0.05)
    p.add_argument("--thr_max", type=float, default=0.95)
    p.add_argument("--thr_steps", type=int, default=19)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = find_project_root()

    pp_dir = Path(args.pp_dir).expanduser().resolve() if args.pp_dir else (root / "Data" / "Processed" / "baseline_preprocess_v2")

    cfg = TrainLogRegConfig(
        pp_dir=pp_dir,
        model_dir=(root / "Models" / "baseline_models"),
        tables_dir=(root / "results" / "tables" / "baseline_train"),
        reports_dir=(root / "results" / "reports" / "baseline_train"),
        solver=args.solver,
        max_iter=args.max_iter,
        thr_min=args.thr_min,
        thr_max=args.thr_max,
        thr_steps=args.thr_steps,
        ovr_n_jobs=1,  # Windows stable
    )

    print("[06] PROJECT_ROOT:", root)
    print("[06] PP_DIR:", cfg.pp_dir)
    print("[06] MODEL_DIR:", cfg.model_dir)
    print("[06] TABLES_DIR:", cfg.tables_dir)
    print("[06] REPORTS_DIR:", cfg.reports_dir)

    paths = run_train_logreg(cfg)

    print("[06] SAVED:")
    for k, v in paths.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()