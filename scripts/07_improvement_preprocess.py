from __future__ import annotations

import argparse

from src.preprocess.improvement_preprocess import ImprovementConfig, run_improvement_preprocess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FAERS v2 improvement preprocess (CSV-only).")
    p.add_argument("--input_csv", type=str, default=None, help="Agar berilsa, shu CSV ishlatiladi.")
    p.add_argument("--top_terms_n", type=int, default=2000, help="0-label top termlar soni.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ImprovementConfig(top_terms_n=args.top_terms_n)
    run_improvement_preprocess(input_csv=args.input_csv, cfg=cfg)


if __name__ == "__main__":
    main()