from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.compare_results import run_compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results/compare", help="Natijalarni yozish papkasi")
    ap.add_argument("--no_files", action="store_true", help="Faylga yozmasin (faqat console)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    run_compare(out_dir=out_dir, write_files=(not args.no_files))


if __name__ == "__main__":
    main()