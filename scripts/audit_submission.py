"""Audit submission tabular KS: format + OOF KS + distribusi + leakage checklist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_ks import fold_ks
from predict import validate_submission


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", help="path submission.csv")
    ap.add_argument("--test", default="data/test_applications.csv")
    ap.add_argument(
        "--oof", default=None, help="optional oof.csv with target,pred,fold"
    )
    args = ap.parse_args()

    sub = pd.read_csv(args.submission)
    try:
        test_ids = pd.read_csv(args.test, usecols=["application_id"])["application_id"]
    except FileNotFoundError:
        print("WARN: test tidak ditemukan, cek format saja")
        test_ids = sub["application_id"]

    errors = validate_submission(sub, test_ids)
    print(f"file: {args.submission}")
    print(f"rows: {len(sub)}, mean_prob: {sub['default_probability'].mean():.4f}")
    print(
        f"deciles: {sub['default_probability'].quantile([0, 0.1, 0.5, 0.9, 1]).to_dict()}"
    )
    if errors:
        print("FAIL:")
        for e in errors:
            print(f" - {e}")
        return 1
    print("format: OK")

    if args.oof:
        oof = pd.read_csv(args.oof)
        scores = fold_ks(
            oof["target"].to_numpy(), oof["pred"].to_numpy(), oof["fold"].to_numpy()
        )
        print(f"OOF KS: {scores}")

    print("leakage checklist: pastikan semua fitur memakai cutoff decision_datetime")
    print("kuota: cek manual di Kaggle (maks 3/hari)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
