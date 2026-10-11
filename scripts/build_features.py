"""Build leakage-safe application-level features for all tables.

Thin orchestrator: per-table builders live in src/features/ (one module
per table). Add new features inside the owning module — never rewrite
another table's builder. See docs/WORKFLOW.md.

Usage:
    python scripts/build_features.py --data data \
        --out-train data/train_features.parquet \
        --out-test data/test_features.parquet

Outputs are ignored by git (data/*.parquet); they are regenerable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.account import agg_account
from src.features.base import base_features, scored_apps
from src.features.bureau import agg_bureau
from src.features.cross import apply_cross_ratios, finalize
from src.features.eco import agg_eco
from src.features.prev import agg_prev
from src.features.repay import agg_repay

BUILDERS = [
    ("prev", agg_prev),
    ("repay", agg_repay),
    ("account", agg_account),
    ("bureau", agg_bureau),
    ("eco", agg_eco),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out-train", default="data/train_features.parquet")
    ap.add_argument("--out-test", default="data/test_features.parquet")
    args = ap.parse_args()
    data = Path(args.data)

    apps = scored_apps(data)
    print(f"scored apps: {len(apps)}")

    feat = base_features(apps, data)
    for name, fn in BUILDERS:
        part = fn(data, apps)
        print(f"{name}: {len(part)} applications with history")
        feat = feat.merge(part, on="application_id", how="left")
    feat = apply_cross_ratios(feat)
    feat = finalize(feat)

    train = feat[feat["is_train"]].drop(columns=["is_train"])
    test = feat[~feat["is_train"]].drop(columns=["is_train", "target"])
    train.to_parquet(args.out_train, index=False)
    test.to_parquet(args.out_test, index=False)
    print(f"train {train.shape} -> {args.out_train}")
    print(f"test {test.shape} -> {args.out_test}")
    print(f"train positive rate: {train['target'].mean():.4f}")


if __name__ == "__main__":
    main()
