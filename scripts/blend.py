"""Probability blend of committed experiment outputs (exp014+).

Usage:
    python scripts/blend.py --member fadhlur --exp exp014 \
        --src fadhlur/exp011 fadhlur/exp013 --weights 0.5 0.5

Reads oof.csv + test_pred.npy from outputs/experiments/<src>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_ks import fold_ks
from utils import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="fadhlur")
    ap.add_argument("--exp", default="exp014")
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--src-base", default="outputs/experiments")
    args = ap.parse_args()
    assert len(args.src) == len(args.weights)
    assert abs(sum(args.weights) - 1.0) < 1e-9

    base = Path(args.out)
    srccase = Path(args.src_base)
    oofs, tests, sub = [], [], None
    for s in args.src:
        d = srccase / s
        o = pd.read_csv(d / "oof.csv")
        oofs.append(o["pred"].to_numpy())
        tests.append(np.load(d / "test_pred.npy"))
        sub = pd.read_csv(d / "submission.csv", usecols=["application_id"])
    oof = sum(w * o for w, o in zip(args.weights, oofs))
    test_pred = sum(w * t for w, t in zip(args.weights, tests))
    ref = pd.read_csv(srccase / args.src[0] / "oof.csv")
    scores = fold_ks(
        ref["target"].to_numpy(), oof, ref["fold"].to_numpy()
    )
    print(f"BLEND OOF KS: {scores['oof']:.5f}")

    outdir = ensure_dir(str(base / args.member / args.exp))
    pd.DataFrame(
        {
            "application_id": ref["application_id"],
            "target": ref["target"],
            "pred": oof,
            "fold": ref["fold"],
        }
    ).to_csv(Path(outdir) / "oof.csv", index=False)
    np.save(Path(outdir) / "test_pred.npy", test_pred)
    sub["default_probability"] = test_pred
    sub.to_csv(Path(outdir) / "submission.csv", index=False)
    (Path(outdir) / "log.md").write_text(
        f"# {args.member}/{args.exp} blend\n\n"
        f"- sources: {args.src}\n- weights: {args.weights}\n"
        f"- OOF KS: {scores['oof']:.5f}\n"
    )
    print(f"artifacts -> {outdir}")


if __name__ == "__main__":
    main()
