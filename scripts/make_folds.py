"""Buat stratified 5-fold untuk train tabular."""

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/train_applications.csv")
    ap.add_argument("--out", default="train_folds.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.input, usecols=["application_id", "target"])
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    df["fold"] = -1
    for fold, (_, val) in enumerate(skf.split(df, df["target"])):
        df.loc[val, "fold"] = fold
    df[["application_id", "fold"]].to_csv(args.out, index=False)
    print(f"wrote {args.out}: {df['fold'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
