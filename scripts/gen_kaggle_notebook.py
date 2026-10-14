"""Generate a Kaggle submission notebook from repo feature code.

Parity by construction: builder source is inlined from src/features/
via inspect, so the notebook always matches repo main. Usage:
    python scripts/gen_kaggle_notebook.py --exp fadhlur/exp052 \\
        --oof 0.51033 --params '{...}' --early-stopping 200 \\
        --dst notebooks/current/fadhlur/logika-ui-2026-notebook-experiment.ipynb
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import account, base, bureau, cross, eco, prev, repay
from src.features.common import with_cutoff


def inline_source(fn) -> str:
    src = inspect.getsource(fn)
    return "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("from src.")
    )


COUNT_PREFIXES_SRC = (
    "COUNT_PREFIXES = (\n"
    '    "prev_n", "repay_n", "repay_overdue_n", "acct_n_months", "bureau_n",\n'
    '    "eco_n_orders", "eco_n_cancelled", "prev_n_approved", "prev_n_rejected",\n'
    ")\n"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--early-stopping", type=int, default=0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    params = json.loads(args.params)

    builders_code = "\n\n".join(
        [
            "import numpy as np",
            "import pandas as pd",
            "from pathlib import Path",
            'CUTOFF = "decision_datetime"',
            inline_source(with_cutoff),
            inline_source(base.scored_apps),
            inline_source(base.base_features),
            inline_source(prev.agg_prev),
            inline_source(repay.agg_repay),
            inline_source(account.agg_account),
            inline_source(bureau.agg_bureau),
            inline_source(eco.agg_eco),
            inline_source(cross.apply_cross_ratios),
            inline_source(cross.finalize),
        ]
    )
    builders_code += "\n\n" + COUNT_PREFIXES_SRC

    train_extra = ""
    if args.early_stopping > 0:
        train_extra = (
            "eval_set=[(train.loc[~tr, features], y[~tr])], "
            "eval_metric=ks_metric, "
            f"callbacks=[lgb.early_stopping({args.early_stopping}, verbose=False)], "
        )

    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip()
    except OSError:
        sha = "unknown"

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "model_mod", Path(__file__).resolve().parents[1] / "src" / "model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    factory_defaults = mod.make_lgbm(seed=42).get_params()
    merged = {**factory_defaults, **params}
    merged.pop("random_state", None)
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3"}
    nb.cells = [
        nbf.v4.new_markdown_cell(
            f"# {args.exp}: submission notebook (KS)\n\n"
            f"Salinan notebook dari eksperimen `{args.exp}` (OOF KS {args.oof}). "
            "Fitur leakage-safe: semua event histori strictly sebelum "
            "`decision_datetime` (bureau memakai `available_datetime`).\n\n"
            "Kode fitur di-inline dari `src/features/` pada repo utama, "
            f"sehingga identik dengan artefak `outputs/experiments/{args.exp}/`."
            f"\nDi-generate dari commit repo {sha} — cocokkan dengan artefak bila meragukan kesegaran.\n"
        ),
        nbf.v4.new_code_cell(
            "!pip install -q lightgbm scikit-learn pandas pyarrow scipy"
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "CANDIDATES = [\n"
            '    "/kaggle/input/competitions/data-science-competition-logika-ui-2026",\n'
            '    "/kaggle/input/data-science-competition-logika-ui-2026",\n'
            "]\n"
            "def find_data():\n"
            "    for c in CANDIDATES:\n"
            '        if os.path.isfile(os.path.join(c, "train_applications.csv")):\n'
            "            return c\n"
            '    for root in ["/kaggle/input", "data"]:\n'
            "        if not os.path.isdir(root):\n"
            "            continue\n"
            "        for dirpath, _, files in os.walk(root):\n"
            '            if "train_applications.csv" in files:\n'
            "                return dirpath\n"
            '    d = os.getcwd()\n'
            '    for _ in range(6):\n'
            '        cand = os.path.join(d, "data", "train_applications.csv")\n'
            '        if os.path.isfile(cand):\n'
            '            return os.path.join(d, "data")\n'
            '        nd = os.path.dirname(d)\n'
            '        if nd == d:\n'
            '            break\n'
            '        d = nd\n'
            '    return None\n'
            "DATA = find_data()\n"
            'assert DATA is not None, ("data tidak ditemukan; "\n'
            '"pastikan dataset kompetisi ter-attach di notebook Kaggle")\n'
            'print("DATA:", DATA)'
        ),
        nbf.v4.new_code_cell(builders_code),
        nbf.v4.new_code_cell(
            "from scipy.stats import ks_2samp\n"
            "from sklearn.model_selection import StratifiedKFold\n"
            "import lightgbm as lgb\n\n"
            "SEED = 42\n"
            f"PARAMS = {merged!r}\n"
            "data = Path(DATA)\n"
            "apps = scored_apps(data)\n"
            "feat = base_features(apps, data)\n"
            'for name, fn in [("prev", agg_prev), ("repay", agg_repay), '
            '("account", agg_account), ("bureau", agg_bureau), ("eco", agg_eco)]:\n'
            "    part = fn(data, apps)\n"
            "    print(name, len(part))\n"
            '    feat = feat.merge(part, on="application_id", how="left")\n'
            "feat = apply_cross_ratios(feat)\n"
            "feat = finalize(feat)\n"
            'train_apps = pd.read_csv(data / "train_applications.csv")\n'
            "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n"
            'folds = train_apps[["application_id"]].copy()\n'
            'folds["fold"] = -1\n'
            'for f, (_, va) in enumerate(skf.split(folds, train_apps["target"])):\n'
            '    folds.loc[va, "fold"] = f\n'
            'train = feat.merge(folds, on="application_id")\n'
            'test = feat[~feat["application_id"].isin(set(train["application_id"]))].copy()\n'
            'drop = {"application_id", "customer_id", "target", "fold", "decision_datetime", "application_datetime", "is_train"}\n'
            "features = sorted(c for c in train.columns if c not in drop)\n"
            'for c in train[features].select_dtypes(include=["object", "string"]).columns:\n'
            '    train[c] = train[c].astype("category")\n'
            '    test[c] = test[c].astype("category")\n'
            'print("features:", len(features), "train:", train.shape, "test:", test.shape)'
        ),
        nbf.v4.new_code_cell(
            'y = train_apps["target"].to_numpy()\n'
            "oof = np.zeros(len(train_apps))\n"
            "test_pred = np.zeros(len(test))\n"
            "ks_metric = lambda yt, yp: ('ks', float(ks_2samp(np.asarray(yp)[np.asarray(yt) == 1], np.asarray(yp)[np.asarray(yt) == 0]).statistic), True)\n"
            f"SEEDS = {args.seeds}\n"
            "for seed in SEEDS:\n"
            "    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)\n"
            '    folds = train_apps[["application_id"]].copy()\n'
            '    folds["fold"] = -1\n'
            '    for f, (_, va) in enumerate(skf.split(folds, train_apps["target"])):\n'
            '        folds.loc[va, "fold"] = f\n'
            '    tr_ids = train.drop(columns=["fold"]).merge(folds, on="application_id")\n'
            "    oof_s = np.zeros(len(train_apps))\n"
            '    for fold in sorted(tr_ids["fold"].unique()):\n'
            '        tr = (tr_ids["fold"] != fold).to_numpy()\n'
            "        va = ~tr\n"
            "        model = lgb.LGBMClassifier(**PARAMS, random_state=seed)\n"
            f"        model.fit(train.loc[tr, features], y[tr], {train_extra})\n"
            "        oof_s[va] = model.predict_proba(train.loc[va, features])[:, 1]\n"
            "        test_pred += model.predict_proba(test[features])[:, 1] / (5 * len(SEEDS))\n"
            "    oof += oof_s / len(SEEDS)\n"
            '    print(f"seed {seed} done", flush=True)\n'
            "ks = ks_2samp(oof[y == 1], oof[y == 0]).statistic\n"
            f'print(f"OOF KS: {{ks:.5f}} (acuan repo: {args.oof})")\n'
            'pd.DataFrame({"application_id": test["application_id"], '
            '"default_probability": test_pred}).to_csv("submission.csv", index=False)\n'
            'print("submission.csv:", test_pred.shape)'
        ),
    ]
    DST = Path(args.dst)
    nbf.write(nb, DST)
    print("wrote", DST, len(nb.cells), "cells")


if __name__ == "__main__":
    main()
