"""Generate a Kaggle notebook from committed experiment code.

Source of truth for member submission notebooks: regenerating beats
hand-editing .ipynb JSON. Usage:
    python scripts/gen_kaggle_notebook.py --src-build /tmp/exp_build.py \\
        --dst notebooks/current/fadhlur/logika-ui-2026-notebook-experiment.ipynb
"""

import argparse
import ast
from pathlib import Path

import nbformat as nbf

ap = argparse.ArgumentParser()
ap.add_argument("--src-build", required=True)
ap.add_argument("--dst", required=True)
a = ap.parse_args()
SRC = Path(a.src_build)
DST = Path(a.dst)

tree = ast.parse(SRC.read_text())
lines = SRC.read_text().splitlines(keepends=True)


def _keep(node):
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
        return True
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "CUTOFF" for t in node.targets
    )


kept = [
    "".join(lines[node.lineno - 1 : node.end_lineno])
    for node in tree.body
    if _keep(node)
]
builders_code = "\n\n".join(kept)
builders_code = builders_code.replace("from __future__ import annotations\n\n", "")

fill_src = ""
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "main":
        for sub in node.body:
            if isinstance(sub, ast.For):
                fill_src = "".join(lines[sub.lineno - 1 : sub.end_lineno])
assert fill_src, "fillna loop not found in main()"

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# exp018: LightGBM + Ecommerce Mix (KS)\n\n"
        "Salinan notebook dari eksperimen `fadhlur/exp018` (OOF KS 0,4543). "
        "Fitur leakage-safe: semua event histori strictly sebelum "
        "`decision_datetime` (bureau memakai `available_datetime`).\n\n"
        "Kode fitur disalin verbatim dari `scripts/build_features.py` "
        "pada commit `e98a3a1` agar submission Kaggle identik dengan artefak "
        "repo (`outputs/experiments/fadhlur/exp018/`)."
    ),
    nbf.v4.new_code_cell(
        "!pip install -q lightgbm==4.7.0 scikit-learn pandas pyarrow scipy"
    ),
    nbf.v4.new_code_cell(
        "import os\n"
        "import os\n"
        "def find_data():\n"
        '    for root in ["/kaggle/input", "data"]:\n'
        "        if not os.path.isdir(root):\n"
        "            continue\n"
        "        for dirpath, _, files in os.walk(root):\n"
        '            if "train_applications.csv" in files:\n'
        "                return dirpath\n"
        "    return None\n"
        "DATA = find_data()\n"
        'assert DATA is not None, ("data tidak ditemukan; "\n'
        '"pastikan dataset kompetisi ter-attach di notebook Kaggle")\n'
        'print("DATA:", DATA)'
    ),
    nbf.v4.new_code_cell(builders_code),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from scipy.stats import ks_2samp\n"
        "from sklearn.model_selection import StratifiedKFold\n"
        "import lightgbm as lgb\n\n"
        "SEED = 42\n"
        "PARAMS = {\n"
        '    "objective": "binary", "learning_rate": 0.03, "num_leaves": 31,\n'
        '    "min_child_samples": 100, "n_estimators": 200,\n'
        '    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,\n'
        '    "scale_pos_weight": 3, "verbose": -1,\n'
        "}\n"
        "data = Path(DATA)\n"
        'train_apps = pd.read_csv(data / "train_applications.csv", '
        'parse_dates=["decision_datetime"])\n'
        "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n"
        'folds = train_apps[["application_id"]].copy()\n'
        'folds["fold"] = -1\n'
        'for f, (_, va) in enumerate(skf.split(folds, train_apps["target"])):\n'
        '    folds.loc[va, "fold"] = f\n'
        "apps = scored_apps(data)\n"
        'profile = pd.read_csv(data / "customer_profiles.csv")\n'
        'feat = apps.merge(profile, on="customer_id", how="left")\n'
        'feat["loan_to_income"] = feat["loan_amount"] / feat["declared_monthly_income"].clip(lower=1)\n'
        'feat["decision_month"] = feat["decision_datetime"].dt.month.astype("int8")\n'
        'feat["decision_weekday"] = feat["decision_datetime"].dt.weekday.astype("int8")\n'
        'for name, fn in [("prev", agg_prev), ("repay", agg_repay), '
        '("account", agg_account), ("bureau", agg_bureau), ("eco", agg_eco)]:\n'
        "    part = fn(data, apps)\n"
        "    print(name, len(part))\n"
        '    feat = feat.merge(part, on="application_id", how="left")\n'
        + fill_src
        + 'train = feat[feat["target"].notna()].merge(folds, on="application_id")\n'
        'test = feat[feat["target"].isna()].drop(columns=["target"])\n'
        'drop = {"application_id", "customer_id", "target", "fold", "decision_datetime"}\n'
        "features = [c for c in train.columns if c not in drop]\n"
        'for c in train[features].select_dtypes(include=["object", "string"]).columns:\n'
        '    train[c] = train[c].astype("category")\n'
        '    test[c] = test[c].astype("category")\n'
        'print("features:", len(features), "train:", train.shape, "test:", test.shape)'
    ),
    nbf.v4.new_code_cell(
        'y = train["target"].to_numpy()\n'
        "oof = np.zeros(len(train))\n"
        "test_pred = np.zeros(len(test))\n"
        'for fold in sorted(train["fold"].unique()):\n'
        '    tr = train["fold"] != fold\n'
        "    model = lgb.LGBMClassifier(**PARAMS, random_state=SEED)\n"
        "    model.fit(train.loc[tr, features], y[tr])\n"
        "    oof[~tr] = model.predict_proba(train.loc[~tr, features])[:, 1]\n"
        "    test_pred += model.predict_proba(test[features])[:, 1] / 5\n"
        '    print(f"fold {fold} done", flush=True)\n'
        "ks = ks_2samp(oof[y == 1], oof[y == 0]).statistic\n"
        'print(f"OOF KS: {ks:.5f} (acuan repo: 0.45429)")\n'
        'pd.DataFrame({"application_id": test["application_id"], '
        '"default_probability": test_pred}).to_csv("submission.csv", index=False)\n'
        'print("submission.csv:", test_pred.shape)'
    ),
]
nbf.write(nb, DST)
print("wrote", DST, len(nb.cells), "cells")
