"""Generate a Kaggle notebook for a rank-blend experiment (analysis + submission).

Reproduces component training (LightGBM / XGBoost / logistic regression)
and the rank blend, then reports component OOF KS, rank correlations and
a 200x bootstrap gate. Parity with scripts/train_baseline.py,
scripts/train_xgb.py, scripts/train_lr.py and scripts/blend.py.

Usage:
    python scripts/gen_blend_notebook.py --exp fadhlur/exp084 --oof 0.51987 \
        --components '[{"name": "xgb", "kind": "xgb", "ref": "...",
                        "seeds": [42], "params": {...}},
                       {"name": "lr", "kind": "lr", "ref": "...",
                        "seeds": [42, 43, 44, 45, 46], "c": 0.03}]' \
        --weights 0.6 0.4 --rank \
        --dst notebooks/experiments/fadhlur/exp084_rank_blend_xgb_lr.ipynb

Resolvers/builders are copied from gen_kaggle_notebook.py on purpose:
that generator is parity-critical, keep both in sync when editing.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import account, base, bureau, cross, eco, prev, repay
from src.features.common import with_cutoff


def inline_source(fn) -> str:
    src = inspect.getsource(fn)
    return "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("from src.")
    )


def repo_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            cwd=ROOT,
        ).stdout.strip()
    except OSError:
        return "unknown"


def lgbm_factory_defaults() -> dict:
    spec = importlib.util.spec_from_file_location(
        "model_mod", ROOT / "src" / "model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_lgbm(seed=42).get_params()


COUNT_PREFIXES_SRC = (
    "COUNT_PREFIXES = (\n"
    '    "prev_n", "repay_n", "repay_overdue_n", "acct_n_months", "bureau_n",\n'
    '    "eco_n_orders", "eco_n_cancelled", "prev_n_approved", "prev_n_rejected",\n'
    ")\n"
)

XGB_BASE = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3,
    "tree_method": "hist",
    "enable_categorical": True,
    "n_jobs": 12,
}


def resolver_code() -> str:
    return (
        "import os\n"
        "\n"
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
        "    d = os.getcwd()\n"
        "    for _ in range(6):\n"
        '        cand = os.path.join(d, "data", "train_applications.csv")\n'
        "        if os.path.isfile(cand):\n"
        '            return os.path.join(d, "data")\n'
        "        nd = os.path.dirname(d)\n"
        "        if nd == d:\n"
        "            break\n"
        "        d = nd\n"
        "    return None\n"
        "DATA = find_data()\n"
        'assert DATA is not None, ("data tidak ditemukan; "\n'
        '"pastikan dataset kompetisi ter-attach di notebook Kaggle")\n'
        'print("DATA:", DATA)'
    )


def builders_code() -> str:
    code = "\n\n".join(
        [
            "from pathlib import Path",
            "import numpy as np\nimport pandas as pd",
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
    return code + "\n\n" + COUNT_PREFIXES_SRC


def data_cell_code() -> str:
    return (
        "from scipy.stats import ks_2samp\n"
        "from sklearn.model_selection import StratifiedKFold\n\n"
        "SEED = 42\n"
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
        'assert (train["application_id"].to_numpy() == '
        'train_apps["application_id"].to_numpy()).all(), "urutan baris tidak sinkron"\n'
        'y = train["target"].to_numpy()\n'
        'test = feat[~feat["application_id"].isin(set(train["application_id"]))].copy()\n'
        'drop = {"application_id", "customer_id", "target", "fold", '
        '"decision_datetime", "application_datetime", "is_train"}\n'
        "features = sorted(c for c in train.columns if c not in drop)\n"
        'for c in train[features].select_dtypes(include=["object", "string"]).columns:\n'
        '    train[c] = train[c].astype("category")\n'
        '    test[c] = test[c].astype("category")\n'
        'print("features:", len(features), "train:", train.shape, "test:", test.shape)\n'
        "\n"
        "def ks_score(yt, yp):\n"
        "    yt = np.asarray(yt)\n"
        "    yp = np.asarray(yp)\n"
        "    bad = yp[yt == 1]\n"
        "    good = yp[yt == 0]\n"
        "    if len(bad) == 0 or len(good) == 0:\n"
        "        return 0.0\n"
        "    return float(ks_2samp(bad, good).statistic)\n"
        "\n"
        'fold_col = train["fold"].to_numpy()\n'
        "\n"
        "def fold_ks(y_, p_):\n"
        "    out = {}\n"
        "    for f in sorted(np.unique(fold_col)):\n"
        "        m = fold_col == f\n"
        "        out[int(f)] = ks_score(y_[m], p_[m])\n"
        '    out["oof"] = ks_score(y_, p_)\n'
        "    return out\n"
        "\n"
        "ks_metric = lambda yt, yp: (\n"
        '    "ks",\n'
        "    float(\n"
        "        ks_2samp(\n"
        "            np.asarray(yp)[np.asarray(yt) == 1],\n"
        "            np.asarray(yp)[np.asarray(yt) == 0],\n"
        "        ).statistic\n"
        "    ),\n"
        "    True,\n"
        ")"
    )


def component_block(comp: dict, index: int) -> str:
    name = comp["name"]
    kind = comp["kind"]
    if kind == "lgbm":
        return (
            f"# --- {name} (parity: scripts/train_baseline.py + src/model.py make_lgbm) ---\n"
            "import lightgbm as lgb\n"
            "\n"
            f"comp = COMPONENTS[{index}]\n"
            "oof_m = np.zeros(len(train))\n"
            "test_m = np.zeros(len(test))\n"
            'for seed in comp["seeds"]:\n'
            "    fold_map = make_folds(seed)\n"
            '    tr_ids = train.drop(columns=["fold"]).merge(fold_map, on="application_id")\n'
            "    oof_s = np.zeros(len(train))\n"
            '    for fold in sorted(tr_ids["fold"].unique()):\n'
            '        tr = (tr_ids["fold"] != fold).to_numpy()\n'
            '        model = lgb.LGBMClassifier(**comp["params"], random_state=seed)\n'
            "        model.fit(\n"
            "            train.loc[tr, features],\n"
            "            y[tr],\n"
            "            eval_set=[(train.loc[~tr, features], y[~tr])],\n"
            "            eval_metric=ks_metric,\n"
            '            callbacks=[lgb.early_stopping(comp["early_stopping"], verbose=False)],\n'
            "        )\n"
            "        oof_s[~tr] = model.predict_proba(train.loc[~tr, features])[:, 1]\n"
            "        test_m += model.predict_proba(test[features])[:, 1] / (\n"
            '            5 * len(comp["seeds"])\n'
            "        )\n"
            '    oof_m += oof_s / len(comp["seeds"])\n'
            f'    print(f"{name} seed {{seed}} done", flush=True)\n'
            f'oofs["{name}"] = oof_m\n'
            f'tests["{name}"] = test_m'
        )
    if kind == "xgb":
        return (
            f"# --- {name} (parity: scripts/train_xgb.py) ---\n"
            "import xgboost as xgb\n"
            "\n"
            f"comp = COMPONENTS[{index}]\n"
            "XGB_BASE = {\n"
            "    "
            + ",\n    ".join(f"{k!r}: {v!r}" for k, v in XGB_BASE.items())
            + ",\n"
            "}\n"
            "oof_m = np.zeros(len(train))\n"
            "test_m = np.zeros(len(test))\n"
            'for seed in comp["seeds"]:\n'
            "    fold_map = make_folds(seed)\n"
            '    tr_ids = train.drop(columns=["fold"]).merge(fold_map, on="application_id")\n'
            "    params = dict(XGB_BASE)\n"
            '    params.update(comp["params"])\n'
            '    params["random_state"] = seed\n'
            '    for fold in sorted(tr_ids["fold"].unique()):\n'
            '        tr = (tr_ids["fold"] != fold).to_numpy()\n'
            "        model = xgb.XGBClassifier(**params)\n"
            "        model.fit(train.loc[tr, features], y[tr])\n"
            "        oof_m[~tr] = model.predict_proba(train.loc[~tr, features])[:, 1]\n"
            "        test_m += model.predict_proba(test[features])[:, 1] / (\n"
            '            5 * len(comp["seeds"])\n'
            "        )\n"
            f'    print(f"{name} seed {{seed}} done", flush=True)\n'
            f'oofs["{name}"] = oof_m\n'
            f'tests["{name}"] = test_m'
        )
    if kind == "lr":
        return (
            f"# --- {name} (parity: scripts/train_lr.py) ---\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.model_selection import cross_val_predict\n"
            "from sklearn.pipeline import make_pipeline\n"
            "from sklearn.preprocessing import OneHotEncoder, QuantileTransformer\n"
            "\n"
            f"comp = COMPONENTS[{index}]\n"
            'cat_cols = list(train[features].select_dtypes("category").columns)\n'
            "num_cols = [c for c in features if c not in cat_cols]\n"
            "X = train[features].copy()\n"
            "X_test = test[features].copy()\n"
            "for c in cat_cols:\n"
            '    X[c] = X[c].astype(str).fillna("MISSING")\n'
            '    X_test[c] = X_test[c].astype(str).fillna("MISSING")\n'
            "\n"
            "def make_lr():\n"
            "    num = make_pipeline(\n"
            "        QuantileTransformer(\n"
            "            n_quantiles=200,\n"
            '            output_distribution="normal",\n'
            "            subsample=200_000,\n"
            '            random_state=comp["seeds"][0],\n'
            "        ),\n"
            '        SimpleImputer(strategy="constant", fill_value=0.0),\n'
            "    )\n"
            "    prep = ColumnTransformer(\n"
            "        [\n"
            '            ("num", num, num_cols),\n'
            '            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),\n'
            "        ]\n"
            "    )\n"
            '    return make_pipeline(prep, LogisticRegression(C=comp["c"], max_iter=2000))\n'
            "\n"
            "oof_list = []\n"
            'for seed in comp["seeds"]:\n'
            "    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)\n"
            "    oof_list.append(\n"
            '        cross_val_predict(make_lr(), X, y, cv=cv, method="predict_proba")[:, 1]\n'
            "    )\n"
            f'    print(f"{name} seed {{seed}} done", flush=True)\n'
            'oofs["' + name + '"] = np.mean(oof_list, axis=0)\n'
            f'tests["{name}"] = make_lr().fit(X, y).predict_proba(X_test)[:, 1]'
        )
    raise ValueError(f"unknown component kind: {kind}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--components", required=True, help="JSON list of components")
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--gate-note", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--slug", default="")
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    components = json.loads(args.components)
    assert len(components) == len(args.weights), "components/weights length mismatch"
    lgbm_defaults = lgbm_factory_defaults()

    clean_components = []
    for comp in components:
        c = {k: v for k, v in comp.items() if k != "ref"}
        if c["kind"] == "lgbm":
            merged = {**lgbm_defaults, **c.get("params", {})}
            merged.pop("random_state", None)
            c["params"] = merged
        clean_components.append(c)

    slug = args.slug or args.exp.split("/")[-1]
    sha = repo_sha()

    comp_lines = "\n".join(
        f"- `{c['name']}` [{c['kind']}]: {c.get('ref', '')}".rstrip()
        for c in components
    )
    weight_lines = ", ".join(
        f"{c['name']} x {w:.4g}" for c, w in zip(components, args.weights)
    )

    title = (
        f"# {slug}: rank-blend analisis\n\n"
        f"Notebook analisis + reproduksi eksperimen `{args.exp}` "
        f"(OOF KS acuan **{args.oof}**). Dijalankan penuh end-to-end; "
        "hasil tiap sel menjadi bahan analisis tim.\n\n"
        f"Komposisi:\n\n{comp_lines}\n\n"
        f"Berat rank-blend: {weight_lines}.\n"
    )
    if args.gate_note:
        title += f"\nGate tercatat: {args.gate_note}\n"
    title += (
        "\nFitur leakage-safe: semua event histori strictly sebelum "
        "`decision_datetime` (bureau memakai `available_datetime`). "
        "Kode fitur di-inline dari `src/features/` pada repo utama, "
        f"sehingga identik dengan artefak `outputs/experiments/{args.exp}/`.\n"
        f"Di-generate dari commit repo {sha}.\n"
    )

    train_cell = (
        "oofs, tests = {}, {}\n\n"
        "def make_folds(seed):\n"
        "    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)\n"
        '    f = train_apps[["application_id"]].copy()\n'
        '    f["fold"] = -1\n'
        '    for k, (_, va) in enumerate(skf.split(f, train_apps["target"])):\n'
        '        f.loc[va, "fold"] = k\n'
        "    return f\n\n"
        + "\n\n".join(component_block(c, i) for i, c in enumerate(clean_components))
        + "\n\n"
        'print("komponen selesai:", {n: round(ks_score(y, o), 5) for n, o in oofs.items()})'
    )

    analysis_cell = (
        "from IPython.display import display\n\n"
        'names = [c["name"] for c in COMPONENTS]\n'
        "w = np.asarray(WEIGHTS, dtype=float)\n"
        "w = w / w.sum()\n\n"
        "rank_oof = {n: pd.Series(oofs[n]).rank(pct=True).to_numpy() for n in names}\n"
        "rank_test = {n: pd.Series(tests[n]).rank(pct=True).to_numpy() for n in names}\n"
        "if RANK:\n"
        "    blend_oof = sum(w[i] * rank_oof[n] for i, n in enumerate(names))\n"
        "    blend_test = sum(w[i] * rank_test[n] for i, n in enumerate(names))\n"
        "else:\n"
        "    blend_oof = sum(w[i] * oofs[n] for i, n in enumerate(names))\n"
        "    blend_test = sum(w[i] * tests[n] for i, n in enumerate(names))\n\n"
        "ks_blend = ks_score(y, blend_oof)\n"
        f'print(f"BLEND OOF KS: {{ks_blend:.5f}} (acuan repo: {args.oof})")\n'
        "blend_folds = fold_ks(y, blend_oof)\n\n"
        "rows = []\n"
        "for i, n in enumerate(names):\n"
        "    s = fold_ks(y, oofs[n])\n"
        "    rows.append(\n"
        "        {\n"
        '            "komponen": n,\n'
        '            "bobot": round(float(w[i]), 4),\n'
        "            **{f\"fold{k}\": round(s[k], 5) for k in sorted(k for k in s if k != 'oof')},\n"
        '            "OOF KS": round(s["oof"], 5),\n'
        "        }\n"
        "    )\n"
        "rows.append(\n"
        "    {\n"
        '        "komponen": "BLEND",\n'
        '        "bobot": 1.0,\n'
        "        **{f\"fold{k}\": round(blend_folds[k], 5) for k in sorted(k for k in blend_folds if k != 'oof')},\n"
        '        "OOF KS": round(ks_blend, 5),\n'
        "    }\n"
        ")\n"
        'display(pd.DataFrame(rows).set_index("komponen"))\n\n'
        "rank_df = pd.DataFrame(rank_oof)\n"
        'rank_df["blend"] = blend_oof\n'
        'print("korelasi spearman antar prediksi (rank):")\n'
        'display(rank_df.corr(method="spearman").round(4))'
    )

    figure_cell = (
        "import matplotlib\n\n"
        'matplotlib.use("Agg")\n'
        "import matplotlib.pyplot as plt\n"
        "from IPython.display import Image\n\n"
        'names = [c["name"] for c in COMPONENTS]\n'
        'fig_dir = Path("outputs/analysis/figures")\n'
        "fig_dir.mkdir(parents=True, exist_ok=True)\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n\n"
        "order = np.argsort(blend_oof)\n"
        "yb = y[order]\n"
        "bad_cum = np.cumsum(yb) / yb.sum()\n"
        "good_cum = np.cumsum(1 - yb) / (1 - yb).sum()\n"
        'axes[0].plot(good_cum, bad_cum, label=f"blend KS={ks_blend:.4f}")\n'
        'axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)\n'
        'axes[0].set_xlabel("kumulatif good (kelas 0)")\n'
        'axes[0].set_ylabel("kumulatif bad (kelas 1)")\n'
        'axes[0].set_title("Kurva KS sederhana (OOF)")\n'
        "axes[0].legend()\n\n"
        'bar_names = names + ["blend"]\n'
        'bar_vals = [fold_ks(y, oofs[n])["oof"] for n in names] + [ks_blend]\n'
        "axes[1].bar(bar_names, bar_vals)\n"
        'axes[1].set_title("OOF KS per komponen")\n'
        "axes[1].set_ylim(min(bar_vals) - 0.01, max(bar_vals) + 0.006)\n"
        "for i, v in enumerate(bar_vals):\n"
        '    axes[1].text(i, v + 0.0006, f"{v:.4f}", ha="center", fontsize=9)\n'
        "plt.tight_layout()\n"
        f'fig_path = fig_dir / "{slug}_analysis.png"\n'
        "plt.savefig(fig_path, dpi=110)\n"
        'print("figure:", fig_path)\n'
        "display(Image(filename=str(fig_path)))"
    )

    bootstrap_cell = (
        "rng = np.random.default_rng(42)\n"
        "n_boot = 200\n"
        'names = [c["name"] for c in COMPONENTS]\n'
        "deltas = {n: np.empty(n_boot) for n in names}\n"
        "for b in range(n_boot):\n"
        "    idx = rng.integers(0, len(y), len(y))\n"
        "    ks_b = ks_score(y[idx], blend_oof[idx])\n"
        "    for n in names:\n"
        "        deltas[n][b] = ks_b - ks_score(y[idx], oofs[n][idx])\n"
        'print("bootstrap 200x (seed 42, resample baris):")\n'
        "for n in names:\n"
        "    d = deltas[n]\n"
        "    lo, hi = np.percentile(d, [2.5, 97.5])\n"
        '    verdict = "PROVEN" if lo > 0 else ("DITOLAK" if hi < 0 else "TIDAK SIGNIFIKAN")\n'
        '    print(f"  vs {n}: delta {d.mean():+.5f} CI95 [{lo:+.5f}, {hi:+.5f}] -> {verdict}")'
    )

    submission_cell = (
        "sub = pd.DataFrame(\n"
        '    {"application_id": test["application_id"], "default_probability": blend_test}\n'
        ")\n"
        'sub.to_csv("submission.csv", index=False)\n'
        'print("submission.csv:", sub.shape, "rows unik:", sub["application_id"].nunique())\n'
        'print(sub["default_probability"].describe().round(6))'
    )

    closing_md = (
        "## Cara membaca & catatan untuk tim\n\n"
        "- Tabel komponen: KS per fold + OOF untuk tiap model dan BLEND. "
        "Bandingkan spread fold: komponen dengan spread besar rawan tidak stabil.\n"
        "- Matriks korelasi spearman: bila korelasi antar komponen > 0,95, "
        "diversitas habis dan blend berikutnya cenderung jenuh (temuan 18/19).\n"
        "- Sel bootstrap: gate 200x resample — blend hanya lolos bila CI95 delta "
        "di atas nol untuk semua pembanding.\n"
        "- `submission.csv` ditulis di akhir; bandingkan distribusinya dengan "
        "artefak di `outputs/experiments/`.\n"
        + (f"\n{args.notes}\n" if args.notes else "")
    )

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3"}
    nb.cells = [
        nbf.v4.new_markdown_cell(title),
        nbf.v4.new_code_cell(
            "!pip install -q lightgbm xgboost scikit-learn pandas pyarrow scipy matplotlib"
        ),
        nbf.v4.new_code_cell(resolver_code()),
        nbf.v4.new_code_cell(builders_code()),
        nbf.v4.new_code_cell(data_cell_code()),
        nbf.v4.new_code_cell(
            "RANK = " + repr(args.rank) + "\n"
            f"WEIGHTS = {args.weights!r}\n"
            + "COMPONENTS = [\n"
            + "".join(
                "    {\n"
                + "".join(f"        {k!r}: {v!r},\n" for k, v in c.items())
                + "    },\n"
                for c in clean_components
            )
            + "]"
        ),
        nbf.v4.new_code_cell(train_cell),
        nbf.v4.new_code_cell(analysis_cell),
        nbf.v4.new_code_cell(figure_cell),
        nbf.v4.new_code_cell(bootstrap_cell),
        nbf.v4.new_code_cell(submission_cell),
        nbf.v4.new_markdown_cell(closing_md),
    ]
    DST = Path(args.dst)
    nbf.write(nb, DST)
    print("wrote", DST, len(nb.cells), "cells")


if __name__ == "__main__":
    main()
