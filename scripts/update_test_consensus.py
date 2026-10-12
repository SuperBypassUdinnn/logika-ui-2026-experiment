"""Test-set consensus census across submissions (adapted from vision workspace).

Usage:
    python scripts/update_test_consensus.py

Reads every submission.csv under outputs/submissions, outputs/experiments,
and outputs/current, computes per-application prediction agreement,
writes outputs/analysis/test_set_consensus.csv + docs/TEST_SET_CONSENSUS.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

THRESHOLD = 0.5


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "outputs").rglob("submission.csv"))
    probs, names = [], []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["application_id", "default_probability"])
        except (ValueError, KeyError):
            continue
        df = df.drop_duplicates("application_id").sort_values("application_id")
        probs.append(df.set_index("application_id")["default_probability"])
        names.append(str(f.relative_to(root / "outputs")))
    mat = pd.concat(probs, axis=1)
    mat.columns = names
    mat.index.name = None
    print(f"submissions: {len(names)} x {len(mat)} apps")

    labels = (mat > THRESHOLD).astype(int)
    agree = labels.eq(labels.iloc[:, 0], axis=0).mean(axis=1)
    consensus = pd.DataFrame(
        {
            "application_id": mat.index,
            "mean_prob": mat.mean(axis=1).round(4),
            "std_prob": mat.std(axis=1).round(4),
            "agreement": agree.round(4),
            "n_models": len(names),
        }
    ).sort_values("application_id")
    out_csv = root / "outputs" / "analysis" / "test_set_consensus.csv"
    consensus.to_csv(out_csv, index=False)

    anchor = (agree == 1.0).sum()
    battle = ((agree >= 0.5) & (agree < 1.0)).sum()
    print(f"anchor (100% sepakat): {anchor} | battleground: {battle}")

    (root / "docs" / "TEST_SET_CONSENSUS.md").write_text(
        "# Sensus Konsensus Test (Bahasa Indonesia)\n\n"
        f"- Submission dihitung: {len(names)}\n"
        f"- Anchor (semua model sepakat kelasnya): {anchor}\n"
        f"- Battleground (prediksi berubah antar model): {battle}\n"
        f"- Ambang kelas: prob > {THRESHOLD}\n\n"
        "File: `outputs/analysis/test_set_consensus.csv`.\n"
        "Gunakan anchor untuk cek stabilitas submission baru "
        "(target >= 98% konsisten), dan fokus analisis error ke battleground.\n"
    )
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
