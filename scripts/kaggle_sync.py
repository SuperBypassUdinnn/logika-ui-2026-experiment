"""Sync helper for current/<member>/ from Kaggle to GitHub.

Usage:
  python scripts/kaggle_sync.py --member fadhlur --pull
  python scripts/kaggle_sync.py --member fadhlur --push

Pull uses `kaggle kernels pull`, push uses `kaggle kernels push`.
Member to kernel slug mapping is read from current/members.json,
with fallback to current/<member>/kernel.json.
"""

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), f"(cwd={cwd})")
    subprocess.run(cmd, cwd=cwd, check=True)


def resolve_slug(root: Path, member: str) -> str:
    members_file = root / "current" / "members.json"
    if members_file.exists():
        data = json.loads(members_file.read_text())
        entry = data.get(member, {})
        slug = entry.get("kaggle_slug")
        if slug:
            return slug
    legacy = root / "current" / member / "kernel.json"
    if legacy.exists():
        return json.loads(legacy.read_text())["slug"]
    raise SystemExit(f"slug untuk {member} belum diisi di current/members.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--push", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    mdir = root / "current" / args.member
    slug = resolve_slug(root, args.member)

    if args.pull:
        run(["kaggle", "kernels", "pull", "-p", str(mdir), slug], root)
    if args.push:
        run(["kaggle", "kernels", "push", "-p", str(mdir)], root)
        print(f"push diminta: {args.push} (commit via git PR harian terpisah)")


if __name__ == "__main__":
    main()
