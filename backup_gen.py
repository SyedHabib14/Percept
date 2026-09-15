from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Back up all repository Python files.")
    parser.add_argument("--version", help="Backup version name, e.g. v1.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    backup_root = root / "backup"
    backup_root.mkdir(exist_ok=True)

    if args.version:
        version = args.version
    else:
        versions = [
            int(p.name[1:])
            for p in backup_root.iterdir()
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
        ]
        version = f"v{max(versions, default=0) + 1}"

    destination = backup_root / version
    destination.mkdir(exist_ok=False)

    for source in root.rglob("*.py"):
        if backup_root in source.parents:
            continue

        target = destination / source.name
        shutil.copy2(source, target)
        print(target.relative_to(root))


if __name__ == "__main__":
    main()