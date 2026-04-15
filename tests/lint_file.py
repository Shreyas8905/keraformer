"""Run Ruff lint checks for a single file or directory.

Usage:
    python tests/lint_file.py path/to/file.py
    python tests/lint_file.py path/to/package_or_directory
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ruff lint checks for a path.")
    parser.add_argument(
        "path",
        type=Path,
        help="File or directory to lint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = args.path

    if not target.exists():
        print(f"error: path does not exist: {target}", file=sys.stderr)
        return 2

    command = [sys.executable, "-m", "ruff", "check", str(target)]
    result = subprocess.run(command, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
