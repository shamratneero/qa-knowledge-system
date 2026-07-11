"""Main entry point for QA Knowledge System."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data_loader import load_knowledge_base, DATA_FILE


def main(argv: list[str] | None = None) -> int:
    """Run the application. Returns exit code."""
    parser = argparse.ArgumentParser(description="QA Knowledge System")
    parser.add_argument("-f", "--file", help="Path to Excel file (optional)", default=None)
    args = parser.parse_args(argv)

    file_path = Path(args.file) if args.file else DATA_FILE

    try:
        df = load_knowledge_base(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return 0
    except Exception as exc:
        print("Error:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
