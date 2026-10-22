"""Command line interface for CtxBeacon."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ctxbeacon.ledger import Ledger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctxbeacon",
        description="CtxBeacon - See what your agent kept—and what it lost.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    # Command: validate
    validate_parser = subparsers.add_parser("validate", help="Validate an append-only JSONL ledger")
    validate_parser.add_argument(
        "--ledger",
        type=str,
        default="ledger/ledger.jsonl",
        help="Path to JSONL ledger file (default: ledger/ledger.jsonl)",
    )

    return parser


def handle_validate(args: argparse.Namespace) -> int:
    ledger_path = Path(args.ledger)
    valid_count, errors = Ledger.validate_file(ledger_path)

    if errors:
        sys.stderr.write(f"Validation failed for '{ledger_path}':\n")
        for err in errors:
            sys.stderr.write(f"  {err}\n")
        sys.stderr.write(f"\nResult: {valid_count} valid events, {len(errors)} error(s).\n")
        return 1

    sys.stdout.write(f"Ledger valid: {valid_count} event(s) successfully verified in '{ledger_path}'.\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "validate":
        return handle_validate(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
