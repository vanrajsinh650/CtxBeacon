"""Command line interface for CtxBeacon."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ctxbeacon.ledger import Ledger
from ctxbeacon.render import render_compaction_diff
from ctxbeacon.scenarios import (
    run_bharatcode_experiments,
    run_clean_scenario,
    run_simulated_compaction_loss,
)


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

    # Command: diff
    diff_parser = subparsers.add_parser("diff", help="Show compaction diff before and after context summarization")
    diff_parser.add_argument(
        "--ledger",
        type=str,
        default="ledger/ledger.jsonl",
        help="Path to JSONL ledger file (default: ledger/ledger.jsonl)",
    )
    diff_parser.add_argument(
        "--turn",
        type=int,
        default=None,
        help="Specific turn number to inspect",
    )
    diff_parser.add_argument(
        "--event-id",
        type=str,
        default=None,
        help="Specific compaction event ID to inspect",
    )
    diff_parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Override configured context token budget",
    )

    # Command: demo
    demo_parser = subparsers.add_parser("demo", help="Run demonstration scenarios")
    demo_parser.add_argument(
        "--scenario",
        choices=["compaction-loss", "clean", "real-experiments"],
        default="compaction-loss",
        help="Scenario to run: compaction-loss (simulated), clean (simulated), real-experiments (BharatCode live)",
    )
    demo_parser.add_argument(
        "--workspace",
        type=str,
        default="examples/demo_workspace",
        help="Target workspace directory for sandboxed tools (default: examples/demo_workspace)",
    )
    demo_parser.add_argument(
        "--ledger",
        type=str,
        default=None,
        help="Output JSONL ledger path (default: ledger/ledger_<scenario>.jsonl)",
    )
    demo_parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Enable full context snapshot storage under ledger/snapshots/ (opt-in)",
    )
    demo_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional BharatCode API key for real-experiments",
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


def handle_diff(args: argparse.Namespace) -> int:
    output = render_compaction_diff(
        ledger_path=args.ledger,
        turn=args.turn,
        event_id=args.event_id,
        budget_override=args.budget,
    )
    sys.stdout.write(output + "\n")
    return 0 if not output.startswith("Error:") else 1


def handle_demo(args: argparse.Namespace) -> int:
    scenario = args.scenario
    workspace = args.workspace
    snapshot = args.snapshot

    if scenario == "compaction-loss":
        ledger_p = args.ledger or "ledger/ledger_simulated_loss.jsonl"
        sys.stdout.write("============================================================\n")
        sys.stdout.write("Running Scenario B: Deterministic Simulated Compaction Loss\n")
        sys.stdout.write("NOTE: This is a controlled simulation testing CtxBeacon's\n")
        sys.stdout.write("detection engine. All events are marked simulation=true.\n")
        sys.stdout.write("============================================================\n\n")

        res = run_simulated_compaction_loss(
            workspace_dir=workspace,
            ledger_path=ledger_p,
            enable_snapshots=snapshot,
        )
        sys.stdout.write(res["diff_output"] + "\n\n")
        sys.stdout.write(f"Scenario complete. Ledger recorded to: {res['ledger_path']}\n")
        return 0

    elif scenario == "clean":
        ledger_p = args.ledger or "ledger/ledger_clean.jsonl"
        sys.stdout.write("============================================================\n")
        sys.stdout.write("Running Scenario: Clean Compaction (Constraints Preserved)\n")
        sys.stdout.write("============================================================\n\n")

        res = run_clean_scenario(
            workspace_dir=workspace,
            ledger_path=ledger_p,
            enable_snapshots=snapshot,
        )
        sys.stdout.write(res["diff_output"] + "\n\n")
        sys.stdout.write(f"Scenario complete. Ledger recorded to: {res['ledger_path']}\n")
        return 0

    elif scenario == "real-experiments":
        ledger_p = args.ledger or "ledger/ledger_bharatcode_experiments.jsonl"
        res = run_bharatcode_experiments(
            api_key=args.api_key,
            ledger_path=ledger_p,
            enable_snapshots=snapshot,
        )
        if res.get("status") == "pending_credentials":
            sys.stdout.write(res["message"] + "\n")
            return 0

        sys.stdout.write("============================================================\n")
        sys.stdout.write("Scenario A: Real BharatCode Compaction Experiments Results\n")
        sys.stdout.write("============================================================\n\n")
        sys.stdout.write(f"Completed genuine runs: {res['completed_runs']}/5\n")
        sys.stdout.write(f"Constraints preserved : {res['preserved_count']}\n")
        sys.stdout.write(f"Constraints missing   : {res['missing_count']}\n")
        sys.stdout.write(f"Blocked / errored     : {res['blocked_count']}\n")
        sys.stdout.write(f"Preservation rate     : {res['preservation_rate']:.1f}%\n")
        sys.stdout.write(f"Total prompt tokens   : {res['total_prompt_tokens']}\n")
        sys.stdout.write(f"Total comp. tokens    : {res['total_completion_tokens']}\n\n")
        return 0

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "validate":
        return handle_validate(args)
    elif args.subcommand == "diff":
        return handle_diff(args)
    elif args.subcommand == "demo":
        return handle_demo(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
