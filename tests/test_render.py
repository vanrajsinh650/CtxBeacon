"""Tests for CtxBeacon compaction diff rendering and terminal output."""

from pathlib import Path
import json

from ctxbeacon.cli import main as cli_main
from ctxbeacon.ledger import Ledger
from ctxbeacon.render import render_compaction_diff
from ctxbeacon.schema import create_event


def test_render_compaction_diff_with_snapshot_disabled(tmp_path: Path):
    ledger_path = tmp_path / "test.jsonl"
    ledger = Ledger(ledger_path)

    # 1. run_started
    ledger.append(
        create_event(
            event_type="run_started",
            run_id="run-100",
            task="Refactor auth",
            budget_tokens=4000,
            simulation=True,
        )
    )

    # 2. compaction_completed
    ledger.append(
        create_event(
            event_type="compaction_completed",
            event_id="evt-comp-10",
            run_id="run-100",
            turn=10,
            context_before_tokens=3120,  # 78% of 4000
            context_after_tokens=1200,  # 30% of 4000
            retained_ids=["msg-008", "msg-009", "msg-010"],
            dropped_ids=["msg-001", "msg-002", "msg-003", "msg-004"],
            summary_ref="sum-auth-1",
            snapshot_ref=None,
            summary_preview="Refactor authentication using the new session API.",
            simulation=True,
        )
    )

    # 3. constraint_check
    ledger.append(
        create_event(
            event_type="constraint_check",
            run_id="run-100",
            turn=10,
            constraint_findings=[
                {
                    "constraint": "Never modify files under migrations/",
                    "status": "possibly_missing",
                    "marker_terms": ["migrations"],
                    "matched_markers": [],
                    "evidence_message_id": "msg-004",
                    "details": "Deterministic keyword check: absent",
                }
            ],
            simulation=True,
        )
    )

    diff_text = render_compaction_diff(ledger_path=ledger_path, turn=10)

    # Check key sections
    assert "TURN 10  CONTEXT 78% (3120/4000 tok) -> COMPACT 30% (1200/4000 tok)" in diff_text
    assert "RETAINED" in diff_text
    assert "msg-008" in diff_text
    assert "msg-010" in diff_text
    assert "SUMMARIZED / DROPPED" in diff_text
    assert "msg-001" in diff_text
    assert "msg-004" in diff_text
    assert "NEW SUMMARY" in diff_text
    assert "Refactor authentication using the new session API." in diff_text
    assert 'POSSIBLY MISSING' in diff_text
    assert '"Never modify files under migrations/"' in diff_text
    assert "Original message: msg-004" in diff_text
    assert "Compaction event: evt-comp-10" in diff_text
    assert "Keyword match: absent" in diff_text
    assert "Snapshot: not enabled (original message content was not saved to disk; IDs and hashes recorded only)" in diff_text
    assert "Note: absence detected by deterministic keyword matching" in diff_text


def test_render_compaction_diff_with_snapshot_available(tmp_path: Path):
    ledger_path = tmp_path / "test_snap.jsonl"
    ledger = Ledger(ledger_path)
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()

    snap_file = snapshots_dir / "snap-abc123.json"
    snap_file.write_text(
        json.dumps({
            "snapshot_ref": "snap-abc123",
            "summary": "Full summary retrieved from snapshot file.",
        }),
        encoding="utf-8",
    )

    ledger.append(
        create_event(
            event_type="compaction_completed",
            event_id="evt-comp-2",
            run_id="run-200",
            turn=5,
            context_before_tokens=3000,
            context_after_tokens=1000,
            retained_ids=["msg-005"],
            dropped_ids=["msg-001"],
            summary_ref="sum-2",
            snapshot_ref="snap-abc123",
            summary_preview="Preview",
            simulation=True,
        )
    )

    diff_text = render_compaction_diff(ledger_path=ledger_path, turn=5, snapshots_dir=snapshots_dir)
    assert "Full summary retrieved from snapshot file." in diff_text
    assert f"available ({snap_file})" in diff_text


def test_render_missing_and_unknown_turns(tmp_path: Path):
    missing_path = tmp_path / "nonexistent.jsonl"
    res = render_compaction_diff(ledger_path=missing_path)
    assert "Error: Ledger file not found" in res

    empty_path = tmp_path / "empty.jsonl"
    empty_path.touch()
    res_empty = render_compaction_diff(ledger_path=empty_path)
    assert "empty" in res_empty

    # Unknown turn
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(create_event(event_type="run_started", run_id="r1", task="t", simulation=True))
    res_unk = render_compaction_diff(ledger_path=tmp_path / "ledger.jsonl", turn=99)
    assert "No compaction event found for turn 99" in res_unk


def test_cli_diff_subcommand(tmp_path: Path, capsys):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    ledger.append(
        create_event(
            event_type="compaction_completed",
            event_id="evt-c1",
            run_id="r1",
            turn=1,
            context_before_tokens=800,
            context_after_tokens=400,
            retained_ids=["msg-002"],
            dropped_ids=["msg-001"],
            summary_ref="sum-1",
            summary_preview="Summary line",
            simulation=True,
        )
    )

    exit_code = cli_main(["diff", "--ledger", str(ledger_path), "--turn", "1", "--budget", "1000"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "TURN 1  CONTEXT 80%" in captured.out
    assert "NEW SUMMARY" in captured.out
    assert "Summary line" in captured.out
