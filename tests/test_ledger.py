"""Tests for CtxBeacon JSONL ledger reader, writer, and validation."""

import json
from pathlib import Path
import pytest

from ctxbeacon.cli import main as cli_main
from ctxbeacon.ledger import Ledger, LedgerParseError
from ctxbeacon.schema import compute_context_hash, create_event


def test_ledger_write_read_and_hashes(tmp_path: Path):
    ledger_path = tmp_path / "test_ledger.jsonl"
    ledger = Ledger(ledger_path)

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User question"},
    ]
    orig_context_hash = compute_context_hash(messages)

    event1 = create_event(
        event_type="run_started",
        run_id="run-1",
        task="Test task",
        simulation=True,
    )
    event2 = create_event(
        event_type="request_sent",
        run_id="run-1",
        turn=0,
        provider="mock",
        model="mock-model",
        request_hash="abcd1234abcd1234",
        context_hash=orig_context_hash,
        simulation=True,
    )

    ledger.append(event1)
    ledger.append(event2)

    events = ledger.read_all()
    assert len(events) == 2
    assert events[0]["event_id"] == event1["event_id"]
    assert events[1]["event_id"] == event2["event_id"]
    assert events[1]["context_hash"] == orig_context_hash


def test_earlier_records_preserved(tmp_path: Path):
    ledger_path = tmp_path / "preserve.jsonl"
    ledger = Ledger(ledger_path)

    evt1 = create_event(event_type="run_started", run_id="run-1", task="Task 1", simulation=True)
    ledger.append(evt1)

    # Re-instantiate ledger
    ledger2 = Ledger(ledger_path)
    evt2 = create_event(event_type="run_completed", run_id="run-1", total_turns=1, simulation=True)
    ledger2.append(evt2)

    all_evts = ledger2.read_all()
    assert len(all_evts) == 2
    assert all_evts[0]["event_type"] == "run_started"
    assert all_evts[1]["event_type"] == "run_completed"


def test_malformed_json_reports_line_number(tmp_path: Path):
    ledger_path = tmp_path / "bad.jsonl"
    evt = create_event(event_type="run_started", run_id="run-1", task="Task", simulation=True)
    
    with open(ledger_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(evt) + "\n")
        f.write("{ broken json line\n")
        f.write(json.dumps(evt) + "\n")

    ledger = Ledger(ledger_path)
    with pytest.raises(LedgerParseError) as exc_info:
        ledger.read_all()
    assert exc_info.value.line_num == 2
    assert "Line 2" in str(exc_info.value)


def test_schema_violation_reports_line_number(tmp_path: Path):
    ledger_path = tmp_path / "schema_bad.jsonl"
    evt1 = create_event(event_type="run_started", run_id="run-1", task="Task", simulation=True)
    bad_evt = {
        "schema_version": "1.0.0",
        "event_id": "evt-123",
        "ts": "2026-10-22T10:00:00Z",
        # missing run_id
        "event_type": "run_started",
        "simulation": True,
        "status": "ok",
        "task": "Task",
    }
    with open(ledger_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(evt1) + "\n")
        f.write(json.dumps(bad_evt) + "\n")

    ledger = Ledger(ledger_path)
    with pytest.raises(LedgerParseError) as exc_info:
        ledger.read_all()
    assert exc_info.value.line_num == 2
    assert "Missing required common field 'run_id'" in str(exc_info.value)


def test_cli_validate_command(tmp_path: Path, capsys):
    valid_file = tmp_path / "valid.jsonl"
    ledger = Ledger(valid_file)
    ledger.append(create_event(event_type="run_started", run_id="run-1", task="Task", simulation=True))

    code = cli_main(["validate", "--ledger", str(valid_file)])
    assert code == 0
    captured = capsys.readouterr()
    assert "1 event(s) successfully verified" in captured.out

    # Corrupt file
    with open(valid_file, "a", encoding="utf-8") as f:
        f.write("corrupted json\n")

    code_err = cli_main(["validate", "--ledger", str(valid_file)])
    assert code_err == 1
    captured_err = capsys.readouterr()
    assert "Validation failed" in captured_err.err
    assert "Line 2" in captured_err.err
