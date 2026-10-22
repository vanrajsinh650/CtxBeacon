"""Tests for demo scenarios: simulated compaction loss, clean run, and experiment runner."""

from pathlib import Path
import pytest

from ctxbeacon.cli import main as cli_main
from ctxbeacon.ledger import Ledger
from ctxbeacon.scenarios import (
    DISTINCT_TASKS,
    run_bharatcode_experiments,
    run_clean_scenario,
    run_simulated_compaction_loss,
)


def test_five_distinct_tasks_defined():
    assert len(DISTINCT_TASKS) == 5
    task_names = set(t["name"] for t in DISTINCT_TASKS)
    assert len(task_names) == 5  # Ensure distinct
    for t in DISTINCT_TASKS:
        assert len(t["constraints"]) > 0
        assert len(t["constraints"][0]["marker_terms"]) > 0


def test_simulated_compaction_loss_scenario(tmp_path: Path):
    ws_dir = tmp_path / "workspace"
    ledger_path = tmp_path / "ledger_loss.jsonl"

    result = run_simulated_compaction_loss(
        workspace_dir=ws_dir,
        ledger_path=ledger_path,
        enable_snapshots=True,
    )

    assert result["simulation"] is True
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["status"] == "violation_observed"
    assert "Possible cause: the original requirement was present before compaction" in finding["details"]

    diff_output = result["diff_output"]
    assert "POSSIBLY MISSING" in diff_output
    assert "migrations" in diff_output

    # Validate resulting ledger
    valid_count, errors = Ledger.validate_file(ledger_path)
    assert len(errors) == 0
    assert valid_count > 0


def test_clean_scenario(tmp_path: Path):
    ws_dir = tmp_path / "workspace"
    ledger_path = tmp_path / "ledger_clean.jsonl"

    result = run_clean_scenario(
        workspace_dir=ws_dir,
        ledger_path=ledger_path,
        enable_snapshots=False,
    )

    assert result["simulation"] is True
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["status"] == "preserved"

    diff_output = result["diff_output"]
    assert "PRESERVED CONSTRAINTS" in diff_output

    valid_count, errors = Ledger.validate_file(ledger_path)
    assert len(errors) == 0
    assert valid_count > 0


def test_bharatcode_experiments_reports_pending_without_key(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("BHARATCODE_API_KEY", raising=False)
    ledger_path = tmp_path / "ledger_exp.jsonl"

    result = run_bharatcode_experiments(
        api_key=None,
        ledger_path=ledger_path,
    )

    assert result["status"] == "pending_credentials"
    assert "BHARATCODE_API_KEY is not configured" in result["message"]
    assert result["completed_runs"] == 0


def test_cli_demo_commands(tmp_path: Path, capsys):
    ws_dir = tmp_path / "ws_cli"
    ledger_p = tmp_path / "cli_loss.jsonl"

    # Test demo --scenario compaction-loss
    code = cli_main([
        "demo",
        "--scenario", "compaction-loss",
        "--workspace", str(ws_dir),
        "--ledger", str(ledger_p),
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "Running Scenario B: Deterministic Simulated Compaction Loss" in captured.out
    assert "POSSIBLY MISSING" in captured.out

    # Test demo --scenario clean
    clean_p = tmp_path / "cli_clean.jsonl"
    code_clean = cli_main([
        "demo",
        "--scenario", "clean",
        "--workspace", str(ws_dir),
        "--ledger", str(clean_p),
    ])
    assert code_clean == 0
    captured_clean = capsys.readouterr()
    assert "Running Scenario: Clean Compaction" in captured_clean.out
    assert "PRESERVED CONSTRAINTS" in captured_clean.out
