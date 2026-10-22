"""Tests for context compaction transformation, snapshot policies, and ID stability."""

from pathlib import Path
import json
import pytest

from ctxbeacon.compaction import Compactor, SNAPSHOT_WARNING
from ctxbeacon.ledger import Ledger
from ctxbeacon.providers.mock import MockProvider


def test_compaction_triggers_and_preserves_system_and_recent(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    provider = MockProvider(summary_text="Summarized: User worked on authentication.")
    compactor = Compactor(
        provider=provider,
        ledger=ledger,
        enable_snapshots=False,
        keep_recent_turns=2,
    )

    # Build 6 messages
    messages = [
        {"id": "msg-001", "role": "system", "content": "You are a coding assistant. Never touch migrations/."},
        {"id": "msg-002", "role": "user", "content": "First instruction: refactor auth."},
        {"id": "msg-003", "role": "assistant", "content": "Working on auth."},
        {"id": "msg-004", "role": "user", "content": "Second instruction: update tokens."},
        {"id": "msg-005", "role": "assistant", "content": "Working on tokens."},
        {"id": "msg-006", "role": "user", "content": "Third instruction: run tests."},
    ]

    # Set small budget so compaction triggers
    result = compactor.compact_context(
        messages=messages,
        run_id="run-comp-1",
        turn=3,
        context_budget=50,   # Very small budget to force trigger
        compaction_threshold=0.5,
    )

    assert result.compacted is True
    assert result.dropped_ids == ["msg-002", "msg-003", "msg-004"]
    # Retained: system msg-001, recent non-system msg-005, msg-006, plus summary
    assert "msg-001" in result.retained_ids
    assert "msg-005" in result.retained_ids
    assert "msg-006" in result.retained_ids
    assert result.summary_ref in result.retained_ids
    assert result.snapshot_ref is None

    # Context hashes should differ before and after
    assert result.context_hash_before != result.context_hash_after

    # Verify ledger recorded compaction events
    events = ledger.read_all()
    ev_types = [e["event_type"] for e in events]
    assert "compaction_started" in ev_types
    assert "compaction_completed" in ev_types

    done_evt = next(e for e in events if e["event_type"] == "compaction_completed")
    assert done_evt["retained_ids"] == result.retained_ids
    assert done_evt["dropped_ids"] == result.dropped_ids
    assert done_evt["snapshot_ref"] is None


def test_snapshots_policy_when_enabled(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    provider = MockProvider(summary_text="Summarized context.")
    snapshots_dir = tmp_path / "snapshots"

    with pytest.warns(UserWarning, match="Snapshots store conversation context"):
        compactor = Compactor(
            provider=provider,
            ledger=ledger,
            snapshots_dir=snapshots_dir,
            enable_snapshots=True,
            keep_recent_turns=1,
        )

    messages = [
        {"id": "msg-001", "role": "system", "content": "System prompt."},
        {"id": "msg-002", "role": "user", "content": "Task A"},
        {"id": "msg-003", "role": "assistant", "content": "Task A done"},
        {"id": "msg-004", "role": "user", "content": "Task B"},
    ]

    result = compactor.compact_context(
        messages=messages,
        run_id="run-snap-1",
        turn=2,
        context_budget=20,
        compaction_threshold=0.5,
    )

    assert result.compacted is True
    assert result.snapshot_ref is not None
    assert result.snapshot_ref.startswith("snap-")

    # Snapshot file exists
    snap_file = snapshots_dir / f"{result.snapshot_ref}.json"
    assert snap_file.exists()
    snap_content = json.loads(snap_file.read_text(encoding="utf-8"))
    assert snap_content["snapshot_ref"] == result.snapshot_ref
    assert snap_content["run_id"] == "run-snap-1"
    assert len(snap_content["original_messages"]) == 4
    assert snap_content["dropped_ids"] == ["msg-002", "msg-003"]
