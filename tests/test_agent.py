"""Tests for CtxBeacon basic agent loop and sandboxed tool execution."""

import json
from pathlib import Path
import pytest

from ctxbeacon.agent import AgentLoop
from ctxbeacon.ledger import Ledger
from ctxbeacon.providers.base import ProviderResponse, ProviderUsage, ToolCall
from ctxbeacon.providers.mock import MockProvider
from ctxbeacon.tools import WorkspaceSandbox


def test_normal_assistant_response_ends_turn(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    provider = MockProvider(default_response="All done!")
    agent = AgentLoop(
        provider=provider,
        ledger=ledger,
        workspace_dir=tmp_path / "workspace",
        max_turns=5,
    )

    result = agent.run("Hello assistant")
    assert result["termination_reason"] == "completed"
    assert result["turns"] == 1
    assert result["total_tool_calls"] == 0

    events = ledger.read_all()
    event_types = [e["event_type"] for e in events]
    assert event_types == ["run_started", "request_sent", "provider_response", "run_completed"]
    assert all(e["simulation"] is True for e in events)


def test_tool_call_and_follow_up_execution(tmp_path: Path):
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()
    ledger = Ledger(tmp_path / "ledger.jsonl")

    # Script provider: 1st turn calls write_file, 2nd turn returns completion text
    provider = MockProvider.with_tool_sequence(
        tool_name="write_file",
        arguments={"path": "hello.txt", "content": "Hello CtxBeacon!"},
        follow_up_content="File created successfully.",
    )

    agent = AgentLoop(
        provider=provider,
        ledger=ledger,
        workspace_dir=ws_dir,
        max_turns=5,
    )

    result = agent.run("Create hello.txt")
    assert result["turns"] == 2
    assert result["total_tool_calls"] == 1

    # Verify file actually written to workspace
    written_file = ws_dir / "hello.txt"
    assert written_file.exists()
    assert written_file.read_text(encoding="utf-8") == "Hello CtxBeacon!"

    events = ledger.read_all()
    event_types = [e["event_type"] for e in events]
    assert "tool_call_started" in event_types
    assert "tool_call_completed" in event_types

    # Find tool_call_completed event
    tc_done = next(e for e in events if e["event_type"] == "tool_call_completed")
    assert tc_done["tool_name"] == "write_file"
    assert tc_done["execution_status"] == "success"
    assert tc_done["target_path"] == "hello.txt"


def test_workspace_escaping_path_is_blocked(tmp_path: Path):
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()
    ledger = Ledger(tmp_path / "ledger.jsonl")

    # Tool call trying to write outside workspace via ../
    provider = MockProvider.with_tool_sequence(
        tool_name="write_file",
        arguments={"path": "../escape.txt", "content": "evil payload"},
        follow_up_content="Attempted write outside.",
    )

    agent = AgentLoop(
        provider=provider,
        ledger=ledger,
        workspace_dir=ws_dir,
        max_turns=5,
    )

    result = agent.run("Escape sandbox")
    assert result["turns"] == 2

    # Check file was NOT written outside
    assert not (tmp_path / "escape.txt").exists()

    events = ledger.read_all()
    event_types = [e["event_type"] for e in events]
    assert "request_blocked" in event_types
    blocked_evt = next(e for e in events if e["event_type"] == "request_blocked")
    assert blocked_evt["status"] == "blocked"
    assert "Security boundary violation" in blocked_evt["reason"]


def test_unknown_tool_and_invalid_arguments_handled_safely(tmp_path: Path):
    sandbox = WorkspaceSandbox(tmp_path / "ws")
    
    # Unknown tool
    res_unk = sandbox.execute_tool("launch_missiles", {"target": "moon"})
    assert res_unk.status == "error"
    assert "Unknown tool" in res_unk.output

    # Invalid arguments JSON
    res_bad_json = sandbox.execute_tool("write_file", "{ not valid json")
    assert res_bad_json.status == "error"
    assert "Invalid JSON arguments" in res_bad_json.output

    # Missing argument
    res_missing = sandbox.execute_tool("write_file", {"path": "test.txt"})
    assert res_missing.status == "error"
    assert "Missing required argument 'content'" in res_missing.output


def test_turn_limits_enforced(tmp_path: Path):
    ws_dir = tmp_path / "workspace"
    ledger = Ledger(tmp_path / "ledger.jsonl")

    # Provider that perpetually calls a tool
    continuous_responses = [
        ProviderResponse(
            content=None,
            tool_calls=[ToolCall(id=f"c_{i}", name="list_files", arguments={})],
            status_code=200,
        )
        for i in range(10)
    ]
    provider = MockProvider(scripted_responses=continuous_responses)

    agent = AgentLoop(
        provider=provider,
        ledger=ledger,
        workspace_dir=ws_dir,
        max_turns=3,
    )

    res = agent.run("Loop forever")
    assert res["turns"] == 3
    assert res["termination_reason"] == "max_turns_reached"

    completed_evt = ledger.read_all()[-1]
    assert completed_evt["event_type"] == "run_completed"
    assert completed_evt["termination_reason"] == "max_turns_reached"
