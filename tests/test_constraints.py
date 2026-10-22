"""Tests for deterministic constraint-loss checker and filesystem violation audits."""

import pytest
from ctxbeacon.constraints import (
    ConstraintChecker,
    ConstraintDefinition,
    KEYWORD_LIMITATION_NOTE,
)
from ctxbeacon.schema import create_event


def test_presence_check_preserved():
    constraint = ConstraintDefinition(
        text="Never modify files under migrations/",
        marker_terms=["migrations", "never modify"],
        evidence_message_id="msg-002",
    )
    context = [
        {"role": "system", "content": "Assistant prompt."},
        {"role": "system", "content": "[Compacted Summary]: Never modify files under migrations/ directory."},
        {"role": "user", "content": "Continue work."},
    ]

    finding = ConstraintChecker.check_presence(constraint, context)
    assert finding.status == "preserved"
    assert "migrations" in finding.matched_markers
    assert KEYWORD_LIMITATION_NOTE in finding.details


def test_presence_check_possibly_missing():
    constraint = ConstraintDefinition(
        text="Never modify files under migrations/",
        marker_terms=["migrations", "never modify"],
        evidence_message_id="msg-002",
    )
    # Context where summary omitted migrations requirement
    context = [
        {"role": "system", "content": "Assistant prompt."},
        {"role": "system", "content": "[Compacted Summary]: Refactor authentication module."},
        {"role": "user", "content": "Continue work."},
    ]

    finding = ConstraintChecker.check_presence(constraint, context)
    assert finding.status == "possibly_missing"
    assert finding.matched_markers == []
    assert finding.evidence_message_id == "msg-002"
    assert "Original requirement was present before compaction" in finding.details


def test_violation_observed_on_completed_prohibited_write():
    constraint = ConstraintDefinition(
        text="Never modify files under migrations/",
        marker_terms=["migrations"],
        prohibited_paths=["migrations", "migrations/"],
        evidence_message_id="msg-002",
    )
    context = [
        {"role": "system", "content": "[Compacted Summary]: Auth refactor in progress."},
    ]
    ledger_events = [
        create_event(
            event_type="tool_call_completed",
            run_id="run-1",
            turn=4,
            tool_name="write_file",
            tool_call_id="call-bad-write",
            execution_status="success",
            target_path="migrations/0002_user.py",
            simulation=True,
        )
    ]

    findings = ConstraintChecker.audit_context_and_ledger(
        constraints=[constraint],
        context_messages=context,
        ledger_events=ledger_events,
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.status == "violation_observed"
    assert finding.evidence_tool_call_id == "call-bad-write"
    assert "Possible cause: the original requirement was present before compaction" in finding.details


def test_rejected_write_is_not_reported_as_completed_violation():
    constraint = ConstraintDefinition(
        text="Never modify files under migrations/",
        marker_terms=["migrations"],
        prohibited_paths=["migrations/"],
        evidence_message_id="msg-002",
    )
    context = [
        {"role": "system", "content": "[Compacted Summary]: Auth refactor."},
    ]
    # Rejected write attempt (e.g. permission/sandbox error)
    ledger_events = [
        create_event(
            event_type="tool_call_completed",
            run_id="run-1",
            turn=4,
            tool_name="write_file",
            tool_call_id="call-blocked",
            execution_status="rejected",
            target_path="migrations/0002_user.py",
            simulation=True,
        )
    ]

    findings = ConstraintChecker.audit_context_and_ledger(
        constraints=[constraint],
        context_messages=context,
        ledger_events=ledger_events,
    )
    assert len(findings) == 1
    # Must NOT be violation_observed
    assert findings[0].status == "possible_violation"
    assert "was blocked" in findings[0].details


def test_unable_to_verify_when_no_markers_declared():
    constraint = ConstraintDefinition(
        text="Some vague rule without keywords",
        marker_terms=[],
    )
    context = [{"role": "user", "content": "hello"}]
    finding = ConstraintChecker.check_presence(constraint, context)
    assert finding.status == "unable_to_verify"
