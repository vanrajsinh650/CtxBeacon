"""Tests for CtxBeacon event schema, validation, and deterministic hashing."""

import json
import pytest

from ctxbeacon.schema import (
    COMMON_REQUIRED_FIELDS,
    EVENT_TYPES,
    SCHEMA_VERSION,
    ValidationError,
    compute_context_hash,
    compute_request_hash,
    create_event,
    sanitize_payload_for_hashing,
    validate_event,
)


def test_create_valid_event():
    event = create_event(
        event_type="run_started",
        run_id="run-12345",
        task="Test run",
        simulation=True,
    )
    assert event["schema_version"] == SCHEMA_VERSION
    assert event["event_type"] == "run_started"
    assert event["run_id"] == "run-12345"
    assert event["simulation"] is True
    assert event["status"] == "ok"
    assert "evt-" in event["event_id"]
    assert event["ts"].endswith("Z")


def test_missing_common_fields_raise_error():
    event = {
        "schema_version": "1.0.0",
        "event_id": "evt-1",
        # missing ts
        "run_id": "run-1",
        "event_type": "run_started",
        "simulation": False,
        "status": "ok",
        "task": "do something",
    }
    with pytest.raises(ValidationError, match="Missing required common field 'ts'"):
        validate_event(event, line_num=42)


def test_invalid_event_type_raises_error():
    event = {
        "schema_version": "1.0.0",
        "event_id": "evt-1",
        "ts": "2026-10-22T10:00:00Z",
        "run_id": "run-1",
        "event_type": "unknown_type",
        "simulation": False,
        "status": "ok",
    }
    with pytest.raises(ValidationError, match="Unknown event_type 'unknown_type'"):
        validate_event(event)


def test_sensitive_keys_rejected():
    event = {
        "schema_version": "1.0.0",
        "event_id": "evt-1",
        "ts": "2026-10-22T10:00:00Z",
        "run_id": "run-1",
        "event_type": "run_started",
        "simulation": False,
        "status": "ok",
        "task": "build app",
        "api_key": "secret-123",  # forbidden
    }
    with pytest.raises(ValidationError, match="Forbidden sensitive field 'api_key'"):
        validate_event(event)


def test_deterministic_context_hash():
    messages_a = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Refactor auth and never touch migrations/."},
    ]
    messages_b = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Refactor auth and never touch migrations/."},
    ]
    messages_modified = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Refactor auth and feel free to touch migrations/."},
    ]

    hash_a = compute_context_hash(messages_a)
    hash_b = compute_context_hash(messages_b)
    hash_mod = compute_context_hash(messages_modified)

    assert hash_a == hash_b
    assert hash_a != hash_mod
    assert len(hash_a) == 64  # SHA-256


def test_deterministic_request_hash_strips_secrets():
    payload_a = {
        "model": "qwen-3.8-27b",
        "input": "test prompt",
        "authorization": "Bearer secret_token_xyz",
    }
    payload_b = {
        "model": "qwen-3.8-27b",
        "input": "test prompt",
        "authorization": "Bearer completely_different_token",
    }
    hash_a = compute_request_hash(payload_a)
    hash_b = compute_request_hash(payload_b)

    # Because authorization is filtered out, both hashes must match
    assert hash_a == hash_b


def test_compaction_completed_validation():
    event = create_event(
        event_type="compaction_completed",
        run_id="run-test",
        turn=5,
        context_before_tokens=3500,
        context_after_tokens=1500,
        retained_ids=["msg-004", "msg-005"],
        dropped_ids=["msg-001", "msg-002", "msg-003"],
        summary_ref="sum-001",
    )
    validate_event(event)

    # Missing retained_ids
    invalid_event = dict(event)
    del invalid_event["retained_ids"]
    with pytest.raises(ValidationError, match="requires 'retained_ids'"):
        validate_event(invalid_event)
