"""Event schema definitions, deterministic hashing, and validation for CtxBeacon."""

from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional, Set

SCHEMA_VERSION = "1.0.0"

EVENT_TYPES: Set[str] = {
    "run_started",
    "request_sent",
    "provider_response",
    "tool_call_started",
    "tool_call_completed",
    "compaction_started",
    "compaction_completed",
    "provider_failover",
    "constraint_check",
    "run_completed",
    "request_blocked",
}

COMMON_REQUIRED_FIELDS: Set[str] = {
    "schema_version",
    "event_id",
    "ts",
    "run_id",
    "event_type",
    "simulation",
    "status",
}

# Sensitive keys that should never be present in hashed payloads or normal ledger events
SENSITIVE_KEYS: Set[str] = {
    "authorization",
    "api_key",
    "apikey",
    "bharatcode_api_key",
    "secret",
    "password",
    "token",
}


def sanitize_payload_for_hashing(obj: Any) -> Any:
    """Recursively strip sensitive credentials/headers before deterministic hashing."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if str(k).lower() in SENSITIVE_KEYS:
                continue
            cleaned[k] = sanitize_payload_for_hashing(v)
        return cleaned
    elif isinstance(obj, list):
        return [sanitize_payload_for_hashing(item) for item in obj]
    return obj


def compute_context_hash(messages: List[Dict[str, Any]]) -> str:
    """
    Calculate deterministic SHA-256 over effective conversation history.
    
    A hash proves payload identity, not what the remote model internally attended to.
    It verifies the context constructed by CtxBeacon; it does not independently
    prove receipt or attention by a remote model.
    """
    clean_messages = sanitize_payload_for_hashing(messages)
    serialized = json.dumps(clean_messages, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_request_hash(payload: Dict[str, Any]) -> str:
    """
    Calculate deterministic SHA-256 over application-side request payload sent to the provider,
    strictly excluding credentials and authorization headers.
    """
    clean_payload = sanitize_payload_for_hashing(payload)
    serialized = json.dumps(clean_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def generate_event_id() -> str:
    """Return a standard UUID4 identifier for events."""
    return f"evt-{uuid.uuid4().hex[:12]}"


def generate_run_id() -> str:
    """Return a standard UUID4 identifier for runs."""
    return f"run-{uuid.uuid4().hex[:12]}"


def current_iso_timestamp() -> str:
    """Return current UTC ISO 8601 timestamp with Z designator."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


class ValidationError(ValueError):
    """Raised when an event does not conform to the CtxBeacon schema."""
    pass


def validate_event(event: Dict[str, Any], line_num: Optional[int] = None) -> None:
    """
    Validate a single ledger event dict against schema rules.
    Raises ValidationError with descriptive message including line number if provided.
    """
    prefix = f"Line {line_num}: " if line_num is not None else ""

    if not isinstance(event, dict):
        raise ValidationError(f"{prefix}Event record must be a JSON object, got {type(event).__name__}")

    # Check common required fields
    for field in COMMON_REQUIRED_FIELDS:
        if field not in event:
            raise ValidationError(f"{prefix}Missing required common field '{field}'")
        if event[field] is None:
            raise ValidationError(f"{prefix}Field '{field}' cannot be null")

    # Validate types of common fields
    if not isinstance(event["schema_version"], str) or not event["schema_version"]:
        raise ValidationError(f"{prefix}'schema_version' must be a non-empty string")

    if not isinstance(event["event_id"], str) or not event["event_id"]:
        raise ValidationError(f"{prefix}'event_id' must be a non-empty string")

    if not isinstance(event["run_id"], str) or not event["run_id"]:
        raise ValidationError(f"{prefix}'run_id' must be a non-empty string")

    if not isinstance(event["simulation"], bool):
        raise ValidationError(f"{prefix}'simulation' must be a boolean")

    if not isinstance(event["status"], str) or not event["status"]:
        raise ValidationError(f"{prefix}'status' must be a non-empty string")

    # Validate timestamp
    ts = event["ts"]
    if not isinstance(ts, str):
        raise ValidationError(f"{prefix}'ts' must be an ISO 8601 string")
    try:
        # Verify it can be parsed as ISO timestamp
        normalized_ts = ts.replace("Z", "+00:00")
        datetime.datetime.fromisoformat(normalized_ts)
    except Exception as e:
        raise ValidationError(f"{prefix}Invalid ISO 8601 timestamp '{ts}': {e}")

    # Validate turn if present
    if "turn" in event and event["turn"] is not None:
        if not isinstance(event["turn"], int) or event["turn"] < 0:
            raise ValidationError(f"{prefix}'turn' must be a non-negative integer or null")

    # Validate event_type
    event_type = event["event_type"]
    if event_type not in EVENT_TYPES:
        raise ValidationError(f"{prefix}Unknown event_type '{event_type}'. Must be one of {sorted(EVENT_TYPES)}")

    # Check for secret leakage in any field
    for k in event.keys():
        if k.lower() in SENSITIVE_KEYS:
            raise ValidationError(f"{prefix}Forbidden sensitive field '{k}' detected in ledger event")

    # Event-type specific validation
    if event_type == "run_started":
        if "task" not in event:
            raise ValidationError(f"{prefix}'run_started' requires 'task'")

    elif event_type == "request_sent":
        for req in ["provider", "model", "request_hash", "context_hash"]:
            if req not in event:
                raise ValidationError(f"{prefix}'request_sent' requires '{req}'")

    elif event_type == "provider_response":
        for req in ["provider", "model", "status_code", "latency_ms"]:
            if req not in event:
                raise ValidationError(f"{prefix}'provider_response' requires '{req}'")

    elif event_type == "tool_call_started":
        for req in ["tool_name", "tool_call_id"]:
            if req not in event:
                raise ValidationError(f"{prefix}'tool_call_started' requires '{req}'")

    elif event_type == "tool_call_completed":
        for req in ["tool_name", "tool_call_id", "execution_status"]:
            if req not in event:
                raise ValidationError(f"{prefix}'tool_call_completed' requires '{req}'")

    elif event_type == "compaction_started":
        if "context_before_tokens" not in event:
            raise ValidationError(f"{prefix}'compaction_started' requires 'context_before_tokens'")

    elif event_type == "compaction_completed":
        for req in ["context_before_tokens", "context_after_tokens", "retained_ids", "dropped_ids"]:
            if req not in event:
                raise ValidationError(f"{prefix}'compaction_completed' requires '{req}'")
        if not isinstance(event["retained_ids"], list):
            raise ValidationError(f"{prefix}'retained_ids' must be a list of message IDs")
        if not isinstance(event["dropped_ids"], list):
            raise ValidationError(f"{prefix}'dropped_ids' must be a list of message IDs")

    elif event_type == "provider_failover":
        for req in ["from_provider", "to_provider", "reason"]:
            if req not in event:
                raise ValidationError(f"{prefix}'provider_failover' requires '{req}'")

    elif event_type == "constraint_check":
        if "constraint_findings" not in event:
            raise ValidationError(f"{prefix}'constraint_check' requires 'constraint_findings'")
        if not isinstance(event["constraint_findings"], list):
            raise ValidationError(f"{prefix}'constraint_findings' must be a list")

    elif event_type == "run_completed":
        if "total_turns" not in event:
            raise ValidationError(f"{prefix}'run_completed' requires 'total_turns'")

    elif event_type == "request_blocked":
        if "reason" not in event:
            raise ValidationError(f"{prefix}'request_blocked' requires 'reason'")


def create_event(
    event_type: str,
    run_id: str,
    turn: Optional[int] = None,
    simulation: bool = False,
    status: str = "ok",
    event_id: Optional[str] = None,
    ts: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Helper to construct and validate a ledger event."""
    event: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id or generate_event_id(),
        "ts": ts or current_iso_timestamp(),
        "run_id": run_id,
        "turn": turn,
        "event_type": event_type,
        "simulation": simulation,
        "status": status,
    }
    for k, v in kwargs.items():
        if v is not None or k in ("snapshot_ref", "summary_ref", "turn"):
            event[k] = v

    validate_event(event)
    return event
