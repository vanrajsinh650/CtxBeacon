"""Tests for provider failover, transient 429 retries, and fallback logging."""

from pathlib import Path
from typing import List
import pytest

from ctxbeacon.ledger import Ledger
from ctxbeacon.providers.base import ProviderResponse
from ctxbeacon.providers.fallback import FallbackProvider
from ctxbeacon.providers.mock import MockProvider
from ctxbeacon.schema import compute_context_hash


def test_simulated_429_retry_and_failover_to_fallback(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    
    sleep_calls: List[float] = []

    def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    # Primary provider returns 429 with Retry-After: 2 on attempt 1 and 2
    primary_responses = [
        ProviderResponse(
            content=None,
            status_code=429,
            error="Rate limit exceeded (simulated)",
            retry_after_seconds=2.0,
        ),
        ProviderResponse(
            content=None,
            status_code=429,
            error="Rate limit exceeded (simulated)",
            retry_after_seconds=2.0,
        ),
    ]
    primary = MockProvider(
        model_name="mock-primary-model",
        scripted_responses=primary_responses,
    )
    primary.name = "mock-primary"

    # Fallback provider returns successful response
    fallback = MockProvider(
        model_name="mock-fallback-model",
        default_response="Response from fallback provider.",
    )
    fallback.name = "mock-fallback"

    provider = FallbackProvider(
        primary=primary,
        fallback=fallback,
        ledger=ledger,
        max_primary_retries=1,
        sleep_fn=mock_sleep,
        run_id="run-failover-test",
    )

    messages = [{"role": "user", "content": "Help with query"}]
    ctx_hash = compute_context_hash(messages)

    resp = provider.generate(messages=messages)

    # 1. Successful response originated from fallback
    assert resp.status_code == 200
    assert resp.content == "Response from fallback provider."

    # 2. Injected sleep was called with 2.0 seconds (without blocking real time)
    assert sleep_calls == [2.0]

    # 3. Provider marked as simulation
    assert provider.is_simulation is True

    # 4. Ledger contains provider_failover event
    events = ledger.read_all()
    failover_evts = [e for e in events if e["event_type"] == "provider_failover"]
    assert len(failover_evts) == 1
    fo = failover_evts[0]
    assert fo["from_provider"] == "mock-primary"
    assert fo["to_provider"] == "mock-fallback"
    assert fo["from_model"] == "mock-primary-model"
    assert fo["to_model"] == "mock-fallback-model"
    assert fo["status_code"] == 429
    assert fo["retry_after_seconds"] == 2.0
    assert fo["simulation"] is True
    assert fo["context_hash"] == ctx_hash


def test_primary_failure_without_fallback_returns_error(tmp_path: Path):
    primary = MockProvider.with_simulated_error(status_code=500, error_message="Internal error")
    provider = FallbackProvider(
        primary=primary,
        fallback=None,
        max_primary_retries=0,
    )

    resp = provider.generate(messages=[{"role": "user", "content": "hi"}])
    assert resp.status_code == 500
    assert "Internal error" in resp.error
