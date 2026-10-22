"""Provider fallback and failover orchestration for CtxBeacon."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from ctxbeacon.ledger import Ledger
from ctxbeacon.providers.base import BaseProvider, ProviderResponse
from ctxbeacon.schema import compute_context_hash, create_event


class FallbackProvider(BaseProvider):
    """
    Wraps a primary and optional fallback provider.
    
    Handles transient rate limits (HTTP 429) and server errors with bounded retries,
    respecting Retry-After headers, and fails over safely while preserving context hashes.
    """

    def __init__(
        self,
        primary: BaseProvider,
        fallback: Optional[BaseProvider] = None,
        ledger: Optional[Ledger] = None,
        max_primary_retries: int = 1,
        sleep_fn: Callable[[float], None] = time.sleep,
        run_id: str = "run-failover",
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.ledger = ledger
        self.max_primary_retries = max_primary_retries
        self.sleep_fn = sleep_fn
        self.run_id = run_id

        self.name = f"fallback({primary.name}->{fallback.name if fallback else 'none'})"
        self.model = primary.model
        self.is_simulation = primary.is_simulation or (fallback.is_simulation if fallback else False)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
    ) -> ProviderResponse:
        context_hash = compute_context_hash(messages)
        attempt = 0
        last_response: Optional[ProviderResponse] = None

        while attempt <= self.max_primary_retries:
            attempt += 1
            try:
                resp = self.primary.generate(
                    messages=messages,
                    tools=tools,
                    system_instruction=system_instruction,
                )
                if resp.status_code < 400:
                    return resp

                last_response = resp
                # If retryable 429 and retries remain, respect Retry-After
                if resp.status_code == 429 and attempt <= self.max_primary_retries:
                    wait_time = resp.retry_after_seconds or 1.0
                    self.sleep_fn(wait_time)
                    continue

                # Non-retryable or retries exhausted -> break to failover
                break

            except Exception as e:
                last_response = ProviderResponse(
                    content=None,
                    status_code=500,
                    error=f"Primary provider exception: {e}",
                )
                break

        # Check if fallback is available
        if not self.fallback:
            return last_response or ProviderResponse(
                content=None,
                status_code=500,
                error="Primary failed and no fallback provider is configured.",
            )

        # Record provider_failover event in ledger
        reason_msg = (
            f"Primary {self.primary.name} ({self.primary.model}) returned HTTP {last_response.status_code if last_response else 'error'}: "
            f"{last_response.error if last_response else 'unknown error'}"
        )

        if self.ledger:
            self.ledger.append(
                create_event(
                    event_type="provider_failover",
                    run_id=self.run_id,
                    from_provider=self.primary.name,
                    from_model=self.primary.model,
                    to_provider=self.fallback.name,
                    to_model=self.fallback.model,
                    status_code=last_response.status_code if last_response else 500,
                    retry_after_seconds=last_response.retry_after_seconds if last_response else None,
                    reason=reason_msg,
                    context_hash=context_hash,
                    simulation=self.is_simulation,
                )
            )

        # Dispatch request to fallback provider
        fb_resp = self.fallback.generate(
            messages=messages,
            tools=tools,
            system_instruction=system_instruction,
        )
        return fb_resp

    def summarize(
        self,
        messages: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
    ) -> ProviderResponse:
        try:
            resp = self.primary.summarize(messages, system_instruction=system_instruction)
            if resp.status_code < 400:
                return resp
        except Exception:
            pass

        if self.fallback:
            return self.fallback.summarize(messages, system_instruction=system_instruction)

        return ProviderResponse(
            content=None,
            status_code=500,
            error="Summarization failed on primary and no fallback available.",
        )
