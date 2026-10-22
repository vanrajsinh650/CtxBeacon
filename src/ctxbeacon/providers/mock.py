"""Deterministic mock provider for CtxBeacon local development and testing."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Union

from ctxbeacon.providers.base import (
    BaseProvider,
    ProviderResponse,
    ProviderUsage,
    ToolCall,
)


class MockProvider(BaseProvider):
    """
    Deterministic mock provider for offline development, local unit tests,
    and simulated failure/compaction scenarios.
    
    Always marks simulation=True and exercises the identical application
    interfaces as live providers.
    """

    name: str = "mock"
    model: str = "mock-qwen-3.8-27b"
    is_simulation: bool = True

    def __init__(
        self,
        default_response: str = "Task completed successfully by mock agent.",
        scripted_responses: Optional[List[Union[ProviderResponse, Exception]]] = None,
        summary_text: str = "Summary of previous context: User constraints and completed steps preserved.",
        model_name: str = "mock-qwen-3.8-27b",
        latency_ms: float = 12.0,
    ) -> None:
        self.default_response = default_response
        self.scripted_responses = list(scripted_responses) if scripted_responses else []
        self.summary_text = summary_text
        self.model = model_name
        self.latency_ms = latency_ms
        self.call_count = 0
        self.history: List[Dict[str, Any]] = []

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
    ) -> ProviderResponse:
        self.call_count += 1
        self.history.append({
            "messages": messages,
            "tools": tools,
            "system_instruction": system_instruction,
            "method": "generate",
        })

        if self.scripted_responses:
            next_item = self.scripted_responses.pop(0)
            if isinstance(next_item, Exception):
                raise next_item
            return next_item

        # Default fallback response
        return ProviderResponse(
            content=self.default_response,
            tool_calls=[],
            usage=ProviderUsage(prompt_tokens=45, completion_tokens=15, total_tokens=60),
            status_code=200,
            latency_ms=self.latency_ms,
        )

    def summarize(
        self,
        messages: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
    ) -> ProviderResponse:
        self.call_count += 1
        self.history.append({
            "messages": messages,
            "system_instruction": system_instruction,
            "method": "summarize",
        })

        if self.scripted_responses:
            next_item = self.scripted_responses.pop(0)
            if isinstance(next_item, Exception):
                raise next_item
            return next_item

        return ProviderResponse(
            content=self.summary_text,
            tool_calls=[],
            usage=ProviderUsage(prompt_tokens=120, completion_tokens=30, total_tokens=150),
            status_code=200,
            latency_ms=self.latency_ms,
        )

    @classmethod
    def with_tool_sequence(
        cls,
        tool_name: str,
        arguments: Dict[str, Any],
        follow_up_content: str = "Tool executed and results verified.",
        call_id: str = "call_mock_1",
    ) -> "MockProvider":
        """Convenience factory for a 2-step tool call sequence."""
        first_resp = ProviderResponse(
            content=None,
            tool_calls=[ToolCall(id=call_id, name=tool_name, arguments=arguments)],
            usage=ProviderUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
            status_code=200,
            latency_ms=10.0,
        )
        second_resp = ProviderResponse(
            content=follow_up_content,
            tool_calls=[],
            usage=ProviderUsage(prompt_tokens=70, completion_tokens=15, total_tokens=85),
            status_code=200,
            latency_ms=10.0,
        )
        return cls(scripted_responses=[first_resp, second_resp])

    @classmethod
    def with_simulated_error(
        cls,
        status_code: int = 429,
        error_message: str = "Rate limit exceeded (simulated)",
        retry_after: float = 2.0,
    ) -> "MockProvider":
        """Factory for a provider that returns a simulated error."""
        error_resp = ProviderResponse(
            content=None,
            status_code=status_code,
            error=error_message,
            retry_after_seconds=retry_after,
            latency_ms=5.0,
        )
        return cls(scripted_responses=[error_resp])
