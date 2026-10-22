"""Base provider interface and data models for CtxBeacon."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """Represents a structured function/tool call requested by a model."""
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderUsage:
    """Token usage metrics reported by a provider."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ProviderResponse:
    """Standardized response from any model provider."""
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Optional[ProviderUsage] = None
    status_code: int = 200
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_after_seconds: Optional[float] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseProvider(ABC):
    """Abstract base class for CtxBeacon model providers."""

    name: str = "base"
    model: str = "base-model"
    is_simulation: bool = False

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
    ) -> ProviderResponse:
        """
        Request a completion from the model.
        Must handle both plain text and structured tool calls.
        """
        pass

    @abstractmethod
    def summarize(
        self,
        messages: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
    ) -> ProviderResponse:
        """
        Generate a concise compaction summary from a list of messages.
        """
        pass
