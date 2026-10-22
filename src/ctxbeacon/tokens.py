"""Token counting, approximation methods, and context budget calculations for CtxBeacon."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# Primary approximation method
TOKEN_COUNT_METHOD_TIKTOKEN = "tiktoken_cl100k_approximation"
TOKEN_COUNT_METHOD_FALLBACK = "char_ratio_4to1_fallback"

_TIKTOKEN_ENCODER = None


def _get_encoder():
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        try:
            import tiktoken
            _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENCODER = False
    return _TIKTOKEN_ENCODER


def count_text_tokens(text: str) -> tuple[int, str]:
    """
    Count tokens for a given string.
    Returns (token_count, method_name).
    """
    if not text:
        return 0, TOKEN_COUNT_METHOD_TIKTOKEN

    encoder = _get_encoder()
    if encoder:
        return len(encoder.encode(text)), TOKEN_COUNT_METHOD_TIKTOKEN
    
    # Documented fallback if tiktoken is not available
    approx = max(1, len(text) // 4)
    return approx, TOKEN_COUNT_METHOD_FALLBACK


def count_message_tokens(message: Dict[str, Any]) -> tuple[int, str]:
    """
    Approximate tokens for a single message dictionary.
    Includes role, content, and tool calls.
    """
    text_chunks: List[str] = []
    
    if "role" in message and message["role"]:
        text_chunks.append(str(message["role"]))
    if "content" in message and message["content"]:
        text_chunks.append(str(message["content"]))
    if "tool_calls" in message and message["tool_calls"]:
        text_chunks.append(json.dumps(message["tool_calls"], sort_keys=True))
    if "tool_call_id" in message and message["tool_call_id"]:
        text_chunks.append(str(message["tool_call_id"]))

    combined = "\n".join(text_chunks)
    count, method = count_text_tokens(combined)
    # Add modest per-message formatting overhead (typical 3-4 tokens)
    return count + 3, method


def count_context_tokens(messages: List[Dict[str, Any]]) -> tuple[int, str]:
    """
    Compute total estimated tokens across all messages in conversation history.
    """
    total = 0
    method_used = TOKEN_COUNT_METHOD_TIKTOKEN
    for msg in messages:
        c, m = count_message_tokens(msg)
        total += c
        if m != TOKEN_COUNT_METHOD_TIKTOKEN:
            method_used = m
    return total, method_used


def is_compaction_triggered(
    current_tokens: int,
    budget_tokens: int,
    threshold_ratio: float = 0.75,
) -> bool:
    """
    Determine whether context size has reached or exceeded compaction threshold.
    """
    threshold_limit = int(budget_tokens * threshold_ratio)
    return current_tokens >= threshold_limit


def calculate_token_variance(estimated_tokens: int, reported_usage_tokens: int) -> float:
    """
    Calculate absolute relative variance between estimated tokens and provider-reported usage.
    Returns ratio (e.g. 0.05 for 5% variance).
    """
    if reported_usage_tokens <= 0:
        return 0.0
    return abs(estimated_tokens - reported_usage_tokens) / reported_usage_tokens
