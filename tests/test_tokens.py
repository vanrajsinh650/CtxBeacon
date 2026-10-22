"""Tests for CtxBeacon token estimation, context budgets, and compaction triggers."""

import os
from ctxbeacon.config import CtxBeaconConfig
from ctxbeacon.tokens import (
    TOKEN_COUNT_METHOD_TIKTOKEN,
    calculate_token_variance,
    count_context_tokens,
    count_message_tokens,
    count_text_tokens,
    is_compaction_triggered,
)


def test_token_counting_uses_tiktoken():
    count, method = count_text_tokens("Hello world! This is a test of CtxBeacon token counting.")
    assert count > 0
    assert method == TOKEN_COUNT_METHOD_TIKTOKEN


def test_context_tokens_calculation():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain context compaction."},
        {"role": "assistant", "content": "Compaction summarizes older turns while keeping recent turns."},
    ]
    total, method = count_context_tokens(messages)
    assert total > 20
    assert method == TOKEN_COUNT_METHOD_TIKTOKEN


def test_compaction_trigger_boundaries():
    budget = 1000
    threshold = 0.75  # 750 tokens

    # Strictly below threshold
    assert not is_compaction_triggered(current_tokens=749, budget_tokens=budget, threshold_ratio=threshold)
    # Exactly at threshold
    assert is_compaction_triggered(current_tokens=750, budget_tokens=budget, threshold_ratio=threshold)
    # Above threshold
    assert is_compaction_triggered(current_tokens=800, budget_tokens=budget, threshold_ratio=threshold)


def test_token_variance_calculation():
    # 100 estimated, 95 reported
    variance = calculate_token_variance(estimated_tokens=100, reported_usage_tokens=95)
    assert round(variance, 4) == round(5 / 95, 4)

    # Identical
    assert calculate_token_variance(estimated_tokens=100, reported_usage_tokens=100) == 0.0


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("CTXBEACON_CONTEXT_BUDGET_TOKENS", "2048")
    monkeypatch.setenv("CTXBEACON_COMPACTION_THRESHOLD", "0.80")
    monkeypatch.setenv("CTXBEACON_KEEP_RECENT_TURNS", "2")
    monkeypatch.setenv("CTXBEACON_MAX_TURNS", "8")

    cfg = CtxBeaconConfig.from_env()
    assert cfg.context_budget_tokens == 2048
    assert cfg.compaction_threshold == 0.80
    assert cfg.threshold_tokens == 1638  # int(2048 * 0.8)
    assert cfg.keep_recent_turns == 2
    assert cfg.max_turns == 8
