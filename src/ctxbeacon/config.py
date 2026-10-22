"""Configuration settings for CtxBeacon context budgets and thresholds."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class CtxBeaconConfig:
    """
    Runtime configuration for CtxBeacon.
    
    Defaults:
    - context_budget_tokens: 4096 (approximate working budget for compaction triggers)
    - compaction_threshold: 0.75 (compaction triggers when context reaches 75% of budget)
    - keep_recent_turns: 3 (recent messages to preserve verbatim)
    - max_turns: 10 (hard limit on agent turns)
    """
    context_budget_tokens: int = 4096
    compaction_threshold: float = 0.75
    keep_recent_turns: int = 3
    max_turns: int = 10

    @classmethod
    def from_env(cls) -> "CtxBeaconConfig":
        budget = int(os.environ.get("CTXBEACON_CONTEXT_BUDGET_TOKENS", "4096"))
        threshold = float(os.environ.get("CTXBEACON_COMPACTION_THRESHOLD", "0.75"))
        recent = int(os.environ.get("CTXBEACON_KEEP_RECENT_TURNS", "3"))
        turns = int(os.environ.get("CTXBEACON_MAX_TURNS", "10"))
        return cls(
            context_budget_tokens=budget,
            compaction_threshold=threshold,
            keep_recent_turns=recent,
            max_turns=turns,
        )

    @property
    def threshold_tokens(self) -> int:
        """The absolute token threshold at which compaction triggers."""
        return int(self.context_budget_tokens * self.compaction_threshold)
