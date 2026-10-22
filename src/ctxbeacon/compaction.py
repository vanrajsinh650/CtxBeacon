"""Context compaction engine and snapshot management for CtxBeacon."""

from __future__ import annotations

import json
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ctxbeacon.ledger import Ledger
from ctxbeacon.providers.base import BaseProvider, ProviderResponse
from ctxbeacon.schema import (
    compute_context_hash,
    create_event,
    sanitize_payload_for_hashing,
)
from ctxbeacon.tokens import count_context_tokens, is_compaction_triggered

SNAPSHOT_WARNING = (
    "WARNING: Snapshots store conversation context and tool outputs on disk under ledger/snapshots/. "
    "Snapshots may contain private prompts, code, or sensitive data. Do not commit or publish snapshots."
)


@dataclass
class CompactionResult:
    """Outcome of a compaction transformation."""
    compacted: bool
    messages: List[Dict[str, Any]]
    context_before_tokens: int
    context_after_tokens: int
    retained_ids: List[str]
    dropped_ids: List[str]
    summary_text: Optional[str] = None
    summary_ref: Optional[str] = None
    snapshot_ref: Optional[str] = None
    context_hash_before: Optional[str] = None
    context_hash_after: Optional[str] = None


class Compactor:
    """
    Identifies older turns exceeding context budget, requests a summary,
    and replaces them with an audited summary message while preserving recent turns.
    """

    def __init__(
        self,
        provider: BaseProvider,
        ledger: Ledger,
        snapshots_dir: Optional[Path] = None,
        enable_snapshots: bool = False,
        keep_recent_turns: int = 3,
    ) -> None:
        self.provider = provider
        self.ledger = ledger
        self.enable_snapshots = enable_snapshots
        self.keep_recent_turns = max(1, keep_recent_turns)
        self.snapshots_dir = snapshots_dir or Path("ledger/snapshots")
        
        if self.enable_snapshots:
            warnings.warn(SNAPSHOT_WARNING, UserWarning)
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def compact_context(
        self,
        messages: List[Dict[str, Any]],
        run_id: str,
        turn: int,
        context_budget: int = 4096,
        compaction_threshold: float = 0.75,
        injected_summary: Optional[str] = None,
    ) -> CompactionResult:
        """
        Check context token budget and execute compaction if threshold reached.
        """
        current_tokens, method_before = count_context_tokens(messages)
        if not is_compaction_triggered(current_tokens, context_budget, compaction_threshold):
            return CompactionResult(
                compacted=False,
                messages=messages,
                context_before_tokens=current_tokens,
                context_after_tokens=current_tokens,
                retained_ids=[m["id"] for m in messages],
                dropped_ids=[],
            )

        # Separate system messages, older messages, and recent messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system_msgs = [m for m in messages if m.get("role") != "system"]

        # Keep recent turns (at least keep_recent_turns messages)
        if len(non_system_msgs) <= self.keep_recent_turns:
            # Cannot safely compact without dropping recent turns
            return CompactionResult(
                compacted=False,
                messages=messages,
                context_before_tokens=current_tokens,
                context_after_tokens=current_tokens,
                retained_ids=[m["id"] for m in messages],
                dropped_ids=[],
            )

        split_index = len(non_system_msgs) - self.keep_recent_turns
        older_msgs = non_system_msgs[:split_index]
        recent_msgs = non_system_msgs[split_index:]

        dropped_ids = [m["id"] for m in older_msgs]
        retained_ids = [m["id"] for m in system_msgs] + [m["id"] for m in recent_msgs]

        context_hash_before = compute_context_hash(messages)

        # 1. Record compaction_started
        self.ledger.append(
            create_event(
                event_type="compaction_started",
                run_id=run_id,
                turn=turn,
                context_before_tokens=current_tokens,
                token_count_method=method_before,
                total_messages_before=len(messages),
                summarize_target_count=len(older_msgs),
                simulation=self.provider.is_simulation,
            )
        )

        # 2. Generate summary
        summary_ref = f"sum-{uuid.uuid4().hex[:8]}"
        if injected_summary is not None:
            summary_text = injected_summary
        else:
            summary_instruction = (
                "Produce a concise summary of the conversation turns above. "
                "You MUST strictly preserve all user constraints, explicit instructions, "
                "approvals, decisions, unresolved requirements, and key tool results verbatim."
            )
            resp: ProviderResponse = self.provider.summarize(older_msgs, system_instruction=summary_instruction)
            summary_text = resp.content or "No summary content generated."

        # 3. Handle snapshot storage if enabled
        snapshot_ref = None
        if self.enable_snapshots:
            snap_id = uuid.uuid4().hex[:12]
            snapshot_ref = f"snap-{snap_id}"
            snapshot_path = self.snapshots_dir / f"{snapshot_ref}.json"
            snapshot_data = {
                "snapshot_ref": snapshot_ref,
                "run_id": run_id,
                "turn": turn,
                "original_messages": sanitize_payload_for_hashing(messages),
                "older_messages_dropped": sanitize_payload_for_hashing(older_msgs),
                "summary": summary_text,
                "retained_ids": retained_ids,
                "dropped_ids": dropped_ids,
            }
            try:
                snapshot_path.write_text(json.dumps(snapshot_data, indent=2), encoding="utf-8")
            except OSError:
                snapshot_ref = None

        # 4. Construct effective new context
        summary_message = {
            "id": summary_ref,
            "role": "system",
            "content": f"[Compacted Summary of Older Turns]:\n{summary_text}",
        }
        retained_ids.append(summary_ref)

        new_messages = list(system_msgs) + [summary_message] + list(recent_msgs)
        after_tokens, method_after = count_context_tokens(new_messages)
        context_hash_after = compute_context_hash(new_messages)

        # 5. Record compaction_completed
        self.ledger.append(
            create_event(
                event_type="compaction_completed",
                run_id=run_id,
                turn=turn,
                context_before_tokens=current_tokens,
                context_after_tokens=after_tokens,
                token_count_method=method_after,
                retained_ids=retained_ids,
                dropped_ids=dropped_ids,
                summary_ref=summary_ref,
                snapshot_ref=snapshot_ref,
                context_hash_before=context_hash_before,
                context_hash_after=context_hash_after,
                summary_preview=summary_text[:200],
                simulation=self.provider.is_simulation,
            )
        )

        return CompactionResult(
            compacted=True,
            messages=new_messages,
            context_before_tokens=current_tokens,
            context_after_tokens=after_tokens,
            retained_ids=retained_ids,
            dropped_ids=dropped_ids,
            summary_text=summary_text,
            summary_ref=summary_ref,
            snapshot_ref=snapshot_ref,
            context_hash_before=context_hash_before,
            context_hash_after=context_hash_after,
        )
