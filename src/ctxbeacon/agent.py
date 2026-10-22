"""Agent execution loop for CtxBeacon."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ctxbeacon.ledger import Ledger
from ctxbeacon.providers.base import BaseProvider, ProviderResponse, ToolCall
from ctxbeacon.schema import (
    compute_context_hash,
    compute_request_hash,
    create_event,
    generate_run_id,
)
from ctxbeacon.tools import DEMO_TOOLS_SCHEMA, ToolExecutionResult, WorkspaceSandbox


@dataclass
class AgentMessage:
    """A single conversation turn with a stable identifier."""
    id: str
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id, "role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d


class AgentLoop:
    """
    Standalone harness executing a controlled agent loop with ledger instrumentation.
    """

    def __init__(
        self,
        provider: BaseProvider,
        ledger: Ledger,
        workspace_dir: str | Path,
        system_instruction: str = "You are a disciplined coding assistant. Follow all user instructions and constraints precisely.",
        run_id: Optional[str] = None,
        max_turns: int = 10,
        max_tool_calls_per_turn: int = 5,
    ) -> None:
        self.provider = provider
        self.ledger = ledger
        self.sandbox = WorkspaceSandbox(workspace_dir)
        self.system_instruction = system_instruction
        self.run_id = run_id or generate_run_id()
        self.max_turns = max_turns
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        
        self.messages: List[AgentMessage] = []
        self._msg_counter = 0
        self.turn = 0
        self.total_tool_calls = 0

    def next_message_id(self) -> str:
        self._msg_counter += 1
        return f"msg-{self._msg_counter:03d}"

    def get_context_dicts(self) -> List[Dict[str, Any]]:
        """Return serialized list of all current messages in context."""
        return [m.to_dict() for m in self.messages]

    def add_message(
        self,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        msg_id: Optional[str] = None,
    ) -> AgentMessage:
        mid = msg_id or self.next_message_id()
        msg = AgentMessage(
            id=mid,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        self.messages.append(msg)
        return msg

    def run(self, task: str) -> Dict[str, Any]:
        """
        Execute the agent loop for the specified task until completion,
        turn limit, or unrecoverable error.
        """
        # 1. Record run_started
        self.ledger.append(
            create_event(
                event_type="run_started",
                run_id=self.run_id,
                turn=None,
                task=task,
                provider=self.provider.name,
                model=self.provider.model,
                simulation=self.provider.is_simulation,
            )
        )

        # 2. Add system and initial user message
        self.add_message("system", self.system_instruction)
        self.add_message("user", task)

        termination_reason = "completed"

        while self.turn < self.max_turns:
            context_dicts = self.get_context_dicts()
            context_hash = compute_context_hash(context_dicts)
            request_payload = {
                "model": self.provider.model,
                "messages": context_dicts,
                "tools": DEMO_TOOLS_SCHEMA,
            }
            request_hash = compute_request_hash(request_payload)

            # Record request_sent
            self.ledger.append(
                create_event(
                    event_type="request_sent",
                    run_id=self.run_id,
                    turn=self.turn,
                    provider=self.provider.name,
                    model=self.provider.model,
                    request_hash=request_hash,
                    context_hash=context_hash,
                    simulation=self.provider.is_simulation,
                )
            )

            # Dispatch to provider
            try:
                response: ProviderResponse = self.provider.generate(
                    messages=context_dicts,
                    tools=DEMO_TOOLS_SCHEMA,
                    system_instruction=self.system_instruction,
                )
            except Exception as e:
                self.ledger.append(
                    create_event(
                        event_type="provider_response",
                        run_id=self.run_id,
                        turn=self.turn,
                        provider=self.provider.name,
                        model=self.provider.model,
                        status_code=500,
                        latency_ms=0.0,
                        reason=f"Provider exception: {e}",
                        status="error",
                        simulation=self.provider.is_simulation,
                    )
                )
                termination_reason = f"provider_error: {e}"
                break

            # Record provider_response
            resp_usage_dict = response.usage.to_dict() if response.usage else None
            self.ledger.append(
                create_event(
                    event_type="provider_response",
                    run_id=self.run_id,
                    turn=self.turn,
                    provider=self.provider.name,
                    model=self.provider.model,
                    status_code=response.status_code,
                    latency_ms=response.latency_ms,
                    provider_usage_tokens=resp_usage_dict,
                    has_tool_calls=response.has_tool_calls,
                    tool_call_count=len(response.tool_calls),
                    retry_after_seconds=response.retry_after_seconds,
                    reason=response.error,
                    status="ok" if response.status_code < 400 else "error",
                    simulation=self.provider.is_simulation,
                )
            )

            if response.status_code >= 400:
                termination_reason = f"provider_status_{response.status_code}"
                break

            # If model returned tool calls, execute them
            if response.has_tool_calls:
                serialized_calls = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in response.tool_calls
                ]
                self.add_message(
                    role="assistant",
                    content=response.content,
                    tool_calls=serialized_calls,
                )

                tool_count_in_turn = 0
                for tc in response.tool_calls:
                    if tool_count_in_turn >= self.max_tool_calls_per_turn:
                        break
                    tool_count_in_turn += 1
                    self.total_tool_calls += 1

                    # Record tool_call_started
                    self.ledger.append(
                        create_event(
                            event_type="tool_call_started",
                            run_id=self.run_id,
                            turn=self.turn,
                            tool_name=tc.name,
                            tool_call_id=tc.id,
                            arguments=tc.arguments,
                            simulation=self.provider.is_simulation,
                        )
                    )

                    # Execute in sandbox
                    exec_res: ToolExecutionResult = self.sandbox.execute_tool(tc.name, tc.arguments)

                    # Record request_blocked if security rejected
                    if exec_res.status == "rejected":
                        self.ledger.append(
                            create_event(
                                event_type="request_blocked",
                                run_id=self.run_id,
                                turn=self.turn,
                                reason=exec_res.error or "Sandbox security boundary violation",
                                target_path=exec_res.target_path,
                                tool_name=tc.name,
                                simulation=self.provider.is_simulation,
                                status="blocked",
                            )
                        )

                    # Record tool_call_completed
                    self.ledger.append(
                        create_event(
                            event_type="tool_call_completed",
                            run_id=self.run_id,
                            turn=self.turn,
                            tool_name=tc.name,
                            tool_call_id=tc.id,
                            execution_status=exec_res.status,
                            output_preview=exec_res.output[:200],
                            target_path=exec_res.target_path,
                            error=exec_res.error,
                            status="ok" if exec_res.status == "success" else "error",
                            simulation=self.provider.is_simulation,
                        )
                    )

                    # Append tool result message to conversation history
                    self.add_message(
                        role="tool",
                        content=exec_res.output,
                        tool_call_id=tc.id,
                    )

                self.turn += 1
                continue

            # Model responded with plain text content without tool calls -> turn done
            self.add_message(role="assistant", content=response.content)
            self.turn += 1
            termination_reason = "completed"
            break

        if self.turn >= self.max_turns and termination_reason == "completed":
            termination_reason = "max_turns_reached"

        # Record run_completed
        self.ledger.append(
            create_event(
                event_type="run_completed",
                run_id=self.run_id,
                turn=self.turn,
                total_turns=self.turn,
                total_tool_calls=self.total_tool_calls,
                termination_reason=termination_reason,
                status="completed" if termination_reason == "completed" else "stopped",
                simulation=self.provider.is_simulation,
            )
        )

        return {
            "run_id": self.run_id,
            "turns": self.turn,
            "total_tool_calls": self.total_tool_calls,
            "termination_reason": termination_reason,
            "messages": self.get_context_dicts(),
        }
