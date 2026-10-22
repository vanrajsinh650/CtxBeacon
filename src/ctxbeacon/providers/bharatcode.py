"""BharatCode official model provider adapter for Qwen-3.8-27b."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from ctxbeacon.providers.base import (
    BaseProvider,
    ProviderResponse,
    ProviderUsage,
    ToolCall,
)

DEFAULT_BASE_URL = "https://bharatcode.ai/api/model/v1"
DEFAULT_MODEL = "qwen-3.8-27b"


class BharatCodeProvider(BaseProvider):
    """
    Adapter for BharatCode API (https://bharatcode.ai/build).
    Interacts with hosted open-weight models like Qwen-3.8-27b via POST /responses.
    
    Credentials are kept strictly external (passed via env or constructor)
    and never leaked in logs, hashes, or exceptions.
    """

    name: str = "bharatcode"
    is_simulation: bool = False

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 45.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("BHARATCODE_API_KEY", "")
        self.base_url = (base_url or os.environ.get("BHARATCODE_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or os.environ.get("BHARATCODE_MODEL", DEFAULT_MODEL)
        self.timeout_seconds = timeout_seconds
        self._client = http_client

    def _get_client(self) -> httpx.Client:
        if self._client:
            return self._client
        return httpx.Client(timeout=self.timeout_seconds)

    def _build_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise ValueError(
                "BharatCode API key is required. Set BHARATCODE_API_KEY environment variable "
                "or pass api_key to BharatCodeProvider."
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _parse_response_body(
        self,
        data: Dict[str, Any],
        status_code: int,
        latency_ms: float,
        retry_after: Optional[float] = None,
    ) -> ProviderResponse:
        """Parse BharatCode /responses output adhering strictly to documented schema."""
        content_text: Optional[str] = None
        tool_calls: List[ToolCall] = []

        # Parse text content
        if "output_text" in data and isinstance(data["output_text"], str):
            content_text = data["output_text"]

        # Parse structured output array (messages and tool/function calls)
        output_items = data.get("output", [])
        if isinstance(output_items, list):
            for item in output_items:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "function_call":
                    tc_id = item.get("call_id") or item.get("id") or f"call_{len(tool_calls)+1}"
                    name = item.get("name", "")
                    raw_args = item.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args)
                        except Exception:
                            parsed_args = {"raw": raw_args}
                    elif isinstance(raw_args, dict):
                        parsed_args = raw_args
                    else:
                        parsed_args = {}
                    tool_calls.append(ToolCall(id=tc_id, name=name, arguments=parsed_args))
                elif item_type == "message" and content_text is None:
                    # Fallback if output_text was absent
                    msg_content = item.get("content", [])
                    if isinstance(msg_content, list):
                        parts = [
                            c.get("text", "")
                            for c in msg_content
                            if isinstance(c, dict) and c.get("type") in ("output_text", "text")
                        ]
                        content_text = "".join(parts)

        # Parse usage strictly if present
        usage_obj: Optional[ProviderUsage] = None
        if "usage" in data and isinstance(data["usage"], dict):
            raw_usage = data["usage"]
            prompt_t = raw_usage.get("prompt_tokens", 0)
            comp_t = raw_usage.get("completion_tokens", 0)
            tot_t = raw_usage.get("total_tokens", prompt_t + comp_t)
            usage_obj = ProviderUsage(
                prompt_tokens=prompt_t,
                completion_tokens=comp_t,
                total_tokens=tot_t,
            )

        return ProviderResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage=usage_obj,
            status_code=status_code,
            latency_ms=latency_ms,
            raw_response=data,
            retry_after_seconds=retry_after,
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
    ) -> ProviderResponse:
        url = f"{self.base_url}/responses"
        headers = self._build_headers()

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "store": False,
        }
        if system_instruction:
            payload["instructions"] = system_instruction
        if tools:
            payload["tools"] = tools

        start_time = time.monotonic()
        client = self._get_client()

        try:
            resp = client.post(url, headers=headers, json=payload)
            latency_ms = (time.monotonic() - start_time) * 1000.0

            # Extract Retry-After header if present on 429
            retry_after = None
            if resp.status_code == 429 and "retry-after" in resp.headers:
                try:
                    retry_after = float(resp.headers["retry-after"])
                except ValueError:
                    retry_after = 2.0

            if resp.status_code >= 400:
                error_msg = f"BharatCode API returned HTTP {resp.status_code}: {resp.text[:200]}"
                return ProviderResponse(
                    content=None,
                    status_code=resp.status_code,
                    latency_ms=latency_ms,
                    error=error_msg,
                    retry_after_seconds=retry_after,
                )

            data = resp.json()
            return self._parse_response_body(
                data=data,
                status_code=resp.status_code,
                latency_ms=latency_ms,
            )

        except httpx.RequestError as e:
            latency_ms = (time.monotonic() - start_time) * 1000.0
            return ProviderResponse(
                content=None,
                status_code=500,
                latency_ms=latency_ms,
                error=f"HTTP network error connecting to BharatCode: {type(e).__name__}",
            )

    def summarize(
        self,
        messages: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
    ) -> ProviderResponse:
        default_inst = (
            "Produce a concise summary of the older conversation turns. "
            "You MUST preserve all user constraints, explicit instructions, approvals, "
            "decisions, and key tool results verbatim."
        )
        instruction = system_instruction or default_inst

        # Format older messages for summarization prompt
        serialized_context = json.dumps(messages, indent=2)
        prompt_input = (
            f"Here are the older conversation turns to compact:\n"
            f"```json\n{serialized_context}\n```\n"
            "Summarize the turns above while strictly preserving all constraints."
        )

        return self.generate(
            messages=[{"role": "user", "content": prompt_input}],
            system_instruction=instruction,
        )
