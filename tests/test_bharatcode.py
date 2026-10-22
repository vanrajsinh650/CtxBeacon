"""Unit tests for BharatCode provider adapter using mocked HTTP responses."""

import json
import httpx
import pytest

from ctxbeacon.providers.bharatcode import BharatCodeProvider


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("BHARATCODE_API_KEY", raising=False)
    provider = BharatCodeProvider(api_key=None)
    with pytest.raises(ValueError, match="BharatCode API key is required"):
        provider.generate(messages=[{"role": "user", "content": "hello"}])


def test_mocked_http_successful_text_response():
    fake_response_data = {
        "id": "resp_test123",
        "object": "response",
        "model": "qwen-3.8-27b",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello from BharatCode!"}],
            }
        ],
        "output_text": "Hello from BharatCode!",
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 6,
            "total_tokens": 21,
        },
    }

    def fake_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/model/v1/responses"
        assert request.headers["Authorization"] == "Bearer test_secret_key"
        body = json.loads(request.content)
        assert body["model"] == "qwen-3.8-27b"
        assert body["store"] is False
        return httpx.Response(200, json=fake_response_data)

    client = httpx.Client(transport=httpx.MockTransport(fake_handler))
    provider = BharatCodeProvider(api_key="test_secret_key", http_client=client)

    resp = provider.generate(messages=[{"role": "user", "content": "Say hello"}])

    assert resp.status_code == 200
    assert resp.content == "Hello from BharatCode!"
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 15
    assert resp.usage.completion_tokens == 6
    assert resp.usage.total_tokens == 21
    assert not resp.has_tool_calls


def test_mocked_http_function_call_response():
    fake_response_data = {
        "id": "resp_tool_1",
        "object": "response",
        "model": "qwen-3.8-27b",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_abc_456",
                "name": "write_file",
                "arguments": json.dumps({"path": "output.txt", "content": "Done"}),
            }
        ],
        "output_text": "",
        "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
    }

    def fake_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fake_response_data)

    client = httpx.Client(transport=httpx.MockTransport(fake_handler))
    provider = BharatCodeProvider(api_key="key_123", http_client=client)

    resp = provider.generate(
        messages=[{"role": "user", "content": "Write output.txt"}],
        tools=[{"type": "function", "function": {"name": "write_file"}}],
    )

    assert resp.status_code == 200
    assert resp.has_tool_calls is True
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.id == "call_abc_456"
    assert tc.name == "write_file"
    assert tc.arguments == {"path": "output.txt", "content": "Done"}


def test_mocked_http_rate_limit_429_with_retry_after():
    def fake_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "3"},
            json={"error": "Too Many Requests", "message": "Rate limit exceeded"},
        )

    client = httpx.Client(transport=httpx.MockTransport(fake_handler))
    provider = BharatCodeProvider(api_key="key_123", http_client=client)

    resp = provider.generate(messages=[{"role": "user", "content": "hello"}])
    assert resp.status_code == 429
    assert resp.retry_after_seconds == 3.0
    assert "BharatCode API returned HTTP 429" in resp.error


def test_credentials_not_leaked_in_error_message():
    def fake_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Invalid API key"})

    client = httpx.Client(transport=httpx.MockTransport(fake_handler))
    secret_key = "super_confidential_secret_999"
    provider = BharatCodeProvider(api_key=secret_key, http_client=client)

    resp = provider.generate(messages=[{"role": "user", "content": "hi"}])
    assert resp.status_code == 401
    assert secret_key not in resp.error
