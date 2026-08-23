"""OpenAI multi-key rotation — clients.py's OpenAIClient.

Constructs real `openai.APIStatusError` instances (status code is all that matters to
the rotation logic) rather than mocking the exception type, so the test tracks the SDK's
actual exception hierarchy.
"""
from __future__ import annotations

import httpx
import openai
import pytest

from research.capability_fitting.clients import OpenAIClient, RetryableError


def _status_error(code: int) -> openai.APIStatusError:
    response = httpx.Response(code, request=httpx.Request("POST", "http://example.invalid"))
    return openai.APIStatusError(f"status {code}", response=response, body=None)


class _FakeResponses:
    """Stands in for `client.responses` — `create` raises according to a per-key script,
    or returns a canned success payload."""

    def __init__(self, key_label: str, script: dict[str, list[int | None]]):
        self._key_label = key_label
        self._script = script  # key_label -> list of status codes to raise; None = succeed

    def create(self, **kwargs):
        codes = self._script[self._key_label]
        code = codes.pop(0)
        if code is None:
            class _Resp:
                def model_dump(self_inner):
                    return {"output": [], "_served_by": self._key_label}

            return _Resp()
        raise _status_error(code)


class _FakeOpenAI:
    def __init__(self, api_key: str, script: dict[str, list[int | None]]):
        self.responses = _FakeResponses(api_key, script)


def _make_client(monkeypatch, keys: list[str], script: dict[str, list[int | None]]) -> OpenAIClient:
    monkeypatch.setattr(
        openai, "OpenAI",
        lambda api_key=None, **kwargs: _FakeOpenAI(api_key, script),
    )
    return OpenAIClient(api_keys=keys)


class TestKeyRotation:
    def test_rotates_to_next_key_on_401(self, monkeypatch):
        script = {"key1": [401], "key2": [None]}
        client = _make_client(monkeypatch, ["key1", "key2"], script)
        raw = client._call("gpt-5.6-luna", {"instructions": "", "input": [], "tools": []}, 0.0)
        assert raw["_served_by"] == "key2"

    def test_rotates_to_next_key_on_429(self, monkeypatch):
        script = {"key1": [429], "key2": [None]}
        client = _make_client(monkeypatch, ["key1", "key2"], script)
        raw = client._call("gpt-5.6-luna", {"instructions": "", "input": [], "tools": []}, 0.0)
        assert raw["_served_by"] == "key2"

    def test_all_keys_exhausted_raises_retryable(self, monkeypatch):
        script = {"key1": [429], "key2": [429]}
        client = _make_client(monkeypatch, ["key1", "key2"], script)
        with pytest.raises(RetryableError):
            client._attempt_all_keys("gpt-5.6-luna", {"instructions": "", "input": [], "tools": []}, 0.0)

    def test_500_is_retryable_without_rotating(self, monkeypatch):
        # a 500 is the provider's fault, not the key's — should surface as RetryableError
        # (so _RETRYABLE backs off and retries) rather than silently jumping keys.
        script = {"key1": [500]}
        client = _make_client(monkeypatch, ["key1"], script)
        with pytest.raises(RetryableError):
            client._attempt_all_keys("gpt-5.6-luna", {"instructions": "", "input": [], "tools": []}, 0.0)

    def test_non_retryable_error_propagates_immediately(self, monkeypatch):
        script = {"key1": [400]}
        client = _make_client(monkeypatch, ["key1"], script)
        with pytest.raises(openai.APIStatusError):
            client._call("gpt-5.6-luna", {"instructions": "", "input": [], "tools": []}, 0.0)

    def test_single_key_still_works(self, monkeypatch):
        script = {"only": [None]}
        client = _make_client(monkeypatch, ["only"], script)
        raw = client._call("gpt-5.6-luna", {"instructions": "", "input": [], "tools": []}, 0.0)
        assert raw["_served_by"] == "only"


class TestNetworkFailuresAreTransient:
    """A dropped connection is not a property of the request — it must not be banked as a
    terminal failure, and it must not trigger key rotation."""

    def test_connection_error_is_retryable(self, monkeypatch):
        class _Boom:
            def create(self, **kwargs):
                raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))

        class _C:
            def __init__(self, *a, **k):
                self.responses = _Boom()

        monkeypatch.setattr(openai, "OpenAI", lambda api_key=None, **k: _C())
        client = OpenAIClient(api_keys=["k1", "k2"])
        with pytest.raises(RetryableError):
            client._attempt_all_keys("gpt-5.6-luna", {"instructions": "", "input": [], "tools": []}, None)

    def test_sdk_internal_retries_are_disabled(self, monkeypatch):
        seen = {}

        class _C:
            def __init__(self, *a, **k):
                seen.update(k)
                self.responses = None

        monkeypatch.setattr(openai, "OpenAI", lambda api_key=None, **k: _C(**k))
        OpenAIClient(api_keys=["k1"])
        # Retry policy must live in exactly one place, not multiply across layers.
        assert seen.get("max_retries") == 0
        assert seen.get("timeout") is not None
