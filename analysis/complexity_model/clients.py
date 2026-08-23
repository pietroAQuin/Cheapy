"""Provider clients — spec §2.1, §2.2.

Thin wrappers: canonical request in, raw response body out. No LangChain (§2.2) — this is
one `prefix -> API call -> parse response` round, not an agent loop, and LangChain's
tool-calling wrappers reshape the payload silently, which is exactly the confound §2
warns about.

SDKs are imported lazily, inside the client classes, so importing this module — and
therefore `capability_model.py`, which does not import it — never requires `anthropic` or
`openai` to be installed. Install them from `requirements-elicit.txt` to actually run
elicitation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from analysis.complexity_model.canonical import CanonicalRequest, render_anthropic, render_openai
from analysis.complexity_model.parser import ANTHROPIC_MODELS

#: Per-request wall-clock cap. The SDK's own default is 10 minutes, which turns a dropped
#: connection into a wedged worker thread — observed in practice: a brief network outage
#: stalled a 6-way-concurrent run for 10 minutes with zero throughput. Large prefixes on
#: reasoning models still finish inside this.
REQUEST_TIMEOUT_S = 180.0

#: Output tokens for the elicitation call. One action, not a conversation — kept small
#: since responses are a single next-step (message or a few tool calls), and every token
#: here is billed at every one of the 7 models on every step.
MAX_OUTPUT_TOKENS = 1024

#: Per-family context ceilings, for the pre-flight guard in `query`. Real published
#: figures — see `pre_processing/model_list.py`'s `_MODEL_CONTEXT_WINDOWS` for sources.
CONTEXT_WINDOW = {
    "claude-opus-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-fable-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "gpt-5.6-sol": 1_050_000,
    "gpt-5.6-terra": 1_050_000,
    "gpt-5.6-luna": 1_050_000,
}


class RetryableError(Exception):
    """Raised by a client wrapper for a transient failure worth retrying — rate limits,
    5xx, timeouts. Anything else (bad request, auth) propagates immediately: retrying a
    malformed request just burns the same budget three more times for the same failure."""


@dataclass(frozen=True)
class QueryResult:
    """What `query` returns: the raw provider body plus the bookkeeping §2.1 asks be
    recorded alongside every response."""

    raw: dict[str, Any]
    effective_temperature: float | None  # None means the parameter was omitted (§2.1)


class Client(Protocol):
    def query(self, model: str, request: CanonicalRequest, temperature: float | None) -> QueryResult: ...
    def probe_temperature(self, model: str) -> bool:
        """True if `model` accepts an explicit `temperature=0`. One cheap call."""
        ...


_RETRYABLE = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(RetryableError),
)


class AnthropicClient:
    """Wraps the Messages API for the four Anthropic candidates."""

    def __init__(self, api_key: str | None = None) -> None:
        import anthropic  # local import — see module docstring

        self._sdk = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    @_RETRYABLE
    def _call(self, model: str, payload: dict, temperature: float | None) -> dict:
        kwargs: dict[str, Any] = dict(model=model, max_tokens=MAX_OUTPUT_TOKENS, **payload)
        if temperature is not None:
            kwargs["temperature"] = temperature
        try:
            response = self._client.messages.create(**kwargs)
        except self._sdk.APIStatusError as exc:
            if exc.status_code in (429, 500, 502, 503, 529):
                raise RetryableError(str(exc)) from exc
            raise
        return response.model_dump()

    def query(self, model: str, request: CanonicalRequest, temperature: float | None) -> QueryResult:
        payload = render_anthropic(request)
        raw = self._call(model, payload, temperature)
        return QueryResult(raw=raw, effective_temperature=temperature)

    def probe_temperature(self, model: str) -> bool:
        """§2.1: `claude-opus-4-8` (and possibly siblings) return a 400 if `temperature`
        is set to any non-default value. Verified per model with one live call rather than
        assumed, since the constraint has moved between releases."""
        probe = CanonicalRequest(system="", items=(), tools=())
        try:
            self._call(model, render_anthropic(probe) | {"messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]}
            ]}, 0.0)
            return True
        except self._sdk.BadRequestError:
            return False


class OpenAIClient:
    """Wraps the Responses API for the three gpt-5.6 candidates.

    Accepts **multiple** keys and rotates across them on auth failure (401/403 — a
    revoked or wrong key) or rate/quota exhaustion (429 — could be per-key throttling or
    an exhausted quota; the API doesn't reliably distinguish the two, so both are treated
    the same: try the next key). Only after every key has failed on the same call does
    the failure become a `RetryableError` for `_RETRYABLE`'s backoff-and-retry, so a
    transient 429 on key 1 doesn't burn the whole batch — it just tries key 2 immediately.
    """

    def __init__(self, api_keys: list[str | None] | None = None) -> None:
        import openai  # local import — see module docstring

        self._sdk = openai
        keys = api_keys or [None]
        # max_retries=0: the SDK retries internally by default, which would multiply with
        # this module's own retry and the key rotation (3 x 4 x 3 attempts for one call).
        # Retry policy lives in exactly one place — `_RETRYABLE` plus `_attempt_all_keys`.
        self._clients = [
            openai.OpenAI(api_key=k, timeout=REQUEST_TIMEOUT_S, max_retries=0) for k in keys
        ]
        self._active = 0

    def _rotate(self) -> None:
        self._active = (self._active + 1) % len(self._clients)

    def _attempt_all_keys(self, model: str, payload: dict, temperature: float | None) -> dict:
        """One pass over every key, in rotation order, with no backoff between them —
        a bad or exhausted key should fail over to the next one immediately, not wait.
        Separated from `_call` so this loop is unit-testable without `_RETRYABLE`'s real
        sleep-and-retry wrapped around it.
        """
        kwargs: dict[str, Any] = dict(model=model, max_output_tokens=MAX_OUTPUT_TOKENS, **payload)
        if temperature is not None:
            kwargs["temperature"] = temperature

        last_exc: Exception | None = None
        for _ in range(len(self._clients)):
            client = self._clients[self._active]
            try:
                response = client.responses.create(**kwargs)
                return response.model_dump()
            except (self._sdk.APITimeoutError, self._sdk.APIConnectionError) as exc:
                # The network, not the key or the request — rotating would not help, and
                # recording it as a terminal failure would permanently poison this
                # (model, step_id) with an outage that has nothing to do with the model.
                raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
            except self._sdk.APIStatusError as exc:
                last_exc = exc
                if exc.status_code in (401, 403, 429):
                    self._rotate()  # bad/exhausted key — try the next one immediately
                    continue
                if exc.status_code in (500, 502, 503):
                    raise RetryableError(str(exc)) from exc
                raise
        # every key failed on this pass — let the outer retry back off, then loop again
        raise RetryableError(f"all {len(self._clients)} OpenAI keys failed: {last_exc}") from last_exc

    @_RETRYABLE
    def _call(self, model: str, payload: dict, temperature: float | None) -> dict:
        return self._attempt_all_keys(model, payload, temperature)

    def query(self, model: str, request: CanonicalRequest, temperature: float | None) -> QueryResult:
        payload = render_openai(request)
        raw = self._call(model, payload, temperature)
        return QueryResult(raw=raw, effective_temperature=temperature)

    def probe_temperature(self, model: str) -> bool:
        probe = CanonicalRequest(system="", items=(), tools=())
        payload = render_openai(probe) | {
            "input": [{"role": "user", "content": "hi"}]
        }
        try:
            self._call(model, payload, 0.0)
            return True
        except self._sdk.BadRequestError:
            return False


def client_for(
    model: str,
    anthropic_key: str | None = None,
    openai_keys: list[str | None] | None = None,
) -> Client:
    if model in ANTHROPIC_MODELS:
        return AnthropicClient(api_key=anthropic_key)
    return OpenAIClient(api_keys=openai_keys)
