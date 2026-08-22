#!/usr/bin/env python3
"""Build the scored list of models the router chooses between.

price_score is intentionally left out: real cache-aware pricing lives in
router_models/price_model.py, which isn't implemented yet. Every ModelLLM
here comes back with price_score at its None default (see data_models/model_llm.py) —
a later pass fills it in, this module doesn't guess at it.

capability_score also isn't a measurement — there is no ground truth for
anonymized model ids. It's seeded from a naming-tier prior (mirrors the
assumption already used in the old baseline router: opus > sonnet > fable;
gpt-5.6 variants have no published order, so they share one tier) and is
marked "assumed" via capability_provenance until judge-model scoring or
revealed-preference analysis (recovered outputs / trace complexity, see
AGENTS.md's "earlier outputs are recoverable" note) replaces it.

The (name -> family) map below was captured from one scan of the only export
chunk available so far (9 ids, 4 families). It's static rather than scanned
from export/ on every run so this module has no dependency on the dataset
being present. Re-derive it (or move to dynamic discovery) if a later chunk
introduces new ids.
"""
from data_models.model_llm import ModelLLM

_MODEL_FAMILIES: dict[str, str] = {
    "claude-opus-5": "claude-opus",
    "claude-opus-4-8": "claude-opus",
    "claude-opus-4-6": "claude-opus",
    "claude-sonnet-5": "claude-sonnet",
    "claude-sonnet-4-6": "claude-sonnet",
    "claude-fable-5": "claude-fable",
    "gpt-5.6-sol": "gpt-5.6",
    "gpt-5.6-terra": "gpt-5.6",
    "gpt-5.6-luna": "gpt-5.6",
}




# Real published per-1M-token rates for the 9 ids above, keyed by field name on
# ModelLLM. Sources:
#   - claude-* rows: Claude API pricing page, "Model pricing" table — Base Input
#     Tokens -> input_price_per_1m, Cache Hits & Refreshes -> cached_input_price_per_1m,
#     Output Tokens -> output_price_per_1m. That page also lists separate "5m Cache
#     Writes" / "1h Cache Writes" rates, but ModelLLM has no cache-write field, so
#     those aren't captured here.
#   - gpt-5.6-* rows: OpenAI pricing page, "Standard" tier, short-context columns.
#     That page also lists a long-context tier (roughly 2x) above an unstated token
#     threshold for gpt-5.6 — not modeled here, since we don't track per-request
#     context length in this dict. State that simplification if you quote these numbers.
#   - cached_output_price_per_1m is 0.0 everywhere, not a placeholder: neither source
#     prices a cached *output* token (only input gets cached/read-discounted).
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5":     {"input_price_per_1m": 5.00,  "cached_input_price_per_1m": 0.50, "output_price_per_1m": 25.00, "cached_output_price_per_1m": 0.0},
    "claude-opus-4-8":   {"input_price_per_1m": 5.00,  "cached_input_price_per_1m": 0.50, "output_price_per_1m": 25.00, "cached_output_price_per_1m": 0.0},
    "claude-opus-4-6":   {"input_price_per_1m": 5.00,  "cached_input_price_per_1m": 0.50, "output_price_per_1m": 25.00, "cached_output_price_per_1m": 0.0},
    "claude-sonnet-5":   {"input_price_per_1m": 2.00,  "cached_input_price_per_1m": 0.20, "output_price_per_1m": 10.00, "cached_output_price_per_1m": 0.0},
    "claude-sonnet-4-6": {"input_price_per_1m": 3.00,  "cached_input_price_per_1m": 0.30, "output_price_per_1m": 15.00, "cached_output_price_per_1m": 0.0},
    "claude-fable-5":    {"input_price_per_1m": 10.00, "cached_input_price_per_1m": 1.00, "output_price_per_1m": 50.00, "cached_output_price_per_1m": 0.0},
    "gpt-5.6-sol":       {"input_price_per_1m": 4.00,  "cached_input_price_per_1m": 0.40, "output_price_per_1m": 20.00, "cached_output_price_per_1m": 0.0},
    "gpt-5.6-terra":     {"input_price_per_1m": 2.00,  "cached_input_price_per_1m": 0.20, "output_price_per_1m": 12.00, "cached_output_price_per_1m": 0.0},
    "gpt-5.6-luna":      {"input_price_per_1m": 0.20,  "cached_input_price_per_1m": 0.02, "output_price_per_1m": 1.20,  "cached_output_price_per_1m": 0.0},
}


def price_for(name: str) -> dict[str, float]:
    """Real per-1M-token prices for a known model id — see _MODEL_PRICING for sources."""
    try:
        return _MODEL_PRICING[name]
    except KeyError:
        raise KeyError(f"no published pricing captured for {name!r} — add it to _MODEL_PRICING") from None


# Published context window sizes for the 9 ids above, in tokens. Sources:
#   - claude-* rows: Claude models overview page, "Context window" row (current models'
#     table + the legacy-models accordion). All Claude ids we track list "1M tokens".
#   - gpt-5.6-* rows: OpenAI models comparison page. All three GPT-5.6 variants list the
#     same 1,050,000-token context window; the page gives no separate figure per variant
#     (mirrors the "no published tier order" caveat already noted for gpt-5.6 pricing above).
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-5":     1_000_000,
    "claude-opus-4-8":   1_000_000,
    "claude-opus-4-6":   1_000_000,
    "claude-sonnet-5":   1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-fable-5":    1_000_000,
    "gpt-5.6-sol":       1_050_000,
    "gpt-5.6-terra":     1_050_000,
    "gpt-5.6-luna":      1_050_000,
}


def context_window_for(name: str) -> int:
    """Published context window size (tokens) for a known model id — see _MODEL_CONTEXT_WINDOWS for sources."""
    try:
        return _MODEL_CONTEXT_WINDOWS[name]
    except KeyError:
        raise KeyError(f"no published context window captured for {name!r} — add it to _MODEL_CONTEXT_WINDOWS") from None


def build_model_list() -> list[ModelLLM]:
    """Return one ModelLLM per known model id. capability_score, price_score, and tier
    are left at 0 (not yet seeded); context window and per-1M-token prices are filled
    in from the published values captured above."""
    return [
        ModelLLM(
            name=name,
            family=family,
            capability_score=0.0,
            price_score=0.0,
            tier=0,
            context_window_size=context_window_for(name),
            **price_for(name),
        )
        for name, family in _MODEL_FAMILIES.items()
    ]


def main() -> None:
    for model in build_model_list():
        print(
            f"{model.name:<20} family={model.family:<14} "
            f"capability={model.capability_score:.2f} ({model.capability_provenance})  "
            f"price=placeholder"
        )


if __name__ == "__main__":
    main()
