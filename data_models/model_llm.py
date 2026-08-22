from pydantic import BaseModel, Field


class ModelLLM(BaseModel):
    """One scored entry per anonymized model id seen in the export.

    Performance and price are kept as two separate scores rather than one
    blended number, so the router can trade them off explicitly instead of
    ranking models on a single opaque figure.
    """

    name: str  # anonymized model id as logged in the export, e.g. "claude-opus-5"
    family: str  # shared prefix across generations, e.g. "claude-opus"

    performance_score: float | None = Field(
        default=None,
        description="0-1, higher = more capable. Set by router_models/performance_model.py "
        "as a function of (Trajectory, ModelLLM) — None means 'not scored yet for this "
        "trajectory', not 'worst possible'; don't treat it as 0 in any trade-off math.",
    )

    price_score: float | None = Field(
        default=None,
        description="0-1, higher = cheaper. Placeholder: real cache-aware pricing "
        "isn't wired up yet (router_models/price_model.py). None means 'not "
        "computed yet', not 'free' — don't treat it as 0 in any trade-off math.",
    )

    final_score: float | None = Field(
        default=None,
        description="Weighted combination w_price·price_score + w_perf·performance_score, "
        "set by router_models/model.py once both inputs exist (see README §5). None until "
        "that stage runs — never fall back to 0.",
    )

    context_window_size: int = Field(
        description="Max number of tokens (input + output) the model can hold in a single request.")

    input_price_per_1m: float = Field(description="USD per 1M uncached input tokens.")

    cached_input_price_per_1m: float = Field(description="USD per 1M cached input tokens.")

    output_price_per_1m: float = Field(description="USD per 1M uncached output tokens.")

    cached_output_price_per_1m: float = Field(description="USD per 1M cached output tokens.")
