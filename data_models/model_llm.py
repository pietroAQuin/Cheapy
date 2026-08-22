from pydantic import BaseModel, Field


class ModelLLM(BaseModel):
    """One scored entry per anonymized model id seen in the export.

    Capability and price are kept as two separate scores rather than one
    blended number, so the router can trade them off explicitly instead of
    ranking models on a single opaque figure.
    """

    name: str  # anonymized model id as logged in the export, e.g. "claude-opus-5"
    family: str  # shared prefix across generations, e.g. "claude-opus"

    capability_score: float = Field(
        description="0-1, higher = more capable. See pre_processing/model_list.py "
        "for how this is currently seeded — a naming-tier prior, not a measurement."
    )

    price_score: float | None = Field(
        default=None,
        description="0-1, higher = cheaper. Placeholder: real cache-aware pricing "
        "isn't wired up yet (router_models/price_model.py). None means 'not "
        "computed yet', not 'free' — don't treat it as 0 in any trade-off math.",
    )
    tier: int = Field(description="Relative capability/cost tier within its family; lower = cheaper/smaller.")

    context_window_size: int = Field(
        description="Max number of tokens (input + output) the model can hold in a single request.")

    input_price_per_1m: float = Field(description="USD per 1M uncached input tokens.")

    cached_input_price_per_1m: float = Field(description="USD per 1M cached input tokens.")

    output_price_per_1m: float = Field(description="USD per 1M uncached output tokens.")

    cached_output_price_per_1m: float = Field(description="USD per 1M cached output tokens.")
