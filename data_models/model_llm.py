from pydantic import BaseModel, Field


class ModelLLM(BaseModel):
    """Describes a single LLM model/variant and everything needed to price and compare it."""

    name: str
    family: str
    tier: int = Field(description="Relative capability/cost tier within its family; lower = cheaper/smaller.")

    cache_window_size: int = Field(description="Max number of tokens that can be served from cache (e.g. context window eligible for prompt caching).")

    input_price_per_1m: float = Field(description="USD per 1M uncached input tokens.")
    cached_input_price_per_1m: float = Field(description="USD per 1M cached input tokens.")
    output_price_per_1m: float = Field(description="USD per 1M uncached output tokens.")
    cached_output_price_per_1m: float = Field(description="USD per 1M cached output tokens.")
