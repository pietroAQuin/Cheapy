"""ModelLLM: field defaults and validation."""
import pytest
from pydantic import ValidationError

from data_models.model_llm import ModelLLM

PRICE_FIELDS = dict(
    input_price_per_1m=3.0,
    cached_input_price_per_1m=0.3,
    output_price_per_1m=15.0,
    cached_output_price_per_1m=0.0,
)


def test_scores_default_to_none_not_zero():
    model = ModelLLM(name="claude-sonnet-5", family="claude-sonnet", context_window_size=1_000_000, **PRICE_FIELDS)
    assert model.performance_score is None
    assert model.price_score is None
    assert model.final_score is None


def test_required_fields_are_enforced():
    with pytest.raises(ValidationError):
        ModelLLM(name="claude-sonnet-5", family="claude-sonnet")  # missing context window + prices


def test_context_window_rejects_non_numeric_string():
    with pytest.raises(ValidationError):
        ModelLLM(
            name="claude-sonnet-5",
            family="claude-sonnet",
            context_window_size="not-a-number",
            **PRICE_FIELDS,
        )
