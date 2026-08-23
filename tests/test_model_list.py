"""src/cheapy/preprocessing/model_list.py: the static candidate-pool builder."""
import pytest

from cheapy.models.llm import ModelLLM
from cheapy.preprocessing.model_list import (
    _MODEL_FAMILIES,
    build_model_list,
    context_window_for,
    price_for,
)


def test_build_model_list_returns_one_entry_per_known_id():
    models = build_model_list()
    assert {m.name for m in models} == set(_MODEL_FAMILIES)
    assert all(isinstance(m, ModelLLM) for m in models)


def test_built_models_are_unscored():
    for model in build_model_list():
        assert model.performance_score is None
        assert model.price_score is None
        assert model.final_score is None


def test_built_models_carry_family_and_pricing():
    by_name = {m.name: m for m in build_model_list()}
    opus = by_name["claude-opus-5"]
    assert opus.family == "claude-opus"
    assert opus.context_window_size == 1_000_000
    assert opus.input_price_per_1m == 5.00
    assert opus.cached_output_price_per_1m == 0.0


def test_price_for_unknown_id_raises():
    with pytest.raises(KeyError):
        price_for("not-a-real-model")


def test_context_window_for_unknown_id_raises():
    with pytest.raises(KeyError):
        context_window_for("not-a-real-model")


def test_every_known_id_has_pricing_and_context_window():
    # Guards against the pool builder silently dropping an id that model_list.py
    # knows about but forgot to price or size — build_model_list() would KeyError
    # on it, not skip it, so this test would fail loudly rather than the pool
    # quietly shrinking.
    for name in _MODEL_FAMILIES:
        price_for(name)
        context_window_for(name)
