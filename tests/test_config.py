"""src/cheapy/config.py: cheapy.yaml loading and the flag > file > default precedence.

Every test writes its own config into `tmp_path`. The repo's real `cheapy.yaml` is never
read here — a test that depended on it would start failing the moment someone dialed the
router to a different operating point, which is exactly what that file is for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cheapy.config import (
    DEFAULT_BETA,
    DEFAULT_CACHE_HIT_RATE,
    DEFAULT_W_COST,
    DEFAULT_W_PERFORMANCE,
    SimulationConfig,
    load_config,
)


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cheapy.yaml"
    path.write_text(body, encoding="utf-8")
    return path


class TestLoad:
    def test_reads_every_key(self, tmp_path):
        path = write_config(
            tmp_path,
            "W_COST: 0.7\nW_PERFORMANCE: 0.3\nBETA: 1.5\nCACHE_HIT_RATE: 0.4\nVERBOSE: true\n",
        )
        config = load_config(path)
        assert (config.w_cost, config.w_performance) == (0.7, 0.3)
        assert (config.beta, config.cache_hit_rate) == (1.5, 0.4)
        assert config.verbose is True

    def test_absent_keys_fall_back_to_defaults(self, tmp_path):
        config = load_config(write_config(tmp_path, "BETA: 0.0\n"))
        assert config.beta == 0.0
        assert config.w_cost == DEFAULT_W_COST
        assert config.w_performance == DEFAULT_W_PERFORMANCE
        assert config.cache_hit_rate == DEFAULT_CACHE_HIT_RATE
        assert config.verbose is False

    def test_empty_file_is_all_defaults(self, tmp_path):
        config = load_config(write_config(tmp_path, "# nothing set\n"))
        assert config == SimulationConfig()

    def test_missing_default_file_is_not_an_error(self, monkeypatch, tmp_path):
        # No path given and no cheapy.yaml on disk: the built-in defaults are the shipped
        # operating point, so a fresh checkout without the file still runs.
        import cheapy.config as config_module

        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "absent.yaml")
        assert load_config() == SimulationConfig()

    def test_missing_explicit_file_is_an_error(self, tmp_path):
        # Asking for a named file that is not there is a typo, not a choice.
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "absent.yaml")

    def test_unknown_key_is_rejected(self, tmp_path):
        # A silently ignored W_PRICE: would read as the router disregarding your settings.
        path = write_config(tmp_path, "W_PRICE: 0.7\n")
        with pytest.raises(ValueError, match="W_PRICE"):
            load_config(path)

    def test_non_numeric_value_names_the_key(self, tmp_path):
        with pytest.raises(ValueError, match="BETA"):
            load_config(write_config(tmp_path, "BETA: three\n"))

    def test_verbose_accepts_a_hand_written_string(self, tmp_path):
        assert load_config(write_config(tmp_path, 'VERBOSE: "yes"\n')).verbose is True
        assert load_config(write_config(tmp_path, 'VERBOSE: "no"\n')).verbose is False


class TestResolve:
    def test_a_flag_beats_the_file(self):
        config = SimulationConfig(w_cost=0.2, w_performance=0.8).resolve(w_cost=0.9)
        assert config.w_cost == 0.9
        assert config.w_performance == 0.8

    def test_none_means_flag_not_given(self):
        # argparse defaults are None precisely so an unset flag cannot outrank the file.
        base = SimulationConfig(beta=1.0, verbose=True)
        assert base.resolve(beta=None, verbose=None) == base

    def test_false_is_an_override_not_an_absence(self):
        # `--quiet` on a config with VERBOSE: true must actually win.
        assert SimulationConfig(verbose=True).resolve(verbose=False).verbose is False

    def test_zero_is_an_override_not_an_absence(self):
        assert SimulationConfig(beta=DEFAULT_BETA).resolve(beta=0.0).beta == 0.0

    def test_unknown_setting_is_rejected(self):
        with pytest.raises(TypeError, match="w_price"):
            SimulationConfig().resolve(w_price=0.5)


class TestValidate:
    def test_defaults_are_valid(self):
        assert SimulationConfig().validate() == SimulationConfig()

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"w_cost": -0.1}, "must be >= 0"),
            ({"w_cost": 0.0, "w_performance": 0.0}, "must be > 0"),
            ({"beta": -1.0}, "BETA"),
            ({"cache_hit_rate": 1.5}, "CACHE_HIT_RATE"),
            ({"cache_hit_rate": -0.5}, "CACHE_HIT_RATE"),
        ],
    )
    def test_rejects_values_the_stages_cannot_interpret(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            SimulationConfig(**kwargs).validate()

    def test_a_one_sided_weighting_is_allowed(self):
        # w=1/0 is a legitimate operating point (pure cost); only 0/0 is undefined.
        assert SimulationConfig(w_cost=1.0, w_performance=0.0).validate().w_performance == 0.0
