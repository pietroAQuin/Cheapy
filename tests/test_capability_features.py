"""Step features — §6.2: leakage guard and the offline/online parity the module exists
to guarantee (same function, called on a Trajectory built either way).
"""
from __future__ import annotations

import inspect

from cheapy.capability.features import FEATURE_NAMES, extract_step_features
from cheapy.preprocessing.trajectory_analyzer import analyze


class TestNoServedModelLeakage:
    """docs/FULL_REPORT.md §4's five leakage features must not be readable from the feature vector."""

    def test_source_never_reads_served_model(self):
        source = inspect.getsource(extract_step_features)
        assert "served_model" not in source

    def test_source_never_reads_raw_tool_identity(self):
        # tool_name is folded into aggregate counts (distinct ratio, repeat runs) only —
        # no per-tool-name branch should appear in the feature function itself.
        source = inspect.getsource(extract_step_features)
        assert "tool_name ==" not in source
        assert '"bash"' not in source
        assert '"apply_patch"' not in source
        assert '"shell_command"' not in source

    def test_identical_feature_vector_for_the_same_content_in_either_encoding(
        self, claude_trajectory_line, gpt_trajectory_line
    ):
        # Not a claim the two fixture trajectories are equivalent content-wise (they
        # aren't) — a smoke check that extraction runs cleanly on both raw encodings
        # trajectory_analyzer.normalize() has to unify, with no encoding-specific branch
        # inside extract_step_features itself.
        for line in (claude_trajectory_line, gpt_trajectory_line):
            trajectory = analyze(line, id=0)
            features = extract_step_features(trajectory)
            assert set(features) == set(FEATURE_NAMES)
            assert all(isinstance(v, float) for v in features.values())


class TestFeatureValues:
    def test_feature_vector_has_the_declared_names_only(self, claude_trajectory_line):
        trajectory = analyze(claude_trajectory_line, id=0)
        features = extract_step_features(trajectory)
        assert set(features) == set(FEATURE_NAMES)

    def test_toolset_size_and_environment_come_from_the_trajectory(self, claude_trajectory_line):
        trajectory = analyze(claude_trajectory_line, id=0)
        features = extract_step_features(trajectory)
        assert features["toolset_size"] == float(trajectory.toolset_size)
        assert features["is_teams"] == 0.0  # claude_trajectory_line is Slack-flavored

    def test_teams_trajectory_sets_is_teams(self, gpt_trajectory_line):
        trajectory = analyze(gpt_trajectory_line, id=0)
        features = extract_step_features(trajectory)
        assert features["is_teams"] == 1.0

    def test_empty_trajectory_does_not_divide_by_zero(self):
        trajectory = analyze({"model": "claude-opus-5", "input": [], "tools": []}, id=0)
        features = extract_step_features(trajectory)
        assert features["assistant_share"] == 0.0
        assert features["tool_output_share"] == 0.0
        assert features["distinct_tool_ratio"] == 1.0
        assert features["max_repeat_run"] == 0.0
