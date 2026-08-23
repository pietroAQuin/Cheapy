"""Where `ModelLLM` and `Trajectory` objects are created.

`model_list.build_model_list()` returns the candidate pool; `trajectory_analyzer.analyze()`
turns one JSONL line into one `Trajectory`. Nothing downstream parses the export.
"""
