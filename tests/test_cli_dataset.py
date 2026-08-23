"""src/cheapy/cli.py: resolving the dataset argument.

The CLI takes a dataset as either a single `.jsonl` file or a directory of them, and the
CSV is written beside whatever it was given — so a reproduction run can point at an export
anywhere on disk without copying it into `data/` first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cheapy.cli import resolve_dataset


def test_a_file_routes_only_that_file(tmp_path):
    chunk = tmp_path / "export.jsonl"
    chunk.write_text("", encoding="utf-8")
    (tmp_path / "other.jsonl").write_text("", encoding="utf-8")

    chunks, output_dir = resolve_dataset(chunk)
    assert chunks == [chunk]
    assert output_dir == tmp_path


def test_a_directory_routes_every_chunk_in_sorted_order(tmp_path):
    # Sorted, because ids come from one counter across all chunks: a run has to be
    # reproducible, and glob order is not.
    for name in ("b.jsonl", "a.jsonl", "notes.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")

    chunks, output_dir = resolve_dataset(tmp_path)
    assert [c.name for c in chunks] == ["a.jsonl", "b.jsonl"]
    assert output_dir == tmp_path


def test_an_empty_directory_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="no \\*.jsonl chunks"):
        resolve_dataset(tmp_path)


def test_a_missing_path_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="dataset not found"):
        resolve_dataset(tmp_path / "absent.jsonl")
