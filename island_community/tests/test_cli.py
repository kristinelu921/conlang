"""Command routing checks that do not execute simulations or neural policies."""

from pathlib import Path
import sys
from types import ModuleType

import pytest

from island_community.__main__ import main


def test_default_output_belongs_to_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Installed commands should write runs under the caller's directory, not site-packages."""
    captured: list[Path] = []
    runner = ModuleType("island_community.runner")

    def capture_run(output: Path, *arguments: object) -> Path:
        captured.append(output)
        return output

    runner.run = capture_run
    monkeypatch.setitem(sys.modules, "island_community.runner", runner)
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--turns", "1"]) == 0
    assert len(captured) == 1
    assert captured[0].is_relative_to(tmp_path / "experiment_log")
    assert not captured[0].exists()
