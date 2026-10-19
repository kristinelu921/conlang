"""Tests for immutable conlang source snapshots."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_staging_keeps_source_and_excludes_data_and_secrets(tmp_path: Path) -> None:
    """A run snapshot should contain code/config but no data, logs, or credentials."""
    source = tmp_path / "source"
    staging_root = tmp_path / "staging"
    (source / "configs").mkdir(parents=True)
    (source / "datasets").mkdir()
    (source / "wandb").mkdir()
    (source / "train.py").write_text("print('source')\n", encoding="utf-8")
    (source / "configs" / "conlang-jax-v5e8-gate1.yaml").write_text("steps: 1\n", encoding="utf-8")
    (source / "datasets" / "train.jsonl").write_text("private data\n", encoding="utf-8")
    (source / "wandb" / "debug.log").write_text("runtime log\n", encoding="utf-8")
    (source / ".git").write_text("gitdir: /outside/worktree\n", encoding="utf-8")
    (source / ".netrc").write_text("machine example.invalid\n", encoding="utf-8")
    environment = {
        **os.environ,
        "ELF_STAGING_SOURCE": str(source),
        "ELF_STAGING_ROOT": str(staging_root),
        "ELF_STAGING_CONFIG": "configs/conlang-jax-v5e8-gate1.yaml",
    }

    completed = subprocess.run(
        ["bash", "tpu_scripts/run_staging.sh", "test-run"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    snapshot = Path(completed.stdout.strip().removeprefix("STAGEDIR="))
    assert (snapshot / "train.py").is_file()
    assert (snapshot / "configs" / "conlang-jax-v5e8-gate1.yaml").is_file()
    assert not (snapshot / "datasets").exists()
    assert not (snapshot / "wandb").exists()
    assert not (snapshot / ".git").exists()
    assert not (snapshot / ".netrc").exists()
