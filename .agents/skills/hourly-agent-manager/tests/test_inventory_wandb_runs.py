"""Tests for the W&B experiment inventory utility."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "inventory_wandb_runs.py"
SPECIFICATION = importlib.util.spec_from_file_location("inventory_wandb_runs", SCRIPT_PATH)
assert SPECIFICATION is not None and SPECIFICATION.loader is not None
inventory_wandb_runs = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(inventory_wandb_runs)


class FakeApi:
    """Record W&B API calls and return deterministic fake projects and runs."""

    def __init__(self, run: Any) -> None:
        self.run = run
        self.calls = []

    def projects(self, entity: str) -> list[Any]:
        """Return one accessible project for the requested entity."""
        return [SimpleNamespace(name="elf")]

    def runs(self, **arguments: Any) -> list[Any]:
        """Record the query and return the configured run."""
        self.calls.append(arguments)
        return [self.run]


def test_redact_sensitive_values_recursively() -> None:
    """Redact nested credentials without removing public configuration."""
    redacted = inventory_wandb_runs.redact_sensitive_values(
        {"learning_rate": 0.1, "auth": {"api_key": "secret"}, "tokens": [{"token": "secret"}]}
    )

    assert redacted == {
        "learning_rate": 0.1,
        "auth": {"api_key": "<redacted>"},
        "tokens": [{"token": "<redacted>"}],
    }


def test_collect_run_inventory_filters_by_authenticated_username() -> None:
    """Collect project runs using stable identity and state filters."""
    run = SimpleNamespace(
        entity="entity",
        project="elf",
        id="run-123",
        name="test-run",
        state="running",
        url="https://wandb.ai/entity/elf/runs/run-123",
        user="user-name",
        created_at="2026-07-17T00:00:00Z",
        lastHistoryStep=7,
        tags=["test"],
        notes=None,
        config={"seed": 42},
        summary_metrics={"train/loss": 1.5},
        metadata={"host": "worker-0"},
    )
    api = FakeApi(run)

    inventory = inventory_wandb_runs.collect_run_inventory(
        api=api,
        entity="entity",
        project_names=None,
        username="user-name",
        states=["running"],
    )

    assert inventory["projects"] == ["elf"]
    assert inventory["runs"][0]["run_id"] == "run-123"
    assert api.calls == [
        {
            "path": "entity/elf",
            "filters": {"username": "user-name", "state": {"$in": ["running"]}},
            "order": "-created_at",
            "per_page": 100,
            "lazy": False,
        }
    ]
