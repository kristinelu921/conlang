"""Keep numerical policy tests restricted to explicitly enabled remote runs."""

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "neural: executes neural inference or training; enable on authorized remote compute")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip neural execution unless the remote-test environment explicitly enables it."""
    if os.environ.get("ISLAND_ALLOW_NEURAL_TESTS") == "1":
        return
    restricted = pytest.mark.skip(reason="Set ISLAND_ALLOW_NEURAL_TESTS=1 on authorized remote compute")
    for item in items:
        if "neural" in item.keywords:
            item.add_marker(restricted)
