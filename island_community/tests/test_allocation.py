"""Rule checks for resource distributions, without policy execution."""

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pytest

from island_community.config import load_config, proportional_resource_counts, validate_config
from island_community.world import World


def test_legacy_global_allocation_retains_recorded_seeded_gifts() -> None:
    """Match the original seed-42 first-turn gift ledger before changing allocation modes."""
    world = World(load_config())
    world.turn = 1
    world._allocate()
    gifts = [(event["island"], event["item"], event["delta"]) for event in world.events
             if event["type"] == "inventory" and event["reason"] == "world_allocation"]
    assert gifts == [(4, "iron", 16), (0, "water", 16), (4, "iron", 8), (0, "water", 4), (2, "timber", 8)]


@pytest.fixture
def configuration() -> dict[str, Any]:
    """Load the complete fifty-founder scenario without starting a simulation."""
    return load_config(Path(__file__).parents[1] / "scenarios" / "population_50.json")


def test_proportional_fifty_founders_cover_every_resource_and_keep_endowments(configuration: dict[str, Any]) -> None:
    """Preserve exact resource quotas, per-agent starting necessities, and seed reproducibility."""
    world = World(configuration)
    counts = Counter(island.specialization[0] for island in world.islands.values())
    assert counts == {item: 13 if item in {"water", "grain"} else 3 for item in configuration["allocation"]["resource_weights"]}
    assert counts == proportional_resource_counts(50, configuration["allocation"]["resource_weights"])
    assert all(len(island.specialization) == 1 for island in world.islands.values())
    assert all(island.inventory["grain"] >= 8 and island.inventory["water"] >= 8 for island in world.islands.values())
    assert world.state_dict() == World(configuration).state_dict()
    configuration["seed"] += 1
    alternative = World(configuration)
    assert [island.specialization for island in world.islands.values()] != [island.specialization for island in alternative.islands.values()]
    assert Counter(island.specialization[0] for island in alternative.islands.values()) == counts


@pytest.mark.parametrize("population", [20, 50, 100])
def test_one_gift_per_living_island_scales_with_population(configuration: dict[str, Any], population: int) -> None:
    """Deliver exactly one independently drawn gift from each island's assigned resource type."""
    configuration["initialization"]["population"] = population
    configuration["allocation"]["quantity_weights"] = [1] + [0] * 10
    world = World(configuration)
    world.turn = 1
    world._allocate()
    gifts = [event for event in world.events if event["type"] == "inventory" and event["reason"] == "world_allocation"]
    assert Counter(event["island"] for event in gifts) == {identifier: 1 for identifier in world.islands}
    assert all(event["delta"] == 1 and event["item"] in world.islands[event["island"]].specialization for event in gifts)


def test_allocation_respects_birth_death_cadence_and_income_effects(configuration: dict[str, Any]) -> None:
    """Include newborn producers on later allocations and never gift dead islands."""
    configuration["allocation"].update(turns_per_minute=2, allocations_per_minute=2,
                                       quantity_weights=[0, 0, 0, 1] + [0] * 7)
    world = World(configuration)
    world._die(1)
    parent = world.islands[0]
    parent.next_birth_turn = 0
    child_id = world._try_birth(0)
    assert child_id is not None
    parent.income_reduced_until = 3
    world.turn = 1
    start = len(world.events)
    world._allocate()
    assert len(world.events) == start
    world.turn = 2
    world._allocate()
    gifts = [event for event in world.events[start:] if event["type"] == "inventory" and event["reason"] == "world_allocation"]
    assert Counter(event["island"] for event in gifts) == {identifier: 2 for identifier in world.islands}
    assert all(event["delta"] == (4 if event["island"] == 0 else 8) for event in gifts)
    assert any(event["island"] == child_id for event in gifts)
    assert all(event["island"] != 1 for event in gifts)


def test_proportional_allocation_restores_exact_random_stream(configuration: dict[str, Any]) -> None:
    """Continue the same producer and gift sequence after JSON checkpoint restoration."""
    world = World(configuration)
    world.turn = 1
    world._allocate()
    restored = World.from_state_dict(configuration, json.loads(json.dumps(world.state_dict())))
    for current in (world, restored):
        current.turn = 2
        current._allocate()
    assert world.state_dict() == restored.state_dict()


@pytest.mark.parametrize("mutation", [
    lambda settings: settings["allocation"].update(scope="unknown"),
    lambda settings: settings["allocation"].update(recipient_rule="uniform_eligible"),
    lambda settings: settings["allocation"].pop("scope"),
    lambda settings: settings["allocation"]["resource_weights"].update(water=0),
    lambda settings: settings["initialization"].update(population=10),
    lambda settings: settings.update(schema_version=6),
])
def test_invalid_population_allocation_settings_are_rejected(configuration: dict[str, Any], mutation: Any) -> None:
    """Reject ambiguous scope, unproductive resource assignments, and unversioned behavior changes."""
    mutation(configuration)
    with pytest.raises(ValueError):
        validate_config(configuration)
