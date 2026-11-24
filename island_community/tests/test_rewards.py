"""Rule-only checks for survival penalties, health budgets, and legacy rewards."""

import json
from typing import Any

import pytest

from island_community.config import load_config, validate_config
from island_community.world import Action, Shipment, World


@pytest.fixture
def configuration() -> dict[str, Any]:
    """Disable unrelated random changes while preserving the default reward settings."""
    settings = load_config()
    settings["allocation"]["allocations_per_minute"] = 0
    settings["events"]["probability_per_turn"] = 0
    settings["reproduction"]["enabled"] = False
    settings["survival"]["unhappiness_damage"] = 0
    return settings


def test_founders_and_newborns_start_with_three_hundred_health(configuration: dict[str, Any]) -> None:
    """Apply the increased health budget consistently to founders and descendants."""
    world = World(configuration)
    assert all(island.health == 300 for island in world.islands.values())
    world.islands[0].next_birth_turn = 0
    child = world._try_birth(0)
    assert child is not None
    assert world.islands[child].health == 300


def test_empty_food_and_health_penalties_repeat_and_keep_resource_accounting(configuration: dict[str, Any]) -> None:
    """Charge both configured terms each empty turn while retaining inventory-only accounting."""
    world = World(configuration)
    island = world.islands[0]
    island.inventory["grain"] = 0
    total = 0.0
    for _ in range(2):
        world.start_turn()
        outcome = world.finish_turn({})
        components = outcome.reward_components[0]
        assert components["food_available"] == 0
        assert components["health_lost"] == 7
        assert components["empty_food_penalty"] == -5
        assert components["health_loss_penalty"] == -3.5
        assert outcome.rewards[0] == pytest.approx(world._resource_reward(0) - 8.5)
        assert island.previous_reward == world._resource_reward(0)
        total += outcome.rewards[0]
    assert island.lifetime_reward == pytest.approx(total)
    assert island.health == 286
    assert json.loads(json.dumps(world.state_dict())) == json.loads(json.dumps(World.from_state_dict(configuration, world.state_dict()).state_dict()))


def test_meals_and_saved_nutrition_count_until_the_last_fed_turn(configuration: dict[str, Any]) -> None:
    """Recognize meal nutrition and distinguish an empty pantry from missing a meal."""
    world = World(configuration)
    island = world.islands[0]
    island.inventory.update(grain=0, meal=1)
    world.start_turn()
    first = world.finish_turn({}).reward_components[0]
    assert first["food_available"] == 1
    assert first["empty_food_penalty"] == first["health_loss_penalty"] == 0
    world.start_turn()
    second = world.finish_turn({}).reward_components[0]
    assert second["food_available"] == 0
    assert second["empty_food_penalty"] == -5
    assert second["health_loss_penalty"] == 0
    assert island.health == 300


@pytest.mark.parametrize("item,necessity", [("grain", "food"), ("water", "water")])
def test_reserved_and_in_transit_goods_cannot_prevent_empty_penalty(configuration: dict[str, Any], item: str, necessity: str) -> None:
    """Exclude promised and undelivered goods from each usable necessity balance."""
    world = World(configuration)
    world.islands[0].inventory[item] = 1
    world.shipments[99] = Shipment(99, 99, 0, 1, 0, {item: 10}, {}, 5)
    world.start_turn()
    outcome = world.finish_turn({0: Action("propose_trade", {"target": 1, "offer": {item: 1}, "request": {"fiber": 1}})})
    assert world.islands[0].inventory[item] == 1
    assert outcome.reward_components[0][f"{necessity}_available"] == 0
    assert outcome.reward_components[0][f"empty_{necessity}_penalty"] == -5
    assert outcome.reward_components[0]["resource_reward"] >= 11 * configuration["items"][item]["reward_value"] * 0.01


@pytest.mark.parametrize("damage_source", ["water", "disease", "aging", "unhappiness"])
def test_health_penalty_uses_actual_net_loss_from_any_cause(configuration: dict[str, Any], damage_source: str) -> None:
    """Measure actual net health change independently of the physical damage source."""
    if damage_source == "aging":
        configuration["aging"]["start_turn"] = 1
    configuration["survival"]["unhappiness_damage"] = 1
    world = World(configuration)
    island = world.islands[0]
    if damage_source == "water":
        island.inventory["water"] = 0
    elif damage_source == "disease":
        island.disease_until = 4
    elif damage_source == "unhappiness":
        island.unhappiness = 0.5
    previous_health = island.health
    world.start_turn()
    components = world.finish_turn({}).reward_components[0]
    assert components["health_lost"] > 0
    assert components["empty_food_penalty"] == 0
    assert components["health_loss_penalty"] == pytest.approx(-0.5 * (previous_health - island.health))


def test_healing_does_not_become_a_negative_health_loss_penalty(configuration: dict[str, Any]) -> None:
    """Allow healing to offset damage without rewarding a positive health change."""
    world = World(configuration)
    island = world.islands[0]
    island.health = 200
    island.inventory["herbs"] = 1
    island.disease_until = 4
    world.start_turn()
    components = world.finish_turn({0: Action("use_item", {"item": "herbs"})}).reward_components[0]
    assert island.health == 201
    assert components["health_lost"] == components["health_loss_penalty"] == 0


@pytest.mark.parametrize("version,expected_reward", [(4, -6), (5, -11)])
def test_terminal_reward_preserves_penalties_and_caps_loss_at_remaining_health(configuration: dict[str, Any], version: int, expected_reward: float) -> None:
    """Retain final survival costs in both the learner transition and deceased lifetime score."""
    configuration["schema_version"] = version
    del configuration["trade_initialization"]
    if version == 4:
        del configuration["reward_shaping"]["empty_water_penalty"]
    world = World(configuration)
    island = world.islands[0]
    island.health = 2
    island.inventory.update(grain=0, water=0)
    world.start_turn()
    outcome = world.finish_turn({})
    assert 0 in outcome.deaths
    assert outcome.reward_components[0]["resource_reward"] == 0
    assert outcome.reward_components[0]["health_lost"] == 2
    assert outcome.rewards[0] == expected_reward
    assert world.deceased[0]["lifetime_reward"] == expected_reward
    assert next(event for event in world.events if event["type"] == "death")["lifetime_reward"] == expected_reward
    if version == 4:
        assert "empty_water_penalty" not in outcome.reward_components[0]


def test_birth_depletion_penalizes_parent_without_penalizing_a_newborn(configuration: dict[str, Any]) -> None:
    """Measure the parent's food after inheritance and leave newborns out of this turn's learning."""
    configuration["reproduction"]["enabled"] = True
    world = World(configuration)
    world.islands[0].next_birth_turn = 1
    world.islands[0].inventory.update(grain=9, water=9)
    world.start_turn()
    outcome = world.finish_turn({})
    assert outcome.births
    assert outcome.reward_components[0]["empty_food_penalty"] == -5
    assert outcome.reward_components[0]["empty_water_penalty"] == -5
    for identifier in outcome.births:
        assert identifier not in outcome.reward_components
        assert world.islands[identifier].lifetime_reward == 0


def test_penalties_do_not_feed_back_into_unhappiness_or_health_damage(configuration: dict[str, Any]) -> None:
    """Keep physical outcomes and resource-based unhappiness identical when penalty magnitudes change."""
    configuration["survival"]["unhappiness_damage"] = 1
    shaped = World(configuration)
    configuration["reward_shaping"] = {setting: 0 for setting in configuration["reward_shaping"]}
    unshaped = World(configuration)
    for world in (shaped, unshaped):
        world.islands[0].inventory.update(grain=0, water=0)
    for _ in range(3):
        for world in (shaped, unshaped):
            world.start_turn()
            world.finish_turn({})
        assert shaped.islands[0].unhappiness == unshaped.islands[0].unhappiness
        assert shaped.islands[0].previous_reward == unshaped.islands[0].previous_reward
        assert shaped.islands[0].health == unshaped.islands[0].health


@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_configurations_keep_inventory_rewards_and_zero_terminal_reward(configuration: dict[str, Any], version: int) -> None:
    """Ensure older rule versions do not acquire survival reward shaping on restore."""
    configuration["schema_version"] = version
    del configuration["trade_initialization"]
    del configuration["reward_shaping"]
    configuration["rewards"]["lifetime_score"] = "sum_resource_rewards"
    if version < 3:
        del configuration["exploration"]
    if version < 2:
        del configuration["item_utility"]
    world = World(configuration)
    world.islands[0].inventory["grain"] = 0
    world.start_turn()
    outcome = world.finish_turn({})
    assert outcome.rewards[0] == world._resource_reward(0)
    assert outcome.reward_components == {}
    world.islands[0].health = 1
    world.start_turn()
    assert world.finish_turn({}).rewards[0] == 0


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), True, "5"])
@pytest.mark.parametrize("setting", ["empty_food_penalty", "empty_water_penalty", "health_loss_penalty_per_point"])
def test_invalid_reward_penalty_is_rejected(configuration: dict[str, Any], setting: str, invalid: Any) -> None:
    """Require finite numeric penalty magnitudes without accepting booleans as numbers."""
    configuration["reward_shaping"][setting] = invalid
    with pytest.raises(ValueError):
        validate_config(configuration)


def test_empty_water_penalty_repeats_independently_of_food_and_health_loss(configuration: dict[str, Any]) -> None:
    """Penalize the last available sip and subsequent empty turns without requiring food shortage."""
    world = World(configuration)
    world.islands[0].inventory["water"] = 1
    for expected_loss in (0, 7, 7):
        world.start_turn()
        result = world.finish_turn({})
        parts = result.reward_components[0]
        assert parts["water_available"] == 0
        assert parts["empty_water_penalty"] == -5
        assert parts["empty_food_penalty"] == 0
        assert parts["health_lost"] == expected_loss
        assert result.rewards[0] == pytest.approx(world._resource_reward(0) - 5 - 0.5 * expected_loss)


def test_both_empty_penalties_add_to_health_loss_and_stop_when_replenished(configuration: dict[str, Any]) -> None:
    """Charge both shortages independently and clear their penalties after enough supplies arrive."""
    world = World(configuration)
    island = world.islands[0]
    island.inventory.update(grain=0, water=0)
    world.start_turn()
    result = world.finish_turn({})
    parts = result.reward_components[0]
    assert parts["empty_food_penalty"] == parts["empty_water_penalty"] == -5
    assert parts["health_lost"] == 14
    assert result.rewards[0] == pytest.approx(world._resource_reward(0) - 17)
    island.inventory.update(grain=2, water=2)
    world.start_turn()
    parts = world.finish_turn({}).reward_components[0]
    assert parts["food_available"] == parts["water_available"] == 1
    assert parts["empty_food_penalty"] == parts["empty_water_penalty"] == parts["health_loss_penalty"] == 0


def test_legacy_version_four_water_shortage_keeps_only_its_original_health_penalty(configuration: dict[str, Any]) -> None:
    """Preserve the pre-water-penalty reward and component schema across checkpoint restoration."""
    configuration["schema_version"] = 4
    del configuration["trade_initialization"]
    del configuration["reward_shaping"]["empty_water_penalty"]
    world = World(configuration)
    world.islands[0].inventory["water"] = 0
    restored = World.from_state_dict(configuration, json.loads(json.dumps(world.state_dict())))
    for current in (world, restored):
        current.start_turn()
        result = current.finish_turn({})
        assert result.rewards[0] == pytest.approx(current._resource_reward(0) - 3.5)
        assert "empty_water_penalty" not in result.reward_components[0]
    assert world.state_dict() == restored.state_dict()


@pytest.mark.parametrize("version", [4, 5])
def test_water_penalty_configuration_is_explicitly_versioned(configuration: dict[str, Any], version: int) -> None:
    """Reject missing version-five settings and accidental additions to historical version-four rules."""
    configuration["schema_version"] = version
    del configuration["trade_initialization"]
    if version == 5:
        del configuration["reward_shaping"]["empty_water_penalty"]
    with pytest.raises(ValueError, match="reward_shaping"):
        validate_config(configuration)
