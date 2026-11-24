"""Check informed staple-trade features and learnable founder initialization."""

import math
from typing import Any

import pytest

from island_community.config import load_config, validate_config
from island_community.world import Action, Proposal, World


@pytest.mark.parametrize("island_id,target,item", [(0, 1, "grain"), (1, 0, "water")])
def test_staple_purchases_prioritize_public_producers(island_id: int, target: int, item: str) -> None:
    """Recognize both necessities without reading another island's private inventory."""
    world = World(load_config())
    purchase = Action("propose_trade", {"target": target, "offer": {}, "request": {item: 1}, "price": 2})
    assert world._action_features(island_id, purchase)[-1] == 1
    world.islands[target].inventory[item] = 0
    assert world._action_features(island_id, purchase)[-1] == 1
    purchase.args["target"] = 2
    assert world._action_features(island_id, purchase)[-1] == 0


def test_bias_does_not_favor_unrelated_actions_or_nonstaple_trades() -> None:
    """Keep messages, manufacture, rejection, cancellation, and unrelated barter unbiased."""
    world = World(load_config())
    world.proposals[9] = Proposal(9, 1, 0, "trade", {"grain": 4}, {"water": 4}, 0, 0, 5)
    for action in (Action("wait"), Action("manufacture", {"recipe": "cook"}),
                   Action("public_request", {"item": "grain"}), Action("reject", {"proposal": 9}),
                   Action("cancel", {"proposal": 9}),
                   Action("propose_trade", {"target": 2, "offer": {"clay": 1}, "request": {"timber": 1}})):
        assert world._action_features(0, action)[-1] == 0


@pytest.mark.parametrize("kind", ["propose_trade", "accept", "counter"])
def test_trade_directions_preserve_a_day_of_outgoing_water(kind: str) -> None:
    """Interpret each actor's actual outgoing side before assessing its reserve."""
    world = World(load_config())
    world.proposals[9] = Proposal(9, 1, 0, "trade", {"grain": 4}, {"water": 4}, 0, 0, 5)
    arguments = {"target": 1, "offer": {"water": 4}, "request": {"grain": 4}}
    if kind == "accept":
        arguments = {"proposal": 9}
    elif kind == "counter":
        arguments["proposal"] = 9
    action = Action(kind, arguments)
    world.islands[0].inventory["water"] = 5
    assert world._action_features(0, action)[-1] == 1
    world.islands[0].inventory["water"] = 4
    assert world._action_features(0, action)[-1] == 0


def test_food_reserve_and_reserved_goods_affect_priority_without_removing_actions() -> None:
    """Count saved nutrition and committed stock while leaving every legal choice available."""
    world = World(load_config())
    world.islands[1].inventory.update(grain=4, meal=0)
    trade = Action("propose_trade", {"target": 0, "offer": {"grain": 4}, "request": {"water": 4}, "price": 0})
    assert world._action_features(1, trade)[-1] == 0
    world.islands[1].food_reserve = 1
    assert world._action_features(1, trade)[-1] == 1
    world.islands[1].food_reserve = 0
    world.islands[1].inventory["grain"] = 5
    assert world._action_features(1, trade)[-1] == 1
    world.proposals[9] = Proposal(9, 1, 2, "trade", {"grain": 1}, {"timber": 1}, 0, 0, 5)
    assert world._action_features(1, trade)[-1] == 0
    world.islands[1].available_time = 1
    assert trade in world._candidates(1)


def test_legacy_features_and_settings_remain_unchanged() -> None:
    """Append one feature only under version six and derive a temperature-aware initial weight."""
    configuration = load_config()
    current = World(configuration)
    assert current.action_feature_size == 109
    assert current.initial_action_biases == {108: 2 * math.log(2)}
    configuration["exploration"]["anneal_updates"] = 0
    assert World(configuration).initial_action_biases == {108: math.log(2)}
    configuration["schema_version"] = 5
    del configuration["trade_initialization"]
    legacy = World(configuration)
    assert legacy.action_feature_size == 108
    assert legacy.initial_action_biases == {}
    action = Action("propose_trade", {"target": 1, "offer": {"water": 4}, "request": {"grain": 4}})
    assert current._action_features(0, action)[:-1] == legacy._action_features(0, action)


@pytest.mark.parametrize("setting,value", [
    ("initial_odds_multiplier", 0.9), ("initial_odds_multiplier", True),
    ("initial_odds_multiplier", float("inf")), ("initial_odds_multiplier", "2"),
    ("reserve_turns", -1), ("reserve_turns", True), ("reserve_turns", float("nan")),
])
def test_invalid_trade_initialization_is_rejected(setting: str, value: Any) -> None:
    """Validate explicit finite trade initialization settings before any policy is allocated."""
    configuration = load_config()
    configuration["trade_initialization"][setting] = value
    with pytest.raises(ValueError):
        validate_config(configuration)


@pytest.mark.neural
@pytest.mark.parametrize("use_vmap", [True, False])
def test_starting_odds_double_and_prior_is_learned_inherited_and_restored(use_vmap: bool) -> None:
    """Verify an actual twofold odds change and normal learning without reapplying the founder bias."""
    import torch
    from island_community.policies import PolicyPopulation

    unbiased = PolicyPopulation(3, 2, hidden_size=8, capacity=2, seed=42, mutation_std=0,
                                initial_temperature=2, anneal_updates=30, use_vmap=use_vmap)
    biased = PolicyPopulation(3, 2, hidden_size=8, capacity=2, seed=42, mutation_std=0,
                              initial_temperature=2, anneal_updates=30, use_vmap=use_vmap,
                              initial_action_biases={1: 2 * math.log(2)})
    observation = {0: [0.2, 0.4, 0.6]}
    candidates = {0: [[0.5, 1.0], [0.5, 0.0], [0.1, 0.0]]}
    odds = []
    for population in (unbiased, biased):
        population.add(0)
        population.act(observation, candidates)
        pending = population._pending
        assert pending is not None
        parameters = {name: value.index_select(0, pending.slots) for name, value in population._parameters.items()}
        scores, _, _ = population._forward(parameters, pending.observations, pending.candidates, pending.valid_mask, pending.previous_hidden)
        logits = population._behavior_logits(scores, pending.valid_mask, pending.temperatures, pending.log_candidate_counts)
        probabilities = logits.softmax(-1)
        odds.append((probabilities[0, 0] / probabilities[0, 1]).item())
    assert odds[1] / odds[0] == pytest.approx(2, rel=1e-6)
    before = biased._parameters["action_bias.weight"][0, 0, 1].item()
    restored = PolicyPopulation.from_state_dict(biased.state_dict())
    for population in (biased, restored):
        population.learn({0: 2}, {}, {0})
        assert population._parameters["action_bias.weight"][0, 0, 1].item() != before
        population.add(1, parent_id=0)
        torch.testing.assert_close(population._parameters["action_bias.weight"][0], population._parameters["action_bias.weight"][1], rtol=0, atol=0)
    for name in biased._parameters:
        torch.testing.assert_close(biased._parameters[name], restored._parameters[name], rtol=0, atol=0)
    assert restored.initial_action_biases == biased.initial_action_biases
    legacy = unbiased.state_dict()
    del legacy["configuration"]["initial_action_biases"]
    assert PolicyPopulation.from_state_dict(legacy).initial_action_biases == {}


@pytest.mark.neural
@pytest.mark.parametrize("biases", [{-1: 1}, {2: 1}, {True: 1}, {1: float("nan")}, {1: True}])
def test_policy_rejects_invalid_initial_biases(biases: dict[int, float]) -> None:
    """Reject invalid sparse feature weights at the policy initialization boundary."""
    from island_community.policies import PolicyPopulation

    with pytest.raises(ValueError, match="Initial action biases"):
        PolicyPopulation(3, 2, initial_action_biases=biases)
