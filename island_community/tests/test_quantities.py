"""Check flexible integer trade amounts against public limits and private stock boundaries."""

from pathlib import Path
from typing import Any

import pytest

from island_community.config import load_config, validate_config
from island_community.world import Action, World


@pytest.fixture
def world() -> World:
    """Prepare a small deterministic neighborhood with explicit available stock."""
    configuration = load_config(Path(__file__).parents[1] / "scenarios" / "neighbors_50.json")
    configuration["initialization"]["population"] = 20
    configuration["events"]["probability_per_turn"] = 0
    configuration["reproduction"]["enabled"] = False
    instance = World(configuration)
    instance.start_turn()
    instance.islands[0].inventory["grain"] = 50
    return instance


def test_every_asymmetric_integer_pair_within_transport_capacity_is_available(world: World) -> None:
    """Include amounts outside the old menu and all ratios, without exposing the target's stock."""
    target = world.neighbors(0)[0]
    template = Action("propose_trade", {"target": target, "offer": {"grain": 1}, "request": {"water": 1}, "price": 0})
    assert template in world._candidates(0)
    choices = world.quantity_choices(0, template)
    pairs = {(action.args["offer"]["grain"], action.args["request"]["water"]) for action in choices}
    assert pairs == {(offered, requested) for offered in range(1, 32) for requested in range(1, 33 - offered)}
    assert (3, 7) in pairs and (31, 1) in pairs
    world.islands[target].inventory["water"] = 0
    assert world.quantity_choices(0, template) == choices
    chosen = next(action for action in choices if action.args["offer"]["grain"] == 3 and action.args["request"]["water"] == 7)
    assert world._propose(0, chosen)
    world.finish_turn({})
    world.start_turn()
    world.islands[target].inventory["water"] = 0
    incoming = [action for action in world._candidates(target) if action.args.get("proposal") == 0]
    assert Action("accept", {"proposal": 0}) not in incoming
    world.islands[target].inventory["water"] = 7
    assert Action("accept", {"proposal": 0}) in world._candidates(target)
    assert world._accept(world.proposals[0])
    assert sorted((shipment.items for shipment in world.shipments.values()), key=str) == sorted([{"grain": 3}, {"water": 7}], key=str)


def test_reservations_and_money_bound_choices_but_partner_money_does_not(world: World) -> None:
    """Respect the proposer's own reservations and recompute each purchase's actual price."""
    target = world.neighbors(0)[0]
    assert world._propose(0, Action("propose_trade", {"target": target, "offer": {"grain": 47}, "request": {"water": 1}}))
    template = Action("propose_trade", {"target": target, "offer": {"grain": 1}, "request": {"water": 1}, "price": 0})
    assert max(action.args["offer"]["grain"] for action in world.quantity_choices(0, template)) == 3
    world.islands[0].currency = 7
    world.config["actions"]["purchase_price_per_value"] = 1
    world.config["items"]["water"]["reward_value"] = 1
    purchase = Action("propose_trade", {"target": target, "offer": {}, "request": {"water": 1}, "price": 1, "payment_from": "proposer"})
    choices = world.quantity_choices(0, purchase)
    assert [action.args["request"]["water"] for action in choices] == list(range(1, 8))
    assert [action.args["price"] for action in choices] == list(range(1, 8))
    world.islands[target].currency = 0
    assert world.quantity_choices(0, purchase) == choices
    assert world._propose(0, choices[-1])
    assert not world.quantity_choices(0, purchase)


def test_capacity_is_shared_and_weighted_and_outages_disable_proposals(world: World) -> None:
    """Mask quantities by load, existing dispatches, contact restrictions, and channel outages."""
    target = world.neighbors(0)[0]
    template = Action("propose_trade", {"target": target, "offer": {"grain": 1}, "request": {"water": 1}, "price": 0})
    channel_id, channel = world._channel(0, target)
    world._channel_used[channel_id] = 26
    world.config["items"]["grain"]["transport_load"] = 2
    choices = world.quantity_choices(0, template)
    assert {(action.args["offer"]["grain"], action.args["request"]["water"]) for action in choices} == {(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2)}
    channel.disabled_until = world.turn
    assert not world.quantity_choices(0, template)
    assert template not in world._candidates(0)
    outsider = next(identifier for identifier in world.islands if identifier != 0 and identifier not in world.neighbors(0))
    assert not world.quantity_choices(0, Action(template.kind, {**template.args, "target": outsider}))


def test_counteroffers_can_reduce_amount_even_when_original_request_is_unaffordable(world: World) -> None:
    """Let recipients negotiate smaller and unequal terms instead of requiring the original amount."""
    target = world.neighbors(0)[0]
    world.islands[target].inventory["water"] = 3
    assert world._propose(0, Action("propose_trade", {"target": target, "offer": {"grain": 8}, "request": {"water": 20}}))
    world.finish_turn({})
    world.start_turn()
    world.islands[target].inventory["water"] = 3
    template = next(action for action in world._candidates(target) if action.kind == "counter")
    choices = world.quantity_choices(target, template)
    chosen = next(action for action in choices if action.args["offer"] == {"water": 2} and action.args["request"] == {"grain": 7})
    assert world._propose(target, chosen)
    assert world.proposals[0].status == "countered"
    assert world.proposals[1].offer == {"water": 2}
    assert world.proposals[1].request == {"grain": 7}


def test_nontrades_and_fixed_mode_do_not_add_quantity_options(world: World) -> None:
    wait = Action("wait")
    assert world.quantity_choices(0, wait) == [wait]
    world.config["actions"].update(trade_quantity_mode="fixed", trade_quantities=[1, 4])
    validate_config(world.config)
    trade = next(action for action in world._candidates(0) if action.kind == "propose_trade")
    assert world.quantity_choices(0, trade) == [trade]


@pytest.mark.parametrize("mode,quantities", [("unbounded", []), ("all_feasible", [1, 4]), ("fixed", []), ("fixed", [True])])
def test_invalid_quantity_configurations_are_rejected(world: World, mode: str, quantities: list[Any]) -> None:
    world.config["actions"].update(trade_quantity_mode=mode, trade_quantities=quantities)
    with pytest.raises(ValueError):
        validate_config(world.config)
