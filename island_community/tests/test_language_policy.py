"""Exercise language action contracts and privacy without running neural models."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pytest

from island_community.config import load_config, validate_config
from island_community.language_policy import TokenChoiceTrie, decision_messages, parse_choice, private_context
from island_community.world import Action, World


@pytest.fixture
def configuration() -> dict[str, Any]:
    """Use the explicit ten-island scenario with deterministic economic fixtures."""
    settings = load_config(Path(__file__).parents[1] / "scenarios" / "qwen_10_frozen.json")
    settings["events"]["probability_per_turn"] = 0
    settings["reproduction"]["enabled"] = False
    return settings


@pytest.mark.parametrize("seed", [0, 1, 42, 77])
def test_ten_island_coverage_is_explicit(configuration: dict[str, Any], seed: int) -> None:
    configuration["seed"] = seed
    world = World(configuration)
    assert len(world.islands) == 10
    assert all(len(world.neighbors(identifier)) == 5 and not world.missing_neighbor_resources(identifier)
               for identifier in world.islands)
    configuration["allocation"]["require_all_resource_producers"] = True
    with pytest.raises(ValueError, match="every resource"):
        validate_config(configuration)


def test_language_templates_require_separate_quantity_selection(configuration: dict[str, Any]) -> None:
    configuration["actions"]["trade_quantity_mode"] = "fixed"
    configuration["actions"]["trade_quantities"] = [1, 4]
    with pytest.raises(ValueError, match="all_feasible trade quantities"):
        validate_config(configuration)


@pytest.mark.parametrize("response", ['{"choice":true}', '{"choice":1.0}', '{"choice":"1"}',
    '{"choice":-1}', '{"choice":3}', '{"choice":01}', '{"choice":1,"choice":0}',
    '{"choice":1,"action":"wait"}', '[1]', '```json\n{"choice":1}\n```',
    'reasoning {"choice":1}', '{"choice":1}\n{"choice":0}', ' {"choice":1}'])
def test_parser_never_coerces_or_repairs_invalid_moves(response: str) -> None:
    with pytest.raises(ValueError):
        parse_choice(response, 3)


def test_choice_grammar_distinguishes_prefixes_and_requires_completion() -> None:
    grammar = TokenChoiceTrie([[1, 2, 3], [1, 2, 4, 3], [1, 5, 3]], 9)
    assert grammar.allowed([]) == [1]
    assert grammar.allowed([1]) == [2, 5]
    assert grammar.allowed([1, 2]) == [3, 4]
    assert grammar.allowed([1, 2, 3]) == [9]
    with pytest.raises(ValueError):
        grammar.allowed([1, 8])
    assert parse_choice('{"choice":2}', 3) == 2


def test_prompt_contains_exact_own_state_without_neighbor_inventory(configuration: dict[str, Any]) -> None:
    world = World(configuration)
    world.start_turn()
    identifier = 0
    history = [{"turn": 0, "action": {"kind": "wait", "args": {}}, "reward": -5.0, "executed": True}]
    original = private_context(world, identifier, history)
    neighbor = world.neighbors(identifier)[0]
    world.islands[neighbor].inventory["water"] = 987654321
    world.islands[neighbor].recent_history = [{"secret": "neighbor-only"}]
    assert private_context(world, identifier, history) == original
    assert original["inventory"] == world.islands[identifier].inventory
    assert original["available"] == world.available(identifier)
    assert original["history"] == history
    history.append({"later": True})
    assert len(original["history"]) == 1
    assert all(set(neighbor) == {"island", "produces", "distance", "capacity", "closed", "transit_turns"}
               for neighbor in original["neighbors"])


def test_requested_quantities_reach_proposal_acceptance_and_shipments(configuration: dict[str, Any]) -> None:
    """Trace parsed IDs through exact action objects, actual contracts, and delivered quantities."""
    world = World(configuration)
    decisions = world.start_turn()
    sender = next(identifier for identifier, island in world.islands.items() if island.specialization == ["water"])
    receiver = next(identifier for identifier in world.neighbors(sender) if world.islands[identifier].specialization == ["grain"])
    template_index = next(index for index, action in enumerate(decisions[sender].candidates)
                          if action.kind == "propose_trade" and action.args["target"] == receiver
                          and action.args["offer"] == {"water": 1} and action.args["request"] == {"grain": 1})
    messages = decision_messages(world, private_context(world, sender, []), decisions[sender].candidates, "action")
    menu = json.loads(messages[1]["content"])["action_options"]
    rows = {row[0]: (group["kind"], dict(zip(group["columns"], row))) for group in menu for row in group["rows"]}
    assert len(rows) == len(decisions[sender].candidates)
    assert rows[template_index][0] == "propose_trade"
    assert rows[template_index][1]["offer_item"] == "water"
    assert rows[template_index][1]["request_item"] == "grain"
    assert rows[template_index][1]["target"] == receiver
    for index, candidate in enumerate(decisions[sender].candidates):
        kind, displayed = rows[index]
        assert kind == candidate.kind
        for name, value in candidate.args.items():
            if name in {"offer", "request"}:
                assert len(value) <= 1
                assert displayed[name + "_item"] == next(iter(value), None)
                assert all(quantity == 1 for quantity in value.values())
            else:
                assert displayed[name] == value
    template = decisions[sender].candidates[parse_choice(f'{{"choice":{template_index}}}', len(rows))]
    quantities = world.quantity_choices(sender, template)
    quantity_index = next(index for index, action in enumerate(quantities)
                          if action.args["offer"] == {"water": 5} and action.args["request"] == {"grain": 3})
    requested = quantities[parse_choice(f'{{"choice":{quantity_index}}}', len(quantities))]
    result = world.finish_turn({sender: requested})
    assert result.action_results[sender][0]["action"] == asdict(requested)
    assert result.action_results[sender][0]["executed"]
    proposal = next(iter(world.proposals.values()))
    assert proposal.offer == {"water": 5} and proposal.request == {"grain": 3}
    next_decisions = world.start_turn()
    accept_index = next(index for index, action in enumerate(next_decisions[receiver].candidates)
                        if action == Action("accept", {"proposal": proposal.id}))
    accepted = next_decisions[receiver].candidates[parse_choice(f'{{"choice":{accept_index}}}', len(next_decisions[receiver].candidates))]
    result = world.finish_turn({receiver: accepted})
    assert result.action_results[receiver][0]["executed"]
    assert proposal.status == "accepted"
    assert {shipment.owner: shipment.items for shipment in world.shipments.values()} == {receiver: {"water": 5}, sender: {"grain": 3}}
    for _ in range(3):
        world.start_turn()
        world.finish_turn({})
    assert all(shipment.status == "delivered" for shipment in world.shipments.values())
    receipts = {(event["island"], event["item"]): event["delta"] for event in world.events
                if event["type"] == "inventory" and event["reason"] == "trade_receipt"}
    assert receipts == {(receiver, "water"): 5, (sender, "grain"): 3}


def test_stale_action_is_reported_as_rejected_without_substitution(configuration: dict[str, Any]) -> None:
    world = World(configuration)
    world.start_turn()
    sender, receiver = 0, world.neighbors(0)[0]
    world.finish_turn({sender: Action("propose_trade", {"target": receiver, "offer": {"water": 1}, "request": {"grain": 1}})})
    decisions = world.start_turn()
    proposal = next(iter(world.proposals.values()))
    action = Action("accept", {"proposal": proposal.id})
    assert action in decisions[receiver].candidates
    proposal.status = "cancelled"
    result = world.finish_turn({receiver: action})
    record = result.action_results[receiver][0]
    assert record["action"] == asdict(action)
    assert record["executed"] is False
    assert world.events[record["event_sequences"][-1]]["type"] == "action_rejected"
