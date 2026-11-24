"""Pure rule tests: no learned inference, training, benchmarks, or experiment runs."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from island_community.config import load_config, survival_bundle, validate_config
from island_community.world import Action, WorkOrder, World


@pytest.fixture
def config() -> dict[str, Any]:
    configuration = load_config()
    configuration["survival"]["maximum_health"] = 100.0
    configuration["allocation"]["allocations_per_minute"] = 0
    configuration["events"]["probability_per_turn"] = 0.0
    configuration["events"]["incoming_contamination_probability"] = 0.0
    configuration["reproduction"]["enabled"] = False
    return configuration


def roundtrip(world: World) -> World:
    """Restore the exact JSON format used by a world checkpoint."""
    return World.from_state_dict(world.config, json.loads(json.dumps(world.state_dict())))


def finish(world: World, actions: dict[int, Action] | None = None) -> None:
    """Resolve one deliberately specified rule-test turn."""
    world.start_turn()
    world.finish_turn(actions or {})


def test_defaults_cover_base_resources_and_five_turn_bundle(config: dict[str, Any]) -> None:
    world = World(config)
    assert len(world.islands) == 5
    assert len(set().union(*(set(island.specialization) for island in world.islands.values()))) == 10
    assert all(len(island.specialization) <= 2 for island in world.islands.values())
    bundle = survival_bundle(config)
    assert bundle == {"grain": 8, "water": 8}
    for island in world.islands.values():
        assert all(island.inventory[item] >= quantity for item, quantity in bundle.items())
    for _ in range(5):
        finish(world)
    assert all(island.inventory["grain"] >= 3 and island.inventory["water"] >= 3 for island in world.islands.values())
    assert all(island.health > 95 for island in world.islands.values())


@pytest.mark.parametrize("mutation", [
    lambda configuration: configuration["items"]["water"].update(reward_value=float("nan")),
    lambda configuration: configuration["items"]["water"].update(reward_value=0),
    lambda configuration: configuration["recipes"]["weave"]["outputs"].update(cloth=100),
    lambda configuration: configuration["recipes"]["weave"]["inputs"].update(cloth=-1),
    lambda configuration: configuration["allocation"].update(turns_per_minute=0),
    lambda configuration: configuration["allocation"].update(allocations_per_minute=1.5),
    lambda configuration: configuration["initialization"].update(population=3),
    lambda configuration: configuration["learning"].update(death_replay=True),
    lambda configuration: configuration["transport"].update(ownership="both"),
    lambda configuration: configuration["policy"].update(learning_rate=-1),
    lambda configuration: configuration["exploration"].update(initial_temperature=0),
    lambda configuration: configuration["exploration"].update(final_temperature=-1),
    lambda configuration: configuration["exploration"].update(initial_temperature=float("inf")),
    lambda configuration: configuration["exploration"].update(initial_entropy_coefficient=-0.1),
    lambda configuration: configuration["exploration"].update(anneal_updates=1.5),
    lambda configuration: configuration["exploration"].update(anneal_updates=-1),
    lambda configuration: configuration["exploration"].update(anneal_updates=True),
    lambda configuration: configuration["exploration"].update(balance_action_kinds=1),
])
def test_invalid_configuration_is_rejected(config: dict[str, Any], mutation: Any) -> None:
    mutation(config)
    with pytest.raises(ValueError):
        validate_config(config)


def test_decision_dimensions_do_not_disclose_neighbor_inventory(config: dict[str, Any]) -> None:
    world = World(config)
    decisions = world.start_turn()
    for decision in decisions.values():
        assert len(decision.observation) == world.observation_size
        assert all(len(features) == world.action_feature_size for features in decision.action_features)
        assert decision.candidates[0] == Action("wait")
    observation = world.observe(0)
    candidates = world._candidates(0)
    features = [world._action_features(0, action) for action in candidates]
    world.islands[1].inventory["grain"] = 1_000_000
    world.islands[1].currency = 1_000_000
    assert observation == world.observe(0)
    assert candidates == world._candidates(0)
    assert features == [world._action_features(0, action) for action in candidates]


def test_work_consumes_inputs_before_outputs_and_carries_forward(config: dict[str, Any]) -> None:
    world = World(config)
    world.islands[0].inventory["fiber"] = 2
    finish(world, {0: Action("manufacture", {"recipe": "weave"})})
    assert world.islands[0].inventory["fiber"] == 0
    assert world.islands[0].inventory["cloth"] == 0
    assert world.islands[0].work is not None
    assert world.islands[0].work.remaining == pytest.approx(0.5)
    finish(world)
    assert world.islands[0].inventory["cloth"] == 1
    assert world.islands[0].work is None
    assert world.islands[0].skills["weaving"] == config["actions"]["skill_gain"]


def test_noncarried_work_and_plan_respect_total_time(config: dict[str, Any]) -> None:
    config["actions"]["work_carries_forward"] = False
    world = World(config)
    world.islands[0].inventory["fiber"] = 2
    finish(world, {0: Action("manufacture", {"recipe": "weave"})})
    assert world.islands[0].inventory["fiber"] == 2
    assert world.islands[0].work is None
    world.islands[0].inventory["grain"] = 20
    world.islands[0].inventory["water"] = 20
    finish(world, {0: Action("plan", {"actions": [
        {"kind": "manufacture", "args": {"recipe": "cook"}},
        {"kind": "manufacture", "args": {"recipe": "cook"}},
        {"kind": "public_request", "args": {"item": "water"}},
    ]})})
    assert world.islands[0].skills["cooking"] == config["actions"]["skill_gain"]
    assert world.islands[0].available_time == pytest.approx(0.28)
    assert len(world.messages) == 1


def test_reservations_prevent_multiple_commitments_and_manufacturing(config: dict[str, Any]) -> None:
    world = World(config)
    world.islands[0].inventory["fiber"] = 2
    finish(world, {0: Action("plan", {"actions": [
        {"kind": "propose_trade", "args": {"target": 1, "offer": {"fiber": 2}, "request": {"iron": 1}}},
        {"kind": "propose_trade", "args": {"target": 2, "offer": {"fiber": 2}, "request": {"iron": 1}}},
        {"kind": "manufacture", "args": {"recipe": "weave"}},
    ]})})
    assert len(world.proposals) == 1
    assert world.reserved(0)["fiber"] == 2
    assert world.available(0)["fiber"] == 0
    assert world.islands[0].inventory["cloth"] == 0
    finish(world, {0: Action("cancel", {"proposal": 0})})
    assert world.available(0)["fiber"] == 2


def test_same_turn_proposal_cannot_be_accepted(config: dict[str, Any]) -> None:
    world = World(config)
    world.islands[0].inventory["timber"] = 1
    world.islands[1].inventory["iron"] = 1
    finish(world, {
        0: Action("propose_trade", {"target": 1, "offer": {"timber": 1}, "request": {"iron": 1}}),
        1: Action("accept", {"proposal": 0}),
    })
    assert world.proposals[0].status == "pending"
    assert not world.shipments


def test_trade_charges_acceptor_and_has_one_owner_in_transit(config: dict[str, Any]) -> None:
    config["transport"]["minimum_transit_turns"] = 2
    world = World(config)
    world.islands[0].inventory["timber"] = 1
    world.islands[1].inventory["iron"] = 1
    finish(world, {0: Action("propose_trade", {"target": 1, "offer": {"timber": 1}, "request": {"iron": 1}, "price": 3})})
    currency_before = {identifier: island.currency for identifier, island in world.islands.items()}
    finish(world, {1: Action("accept", {"proposal": 0})})
    fee = math.ceil(world._channel(0, 1)[1].distance * config["transport"]["fee_per_distance"])
    assert world.islands[0].currency == currency_before[0] + 3
    assert world.islands[1].currency == currency_before[1] - 3 - fee
    assert world.islands[0].inventory["timber"] == 0
    assert world.islands[1].inventory["iron"] == 0
    assert world.shipments[0].owner == 1
    assert world.shipments[1].owner == 0
    local_value = sum(quantity * config["items"][item]["reward_value"] for island in world.islands.values() for item, quantity in island.inventory.items())
    total_value = world.snapshot()["metrics"]["total_resource_value"]
    assert total_value == pytest.approx(local_value + 6)
    finish(world)
    assert world.islands[1].inventory["timber"] == 0
    world.start_turn()
    assert world.islands[1].inventory["timber"] == 1
    assert world.islands[0].inventory["iron"] == 1
    assert world.contracts[0].status == "completed"


def test_bidirectional_capacity_is_shared_and_rejected_trade_is_atomic(config: dict[str, Any]) -> None:
    config["transport"]["capacity"] = 3
    world = World(config)
    world.islands[0].inventory["timber"] = 1
    world.islands[1].inventory["iron"] = 1
    finish(world, {0: Action("propose_trade", {"target": 1, "offer": {"timber": 1}, "request": {"iron": 1}})})
    balance = world.islands[1].currency
    finish(world, {1: Action("accept", {"proposal": 0})})
    assert world.proposals[0].status == "pending"
    assert world.islands[1].currency == balance
    assert not world.shipments


def test_counter_releases_old_reservation_and_waits_for_next_turn(config: dict[str, Any]) -> None:
    world = World(config)
    world.islands[0].inventory["timber"] = 2
    world.islands[1].inventory["iron"] = 1
    finish(world, {0: Action("propose_trade", {"target": 1, "offer": {"timber": 1}, "request": {"iron": 1}})})
    finish(world, {1: Action("counter", {"proposal": 0, "offer": {"iron": 1}, "request": {"timber": 2}})})
    assert world.proposals[0].status == "countered"
    assert world.reserved(0)["timber"] == 0
    assert world.proposals[1].acceptor == 0
    assert world.reserved(1)["iron"] == 1


def test_service_uses_supplier_work_and_pays_after_delivery(config: dict[str, Any]) -> None:
    world = World(config)
    world.islands[0].inventory["fiber"] = 2
    finish(world, {0: Action("propose_service", {"target": 1, "recipe": "weave", "price": 4})})
    supplier_balance = world.islands[0].currency
    finish(world, {1: Action("accept", {"proposal": 0})})
    assert world.contracts[0].escrow == 4
    assert world.islands[0].currency == supplier_balance
    assert world.islands[0].work is not None
    assert world.islands[0].inventory["fiber"] == 0
    finish(world)
    assert world.islands[0].work is None
    assert world.shipments[0].items == {"cloth": 1}
    while world.turn < world.shipments[0].arrival_turn:
        finish(world)
    assert world.islands[1].inventory["cloth"] == 1
    assert world.islands[0].currency == supplier_balance + 4
    assert world.contracts[0].escrow == 0


def test_teaching_retains_skill_and_transfer_relinquishes_it(config: dict[str, Any]) -> None:
    world = World(config)
    world.islands[0].skills["weaving"] = 1.0
    finish(world, {0: Action("propose_teaching", {"target": 1, "skill": "weaving", "price": 2})})
    finish(world, {1: Action("accept", {"proposal": 0})})
    assert world.islands[0].skills["weaving"] == 1.0
    assert world.islands[1].skills["weaving"] == config["actions"]["teaching_gain"]
    finish(world, {0: Action("propose_transfer", {"target": 1, "skill": "weaving", "transfer_capability": True})})
    finish(world, {1: Action("accept", {"proposal": 1})})
    assert world.islands[0].skills["weaving"] == 0.75
    assert world.islands[1].skills["weaving"] == pytest.approx(0.35)
    assert "weaving" not in world.islands[0].capabilities
    assert "weaving" in world.islands[1].capabilities


def test_birth_needs_no_parent_reserve_and_newborn_does_not_consume(config: dict[str, Any]) -> None:
    config["reproduction"]["enabled"] = True
    config["reproduction"]["interval"] = 1
    config["reproduction"]["additional_inheritance_factor"] = 0
    world = World(config)
    for island in world.islands.values():
        island.inventory = {item: 0 for item in world.item_ids}
    world.islands[0].inventory.update(grain=9, water=9)
    world.start_turn()
    result = world.finish_turn({})
    assert result.births == {5: 0}
    assert world.islands[0].inventory["grain"] == 0
    assert world.islands[0].inventory["water"] == 0
    assert world.islands[5].inventory["grain"] == 8
    assert world.islands[5].inventory["water"] == 8
    assert world.islands[5].age == 0
    assert world.islands[5].parent_id == 0
    assert world.islands[0].descendants == [5]
    assert len(world.channels) == 15


def test_reserved_resources_cannot_fund_birth(config: dict[str, Any]) -> None:
    config["reproduction"]["enabled"] = True
    config["reproduction"]["interval"] = 1
    world = World(config)
    for island in world.islands.values():
        island.next_birth_turn = 100
    world.islands[0].next_birth_turn = 1
    world.islands[0].inventory.update(grain=9, water=9)
    finish(world, {0: Action("propose_trade", {"target": 1, "offer": {"grain": 2}, "request": {"iron": 1}})})
    assert world.total_births == 0


def test_death_terminal_zero_explicit_assets_and_extinction(config: dict[str, Any]) -> None:
    world = World(config)
    for island in world.islands.values():
        island.health = -100
    world.start_turn()
    result = world.finish_turn({})
    assert sorted(result.deaths) == list(range(5))
    assert result.rewards == {identifier: 0.0 for identifier in range(5)}
    assert not world.islands
    assert len(world.deceased) == 5
    assert world.start_turn() == {}
    assert any(event.get("reason") == "deceased_asset_destruction" for event in world.events)


def test_private_messages_and_public_requests_have_next_turn_visibility(config: dict[str, Any]) -> None:
    world = World(config)
    world.start_turn()
    before = world.observe(1)
    world._execute(0, Action("private_request", {"target": 1, "item": "water"}))
    assert world.observe(1) == before
    world.finish_turn({})
    world.start_turn()
    private_index = 14 + 4 * len(world.item_ids) + world.item_ids.index("water")
    assert world.observe(1)[private_index] > 0
    assert world.observe(2)[private_index] == 0


def test_contaminated_food_causes_disease_and_aging_reduces_skill(config: dict[str, Any]) -> None:
    config["aging"]["start_turn"] = 1
    world = World(config)
    world.islands[0].contaminated["grain"] = 1
    world.islands[0].skills["cooking"] = 1.0
    finish(world)
    assert world.islands[0].health == 100 - config["survival"]["disease_damage"] - config["aging"]["health_damage"]
    assert world.islands[0].disease_until == config["survival"]["disease_duration"]
    assert world.islands[0].skills["cooking"] == config["aging"]["skill_retention"]


def test_missing_food_damages_health_without_immediate_death(config: dict[str, Any]) -> None:
    world = World(config)
    world.islands[0].inventory["grain"] = 0
    finish(world)
    assert world.islands[0].health == 100 - config["survival"]["shortage_damage"]
    assert 0 in world.islands


def test_allocation_is_base_only_and_respects_specialization(config: dict[str, Any]) -> None:
    config["allocation"]["allocations_per_minute"] = 3
    world = World(config)
    world.start_turn()
    allocations = [event for event in world.events if event.get("reason") == "world_allocation"]
    assert len(allocations) == 3
    for event in allocations:
        assert event["item"] in world.islands[event["island"]].specialization
        assert config["items"][event["item"]]["category"] == "base"
        assert event["delta"] in config["allocation"]["quantities"]


def test_checkpoint_reproduces_pending_work_and_random_continuation(config: dict[str, Any]) -> None:
    config["allocation"]["allocations_per_minute"] = 3
    config["events"]["probability_per_turn"] = 1.0
    world = World(config)
    world.islands[0].inventory["fiber"] = 2
    finish(world, {0: Action("manufacture", {"recipe": "weave"})})
    restored = roundtrip(world)
    assert world.start_turn() == restored.start_turn()
    assert world.finish_turn({}) == restored.finish_turn({})
    assert json.loads(json.dumps(world.state_dict())) == json.loads(json.dumps(restored.state_dict()))


def test_preview_preserves_state_and_matches_next_observation(config: dict[str, Any]) -> None:
    config["allocation"]["allocations_per_minute"] = 3
    world = World(config)
    finish(world)
    saved = json.loads(json.dumps(world.state_dict()))
    preview = world.preview_next_observations()
    assert saved == json.loads(json.dumps(world.state_dict()))
    actual = world.start_turn()
    assert preview == {identifier: decision.observation for identifier, decision in actual.items()}


def test_inventory_ledger_reconciles_current_stock(config: dict[str, Any]) -> None:
    world = World(config)
    finish(world, {0: Action("manufacture", {"recipe": "cook"})})
    balances = {(identifier, item): 0 for identifier in world.islands for item in world.item_ids}
    for event in world.events:
        if event["type"] == "inventory":
            balances[event["island"], event["item"]] += event["delta"]
    assert all(balances[identifier, item] == quantity for identifier, island in world.islands.items() for item, quantity in island.inventory.items())


def test_accept_candidates_do_not_leak_supplier_skills_work_or_stock(config: dict[str, Any]) -> None:
    world = World(config)
    world.islands[0].skills["cooking"] = 1
    finish(world, {0: Action("propose_teaching", {"target": 1, "skill": "cooking"})})
    world.start_turn()
    original = world._candidates(1)
    original_features = [world._action_features(1, action) for action in original]
    world.islands[0].skills["cooking"] = 0
    world.islands[0].available_time = 0
    world.islands[0].inventory["water"] = 0
    world.islands[0].work = WorkOrder("weave", 10, {"cloth": 1})
    assert original == world._candidates(1)
    assert original_features == [world._action_features(1, action) for action in world._candidates(1)]
    balance = world.islands[1].currency
    world.finish_turn({1: Action("accept", {"proposal": 0})})
    assert world.proposals[0].status == "pending"
    assert world.islands[1].currency == balance


def test_accept_features_distinguish_teaching_from_skill_transfer(config: dict[str, Any]) -> None:
    world = World(config)
    world.islands[0].skills["weaving"] = 1.0
    finish(world, {0: Action("plan", {"actions": [
        {"kind": "propose_teaching", "args": {"target": 1, "skill": "weaving", "price": 2}},
        {"kind": "propose_transfer", "args": {"target": 1, "skill": "weaving", "price": 2}},
    ]})})
    world.start_turn()
    teaching = world._action_features(1, Action("accept", {"proposal": 0}))
    transfer = world._action_features(1, Action("accept", {"proposal": 1}))
    assert teaching[:-len(world.proposal_kinds)] == transfer[:-len(world.proposal_kinds)]
    assert teaching != transfer
    assert len(teaching) == world.action_feature_size


def test_private_request_requires_a_recipient(config: dict[str, Any]) -> None:
    world = World(config)
    finish(world, {0: Action("private_request", {"item": "grain"})})
    assert not world.messages


def test_visible_requests_identify_the_requested_item_and_partner(config: dict[str, Any]) -> None:
    world = World(config)
    purchase = Action("propose_trade", {"target": 0, "offer": {}, "request": {"iron": 1}, "price": 4, "payment_from": "proposer"})
    world.start_turn()
    before = world._action_features(1, purchase)
    world.finish_turn({0: Action("private_request", {"target": 1, "item": "fiber"})})
    world.start_turn()
    after = world._action_features(1, purchase)
    assert before != after
    assert len(after) == world.action_feature_size
    target_features = len(world.action_kinds) + 4 * len(world.item_ids)
    assert after[target_features + world.item_ids.index("fiber")] > 0


def test_currency_purchase_reserves_money_and_can_buy_manufacturing_inputs(config: dict[str, Any]) -> None:
    world = World(config)
    world.islands[0].currency = 5
    world.islands[1].inventory["iron"] = 1
    finish(world, {0: Action("plan", {"actions": [
        {"kind": "propose_trade", "args": {"target": 1, "request": {"iron": 1}, "price": 4, "payment_from": "proposer"}},
        {"kind": "propose_trade", "args": {"target": 2, "request": {"iron": 1}, "price": 4, "payment_from": "proposer"}},
    ]})})
    assert len(world.proposals) == 1
    assert world.available_currency(0) == 1
    seller_balance = world.islands[1].currency
    finish(world, {1: Action("accept", {"proposal": 0})})
    fee = math.ceil(world._channel(0, 1)[1].distance * config["transport"]["fee_per_distance"])
    assert world.islands[0].currency == 1
    assert world.islands[1].currency == seller_balance + 4 - fee
    assert world.shipments[0].owner == 0
    assert world.shipments[0].items == {"iron": 1}


def test_service_failure_refunds_escrow_and_destroys_no_promised_stock_twice(config: dict[str, Any]) -> None:
    config["actions"]["contract_deadline"] = 1
    config["recipes"]["weave"]["base_time"] = 10
    world = World(config)
    world.islands[0].inventory["fiber"] = 2
    finish(world, {0: Action("propose_service", {"target": 1, "recipe": "weave", "price": 4})})
    balance = world.islands[1].currency
    finish(world, {1: Action("accept", {"proposal": 0})})
    fee = math.ceil(world._channel(0, 1)[1].distance * config["transport"]["fee_per_distance"])
    finish(world)
    world.start_turn()
    assert world.contracts[0].status == "failed"
    assert world.contracts[0].escrow == 0
    assert world.islands[1].currency == balance - fee
    assert world.islands[0].work is None
    assert world.islands[0].inventory["fiber"] == 0
    assert world.islands[0].inventory["cloth"] == 0


def test_late_service_arrival_fails_before_delivery_and_refunds_escrow(config: dict[str, Any]) -> None:
    config["actions"]["contract_deadline"] = 1
    config["transport"]["minimum_transit_turns"] = 2
    world = World(config)
    finish(world, {0: Action("propose_service", {"target": 1, "recipe": "cook", "price": 4})})
    supplier_balance = world.islands[0].currency
    customer_balance = world.islands[1].currency
    finish(world, {1: Action("accept", {"proposal": 0})})
    assert world.contracts[0].deadline == 3
    assert world.shipments[0].arrival_turn == 4
    finish(world)
    world.start_turn()
    fee = math.ceil(world._channel(0, 1)[1].distance * config["transport"]["fee_per_distance"])
    assert world.contracts[0].status == "failed"
    assert world.contracts[0].escrow == 0
    assert world.islands[0].currency == supplier_balance
    assert world.islands[1].currency == customer_balance - fee
    assert world.shipments[0].status == "delivered"
    assert world.islands[1].inventory["meal"] == 2


def test_channel_outage_delays_shipment_but_preserves_owned_reward(config: dict[str, Any]) -> None:
    world = World(config)
    world.islands[0].inventory["timber"] = 1
    finish(world, {0: Action("propose_trade", {"target": 1, "offer": {"timber": 1}})})
    finish(world, {1: Action("accept", {"proposal": 0})})
    shipment = world.shipments[0]
    channel = world._channel(0, 1)[1]
    channel.disabled_until = shipment.arrival_turn + 1
    while world.turn < shipment.arrival_turn - 1:
        finish(world)
    expected_arrival = shipment.arrival_turn
    world.start_turn()
    assert shipment.status == "in_transit"
    assert shipment.arrival_turn == expected_arrival + 1
    assert world.islands[1].inventory["timber"] == 0
    local_value = sum(quantity * config["items"][item]["reward_value"] for item, quantity in world.islands[1].inventory.items())
    assert world._resource_reward(1) == pytest.approx((local_value + 2) * config["rewards"]["scale"])


def test_supplier_death_does_not_destroy_goods_already_owned_by_customer(config: dict[str, Any]) -> None:
    config["transport"]["minimum_transit_turns"] = 2
    world = World(config)
    world.islands[0].inventory["timber"] = 1
    finish(world, {0: Action("propose_trade", {"target": 1, "offer": {"timber": 1}})})
    world.islands[0].health = -100
    finish(world, {1: Action("accept", {"proposal": 0})})
    assert 0 not in world.islands
    assert world.shipments[0].status == "in_transit"
    while world.turn < world.shipments[0].arrival_turn:
        finish(world)
    assert world.islands[1].inventory["timber"] == 1


def test_goods_owned_by_deceased_recipient_are_destroyed(config: dict[str, Any]) -> None:
    world = World(config)
    world.islands[0].inventory["timber"] = 1
    finish(world, {0: Action("propose_trade", {"target": 1, "offer": {"timber": 1}})})
    world.islands[1].health = -100
    finish(world, {1: Action("accept", {"proposal": 0})})
    assert 1 not in world.islands
    assert world.shipments[0].status == "destroyed"


def test_serialization_can_resume_a_frozen_decision_phase(config: dict[str, Any]) -> None:
    world = World(config)
    world.start_turn()
    restored = roundtrip(world)
    actions = {0: Action("manufacture", {"recipe": "cook"})}
    assert world.finish_turn(actions) == restored.finish_turn(actions)
    assert world.snapshot() == restored.snapshot()


def test_meal_nutrition_carries_forward_but_water_is_needed_every_turn(config: dict[str, Any]) -> None:
    world = World(config)
    island = world.islands[0]
    island.inventory.update(grain=0, meal=2, water=4)
    for remaining_meals, reserve in ((1, 1), (1, 0), (0, 1), (0, 0)):
        finish(world)
        assert island.inventory["meal"] == remaining_meals
        assert island.food_reserve == reserve
    assert island.inventory["water"] == 0
    assert not any(event["reason"] == "survival_consumption" and event["item"] == "grain"
                   for event in world.events if event["type"] == "inventory" and event["island"] == 0)
    island.food_reserve = 1
    previous_health = island.health
    finish(world)
    assert island.health <= previous_health - config["survival"]["shortage_damage"]


def test_item_rules_are_isolated_from_the_callers_configuration(config: dict[str, Any]) -> None:
    world = World(config)
    config["item_utility"]["uses"]["tool"]["bonuses"]["productivity"] = 0.9
    world.islands[0].inventory["tool"] = 1
    finish(world, {0: Action("use_item", {"item": "tool"})})
    assert world.item_bonuses(world.islands[0])["productivity"] == 0.5


def test_legacy_rules_preserve_meal_consumption_and_policy_dimensions(config: dict[str, Any]) -> None:
    config["schema_version"] = 1
    del config["trade_initialization"]
    del config["item_utility"]
    del config["exploration"]
    del config["reward_shaping"]
    config["rewards"]["lifetime_score"] = "sum_resource_rewards"
    world = World(config)
    island = world.islands[0]
    island.inventory.update(grain=0, meal=2, water=4, tool=1)
    decisions = world.start_turn()
    assert world.observation_size == 92
    assert world.action_feature_size == 107
    assert not any(action.kind == "use_item" for action in decisions[0].candidates)
    assert not world._execute(0, Action("use_item", {"item": "tool"}))
    world.finish_turn({})
    finish(world)
    assert island.inventory["meal"] == 0
    assert "food_reserve" not in world.state_dict()["islands"][0]
    assert json.loads(json.dumps(world.snapshot())) == json.loads(json.dumps(roundtrip(world).snapshot()))


@pytest.mark.parametrize("item", ["timber", "stone", "iron", "clay", "fiber", "herbs", "salt", "fuel", "cloth", "tool", "medicine"])
def test_every_nonfood_item_has_a_consumable_effect_visible_to_its_policy(config: dict[str, Any], item: str) -> None:
    world = World(config)
    island = world.islands[0]
    island.inventory[item] = 1
    island.health = 60
    island.disease_until = 4
    decisions = world.start_turn()
    action = Action("use_item", {"item": item})
    assert action in decisions[0].candidates
    effect = config["item_utility"]["uses"][item]
    assert world._execute(0, action)
    assert island.inventory[item] == 0
    assert island.available_time == pytest.approx(0.9)
    assert island.health == 60 + effect["health"]
    assert island.disease_until == 4 - effect["disease_turns"]
    for name, amount in effect["bonuses"].items():
        assert world.item_bonuses(island)[name] == amount
    assert world.observe(0) != decisions[0].observation
    assert world.snapshot()["islands"][0]["item_bonuses"] == world.item_bonuses(island)
    usage = next(event for event in world.events if event["type"] == "inventory" and event.get("reason") == "item_use")
    assert usage["item"] == item and usage["delta"] == -1


def test_item_use_respects_reservations_time_health_cap_and_active_effects(config: dict[str, Any]) -> None:
    world = World(config)
    island = world.islands[0]
    island.inventory.update(tool=2, medicine=2)
    world.start_turn()
    assert world._execute(0, Action("propose_trade", {"target": 1, "offer": {"tool": 2}, "request": {}, "price": 0}))
    assert not world._execute(0, Action("use_item", {"item": "tool"}))
    world.proposals[0].status = "cancelled"
    island.available_time = 0
    assert not world._execute(0, Action("use_item", {"item": "tool"}))
    island.available_time = 1
    assert world._execute(0, Action("use_item", {"item": "tool"}))
    assert not world._execute(0, Action("use_item", {"item": "tool"}))
    assert island.inventory["tool"] == 1
    assert not world._execute(0, Action("use_item", {"item": "medicine"}))
    island.health = 99
    assert world._execute(0, Action("use_item", {"item": "medicine"}))
    assert island.health == 100
    assert not world._execute(0, Action("use_item", {"item": "unknown"}))


def test_improvements_expire_without_stacking_and_caps_bound_combined_effects(config: dict[str, Any]) -> None:
    config["item_utility"]["uses"]["tool"]["duration"] = 2
    world = World(config)
    island = world.islands[0]
    island.inventory.update(tool=2, cloth=1, fiber=1)
    world.start_turn()
    for item in ("tool", "cloth", "fiber"):
        assert world._execute(0, Action("use_item", {"item": item}))
    assert world.item_bonuses(island)["comfort"] == 0.8
    world.finish_turn({})
    world.start_turn()
    assert world.item_bonuses(island)["productivity"] == 0.5
    assert not world._can_use_item(0, "tool")
    world.finish_turn({})
    decisions = world.start_turn()
    assert world.item_bonuses(island)["productivity"] == 0
    assert Action("use_item", {"item": "tool"}) in decisions[0].candidates
    assert any(event["type"] == "item_effect_expired" and event["item"] == "tool" for event in world.events)


@pytest.mark.parametrize("item, bonus", [("iron", 0.15), ("tool", 0.5)])
def test_workshop_items_reduce_real_production_time(config: dict[str, Any], item: str, bonus: float) -> None:
    world = World(config)
    island = world.islands[0]
    island.inventory.update({item: 1, "fiber": 2})
    world.start_turn()
    assert world._execute(0, Action("use_item", {"item": item}))
    assert world._execute(0, Action("manufacture", {"recipe": "weave"}))
    assert island.work.remaining == pytest.approx(1.5 / (1 + bonus) - 0.9)
    world.finish_turn({})


@pytest.mark.parametrize("item, expected_gain", [("clay", 2), ("fuel", 3)])
def test_recovery_items_heal_only_when_fed_and_watered(config: dict[str, Any], item: str, expected_gain: float) -> None:
    world = World(config)
    island = world.islands[0]
    island.inventory[item] = 1
    island.health = 50
    finish(world, {0: Action("use_item", {"item": item})})
    assert island.health == 50 + expected_gain
    island.inventory["water"] = 0
    previous_health = island.health
    finish(world)
    assert island.health <= previous_health - config["survival"]["shortage_damage"]


@pytest.mark.parametrize("item, resistance", [("salt", 0.5), ("cloth", 0.2)])
def test_protective_items_reduce_actual_disease_damage(config: dict[str, Any], item: str, resistance: float) -> None:
    world = World(config)
    island = world.islands[0]
    island.inventory[item] = 1
    island.disease_until = 4
    finish(world, {0: Action("use_item", {"item": item})})
    assert island.health == 100 - 3 * (1 - resistance)


@pytest.mark.parametrize("item, comfort", [("fiber", 0.3), ("cloth", 0.6)])
def test_comfort_items_reduce_actual_unhappiness_damage(config: dict[str, Any], item: str, comfort: float) -> None:
    world = World(config)
    island = world.islands[0]
    island.inventory[item] = 1
    island.unhappiness = 0.5
    finish(world, {0: Action("use_item", {"item": item})})
    assert island.health == 100 - 0.5 * (1 - comfort)


@pytest.mark.parametrize("item, expected_loss", [("timber", 3), ("stone", 2)])
def test_shelters_reduce_event_losses_without_creating_resources(config: dict[str, Any], item: str, expected_loss: int) -> None:
    world = World(config)
    world.start_turn()
    for island in world.islands.values():
        island.inventory.update({item: 1, "grain": 10})
        assert world._execute(island.id, Action("use_item", {"item": item}))
    world.config["events"].update(probability_per_turn=1, affected_items=["grain"],
                                  type_weights={"windfall": 0, "loss": 1, "channel_outage": 0, "income_reduction": 0, "disease": 0})
    world._run_event()
    loss = next(event for event in world.events if event.get("reason") == "negative_event")
    assert loss["delta"] == -expected_loss
    assert world.islands[loss["island"]].inventory["grain"] == 10 - expected_loss


def test_item_state_survives_checkpoint_but_is_not_inherited(config: dict[str, Any]) -> None:
    config["reproduction"].update(enabled=True, interval=1, additional_inheritance_factor=0)
    world = World(config)
    island = world.islands[0]
    island.inventory.update(grain=20, water=20, tool=1, meal=1)
    finish(world, {0: Action("use_item", {"item": "tool"})})
    assert island.food_reserve == 1
    child = world.islands[island.descendants[0]]
    assert child.food_reserve == 0 and child.active_effects == {}
    restored = roundtrip(world)
    assert json.loads(json.dumps(restored.snapshot())) == json.loads(json.dumps(world.snapshot()))
    assert restored.start_turn() == world.start_turn()
    restored.finish_turn({})
    world.finish_turn({})
    assert json.loads(json.dumps(restored.state_dict())) == json.loads(json.dumps(world.state_dict()))


@pytest.mark.parametrize("mutation", [
    lambda configuration: configuration["item_utility"]["uses"].pop("stone"),
    lambda configuration: configuration["item_utility"].update(action_time=0),
    lambda configuration: configuration["item_utility"]["bonus_caps"].update(comfort=1),
    lambda configuration: configuration["item_utility"]["uses"]["tool"].update(duration=0),
    lambda configuration: configuration["item_utility"]["uses"]["medicine"].update(health=-1),
    lambda configuration: configuration["item_utility"]["uses"]["herbs"].update(disease_turns=0.5),
    lambda configuration: configuration["item_utility"]["uses"]["fiber"]["bonuses"].update(unknown=1),
])
def test_invalid_or_useless_item_effects_are_rejected(config: dict[str, Any], mutation: Any) -> None:
    mutation(config)
    with pytest.raises(ValueError):
        validate_config(config)
