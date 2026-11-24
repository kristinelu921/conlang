"""Authoritative CPU world with integer inventory, simultaneous observations, and an audit log."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import ITEM_BONUS_NAMES, proportional_resource_counts, survival_bundle, validate_config
from .network import build_neighbor_graph


@dataclass(frozen=True)
class Action:
    kind: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    observation: list[float]
    candidates: list[Action]
    action_features: list[list[float]]


@dataclass
class TurnResult:
    rewards: dict[int, float]
    births: dict[int, int]
    deaths: list[int]
    learning_strengths: dict[int, float]
    reward_components: dict[int, dict[str, float]] = field(default_factory=dict)
    action_results: dict[int, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass
class WorkOrder:
    recipe: str
    remaining: float
    outputs: dict[str, int]
    contaminated: bool = False
    contract_id: int | None = None


@dataclass
class Island:
    id: int
    location: list[float]
    specialization: list[str]
    inventory: dict[str, int]
    contaminated: dict[str, int]
    capabilities: list[str]
    skills: dict[str, float]
    currency: float
    health: float
    parent_id: int | None = None
    descendants: list[int] = field(default_factory=list)
    age: int = 0
    unhappiness: float = 0.0
    available_time: float = 0.0
    work: WorkOrder | None = None
    next_birth_turn: int = 0
    previous_reward: float = 0.0
    lifetime_reward: float = 0.0
    disease_until: int = 0
    income_reduced_until: int = 0
    food_reserve: float = 0.0
    active_effects: dict[str, int] = field(default_factory=dict)
    recent_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Channel:
    source: int
    target: int
    distance: float
    capacity: float
    disabled_until: int = 0


@dataclass
class Proposal:
    id: int
    proposer: int
    acceptor: int
    kind: str
    offer: dict[str, int]
    request: dict[str, int]
    price: float
    created_turn: int
    expires_turn: int
    recipe: str | None = None
    skill: str | None = None
    transfer_capability: bool = False
    status: str = "pending"
    payment_from: str = "acceptor"


@dataclass
class Contract:
    id: int
    proposal_id: int
    supplier: int
    customer: int
    kind: str
    deadline: int
    escrow: float = 0.0
    shipments: list[int] = field(default_factory=list)
    status: str = "active"


@dataclass
class Shipment:
    id: int
    contract_id: int
    owner: int
    source: int
    target: int
    items: dict[str, int]
    contaminated: dict[str, int]
    arrival_turn: int
    status: str = "in_transit"


class World:
    """Resolve world rules while policies remain external holders of island decisions."""

    action_kinds = (
        "wait", "manufacture", "private_request", "public_request", "propose_trade",
        "accept", "reject", "counter", "cancel", "propose_service", "propose_teaching", "propose_transfer",
    )
    proposal_kinds = ("trade", "service", "teaching", "transfer")

    def __init__(self, config: dict[str, Any]) -> None:
        validate_config(config)
        self.config = copy.deepcopy(config)
        self.random = random.Random(config["seed"])
        self.turn = 0
        self.islands: dict[int, Island] = {}
        self.deceased: dict[int, dict[str, Any]] = {}
        self.channels: dict[str, Channel] = {}
        self.proposals: dict[int, Proposal] = {}
        self.contracts: dict[int, Contract] = {}
        self.shipments: dict[int, Shipment] = {}
        self.messages: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.total_births = 0
        self.total_deaths = 0
        self._next_island = config["initialization"]["population"]
        self._next_proposal = 0
        self._next_contract = 0
        self._next_shipment = 0
        self._active = False
        self._turn_actors: list[int] = []
        self._visible_proposals: set[int] = set()
        self._channel_used: dict[str, float] = {}
        self._neighbors: dict[int, set[int]] = {}
        self.limited_neighbors = config["transport"]["topology"] == "resource_neighbors"
        self.history_turns = config.get("memory", {}).get("history_turns", 0)
        self.item_ids = list(config["items"])
        self.recipe_ids = list(config["recipes"])
        self.skill_ids = sorted({recipe["skill"] for recipe in config["recipes"].values()})
        self.capability_ids = sorted({recipe["capability"] for recipe in config["recipes"].values()})
        self.item_utility = self.config.get("item_utility")
        if self.item_utility is not None:
            self.action_kinds = (*self.action_kinds, "use_item")
        self.observation_size = 14 + 5 * len(self.item_ids) + len(self.skill_ids) + len(self.capability_ids)
        if self.item_utility is not None:
            self.observation_size += 1 + len(ITEM_BONUS_NAMES) + len(self.item_ids)
        self.action_feature_size = len(self.action_kinds) + len(self.proposal_kinds) + 5 * len(self.item_ids) + len(self.recipe_ids) + len(self.skill_ids) + 13
        self.initial_action_biases: dict[int, float] = {}
        if "trade_initialization" in self.config:
            exploration = config["exploration"]
            temperature = exploration["initial_temperature"] if exploration["anneal_updates"] else exploration["final_temperature"]
            self.initial_action_biases[self.action_feature_size] = temperature * math.log(config["trade_initialization"]["initial_odds_multiplier"])
            self.action_feature_size += 1
        self.observation_size += self.history_turns * (self.action_feature_size + 5)
        base_ids = [identifier for identifier in self.item_ids if config["items"][identifier]["category"] == "base"]
        population = config["initialization"]["population"]
        bundle = survival_bundle(config)
        producer_assignments: list[str] = []
        if config["initialization"]["specialization_rule"] == "proportional_single_resource":
            counts = proportional_resource_counts(population, config["allocation"]["resource_weights"])
            producer_assignments = [item for item, count in counts.items() for _ in range(count)]
            self.random.shuffle(producer_assignments)
        for island_id in range(population):
            if producer_assignments:
                specialization = [producer_assignments[island_id]]
            else:
                specialization = [identifier for index, identifier in enumerate(base_ids) if index % population == island_id]
                if not specialization:
                    specialization = [base_ids[island_id % len(base_ids)]]
            angle = 2 * math.pi * island_id / population
            radius = config["initialization"]["location_radius"]
            island = Island(
                id=island_id, location=[radius * math.cos(angle), radius * math.sin(angle)],
                specialization=specialization, inventory={identifier: 0 for identifier in self.item_ids},
                contaminated={identifier: 0 for identifier in self.item_ids},
                capabilities=list(config["initialization"]["capabilities"]),
                skills={identifier: 0.0 for identifier in self.skill_ids},
                currency=float(config["initialization"]["currency"]), health=config["survival"]["maximum_health"],
                next_birth_turn=config["reproduction"]["interval"],
            )
            self.islands[island_id] = island
            self._record("founding", island=island_id, currency=island.currency)
            for identifier, quantity in bundle.items():
                self._change_inventory(island_id, identifier, quantity, "initial_survival_endowment")
            for identifier in specialization:
                self._change_inventory(island_id, identifier, config["initialization"]["specialization_quantity"], "initial_specialization_endowment")
        if self.limited_neighbors:
            self._rebuild_neighbors()
            if any(self.missing_neighbor_resources(identifier) for identifier in self.islands):
                raise ValueError("Could not construct founding grain/water neighbor coverage within the degree limit")
        else:
            for island_id in self.islands:
                self._connect(island_id)
        for island in self.islands.values():
            island.previous_reward = self._resource_reward(island.id)

    def _record(self, event_type: str, **fields: Any) -> None:
        self.events.append({"sequence": len(self.events), "turn": self.turn, "type": event_type, **fields})

    def _connect(self, island_id: int) -> None:
        """Add all configured undirected links without replacing existing channels."""
        for neighbor in self.islands.values():
            if neighbor.id == island_id:
                continue
            source, target = sorted((island_id, neighbor.id))
            identifier = f"{source}:{target}"
            if identifier not in self.channels:
                self.channels[identifier] = Channel(
                    source, target, math.dist(self.islands[source].location, self.islands[target].location),
                    self.config["transport"]["capacity"],
                )

    def _channel(self, source: int, target: int) -> tuple[str, Channel]:
        identifier = ":".join(str(value) for value in sorted((source, target)))
        return identifier, self.channels[identifier]

    def neighbors(self, island_id: int) -> list[int]:
        """Return current mutual communication partners without exposing their inventories."""
        if self.limited_neighbors:
            return sorted(self._neighbors.get(island_id, set()) & self.islands.keys())
        return sorted(identifier for identifier in self.islands if identifier != island_id)

    def missing_neighbor_resources(self, island_id: int) -> list[str]:
        """Identify producer access that the surviving network cannot currently provide."""
        available = {item for neighbor in self.neighbors(island_id) for item in self.islands[neighbor].specialization}
        return [item for item in self.config["transport"].get("required_neighbor_resources", []) if item not in available]

    def _rebuild_neighbors(self) -> None:
        """Repair bounded contacts after population changes while preserving existing cargo routes."""
        previous_edges = {(source, target) for source, targets in self._neighbors.items() for target in targets
                          if source < target and source in self.islands and target in self.islands}
        self._neighbors = build_neighbor_graph(
            {identifier: island.specialization[0] for identifier, island in self.islands.items()},
            {identifier: island.location for identifier, island in self.islands.items()},
            self.config["transport"]["maximum_neighbors"], self.config["transport"]["required_neighbor_resources"],
            self.config["seed"] + self.turn * 1_000_003 + self._next_island, previous_edges,
        )
        for source, targets in self._neighbors.items():
            for target in sorted(targets):
                if source >= target:
                    continue
                identifier = f"{source}:{target}"
                if identifier not in self.channels:
                    self.channels[identifier] = Channel(source, target,
                        math.dist(self.islands[source].location, self.islands[target].location), self.config["transport"]["capacity"])
        for proposal in self.proposals.values():
            if proposal.status == "pending" and proposal.acceptor not in self.neighbors(proposal.proposer):
                proposal.status = "cancelled"
                self._record("proposal_cancelled", proposal=proposal.id, reason="no_longer_neighbors")
        self._record("neighbors_updated", neighbors={str(identifier): sorted(targets) for identifier, targets in self._neighbors.items()},
                     missing_resources={str(identifier): self.missing_neighbor_resources(identifier) for identifier in self.islands
                                        if self.missing_neighbor_resources(identifier)})

    def _visible_messages(self, island_id: int) -> list[dict[str, Any]]:
        """Expose only messages addressed to this island within its permitted neighborhood."""
        neighbors = set(self.neighbors(island_id))
        return [message for message in self.messages if message["turn"] < self.turn
                and message["receiver"] in (None, island_id)
                and (not self.limited_neighbors or (message["sender"] in neighbors
                     and (message["receiver"] is not None or island_id in message["recipients"])))]

    def _remember(self, island: Island, action: Action, features: list[float], components: dict[str, float]) -> None:
        """Retain a bounded private record of a chosen action and its realized turn outcome."""
        island.recent_history.append({"turn": self.turn, "action": asdict(action), "action_features": features,
            "reward": components["total"], "health": island.health,
            "food_available": components["food_available"], "water_available": components["water_available"]})
        del island.recent_history[:-self.history_turns]

    def _change_inventory(self, island_id: int, item: str, quantity: int, reason: str, contaminated: int = 0) -> int:
        """Apply a complete inventory ledger entry and return removed contamination."""
        island = self.islands[island_id]
        if quantity < 0:
            if island.inventory[item] < -quantity:
                raise ValueError("Inventory change would create a negative balance")
            removed_contamination = min(island.contaminated[item], -quantity)
            island.contaminated[item] -= removed_contamination
        else:
            removed_contamination = 0
            if not 0 <= contaminated <= quantity:
                raise ValueError("Invalid contamination quantity")
            island.contaminated[item] += contaminated
        island.inventory[item] += quantity
        self._record("inventory", island=island_id, item=item, delta=quantity, reason=reason,
                     contamination_delta=contaminated if quantity >= 0 else -removed_contamination)
        return removed_contamination

    def _change_currency(self, island_id: int, quantity: float, reason: str) -> None:
        """Apply currency changes with no overdraft and a corresponding audit entry."""
        island = self.islands[island_id]
        if island.currency + quantity < -1e-9:
            raise ValueError("Currency change would create a negative balance")
        island.currency = max(0.0, island.currency + quantity)
        self._record("currency", island=island_id, delta=quantity, reason=reason)

    def reserved(self, island_id: int) -> dict[str, int]:
        """Return inventory committed to outstanding proposals for this island."""
        reserved = {identifier: 0 for identifier in self.item_ids}
        for proposal in self.proposals.values():
            if proposal.proposer == island_id and proposal.status == "pending":
                for identifier, quantity in proposal.offer.items():
                    reserved[identifier] += quantity
        return reserved

    def available(self, island_id: int) -> dict[str, int]:
        """Return uncommitted local goods usable for work, survival, or reproduction."""
        reserved = self.reserved(island_id)
        return {identifier: quantity - reserved[identifier] for identifier, quantity in self.islands[island_id].inventory.items()}

    def available_currency(self, island_id: int) -> float:
        """Exclude currency promised in outstanding purchase proposals."""
        committed = sum(proposal.price for proposal in self.proposals.values()
                        if proposal.proposer == island_id and proposal.status == "pending" and proposal.payment_from == "proposer")
        return self.islands[island_id].currency - committed

    def _work_duration(self, island: Island, recipe: dict[str, Any]) -> float:
        """Price a new job using current skill, age, and equipped workshop improvements."""
        aging = self.config["aging"]
        periods = max(0, (island.age - aging["start_turn"]) // aging["interval"] + 1) if island.age >= aging["start_turn"] else 0
        productivity = max(aging["minimum_productivity"], 1 - periods * aging["productivity_decay"])
        speed = (1 + self.config["actions"]["skill_speed_factor"] * island.skills[recipe["skill"]]) * productivity
        speed *= 1 + self.item_bonuses(island)["productivity"]
        return max(self.config["actions"]["minimum_work_time"], recipe["base_time"] / speed)

    def item_bonuses(self, island: Island) -> dict[str, float]:
        """Sum active improvements with a cap on each stat and one effect per item type."""
        bonuses = dict.fromkeys(ITEM_BONUS_NAMES, 0.0)
        if self.item_utility is not None:
            for item, expires_turn in island.active_effects.items():
                if expires_turn >= self.turn:
                    for name, amount in self.item_utility["uses"][item]["bonuses"].items():
                        bonuses[name] += amount
            bonuses = {name: min(amount, self.item_utility["bonus_caps"][name]) for name, amount in bonuses.items()}
        return bonuses

    def _can_use_item(self, island_id: int, item: str) -> bool:
        """Allow useful item applications only from uncommitted stock and available time."""
        if self.item_utility is None or item not in self.item_utility["uses"]:
            return False
        island = self.islands[island_id]
        effect = self.item_utility["uses"][item]
        if self.available(island_id)[item] < 1 or island.available_time < self.item_utility["action_time"]:
            return False
        if island.active_effects.get(item, -1) >= self.turn:
            return False
        return bool(effect["bonuses"] or (effect["health"] and island.health < self.config["survival"]["maximum_health"])
                    or (effect["disease_turns"] and island.disease_until >= self.turn))

    def _use_item(self, island_id: int, item: str) -> bool:
        """Spend one item to heal, shorten illness, or activate a temporary improvement."""
        if not self._can_use_item(island_id, item):
            return False
        island = self.islands[island_id]
        effect = self.item_utility["uses"][item]
        previous_health = island.health
        previous_disease_turns = max(0, island.disease_until - self.turn + 1)
        self._change_inventory(island_id, item, -1, "item_use")
        island.available_time -= self.item_utility["action_time"]
        island.health = min(self.config["survival"]["maximum_health"], island.health + effect["health"])
        if previous_disease_turns:
            island.disease_until = max(self.turn - 1, island.disease_until - effect["disease_turns"])
        if effect["duration"]:
            island.active_effects[item] = self.turn + effect["duration"] - 1
        self._record("item_used", island=island_id, item=item, health_gained=island.health - previous_health,
                     disease_turns_removed=previous_disease_turns - max(0, island.disease_until - self.turn + 1),
                     expires_turn=island.active_effects.get(item), bonuses=dict(effect["bonuses"]))
        return True

    def _can_manufacture(self, island_id: int, recipe_id: str) -> bool:
        island = self.islands[island_id]
        recipe = self.config["recipes"][recipe_id]
        available = self.available(island_id)
        return (
            island.work is None and recipe["capability"] in island.capabilities
            and all(available[item] >= quantity for item, quantity in recipe["inputs"].items())
            and island.available_time > 1e-9
            and (self.config["actions"]["work_carries_forward"] or self._work_duration(island, recipe) <= island.available_time + 1e-9)
        )

    def _start_work(self, island_id: int, recipe_id: str, contract_id: int | None = None) -> bool:
        """Consume inputs before starting one recipe and spend remaining turn time."""
        if not self._can_manufacture(island_id, recipe_id):
            return False
        island = self.islands[island_id]
        recipe = self.config["recipes"][recipe_id]
        contaminated = False
        for item, quantity in recipe["inputs"].items():
            contaminated = bool(self._change_inventory(island_id, item, -quantity, "manufacturing_input")) or contaminated
        island.work = WorkOrder(recipe_id, self._work_duration(island, recipe), dict(recipe["outputs"]), contaminated, contract_id)
        self._record("work_started", island=island_id, recipe=recipe_id, duration=island.work.remaining, contract=contract_id)
        self._advance_work(island_id)
        return True

    def _advance_work(self, island_id: int) -> None:
        """Carry production forward and complete outputs only after all required work."""
        island = self.islands[island_id]
        work = island.work
        if work is None:
            return
        spent = min(work.remaining, island.available_time)
        work.remaining -= spent
        island.available_time -= spent
        if work.remaining > 1e-9:
            return
        recipe = self.config["recipes"][work.recipe]
        if work.contract_id is not None:
            contract = self.contracts[work.contract_id]
            if contract.status != "active":
                island.work = None
                self._record("work_lost", island=island_id, recipe=work.recipe, reason="failed_contract")
                return
            if not self._can_dispatch(island_id, contract.customer, work.outputs):
                return
            self._dispatch(contract, island_id, contract.customer, work.outputs,
                           {item: quantity if work.contaminated and self.config["items"][item]["disease_eligible"] else 0 for item, quantity in work.outputs.items()},
                           manufactured=True)
        else:
            for item, quantity in work.outputs.items():
                contamination = quantity if work.contaminated and self.config["items"][item]["disease_eligible"] else 0
                self._change_inventory(island_id, item, quantity, "manufacturing_output", contamination)
        island.skills[recipe["skill"]] += self.config["actions"]["skill_gain"]
        island.work = None
        self._record("work_completed", island=island_id, recipe=work.recipe, contract=work.contract_id)

    def start_turn(self) -> dict[int, Decision]:
        """Resolve arrivals and exogenous changes, then freeze simultaneous observations."""
        if self._active:
            raise RuntimeError("finish_turn must precede the next start_turn")
        if not self.islands:
            return {}
        self.turn += 1
        self._active = True
        self._channel_used = {}
        self._turn_actors = sorted(self.islands)
        for island in self.islands.values():
            island.available_time = self.config["actions"]["time_budget"]
            for item, expires_turn in list(island.active_effects.items()):
                if expires_turn < self.turn:
                    del island.active_effects[item]
                    self._record("item_effect_expired", island=island.id, item=item)
        for proposal in self.proposals.values():
            if proposal.status == "pending" and proposal.expires_turn < self.turn:
                proposal.status = "expired"
                self._record("proposal_expired", proposal=proposal.id)
        for contract in self.contracts.values():
            if contract.status == "active" and contract.deadline < self.turn:
                self._fail_contract(contract, "deadline")
        self._resolve_arrivals()
        self._allocate()
        self._run_event()
        message_lifetime = self.config["actions"]["message_lifetime"]
        self.messages = [message for message in self.messages if self.turn - message["turn"] <= message_lifetime]
        self._visible_proposals = {identifier for identifier, proposal in self.proposals.items() if proposal.status == "pending"}
        decisions = {}
        for island_id in self._turn_actors:
            candidates = self._candidates(island_id)
            decisions[island_id] = Decision(self.observe(island_id), candidates, [self._action_features(island_id, action) for action in candidates])
        return decisions

    def finish_turn(self, actions: dict[int, Action]) -> TurnResult:
        """Resolve submitted commands, upkeep, death, automatic births, and resource rewards."""
        if not self._active:
            raise RuntimeError("start_turn must precede finish_turn")
        if not set(actions).issubset(self._turn_actors):
            raise ValueError("Actions contain an island outside the current decision snapshot")
        health_before_actions = {identifier: self.islands[identifier].health for identifier in self._turn_actors}
        history_features = {identifier: self._action_features(identifier, actions.get(identifier, Action("wait")))
                            for identifier in self._turn_actors} if self.history_turns else {}
        reward_components: dict[int, dict[str, float]] = {}
        action_results: dict[int, list[dict[str, Any]]] = {}
        actor_order = list(self._turn_actors)
        self.random.shuffle(actor_order)
        for island_id in actor_order:
            self._advance_work(island_id)
        for island_id in actor_order:
            action = actions.get(island_id, Action("wait"))
            commands = [action]
            if action.kind == "plan":
                commands = [Action(**command) for command in action.args["actions"]]
            for command in commands:
                first_event = len(self.events)
                executed = self._execute(island_id, command)
                if not executed:
                    self._record("action_rejected", island=island_id, action=asdict(command))
                action_results.setdefault(island_id, []).append({
                    "action": asdict(command), "executed": executed,
                    "event_sequences": list(range(first_event, len(self.events))),
                })
        for island_id in self._turn_actors:
            self._upkeep(island_id)
        deaths = [island_id for island_id in self._turn_actors if self.islands[island_id].health <= self.config["survival"]["death_threshold"]]
        for island_id in deaths:
            if "reward_shaping" in self.config:
                components = self._learning_reward_components(island_id, health_before_actions[island_id], terminal=True)
                reward_components[island_id] = components
                self.islands[island_id].lifetime_reward += components["total"]
                if self.history_turns:
                    self._remember(self.islands[island_id], actions.get(island_id, Action("wait")), history_features[island_id], components)
            self._die(island_id)
        births: dict[int, int] = {}
        if self.config["reproduction"]["enabled"]:
            for island_id in list(self.islands):
                child = self._try_birth(island_id)
                if child is not None:
                    births[child] = island_id
        rewards = {island_id: self._resource_reward(island_id) for island_id in self.islands}
        strengths = {}
        for island_id, reward in rewards.items():
            island = self.islands[island_id]
            # Keep unhappiness tied to resource shortfalls so reward penalties cannot amplify health damage.
            if island_id not in births:
                shortfall = max(0.0, (island.previous_reward - reward) / max(1e-6, island.previous_reward))
                smoothing = self.config["survival"]["unhappiness_smoothing"]
                island.unhappiness = (1 - smoothing) * island.unhappiness + smoothing * min(1.0, shortfall)
            island.previous_reward = reward
            if "reward_shaping" in self.config:
                if island_id not in births:
                    components = self._learning_reward_components(island_id, health_before_actions[island_id])
                    reward_components[island_id] = components
                    rewards[island_id] = components["total"]
                    island.lifetime_reward += components["total"]
                    if self.history_turns:
                        self._remember(island, actions.get(island_id, Action("wait")), history_features[island_id], components)
            else:
                island.lifetime_reward += reward
            strengths[island_id] = 1 + self.config["survival"]["learning_strength_gain"] * island.unhappiness
        for island_id in deaths:
            rewards[island_id] = reward_components[island_id]["total"] if island_id in reward_components else 0.0
            strengths[island_id] = 1.0
        for island_id, components in reward_components.items():
            self._record("learning_reward", island=island_id, **components)
        if self.limited_neighbors and (births or deaths):
            self._rebuild_neighbors()
        self._record("turn_completed", rewards=rewards, births=births, deaths=deaths, population=len(self.islands))
        self._active = False
        return TurnResult(rewards, births, deaths, strengths, reward_components, action_results)

    def _allocate(self) -> None:
        """Draw resource gifts globally or once per configured allocation for each living island."""
        configuration = self.config["allocation"]
        if self.turn % configuration["turns_per_minute"]:
            return
        resources = list(configuration["resource_weights"])
        recipients: list[Island | None] = [None] * configuration["allocations_per_minute"]
        if configuration.get("scope", "global") == "per_island":
            recipients = [island for island in self.islands.values() for _ in range(configuration["allocations_per_minute"])]
        for recipient in recipients:
            if recipient is None:
                item = self.random.choices(resources, weights=list(configuration["resource_weights"].values()), k=1)[0]
                eligible = [island for island in self.islands.values() if item in island.specialization]
                if not eligible:
                    self._record("allocation_skipped", item=item, reason="no_eligible_island")
                    continue
                island = self.random.choice(eligible)
            else:
                island = recipient
                item = self.random.choices(island.specialization,
                                           weights=[configuration["resource_weights"][resource] for resource in island.specialization], k=1)[0]
            quantity = self.random.choices(configuration["quantities"], weights=configuration["quantity_weights"], k=1)[0]
            if island.income_reduced_until >= self.turn:
                quantity = math.floor(quantity * self.config["events"]["income_multiplier"])
            self._change_inventory(island.id, item, quantity, "world_allocation")

    def _run_event(self) -> None:
        """Apply a reproducible configured event without touching committed inventory."""
        events = self.config["events"]
        if self.random.random() >= events["probability_per_turn"]:
            return
        event_type = self.random.choices(list(events["type_weights"]), weights=list(events["type_weights"].values()), k=1)[0]
        aging = self.config["aging"]
        islands = list(self.islands.values())
        susceptibility = [1 + max(0, island.age - aging["start_turn"]) / aging["interval"] * aging["event_susceptibility"] for island in islands]
        island = self.random.choice(islands) if event_type == "windfall" else self.random.choices(islands, weights=susceptibility, k=1)[0]
        duration = events["duration"]
        if event_type == "windfall":
            item = self.random.choice(events["windfall_items"])
            self._change_inventory(island.id, item, events["resource_quantity"], "positive_event")
        elif event_type == "loss":
            item = self.random.choice(events["affected_items"])
            unprotected_loss = min(events["resource_quantity"], self.available(island.id)[item])
            quantity = math.ceil(unprotected_loss * (1 - self.item_bonuses(island)["loss_protection"]))
            self._change_inventory(island.id, item, -quantity, "negative_event")
        elif event_type == "income_reduction":
            island.income_reduced_until = self.turn + duration - 1
        elif event_type == "disease":
            island.disease_until = self.turn + duration - 1
        else:
            incident = [channel for channel in self.channels.values() if island.id in (channel.source, channel.target)
                        and channel.source in self.islands and channel.target in self.islands
                        and (not self.limited_neighbors or channel.target in self._neighbors[channel.source])]
            if incident:
                self.random.choice(incident).disabled_until = self.turn + duration - 1
        self._record("world_event", event=event_type, island=island.id, duration=duration)

    def _execute(self, island_id: int, action: Action) -> bool:
        """Validate a structured command against current resources and frozen proposal visibility."""
        island = self.islands[island_id]
        arguments = action.args
        if action.kind == "wait":
            return True
        if action.kind == "use_item":
            item = arguments.get("item")
            return isinstance(item, str) and self._use_item(island_id, item)
        if action.kind == "manufacture":
            recipe_id = arguments.get("recipe")
            if recipe_id not in self.config["recipes"]:
                return False
            batches = arguments.get("batches", 1)
            if not isinstance(batches, int) or batches < 1:
                return False
            completed = False
            for _ in range(batches):
                if not self._start_work(island_id, recipe_id):
                    break
                completed = True
                if island.work is not None:
                    break
            return completed
        if action.kind in {"private_request", "public_request"}:
            target = arguments.get("target") if action.kind == "private_request" else None
            item = arguments.get("item")
            cost = self.config["actions"]["communication_time"]
            if item not in self.item_ids or (action.kind == "private_request" and target not in self.neighbors(island_id)) or island.available_time < cost:
                return False
            island.available_time -= cost
            message = {"turn": self.turn, "sender": island_id, "receiver": target, "item": item, "intent": "request"}
            if self.limited_neighbors and target is None:
                message["recipients"] = self.neighbors(island_id)
            self.messages.append(message)
            self._record("message", **message)
            return True
        if action.kind.startswith("propose_") or action.kind == "counter":
            return self._propose(island_id, action)
        if action.kind in {"accept", "reject", "cancel"}:
            identifier = arguments.get("proposal")
            proposal = self.proposals.get(identifier)
            if identifier not in self._visible_proposals or proposal is None or proposal.status != "pending":
                return False
            if action.kind == "cancel" and proposal.proposer == island_id:
                cost = self.config["actions"]["communication_time"]
                if island.available_time < cost:
                    return False
                island.available_time -= cost
                proposal.status = "cancelled"
                self._record("proposal_cancelled", proposal=identifier)
                return True
            if proposal.acceptor != island_id:
                return False
            if action.kind == "reject":
                cost = self.config["actions"]["communication_time"]
                if island.available_time < cost:
                    return False
                island.available_time -= cost
                proposal.status = "rejected"
                self._record("proposal_rejected", proposal=identifier)
                return True
            if action.kind == "accept":
                return self._accept(proposal)
        return False

    def _propose(self, island_id: int, action: Action) -> bool:
        """Create an explicitly priced proposal and reserve the supplier's promised inputs."""
        arguments = action.args
        countered: Proposal | None = None
        if action.kind == "counter":
            countered = self.proposals.get(arguments.get("proposal"))
            if countered is None or countered.id not in self._visible_proposals or countered.status != "pending" or countered.acceptor != island_id:
                return False
            target = countered.proposer
            kind = "trade"
        else:
            target = arguments.get("target")
            kind = {"propose_trade": "trade", "propose_service": "service", "propose_teaching": "teaching", "propose_transfer": "transfer"}.get(action.kind)
        if target not in self.neighbors(island_id) or kind is None:
            return False
        island = self.islands[island_id]
        cost = self.config["actions"]["communication_time"]
        if island.available_time < cost:
            return False
        price = arguments.get("price", 0)
        if not isinstance(price, (int, float)) or not math.isfinite(price) or price < 0:
            return False
        payment_from = arguments.get("payment_from", "acceptor")
        if payment_from not in {"acceptor", "proposer"} or (kind != "trade" and payment_from != "acceptor"):
            return False
        if payment_from == "proposer" and self.available_currency(island_id) < price:
            return False
        offer = arguments.get("offer", {})
        request = arguments.get("request", {})
        recipe_id = arguments.get("recipe")
        skill = arguments.get("skill")
        if kind == "service":
            recipe = self.config["recipes"].get(recipe_id)
            if recipe is None or recipe["capability"] not in island.capabilities:
                return False
            offer = dict(recipe["inputs"])
            request = {}
        elif kind in {"teaching", "transfer"}:
            if skill not in island.skills or island.skills[skill] <= 0:
                return False
            if kind == "transfer" and island.skills[skill] < self.config["actions"]["transfer_units"]:
                return False
            offer, request = {}, {}
        if not isinstance(offer, dict) or not isinstance(request, dict):
            return False
        if any(item not in self.item_ids or not isinstance(quantity, int) or quantity <= 0 for item, quantity in list(offer.items()) + list(request.items())):
            return False
        available = self.available(island_id)
        if any(available[item] < quantity for item, quantity in offer.items()):
            return False
        if kind == "trade" and not offer and not request:
            return False
        if countered is not None:
            countered.status = "countered"
            self._record("proposal_countered", proposal=countered.id)
        proposal = Proposal(
            self._next_proposal, island_id, target, kind, dict(offer), dict(request), float(price),
            self.turn, self.turn + self.config["actions"]["proposal_lifetime"], recipe_id, skill,
            bool(arguments.get("transfer_capability", False)),
            payment_from=payment_from,
        )
        self._next_proposal += 1
        self.proposals[proposal.id] = proposal
        island.available_time -= cost
        self._record("proposal", proposal=asdict(proposal))
        return True

    def _can_accept_visible(self, proposal: Proposal) -> bool:
        """Mask acceptance only using disclosed terms, public channels, and the acceptor's state."""
        if proposal.proposer not in self.islands or proposal.acceptor not in self.islands:
            return False
        if proposal.acceptor not in self.neighbors(proposal.proposer):
            return False
        customer = self.islands[proposal.acceptor]
        _, channel = self._channel(proposal.proposer, customer.id)
        fee = math.ceil(channel.distance * self.config["transport"]["fee_per_distance"])
        extra_cost = self.config["actions"]["teaching_cost"] if proposal.kind == "teaching" else self.config["actions"]["transfer_cost"] if proposal.kind == "transfer" else 0
        payment = proposal.price if proposal.payment_from == "acceptor" else 0
        if self.available_currency(customer.id) < fee + payment + extra_cost or customer.available_time < self.config["actions"]["agreement_time"]:
            return False
        if channel.disabled_until >= self.turn:
            return False
        if proposal.kind == "trade":
            available = self.available(customer.id)
            combined = {item: proposal.offer.get(item, 0) + proposal.request.get(item, 0) for item in self.item_ids}
            return all(available[item] >= quantity for item, quantity in proposal.request.items()) and self._can_dispatch(proposal.proposer, customer.id, combined)
        if proposal.kind in {"teaching", "transfer"}:
            duration = self.config["actions"]["teaching_time" if proposal.kind == "teaching" else "transfer_time"]
            return customer.available_time >= duration + self.config["actions"]["agreement_time"]
        return True

    def _can_accept(self, proposal: Proposal) -> bool:
        """Revalidate private supplier resources atomically only when resolving an agreement."""
        if not self._can_accept_visible(proposal):
            return False
        supplier = self.islands[proposal.proposer]
        customer = self.islands[proposal.acceptor]
        if any(supplier.inventory[item] < quantity for item, quantity in proposal.offer.items()):
            return False
        if proposal.payment_from == "proposer" and supplier.currency < proposal.price:
            return False
        if proposal.kind == "trade":
            return True
        if proposal.kind == "service":
            recipe = self.config["recipes"][proposal.recipe]
            duration = self._work_duration(supplier, recipe)
            _, channel = self._channel(supplier.id, customer.id)
            output_load = sum(quantity * self.config["items"][item]["transport_load"] for item, quantity in recipe["outputs"].items())
            return output_load <= channel.capacity and supplier.work is None and recipe["capability"] in supplier.capabilities and supplier.available_time > 0 and (self.config["actions"]["work_carries_forward"] or duration <= supplier.available_time)
        duration = self.config["actions"]["teaching_time" if proposal.kind == "teaching" else "transfer_time"]
        if min(supplier.available_time, customer.available_time - self.config["actions"]["agreement_time"]) < duration:
            return False
        skill = proposal.skill
        if proposal.kind == "teaching":
            return supplier.skills[skill] > customer.skills[skill]
        if supplier.skills[skill] < self.config["actions"]["transfer_units"]:
            return False
        if proposal.transfer_capability:
            capabilities = {recipe["capability"] for recipe in self.config["recipes"].values() if recipe["skill"] == skill}
            return capabilities.issubset(supplier.capabilities) and not any(
                other.status == "pending" and other.id != proposal.id and other.proposer == supplier.id
                and other.kind == "service" and self.config["recipes"][other.recipe]["capability"] in capabilities
                for other in self.proposals.values()
            ) and supplier.work is None
        return True

    def _accept(self, proposal: Proposal) -> bool:
        """Charge the acceptor's fee and execute an atomic agreement or escrowed service."""
        if not self._can_accept(proposal):
            return False
        supplier = self.islands[proposal.proposer]
        customer = self.islands[proposal.acceptor]
        _, channel = self._channel(supplier.id, customer.id)
        fee = math.ceil(channel.distance * self.config["transport"]["fee_per_distance"])
        self._change_currency(customer.id, -fee, "channel_fee")
        customer.available_time -= self.config["actions"]["agreement_time"]
        proposal.status = "accepted"
        contract = Contract(self._next_contract, proposal.id, supplier.id, customer.id, proposal.kind,
                            self.turn + self.config["actions"]["contract_deadline"])
        self._next_contract += 1
        self.contracts[contract.id] = contract
        payer = customer.id if proposal.payment_from == "acceptor" else supplier.id
        payee = supplier.id if proposal.payment_from == "acceptor" else customer.id
        self._change_currency(payer, -proposal.price, "service_escrow" if proposal.kind == "service" else "trade_payment")
        if proposal.kind == "service":
            contract.escrow = proposal.price
            if not self._start_work(supplier.id, proposal.recipe, contract.id):
                raise RuntimeError("Validated service could not start")
        else:
            self._change_currency(payee, proposal.price, "trade_payment")
            if proposal.kind == "trade":
                if proposal.offer:
                    self._dispatch(contract, supplier.id, customer.id, proposal.offer)
                if proposal.request:
                    self._dispatch(contract, customer.id, supplier.id, proposal.request)
            else:
                duration_key = "teaching_time" if proposal.kind == "teaching" else "transfer_time"
                duration = self.config["actions"][duration_key]
                supplier.available_time -= duration
                customer.available_time -= duration
                skill = proposal.skill
                if proposal.kind == "teaching":
                    customer.skills[skill] = min(supplier.skills[skill], customer.skills[skill] + self.config["actions"]["teaching_gain"])
                    self._change_currency(customer.id, -self.config["actions"]["teaching_cost"], "teaching_cost")
                else:
                    units = self.config["actions"]["transfer_units"]
                    supplier.skills[skill] -= units
                    customer.skills[skill] += units
                    self._change_currency(customer.id, -self.config["actions"]["transfer_cost"], "skill_transfer_cost")
                    if proposal.transfer_capability:
                        capabilities = {recipe["capability"] for recipe in self.config["recipes"].values() if recipe["skill"] == skill}
                        for capability in sorted(capabilities):
                            supplier.capabilities.remove(capability)
                            if capability not in customer.capabilities:
                                customer.capabilities.append(capability)
                contract.status = "completed"
                self._record("skill_transaction", contract=contract.id, kind=proposal.kind, supplier=supplier.id, customer=customer.id, skill=skill)
        self._record("contract_accepted", contract=asdict(contract), channel_fee=fee)
        return True

    def _can_dispatch(self, source: int, target: int, items: dict[str, int]) -> bool:
        identifier, channel = self._channel(source, target)
        load = sum(self.config["items"][item]["transport_load"] * quantity for item, quantity in items.items())
        return channel.disabled_until < self.turn and self._channel_used.get(identifier, 0) + load <= channel.capacity + 1e-9

    def _dispatch(self, contract: Contract, source: int, target: int, items: dict[str, int],
                  contamination: dict[str, int] | None = None, manufactured: bool = False) -> None:
        """Move unique ownership to the recipient and occupy shared bidirectional capacity."""
        identifier, channel = self._channel(source, target)
        contamination = dict(contamination or {})
        if not manufactured:
            contamination = {item: self._change_inventory(source, item, -quantity, "trade_dispatch") for item, quantity in items.items()}
        load = sum(self.config["items"][item]["transport_load"] * quantity for item, quantity in items.items())
        self._channel_used[identifier] = self._channel_used.get(identifier, 0) + load
        transit = max(self.config["transport"]["minimum_transit_turns"], math.ceil(channel.distance / self.config["transport"]["distance_per_turn"]))
        shipment = Shipment(self._next_shipment, contract.id, target, source, target, dict(items), contamination, self.turn + transit)
        self.shipments[shipment.id] = shipment
        self._next_shipment += 1
        contract.shipments.append(shipment.id)
        self._record("shipment_dispatched", shipment=asdict(shipment), manufactured=manufactured)

    def _resolve_arrivals(self) -> None:
        """Deliver recipient-owned cargo, delaying it while its channel is disabled."""
        for shipment in self.shipments.values():
            if shipment.status != "in_transit" or shipment.arrival_turn > self.turn:
                continue
            if shipment.target not in self.islands:
                shipment.status = "destroyed"
                self._record("shipment_destroyed", shipment=shipment.id, reason="owner_deceased")
                continue
            _, channel = self._channel(shipment.source, shipment.target)
            if channel.disabled_until >= self.turn:
                shipment.arrival_turn += 1
                continue
            for item, quantity in shipment.items.items():
                contamination = shipment.contaminated.get(item, 0)
                if self.config["items"][item]["disease_eligible"] and self.random.random() < self.config["events"]["incoming_contamination_probability"]:
                    contamination = quantity
                self._change_inventory(shipment.target, item, quantity, "trade_receipt", contamination)
            shipment.status = "delivered"
            self._record("shipment_arrived", shipment=shipment.id, owner=shipment.owner)
            contract = self.contracts[shipment.contract_id]
            if contract.status == "active" and all(self.shipments[identifier].status == "delivered" for identifier in contract.shipments):
                contract.status = "completed"
                if contract.escrow:
                    self._change_currency(contract.supplier, contract.escrow, "service_payment")
                    contract.escrow = 0.0
                self._record("contract_completed", contract=contract.id)

    def _fail_contract(self, contract: Contract, reason: str) -> None:
        """Refund unearned service escrow while preserving already-dispatched ownership."""
        contract.status = "failed"
        if contract.escrow and contract.customer in self.islands:
            self._change_currency(contract.customer, contract.escrow, "service_refund")
        elif contract.escrow:
            self._record("currency_destroyed", owner=contract.customer, amount=contract.escrow, reason="dead_escrow_owner")
        contract.escrow = 0.0
        supplier = self.islands.get(contract.supplier)
        if supplier is not None and supplier.work is not None and supplier.work.contract_id == contract.id:
            self._record("work_lost", island=supplier.id, recipe=supplier.work.recipe, reason=reason)
            supplier.work = None
        self._record("contract_failed", contract=contract.id, reason=reason)

    def _upkeep(self, island_id: int) -> None:
        """Consume usable uncommitted necessities and apply gradual health and age changes."""
        island = self.islands[island_id]
        survival = self.config["survival"]
        bonuses = self.item_bonuses(island)
        shortage = 0.0
        for necessity, required in survival["consumption"].items():
            remaining = float(required)
            if necessity == "food" and self.item_utility is not None:
                stored_food = min(remaining, island.food_reserve)
                island.food_reserve -= stored_food
                remaining -= stored_food
            edible = sorted((item for item in self.item_ids if self.config["items"][item]["survival"].get(necessity, 0) > 0),
                            key=lambda item: (-self.config["items"][item]["survival"][necessity], item))
            for item in edible:
                contribution = self.config["items"][item]["survival"][necessity]
                quantity = min(self.available(island_id)[item], math.ceil(max(0.0, remaining) / contribution))
                if quantity:
                    contamination = self._change_inventory(island_id, item, -quantity, "survival_consumption")
                    remaining -= quantity * contribution
                    if contamination:
                        island.disease_until = max(island.disease_until, self.turn + survival["disease_duration"] - 1)
            if necessity == "food" and self.item_utility is not None:
                island.food_reserve += max(0.0, -remaining)
            shortage += max(0.0, remaining) / required
        if shortage:
            island.health -= shortage * survival["shortage_damage"]
        else:
            island.health = min(survival["maximum_health"], island.health + survival["recovery"] + bonuses["recovery"])
        island.health -= island.unhappiness * survival["unhappiness_damage"] * (1 - bonuses["comfort"])
        if island.disease_until >= self.turn:
            island.health -= survival["disease_damage"] * (1 - bonuses["disease_resistance"])
        island.age += 1
        aging = self.config["aging"]
        if island.age >= aging["start_turn"]:
            island.health -= aging["health_damage"]
            if (island.age - aging["start_turn"]) % aging["interval"] == 0:
                island.skills = {skill: value * aging["skill_retention"] for skill, value in island.skills.items()}
        island.health = max(0.0, island.health)

    def _die(self, island_id: int) -> None:
        """Archive one identity and explicitly destroy its remaining owned assets."""
        island = self.islands[island_id]
        for proposal in self.proposals.values():
            if proposal.status == "pending" and island_id in (proposal.proposer, proposal.acceptor):
                proposal.status = "cancelled"
                self._record("proposal_cancelled", proposal=proposal.id, reason="party_deceased")
        for contract in self.contracts.values():
            if contract.status == "active" and island_id in (contract.supplier, contract.customer):
                self._fail_contract(contract, "party_deceased")
        self.deceased[island_id] = {**self._island_state(island), "death_turn": self.turn}
        if self.item_utility is not None:
            self.deceased[island_id]["item_bonuses"] = self.item_bonuses(island)
        for item, quantity in list(island.inventory.items()):
            if quantity:
                self._change_inventory(island_id, item, -quantity, "deceased_asset_destruction")
        if island.currency:
            self._change_currency(island_id, -island.currency, "deceased_asset_destruction")
        for shipment in self.shipments.values():
            if shipment.owner == island_id and shipment.status == "in_transit":
                shipment.status = "destroyed"
                self._record("shipment_destroyed", shipment=shipment.id, reason="owner_deceased")
        if island.work is not None:
            self._record("work_lost", island=island_id, recipe=island.work.recipe, reason="owner_deceased")
        del self.islands[island_id]
        self.total_deaths += 1
        self._record("death", island=island_id, lifetime_reward=island.lifetime_reward)

    def _try_birth(self, parent_id: int) -> int | None:
        """Fund an automatic child without creating resources or retaining a parent reserve."""
        parent = self.islands[parent_id]
        if parent.next_birth_turn > self.turn:
            return None
        bundle = survival_bundle(self.config)
        available = self.available(parent_id)
        if any(available[item] < quantity for item, quantity in bundle.items()):
            return None
        reproduction = self.config["reproduction"]
        transfer = {}
        for item, quantity in available.items():
            minimum = bundle.get(item, 0)
            extra = math.floor(reproduction["additional_inheritance_factor"] * math.sqrt(quantity - minimum))
            transfer[item] = minimum + min(quantity - minimum, extra)
        identifier = self._next_island
        self._next_island += 1
        child = Island(
            id=identifier, location=[coordinate + self.random.uniform(-reproduction["location_jitter"], reproduction["location_jitter"]) for coordinate in parent.location],
            specialization=list(parent.specialization), inventory={item: 0 for item in self.item_ids}, contaminated={item: 0 for item in self.item_ids},
            capabilities=list(parent.capabilities) if reproduction["inherit_capabilities"] else [],
            skills=dict(parent.skills) if reproduction["inherit_skills"] else {skill: 0.0 for skill in self.skill_ids},
            currency=0.0, health=self.config["survival"]["maximum_health"], parent_id=parent_id,
            next_birth_turn=self.turn + reproduction["interval"],
        )
        self.islands[identifier] = child
        for item, quantity in transfer.items():
            if quantity:
                contamination = self._change_inventory(parent_id, item, -quantity, "child_transfer_out")
                self._change_inventory(identifier, item, quantity, "child_transfer_in", contamination)
        ancestor: int | None = parent_id
        while ancestor is not None:
            if ancestor in self.islands:
                self.islands[ancestor].descendants.append(identifier)
                ancestor = self.islands[ancestor].parent_id
            else:
                self.deceased[ancestor]["descendants"].append(identifier)
                ancestor = self.deceased[ancestor]["parent_id"]
        parent.next_birth_turn = self.turn + reproduction["interval"]
        if not self.limited_neighbors:
            self._connect(identifier)
        self.total_births += 1
        self._record("birth", child=identifier, parent=parent_id, transfer=transfer)
        return identifier

    def _resource_reward(self, island_id: int) -> float:
        """Value each currently owned physical unit once, including recipient-owned transit."""
        island = self.islands[island_id]
        value = sum(quantity * self.config["items"][item]["reward_value"] for item, quantity in island.inventory.items())
        value += sum(quantity * self.config["items"][item]["reward_value"] for shipment in self.shipments.values()
                     if shipment.owner == island_id and shipment.status == "in_transit" for item, quantity in shipment.items.items())
        currency = island.currency + sum(contract.escrow for contract in self.contracts.values() if contract.customer == island_id)
        return (value + currency * self.config["rewards"]["currency_value"]) * self.config["rewards"]["scale"]

    def _learning_reward_components(self, island_id: int, previous_health: float, terminal: bool = False) -> dict[str, float]:
        """Deduct empty-necessity and net-health-loss costs without changing resource accounting."""
        island = self.islands[island_id]
        settings = self.config["reward_shaping"]
        available = self.available(island_id)
        food_available = island.food_reserve + sum(
            available[item] * self.config["items"][item]["survival"].get("food", 0) for item in self.item_ids
        )
        health_lost = max(0.0, previous_health - island.health)
        resource_reward = self.config["rewards"]["terminal_reward"] if terminal else self._resource_reward(island_id)
        food_penalty = -settings["empty_food_penalty"] if food_available <= 0 else 0.0
        health_penalty = -settings["health_loss_penalty_per_point"] * health_lost
        components = {
            "resource_reward": resource_reward, "food_available": food_available, "health_lost": health_lost,
            "empty_food_penalty": food_penalty, "health_loss_penalty": health_penalty,
            "total": resource_reward + food_penalty + health_penalty,
        }
        if self.config["schema_version"] >= 5:
            water_available = sum(
                available[item] * self.config["items"][item]["survival"].get("water", 0) for item in self.item_ids
            )
            water_penalty = -settings["empty_water_penalty"] if water_available <= 0 else 0.0
            components.update(water_available=water_available, empty_water_penalty=water_penalty)
            components["total"] += water_penalty
        return components

    def observe(self, island_id: int) -> list[float]:
        """Encode private state and deliberately public signals without other islands' inventories."""
        island = self.islands[island_id]
        reserved = self.reserved(island_id)
        messages = self._visible_messages(island_id)
        observations = [
            math.tanh(self.turn / 100), math.tanh(island.age / 100), island.health / self.config["survival"]["maximum_health"],
            island.unhappiness, math.tanh(island.currency / 100), island.available_time / self.config["actions"]["time_budget"],
            math.tanh(len(self.islands) / 20), float(island.parent_id is not None),
            float(island.disease_until >= self.turn and self.turn > 0),
            float(island.income_reduced_until >= self.turn and self.turn > 0),
            math.tanh(island.work.remaining if island.work is not None else 0),
            math.tanh(max(0, island.next_birth_turn - self.turn) / 20),
            math.tanh(sum(proposal.status == "pending" and proposal.acceptor == island_id for proposal in self.proposals.values()) / 10),
            math.tanh(island.previous_reward),
        ]
        observations.extend(math.tanh(island.inventory[item] / 16) for item in self.item_ids)
        observations.extend(math.tanh(reserved[item] / 16) for item in self.item_ids)
        observations.extend(math.tanh(island.contaminated[item] / 16) for item in self.item_ids)
        observations.extend(float(item in island.specialization) for item in self.item_ids)
        observations.extend(math.tanh(sum(message["item"] == item for message in messages) / 5) for item in self.item_ids)
        observations.extend(math.tanh(island.skills[skill]) for skill in self.skill_ids)
        observations.extend(float(capability in island.capabilities) for capability in self.capability_ids)
        if self.item_utility is not None:
            observations.append(math.tanh(island.food_reserve / self.config["survival"]["consumption"]["food"]))
            bonuses = self.item_bonuses(island)
            observations.extend(bonuses[name] / self.item_utility["bonus_caps"][name] for name in ITEM_BONUS_NAMES)
            observations.extend(math.tanh(max(0, island.active_effects.get(item, -1) - self.turn + 1) / 10) for item in self.item_ids)
        if self.history_turns:
            observations.extend([0.0] * ((self.history_turns - len(island.recent_history)) * (self.action_feature_size + 5)))
            for experience in island.recent_history:
                observations.extend([1.0, *experience["action_features"], math.tanh(experience["reward"] / 10),
                    experience["health"] / self.config["survival"]["maximum_health"],
                    math.tanh(experience["food_available"] / 16), math.tanh(experience["water_available"] / 16)])
        return observations

    def _candidates(self, island_id: int) -> list[Action]:
        """Offer finite structured choices without conditioning on a neighbor's hidden stock."""
        island = self.islands[island_id]
        available = self.available(island_id)
        flexible_quantities = self.config["actions"].get("trade_quantity_mode") == "all_feasible"
        quantities = [1] if flexible_quantities else self.config["actions"]["trade_quantities"]
        candidates = [Action("wait")]
        if self.item_utility is not None:
            candidates.extend(Action("use_item", {"item": item}) for item in self.item_ids if self._can_use_item(island_id, item))
        for recipe_id in self.recipe_ids:
            if self._can_manufacture(island_id, recipe_id):
                candidates.append(Action("manufacture", {"recipe": recipe_id}))
        for proposal in self.proposals.values():
            if proposal.id not in self._visible_proposals or proposal.status != "pending":
                continue
            if proposal.proposer == island_id:
                candidates.append(Action("cancel", {"proposal": proposal.id}))
            if proposal.acceptor == island_id:
                candidates.append(Action("reject", {"proposal": proposal.id}))
                if self._can_accept_visible(proposal):
                    candidates.append(Action("accept", {"proposal": proposal.id}))
                counter_offer = {item: 1 if flexible_quantities else quantity for item, quantity in proposal.request.items()}
                counter_request = {item: 1 if flexible_quantities else quantity for item, quantity in proposal.offer.items()}
                if proposal.kind == "trade" and all(available[item] >= quantity for item, quantity in counter_offer.items()):
                    counter = Action("counter", {"proposal": proposal.id, "offer": counter_offer, "request": counter_request, "price": 0,
                        **({"payment_from": "acceptor"} if flexible_quantities else {})})
                    if not flexible_quantities or (island.available_time >= self.config["actions"]["communication_time"] and self._trade_template_fits(island_id, counter)):
                        candidates.append(counter)
        communication_time = self.config["actions"]["communication_time"]
        if island.available_time < communication_time:
            return candidates
        for item in self.item_ids:
            candidates.append(Action("public_request", {"item": item}))
        for target in self.neighbors(island_id):
            for wanted in self.item_ids:
                candidates.append(Action("private_request", {"target": target, "item": wanted}))
                for quantity in quantities:
                    price = math.ceil(quantity * self.config["items"][wanted]["reward_value"] * self.config["actions"]["purchase_price_per_value"])
                    if self.available_currency(island_id) >= price:
                        candidates.append(Action("propose_trade", {"target": target, "offer": {}, "request": {wanted: quantity}, "price": price, "payment_from": "proposer"}))
                for offered in self.item_ids:
                    if offered == wanted:
                        continue
                    for quantity in quantities:
                        if available[offered] >= quantity:
                            candidates.append(Action("propose_trade", {"target": target, "offer": {offered: quantity}, "request": {wanted: quantity}, "price": 0}))
            for recipe_id in self.recipe_ids:
                recipe = self.config["recipes"][recipe_id]
                if recipe["capability"] in island.capabilities and all(available[item] >= quantity for item, quantity in recipe["inputs"].items()):
                    candidates.append(Action("propose_service", {"target": target, "recipe": recipe_id, "price": self.config["actions"]["service_price"]}))
            for skill in self.skill_ids:
                if island.skills[skill] > 0:
                    candidates.append(Action("propose_teaching", {"target": target, "skill": skill, "price": self.config["actions"]["service_price"]}))
                if island.skills[skill] >= self.config["actions"]["transfer_units"]:
                    candidates.append(Action("propose_transfer", {"target": target, "skill": skill, "price": self.config["actions"]["service_price"]}))
                    capabilities = {recipe["capability"] for recipe in self.config["recipes"].values() if recipe["skill"] == skill}
                    if capabilities.issubset(island.capabilities):
                        candidates.append(Action("propose_transfer", {"target": target, "skill": skill, "price": self.config["actions"]["service_price"], "transfer_capability": True}))
        if flexible_quantities:
            candidates = [action for action in candidates if action.kind not in {"propose_trade", "counter"} or self._trade_template_fits(island_id, action)]
        return candidates

    def _trade_template_fits(self, island_id: int, action: Action) -> bool:
        """Check minimum trade terms against public transport and the proposer's own stock."""
        arguments = action.args
        target = self.proposals[arguments["proposal"]].proposer if action.kind == "counter" else arguments["target"]
        offer, request = arguments["offer"], arguments["request"]
        combined = {item: offer.get(item, 0) + request.get(item, 0) for item in offer.keys() | request.keys()}
        return target in self.neighbors(island_id) and self._can_dispatch(island_id, target, combined)

    def quantity_choices(self, island_id: int, action: Action) -> list[Action]:
        """Expand a selected trade into every feasible whole-number pair without reading partner stock."""
        if self.config["actions"].get("trade_quantity_mode") != "all_feasible" or action.kind not in {"propose_trade", "counter"}:
            return [action]
        arguments = action.args
        offer, request = arguments["offer"], arguments["request"]
        # Generated proposals exchange one item per side; external baskets retain their explicit terms.
        if len(offer) > 1 or len(request) > 1:
            return [action]
        target = self.proposals[arguments["proposal"]].proposer if action.kind == "counter" else arguments["target"]
        if target not in self.neighbors(island_id):
            return []
        channel_id, channel = self._channel(island_id, target)
        if channel.disabled_until >= self.turn:
            return []
        capacity = channel.capacity - self._channel_used.get(channel_id, 0)
        offered_item = next(iter(offer), None)
        requested_item = next(iter(request), None)
        offered_load = self.config["items"][offered_item]["transport_load"] if offered_item else 0
        requested_load = self.config["items"][requested_item]["transport_load"] if requested_item else 0
        maximum_offer = min(self.available(island_id)[offered_item], math.floor((capacity + 1e-9) / offered_load)) if offered_item else 0
        maximum_request = math.floor((capacity + 1e-9) / requested_load) if requested_item else 0
        purchase = arguments.get("payment_from") == "proposer"
        unit_price = self.config["items"][requested_item]["reward_value"] * self.config["actions"]["purchase_price_per_value"] if purchase and requested_item else 0
        currency = self.available_currency(island_id)
        if purchase and unit_price > 0:
            maximum_request = min(maximum_request, math.floor(currency / unit_price))
        choices = []
        for offered_quantity in range(1, maximum_offer + 1) if offered_item else [0]:
            for requested_quantity in range(1, maximum_request + 1) if requested_item else [0]:
                if offered_quantity * offered_load + requested_quantity * requested_load > capacity + 1e-9:
                    continue
                price = math.ceil(requested_quantity * unit_price) if purchase else arguments.get("price", 0)
                if purchase and price > currency:
                    continue
                choices.append(Action(action.kind, {
                    **arguments,
                    "offer": {offered_item: offered_quantity} if offered_item else {},
                    "request": {requested_item: requested_quantity} if requested_item else {},
                    "price": price,
                }))
        return choices

    def _action_features(self, island_id: int, action: Action) -> list[float]:
        """Encode action terms and public target geography with a population-independent width."""
        arguments = dict(action.args)
        proposal = self.proposals.get(arguments.get("proposal"))
        if proposal is not None:
            arguments = {**asdict(proposal), **arguments}
            arguments["target"] = proposal.proposer if proposal.acceptor == island_id else proposal.acceptor
        target = self.islands.get(arguments.get("target"))
        island = self.islands[island_id]
        offer = arguments.get("offer", {})
        request = arguments.get("request", {})
        recipe_id = arguments.get("recipe")
        if recipe_id in self.config["recipes"]:
            recipe = self.config["recipes"][recipe_id]
            offer = recipe["inputs"]
            request = recipe["outputs"]
        features = [float(action.kind == kind) for kind in self.action_kinds]
        features.extend(math.tanh(offer.get(item, 0) / 16) for item in self.item_ids)
        features.extend(math.tanh(request.get(item, 0) / 16) for item in self.item_ids)
        features.extend(float(arguments.get("item") == item) for item in self.item_ids)
        features.extend(float(target is not None and item in target.specialization) for item in self.item_ids)
        visible_requests = [message for message in self._visible_messages(island_id) if target is not None and message["sender"] == target.id]
        features.extend(math.tanh(sum(message["item"] == item for message in visible_requests) / 5) for item in self.item_ids)
        features.extend(float(recipe_id == identifier) for identifier in self.recipe_ids)
        features.extend(float(arguments.get("skill") == skill) for skill in self.skill_ids)
        distance = math.dist(island.location, target.location) if target is not None else 0.0
        features.extend([
            math.tanh(distance / 5), math.tanh(arguments.get("price", 0) / 10),
            math.tanh(math.ceil(distance * self.config["transport"]["fee_per_distance"]) / 10),
            float(target is not None and target.parent_id == island_id),
            float(target is not None and island.parent_id == target.id),
            len(set(target.specialization) & set(island.specialization)) / 10 if target is not None else 0,
            math.tanh(self.config["recipes"][recipe_id]["base_time"]) if recipe_id in self.config["recipes"] else 0,
            math.tanh((proposal.expires_turn - self.turn) / 5) if proposal is not None else 0,
            float(arguments.get("transfer_capability", False)),
            math.tanh((target.location[0] - island.location[0]) / 5) if target is not None else 0,
            math.tanh((target.location[1] - island.location[1]) / 5) if target is not None else 0,
            math.tanh(target.id / 100) if target is not None else 0,
            float(arguments.get("payment_from", "acceptor") == "proposer"),
        ])
        proposal_kind = proposal.kind if proposal is not None else {
            "propose_trade": "trade", "propose_service": "service",
            "propose_teaching": "teaching", "propose_transfer": "transfer",
        }.get(action.kind)
        if action.kind == "counter":
            proposal_kind = "trade"
        features.extend(float(proposal_kind == kind) for kind in self.proposal_kinds)
        if "trade_initialization" in self.config:
            features.append(float(self._prioritize_necessity_trade(island_id, action.kind, proposal_kind, target, offer, request)))
        return features

    def _prioritize_necessity_trade(
        self, island_id: int, action_kind: str, proposal_kind: str | None,
        target: Island | None, offer: dict[str, int], request: dict[str, int],
    ) -> bool:
        """Flag staple trades using own reserves, visible terms, and public producer specializations."""
        if proposal_kind != "trade" or action_kind not in {"propose_trade", "counter", "accept"} or target is None:
            return False
        staples = set(self.config["survival"]["bundle_items"].values())
        if not staples.intersection(offer.keys() | request.keys()):
            return False
        # Offers use the proposer's perspective; an acceptance pays the request and receives the offer.
        outgoing = request if action_kind == "accept" else offer
        if action_kind != "accept" and not set(request).issubset(target.specialization):
            return False
        available = self.available(island_id)
        reserve_turns = self.config["trade_initialization"]["reserve_turns"]
        for necessity, consumption in self.config["survival"]["consumption"].items():
            spent = sum(quantity * self.config["items"][item]["survival"].get(necessity, 0) for item, quantity in outgoing.items())
            if spent <= 0:
                continue
            remaining = sum(available[item] * self.config["items"][item]["survival"].get(necessity, 0) for item in self.item_ids) - spent
            if necessity == "food":
                remaining += self.islands[island_id].food_reserve
            if remaining < consumption * reserve_turns:
                return False
        return True

    def preview_next_observations(self) -> dict[int, list[float]]:
        """Preview the actual next observations without advancing the original world or RNG."""
        if self._active:
            raise RuntimeError("Preview is only defined between completed turns")
        preview = World.from_state_dict(self.config, self.state_dict())
        return {identifier: decision.observation for identifier, decision in preview.start_turn().items()}

    def _island_state(self, island: Island) -> dict[str, Any]:
        """Serialize island state while preserving the original schema for legacy runs."""
        state = asdict(island)
        if self.item_utility is None:
            del state["food_reserve"]
            del state["active_effects"]
        if not self.history_turns:
            del state["recent_history"]
        return state

    def snapshot(self) -> dict[str, Any]:
        """Return an inspector snapshot; this privileged view is never passed to island policies."""
        return {
            "schema_version": 1, "turn": self.turn,
            "metrics": {"population": len(self.islands), "births": self.total_births, "deaths": self.total_deaths,
                        "total_resource_value": sum(self._resource_reward(identifier) for identifier in self.islands) / self.config["rewards"]["scale"],
                        "mean_health": sum(island.health for island in self.islands.values()) / max(1, len(self.islands))},
            "islands": [{**self._island_state(island), **({"item_bonuses": self.item_bonuses(island)} if self.item_utility is not None else {}),
                         **({"neighbors": self.neighbors(island.id), "missing_neighbor_resources": self.missing_neighbor_resources(island.id)} if self.limited_neighbors else {}),
                         "reserved": self.reserved(island.id), "resource_reward": self._resource_reward(island.id),
                         "resource_value": self._resource_reward(island.id) / self.config["rewards"]["scale"]} for island in self.islands.values()],
            "channels": [asdict(channel) for channel in self.channels.values() if channel.source in self.islands and channel.target in self.islands
                         and (not self.limited_neighbors or channel.target in self._neighbors[channel.source])],
            "shipments": [asdict(shipment) for shipment in self.shipments.values() if shipment.status == "in_transit"],
            "proposals": [asdict(proposal) for proposal in self.proposals.values() if proposal.status == "pending"],
            "contracts": [asdict(contract) for contract in self.contracts.values() if contract.status == "active"],
            "messages": copy.deepcopy(self.messages), "events": copy.deepcopy(self.events[-100:]),
            "deceased": list(self.deceased.values()),
        }

    def state_dict(self) -> dict[str, Any]:
        """Serialize all causal state, including partial turns, pending work, and randomness."""
        return {
            "schema_version": 1, "turn": self.turn, "random_state": self.random.getstate(),
            "islands": [self._island_state(island) for island in self.islands.values()], "deceased": copy.deepcopy(self.deceased),
            "channels": {identifier: asdict(channel) for identifier, channel in self.channels.items()},
            "proposals": [asdict(proposal) for proposal in self.proposals.values()],
            "contracts": [asdict(contract) for contract in self.contracts.values()],
            "shipments": [asdict(shipment) for shipment in self.shipments.values()],
            "messages": copy.deepcopy(self.messages), "events": copy.deepcopy(self.events),
            "total_births": self.total_births, "total_deaths": self.total_deaths,
            "next_island": self._next_island, "next_proposal": self._next_proposal,
            "next_contract": self._next_contract, "next_shipment": self._next_shipment,
            "active": self._active, "turn_actors": list(self._turn_actors),
            "visible_proposals": sorted(self._visible_proposals), "channel_used": dict(self._channel_used),
            **({"neighbors": {identifier: sorted(targets) for identifier, targets in self._neighbors.items()}} if self.limited_neighbors else {}),
        }

    @classmethod
    def from_state_dict(cls, config: dict[str, Any], state: dict[str, Any]) -> World:
        """Restore a JSON-roundtripped world without losing integer identity or RNG state."""
        if state["schema_version"] != 1:
            raise ValueError("Unsupported world checkpoint schema")
        world = cls(config)
        world.turn = state["turn"]
        random_state = state["random_state"]
        world.random.setstate((random_state[0], tuple(random_state[1]), random_state[2]))
        world.islands = {}
        for saved in state["islands"]:
            fields = copy.deepcopy(saved)
            if fields["work"] is not None:
                fields["work"] = WorkOrder(**fields["work"])
            island = Island(**fields)
            world.islands[island.id] = island
        world.deceased = {int(identifier): copy.deepcopy(saved) for identifier, saved in state["deceased"].items()}
        world.channels = {identifier: Channel(**saved) for identifier, saved in state["channels"].items()}
        world.proposals = {saved["id"]: Proposal(**saved) for saved in state["proposals"]}
        world.contracts = {saved["id"]: Contract(**saved) for saved in state["contracts"]}
        world.shipments = {saved["id"]: Shipment(**saved) for saved in state["shipments"]}
        world.messages = copy.deepcopy(state["messages"])
        world.events = copy.deepcopy(state["events"])
        world.total_births = state["total_births"]
        world.total_deaths = state["total_deaths"]
        world._next_island = state["next_island"]
        world._next_proposal = state["next_proposal"]
        world._next_contract = state["next_contract"]
        world._next_shipment = state["next_shipment"]
        world._active = state["active"]
        world._turn_actors = list(state["turn_actors"])
        world._visible_proposals = set(state["visible_proposals"])
        world._channel_used = dict(state["channel_used"])
        if world.limited_neighbors:
            world._neighbors = {int(identifier): set(targets) for identifier, targets in state["neighbors"].items()}
        return world
