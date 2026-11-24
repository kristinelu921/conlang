"""Validate the explicit, versioned Island Community rules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ALLOCATION_QUANTITIES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
ITEM_BONUS_NAMES = ("productivity", "recovery", "comfort", "disease_resistance", "loss_protection")


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load and validate a complete configuration without silently merging defaults."""
    selected_path = path or Path(__file__).with_name("default.json")
    configuration = json.loads(selected_path.read_text())
    assert isinstance(configuration, dict)
    validate_config(configuration)
    return configuration


def survival_bundle(configuration: dict[str, Any]) -> dict[str, int]:
    """Calculate immediately edible and drinkable five-turn endowments."""
    bundle: dict[str, int] = {}
    for necessity in ("food", "water"):
        item_identifier = configuration["survival"]["bundle_items"][necessity]
        contribution = configuration["items"][item_identifier]["survival"][necessity]
        required = math.ceil(
            configuration["survival"]["consumption"][necessity]
            * configuration["initialization"]["survival_turns"] / contribution
        )
        quantity = next((value for value in ALLOCATION_QUANTITIES if value >= required), None)
        if quantity is None:
            raise ValueError("The survival bundle exceeds the configured logarithmic endowment scale")
        bundle[item_identifier] = bundle.get(item_identifier, 0) + quantity
    return bundle


def proportional_resource_counts(population: int, resource_weights: dict[str, float]) -> dict[str, int]:
    """Assign integer producer quotas by largest remainder, preserving registry order on ties."""
    total_weight = sum(resource_weights.values())
    quotas = {item: population * weight / total_weight for item, weight in resource_weights.items()}
    counts = {item: math.floor(quota) for item, quota in quotas.items()}
    remainder_order = sorted(quotas, key=lambda item: quotas[item] - counts[item], reverse=True)
    for item in remainder_order[:population - sum(counts.values())]:
        counts[item] += 1
    return counts


def validate_config(configuration: dict[str, Any]) -> None:
    """Reject unsupported rule choices and inconsistent accounting or survival settings."""
    pending_values: list[Any] = [configuration]
    while pending_values:
        value = pending_values.pop()
        if isinstance(value, dict):
            pending_values.extend(value.values())
        elif isinstance(value, list):
            pending_values.extend(value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Configuration numbers must be finite")
    required_sections = {
        "schema_version", "seed", "initialization", "items", "recipes", "allocation",
        "actions", "survival", "aging", "transport", "reproduction", "rewards",
        "events", "death", "learning", "policy",
    }
    version = configuration.get("schema_version")
    if type(version) is not int or not 1 <= version <= 9:
        raise ValueError("Only configuration schema versions 1 through 9 are supported")
    if version >= 2:
        required_sections.add("item_utility")
    if version >= 3:
        required_sections.add("exploration")
    if version >= 4:
        required_sections.add("reward_shaping")
    if version >= 6:
        required_sections.add("trade_initialization")
    if version >= 8:
        required_sections.add("memory")
    if version >= 9:
        required_sections.add("language_model")
    if set(configuration) != required_sections:
        raise ValueError(f"Configuration sections must be exactly {sorted(required_sections)}")
    template = json.loads(Path(__file__).with_name("default.json").read_text())
    for section in required_sections - {"schema_version", "seed", "items", "recipes"}:
        if section == "language_model":
            continue
        expected_keys = {"history_turns"} if section == "memory" else set(template[section])
        if section == "reward_shaping" and version == 4:
            expected_keys.remove("empty_water_penalty")
        if section == "allocation" and version >= 7:
            expected_keys.add("scope")
        if section == "allocation" and version >= 9:
            expected_keys.add("require_all_resource_producers")
        if section == "transport" and version >= 8:
            expected_keys.update({"maximum_neighbors", "required_neighbor_resources"})
        if section == "actions" and version >= 8:
            expected_keys.add("trade_quantity_mode")
        if set(configuration[section]) != expected_keys:
            raise ValueError(f"Unsupported or missing keys in {section}: expected {sorted(expected_keys)}")
    if type(configuration["seed"]) is not int:
        raise ValueError("seed must be an integer")
    integer_settings = {
        "initialization": ("population", "specialization_limit", "survival_turns", "specialization_quantity"),
        "allocation": ("turns_per_minute", "allocations_per_minute"),
        "actions": ("proposal_lifetime", "contract_deadline", "message_lifetime"),
        "survival": ("disease_duration",), "aging": ("start_turn", "interval"),
        "transport": ("minimum_transit_turns",), "reproduction": ("interval",),
        "events": ("duration", "resource_quantity"), "policy": ("hidden_size", "initial_capacity"),
    }
    for section, keys in integer_settings.items():
        if any(type(configuration[section][key]) is not int for key in keys):
            raise ValueError(f"{section} counts and durations must be integers")
    items = configuration["items"]
    base_items = [identifier for identifier, item in items.items() if item["category"] == "base"]
    if len(base_items) != 10:
        raise ValueError("Exactly ten base resource types are required")
    for identifier, item in items.items():
        if set(item) != set(template["items"]["water"]):
            raise ValueError(f"Unsupported or missing item properties in {identifier}")
        if not identifier or item["category"] not in {"base", "manufactured"}:
            raise ValueError(f"Invalid item {identifier}")
        if item["fractional"] is not False:
            raise ValueError("Use integer smallest units; fractional inventory is unsupported")
        if item["transport_load"] <= 0 or item["reward_value"] <= 0:
            raise ValueError("Item loads and conservative-accounting reward values must be positive")
        if any(value < 0 for value in item["survival"].values()):
            raise ValueError("Survival contributions cannot be negative")
        if not set(item["survival"]).issubset({"food", "water"}):
            raise ValueError("Only food and water survival contributions are supported")
        if item["survival"].get("food", 0) > 0 and not item["disease_eligible"]:
            raise ValueError("Every food item must support contamination")
    initialization = configuration["initialization"]
    population = initialization["population"]
    if not isinstance(population, int) or population < 2:
        raise ValueError("Initial population must be an integer of at least two")
    specialization_rules = {"round_robin_cover"}
    if version >= 7:
        specialization_rules.add("proportional_single_resource")
    if initialization["specialization_rule"] not in specialization_rules:
        raise ValueError(f"Specialization rule must be one of {sorted(specialization_rules)}")
    if not 1 <= initialization["specialization_limit"] < 10:
        raise ValueError("Specialization limit must be between one and nine")
    if population * initialization["specialization_limit"] < 10:
        raise ValueError("Initial specializations cannot cover all ten base resources")
    if initialization["survival_turns"] < 5 or initialization["endowment_rounding"] != "next_power_of_two":
        raise ValueError("Endowments require at least five turns and next_power_of_two rounding")
    if initialization["specialization_quantity"] not in ALLOCATION_QUANTITIES:
        raise ValueError("Specialization endowment must use the allocation quantity scale")
    if initialization["currency"] < 0 or initialization["location_radius"] <= 0:
        raise ValueError("Initial currency and location radius are invalid")
    recipes = configuration["recipes"]
    for recipe_identifier, recipe in recipes.items():
        if set(recipe) != set(template["recipes"]["cook"]):
            raise ValueError(f"Unsupported or missing recipe properties in {recipe_identifier}")
        if recipe["base_time"] <= 0 or not recipe["inputs"] or not recipe["outputs"]:
            raise ValueError(f"Recipe {recipe_identifier} needs positive time, inputs, and outputs")
        for item_identifier, quantity in list(recipe["inputs"].items()) + list(recipe["outputs"].items()):
            if item_identifier not in items or not isinstance(quantity, int) or quantity <= 0:
                raise ValueError(f"Invalid recipe quantity in {recipe_identifier}")
        input_value = sum(items[item]["reward_value"] * quantity for item, quantity in recipe["inputs"].items())
        output_value = sum(items[item]["reward_value"] * quantity for item, quantity in recipe["outputs"].items())
        if output_value > input_value:
            raise ValueError(f"Recipe {recipe_identifier} violates the conservative no-value-amplification rule")
    known_capabilities = {recipe["capability"] for recipe in recipes.values()}
    if not set(initialization["capabilities"]).issubset(known_capabilities):
        raise ValueError("Unknown initial manufacturing capability")
    actions = configuration["actions"]
    if actions["time_budget"] <= 0 or actions["minimum_work_time"] <= 0:
        raise ValueError("Action time budgets must be positive")
    if any(actions[key] < 0 for key in ("communication_time", "agreement_time", "skill_gain", "skill_speed_factor", "teaching_gain", "teaching_cost", "transfer_cost", "service_price", "message_lifetime", "purchase_price_per_value")):
        raise ValueError("Action times, costs, and gains cannot be negative")
    if actions["teaching_time"] <= 0 or actions["transfer_time"] <= 0 or actions["transfer_units"] <= 0:
        raise ValueError("Teaching/transfer time and transferred skill must be positive")
    if actions["proposal_lifetime"] < 1 or actions["contract_deadline"] < 1:
        raise ValueError("Proposal and contract durations must be positive")
    quantity_mode = actions.get("trade_quantity_mode", "fixed")
    if quantity_mode not in {"fixed", "all_feasible"}:
        raise ValueError("Trade quantity mode must be fixed or all_feasible")
    if quantity_mode == "all_feasible" and actions["trade_quantities"] != []:
        raise ValueError("All-feasible trade quantities require an empty fixed quantity list")
    if quantity_mode == "fixed" and (not actions["trade_quantities"] or any(type(quantity) is not int or quantity <= 0 for quantity in actions["trade_quantities"])):
        raise ValueError("Trade quantities must be positive integers")
    if actions["work_carries_forward"] not in (True, False):
        raise ValueError("work_carries_forward must be boolean")
    allocation = configuration["allocation"]
    if allocation["turns_per_minute"] < 1 or allocation["allocations_per_minute"] < 0:
        raise ValueError("Allocation timing must be nonnegative with positive turns per minute")
    if allocation["quantities"] != ALLOCATION_QUANTITIES or set(allocation["resource_weights"]) != set(base_items):
        raise ValueError("Allocation requires the complete logarithmic scale and all ten base resources")
    weights = allocation["quantity_weights"]
    if len(weights) != len(ALLOCATION_QUANTITIES) or any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Invalid allocation quantity weights")
    if any(weight < 0 for weight in allocation["resource_weights"].values()) or sum(allocation["resource_weights"].values()) <= 0:
        raise ValueError("Invalid resource allocation weights")
    scope = allocation.get("scope", "global")
    if scope not in {"global", "per_island"}:
        raise ValueError("Allocation scope must be global or per_island")
    expected_recipient_rule = "uniform_eligible" if scope == "global" else "self"
    if allocation["recipient_rule"] != expected_recipient_rule:
        raise ValueError(f"Allocation scope {scope} requires recipient_rule={expected_recipient_rule}")
    if scope == "per_island" and any(weight <= 0 for weight in allocation["resource_weights"].values()):
        raise ValueError("Per-island allocation requires a positive weight for every resource")
    if initialization["specialization_rule"] == "proportional_single_resource":
        counts = proportional_resource_counts(population, allocation["resource_weights"])
        require_all = allocation.get("require_all_resource_producers", True)
        if type(require_all) is not bool:
            raise ValueError("require_all_resource_producers must be a boolean")
        if require_all and any(count < 1 for count in counts.values()):
            raise ValueError("Population and proportional resource weights must assign at least one producer to every resource")
    survival = configuration["survival"]
    if set(survival["consumption"]) != {"food", "water"} or set(survival["bundle_items"]) != {"food", "water"}:
        raise ValueError("Survival consumption and bundle items must specify food and water")
    if survival["unhappiness_formula"] != "relative_reward_shortfall_ema":
        raise ValueError("Unsupported unhappiness formula")
    for necessity in ("food", "water"):
        item_identifier = survival["bundle_items"][necessity]
        if item_identifier not in items or items[item_identifier]["survival"].get(necessity, 0) <= 0:
            raise ValueError("Survival bundle must contain immediately usable necessities")
        if items[item_identifier]["category"] != "base":
            raise ValueError("Initial endowment can introduce only base resources")
        if survival["consumption"][necessity] <= 0:
            raise ValueError("Survival consumption must be positive")
    if not 0 < survival["maximum_health"] or not 0 <= survival["death_threshold"] < survival["maximum_health"]:
        raise ValueError("Invalid health thresholds")
    if not 0 <= survival["unhappiness_smoothing"] <= 1:
        raise ValueError("Unhappiness smoothing must be a probability")
    if any(survival[key] < 0 for key in ("shortage_damage", "unhappiness_damage", "recovery", "disease_damage", "learning_strength_gain")):
        raise ValueError("Health and learning coefficients cannot be negative")
    if survival["disease_duration"] < 1:
        raise ValueError("Disease duration must be positive")
    aging = configuration["aging"]
    if aging["interval"] <= 0 or aging["start_turn"] < 0 or not 0 <= aging["skill_retention"] <= 1:
        raise ValueError("Invalid aging schedule")
    if aging["health_damage"] < 0 or not 0 < aging["minimum_productivity"] <= 1 or aging["productivity_decay"] < 0 or aging["event_susceptibility"] < 0:
        raise ValueError("Invalid aging effects")
    transport = configuration["transport"]
    choices = {
        "topology": "resource_neighbors" if version >= 8 else "complete", "direction": "bidirectional", "distance": "euclidean",
        "congestion": "seeded_priority", "fee_denomination": "currency", "ownership": "recipient_at_dispatch",
        "late_failure": "retain_recipient_owned_shipments", "cancellation": "pending_only",
    }
    if any(transport[key] != expected for key, expected in choices.items()):
        raise ValueError("Unsupported topology, transport accounting, or cancellation rule")
    if transport["capacity"] <= 0 or transport["distance_per_turn"] <= 0 or transport["fee_per_distance"] < 0 or transport["minimum_transit_turns"] < 1:
        raise ValueError("Invalid channel capacity, speed, fee, or transit time")
    if version >= 8:
        maximum_neighbors = transport["maximum_neighbors"]
        if type(maximum_neighbors) is not int or not 2 <= maximum_neighbors <= 5:
            raise ValueError("The mutual-neighbor limit must be between two and five")
        if transport["required_neighbor_resources"] != ["grain", "water"]:
            raise ValueError("Resource neighbors require grain and water producers")
        if initialization["specialization_rule"] != "proportional_single_resource":
            raise ValueError("Resource neighbors require single-resource producer assignments")
        if scope != "per_island" or allocation["turns_per_minute"] != 1 or allocation["allocations_per_minute"] < 1:
            raise ValueError("Resource neighbors require gifts to every island each turn")
        if not set(transport["required_neighbor_resources"]).issubset(counts):
            raise ValueError("The resource registry must include grain and water producers")
        for resource in transport["required_neighbor_resources"]:
            producer_count = counts[resource]
            if producer_count < 2 or maximum_neighbors * producer_count < population + producer_count % 2:
                raise ValueError(f"Too few {resource} producers for mutual neighbor coverage")
        history_turns = configuration["memory"]["history_turns"]
        if type(history_turns) is not int or not 0 <= history_turns <= 32:
            raise ValueError("History length must be an integer between zero and 32")
    reproduction = configuration["reproduction"]
    if reproduction["interval"] < 1 or reproduction["additional_inheritance_factor"] < 0:
        raise ValueError("Invalid reproduction interval or inheritance factor")
    if reproduction["inheritance_formula"] != "floor_factor_sqrt_available" or reproduction["specialization"] != "inherit_parent" or reproduction["location"] != "near_parent" or reproduction["channels"] != ("match_topology" if version >= 8 else "connect_all"):
        raise ValueError("Unsupported birth inheritance, specialization, location, or channel rule")
    if reproduction["location_jitter"] < 0 or reproduction["policy_mutation_std"] < 0:
        raise ValueError("Mutation and location jitter cannot be negative")
    rewards = configuration["rewards"]
    lifetime_score = "sum_learning_rewards" if version >= 4 else "sum_resource_rewards"
    if rewards["measurement"] != "after_deaths_and_births" or rewards["lifetime_score"] != lifetime_score or rewards["work_in_progress"] != "excluded":
        raise ValueError("Unsupported reward measurement, lifetime score, or work valuation")
    if rewards["recipe_validation"] != "positive_values_no_amplification":
        raise ValueError("Only positive_values_no_amplification recipe validation is supported")
    if rewards["scale"] <= 0 or rewards["currency_value"] < 0 or rewards["terminal_reward"] != 0:
        raise ValueError("Reward scale must be positive, currency nonnegative, terminal reward zero")
    if rewards["committed_inventory"] != "included" or rewards["in_transit"] != "recipient_owned":
        raise ValueError("Unsupported reservation or transit reward ownership")
    if configuration["death"]["assets"] != "destroy" or configuration["death"]["extinction"] != "stop":
        raise ValueError("Only explicit asset destruction and stop-on-extinction are supported")
    if configuration["learning"]["death_replay"] is not False or configuration["learning"]["ancestral_replay"] is not False:
        raise ValueError("Death and ancestral replay have been removed")
    events = configuration["events"]
    if not 0 <= events["probability_per_turn"] <= 1 or events["duration"] < 1:
        raise ValueError("Invalid event probability or duration")
    if any(value < 0 for value in events["type_weights"].values()) or sum(events["type_weights"].values()) <= 0:
        raise ValueError("Invalid event weights")
    if set(events["type_weights"]) != {"windfall", "loss", "channel_outage", "income_reduction", "disease"}:
        raise ValueError("Unsupported event types")
    if any(item not in items for item in events["affected_items"]):
        raise ValueError("Unknown event item")
    if any(items[item]["category"] != "base" for item in events["windfall_items"]):
        raise ValueError("The world can create only base resources")
    if not events["windfall_items"] or not events["affected_items"] or events["resource_quantity"] < 0 or not 0 <= events["income_multiplier"] <= 1 or not 0 <= events["incoming_contamination_probability"] <= 1:
        raise ValueError("Invalid event quantity, eligibility, or probabilities")
    policy = configuration["policy"]
    if version >= 9:
        from island_community.language_policy import validate_language_settings

        validate_language_settings(configuration["language_model"])
        if configuration["actions"]["trade_quantity_mode"] != "all_feasible":
            raise ValueError("Language action templates require all_feasible trade quantities")
    if policy["hidden_size"] < 1 or policy["initial_capacity"] < 1 or policy["learning_rate"] <= 0 or policy["gradient_clip"] <= 0:
        raise ValueError("Policy size, capacity, learning rate, and gradient clip must be positive")
    if not 0 <= policy["discount"] <= 1 or policy["entropy_coefficient"] < 0 or policy["value_coefficient"] < 0:
        raise ValueError("Policy discount and loss coefficients are invalid")
    if not 0 <= policy["adam_beta1"] < 1 or not 0 <= policy["adam_beta2"] < 1 or policy["adam_epsilon"] <= 0:
        raise ValueError("Invalid Adam moment decay or epsilon")
    survival_bundle(configuration)
    if version >= 2:
        validate_item_utility(configuration)
    if version >= 3:
        exploration = configuration["exploration"]
        if exploration["initial_temperature"] <= 0 or exploration["final_temperature"] <= 0:
            raise ValueError("Exploration temperatures must be positive")
        if exploration["initial_entropy_coefficient"] < 0:
            raise ValueError("Initial entropy coefficient cannot be negative")
        if type(exploration["anneal_updates"]) is not int or exploration["anneal_updates"] < 0:
            raise ValueError("Exploration annealing needs a nonnegative integer update count")
        if type(exploration["balance_action_kinds"]) is not bool:
            raise ValueError("Action-kind balancing must be boolean")
    if version >= 4:
        if any(type(value) not in (int, float) or value < 0 for value in configuration["reward_shaping"].values()):
            raise ValueError("Survival reward penalties must be finite nonnegative numbers")
    if version >= 6:
        trade_initialization = configuration["trade_initialization"]
        if any(type(value) not in (int, float) for value in trade_initialization.values()):
            raise ValueError("Trade initialization settings must be finite numbers")
        if trade_initialization["initial_odds_multiplier"] < 1 or trade_initialization["reserve_turns"] < 0:
            raise ValueError("Trade odds multiplier must be at least one and reserve turns nonnegative")


def validate_item_utility(configuration: dict[str, Any]) -> None:
    """Require useful, bounded item effects with explicit durations and costs."""
    utility = configuration["item_utility"]
    caps = utility["bonus_caps"]
    if not 0 < utility["action_time"] <= configuration["actions"]["time_budget"]:
        raise ValueError("Item use must cost a positive amount of available turn time")
    if set(caps) != set(ITEM_BONUS_NAMES) or any(value <= 0 for value in caps.values()):
        raise ValueError("Item bonus caps must specify every supported stat with a positive limit")
    if any(caps[name] >= 1 for name in ("comfort", "disease_resistance", "loss_protection")):
        raise ValueError("Protection caps must be less than one; needs and damage cannot be eliminated")
    items = configuration["items"]
    uses = utility["uses"]
    if not set(uses).issubset(items):
        raise ValueError("Unknown usable item")
    for identifier, effect in uses.items():
        if set(effect) != {"purpose", "health", "disease_turns", "duration", "bonuses"}:
            raise ValueError(f"Unsupported item-use properties for {identifier}")
        if not isinstance(effect["purpose"], str) or not effect["purpose"].strip():
            raise ValueError("Item uses need a descriptive purpose")
        if any(type(effect[key]) is not int or effect[key] < 0 for key in ("duration", "disease_turns")) or effect["health"] < 0:
            raise ValueError("Item healing and integer durations cannot be negative")
        bonuses = effect["bonuses"]
        if not set(bonuses).issubset(caps) or any(not 0 < amount <= caps[name] for name, amount in bonuses.items()):
            raise ValueError("Item bonuses must be positive, supported, and within their caps")
        if bool(bonuses) != (effect["duration"] > 0):
            raise ValueError("Temporary item bonuses need a positive duration, and instant effects need zero duration")
        if not bonuses and not effect["health"] and not effect["disease_turns"]:
            raise ValueError("Every item use must have an effect")
    for identifier, item in items.items():
        if identifier not in uses and not any(item["survival"].values()):
            raise ValueError(f"Item {identifier} needs a survival contribution or a usable effect")
