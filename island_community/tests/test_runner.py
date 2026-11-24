"""Remote numerical integration checks for inheritance and exact continuation."""

import json
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.neural


@pytest.mark.parametrize("rules_version", [1, 2, 3, 4, 5, 6, 7])
def test_run_checkpoint_resume_matches_uninterrupted_world(tmp_path: Path, rules_version: int) -> None:
    import torch

    from island_community.config import load_config
    from island_community.runner import run
    from island_community.storage import load_checkpoint

    configuration = load_config()
    configuration["schema_version"] = rules_version
    if rules_version >= 7:
        configuration["allocation"].update(scope="per_island", allocations_per_minute=1, recipient_rule="self")
    if rules_version < 6:
        del configuration["trade_initialization"]
    if rules_version == 4:
        del configuration["reward_shaping"]["empty_water_penalty"]
    if rules_version < 4:
        del configuration["reward_shaping"]
        configuration["rewards"]["lifetime_score"] = "sum_resource_rewards"
        configuration["survival"]["maximum_health"] = 100.0
    if rules_version < 3:
        del configuration["exploration"]
    if rules_version == 1:
        del configuration["item_utility"]
    configuration["policy"]["hidden_size"] = 8
    configuration["reproduction"]["interval"] = 2
    configuration["events"]["probability_per_turn"] = 0
    # Give one founder both necessities and enough stock to fund birth despite its first two actions.
    item_ids = list(configuration["items"])
    item_ids[1], item_ids[5] = item_ids[5], item_ids[1]
    configuration["items"] = {identifier: configuration["items"][identifier] for identifier in item_ids}
    configuration["initialization"]["specialization_quantity"] = 32
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(configuration))
    whole_path = run(tmp_path / "whole", turns=4, config_path=config_path, checkpoint_every=1)
    first_path = run(tmp_path / "first", turns=2, config_path=config_path, checkpoint_every=1)
    second_path = run(tmp_path / "second", turns=2, resume=first_path / "checkpoint.pt", checkpoint_every=1)
    whole = load_checkpoint(whole_path / "checkpoint.pt")
    resumed = load_checkpoint(second_path / "checkpoint.pt")
    assert whole["world"] == resumed["world"]
    if rules_version >= 6:
        assert whole["policies"]["configuration"]["initial_action_biases"] == {108: 2 * math.log(2)}
        assert resumed["policies"]["configuration"]["initial_action_biases"] == {108: 2 * math.log(2)}
    assert whole["world"]["total_births"] > 0
    assert whole["policies"]["island_slots"] == resumed["policies"]["island_slots"]
    for name, parameter in whole["policies"]["parameters"].items():
        assert torch.equal(parameter, resumed["policies"]["parameters"][name])
    assert json.loads((second_path / "manifest.json").read_text())["status"] == "completed"
    whole_experiences = [json.loads(line) for line in (whole_path / "experiences.jsonl").read_text().splitlines()]
    split_experiences = [
        json.loads(line)
        for path in (first_path, second_path)
        for line in (path / "experiences.jsonl").read_text().splitlines()
    ]
    assert whole_experiences == split_experiences
    if rules_version >= 4:
        assert all(experience["reward"] == experience["reward_components"]["total"] for experience in whole_experiences)
    assert all(experience["candidate_counts"][experience["action"]["kind"]] > 0 for experience in whole_experiences)
    assert len(whole_experiences) > configuration["initialization"]["population"] * 4
    assert all(len(experience["observation"]) == whole["policies"]["configuration"]["observation_size"] for experience in whole_experiences)
    whole_frames = [json.loads(line) for line in (whole_path / "frames.jsonl").read_text().splitlines()]
    first_frames = [json.loads(line) for line in (first_path / "frames.jsonl").read_text().splitlines()]
    second_frames = [json.loads(line) for line in (second_path / "frames.jsonl").read_text().splitlines()]
    assert whole_frames == first_frames + second_frames[1:]
    assert second_frames[0]["turn"] == 2
    assert second_frames[-1]["turn"] == 4


def test_run_does_not_overwrite_previous_segment(tmp_path: Path) -> None:
    from island_community.runner import run

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        run(existing, turns=1)


def test_survival_penalties_reach_neural_learning_and_terminal_records(tmp_path: Path) -> None:
    """Exercise negative terminal rewards through the complete recording and learner pipeline."""
    from island_community.config import load_config
    from island_community.runner import run

    configuration = load_config()
    configuration["policy"]["hidden_size"] = 8
    configuration["survival"]["maximum_health"] = 1
    configuration["initialization"]["specialization_quantity"] = 1
    configuration["allocation"]["allocations_per_minute"] = 0
    configuration["events"]["probability_per_turn"] = 0
    configuration["reproduction"]["enabled"] = False
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(configuration))
    output = run(tmp_path / "penalties", turns=12, config_path=config_path)
    experiences = [json.loads(line) for line in (output / "experiences.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert any(experience["terminal"] and experience["reward"] < 0 for experience in experiences)
    for metric in metrics[1:]:
        rewards = [experience["reward"] for experience in experiences if experience["turn"] == metric["turn"]]
        assert metric["learning"]["mean_reward"] == pytest.approx(sum(rewards) / len(rewards))
    assert all(experience["reward"] == experience["reward_components"]["total"] for experience in experiences)


def test_neural_policy_can_use_items_and_learn_from_the_updated_observation() -> None:
    """Exercise a real batched transition with item effects and the expanded feature layout."""
    from island_community.config import load_config
    from island_community.policies import PolicyPopulation
    from island_community.world import Action, World

    configuration = load_config()
    configuration["allocation"]["allocations_per_minute"] = 0
    configuration["events"]["probability_per_turn"] = 0
    configuration["reproduction"]["enabled"] = False
    world = World(configuration)
    for island in world.islands.values():
        island.inventory["tool"] = 1
    decisions = world.start_turn()
    population = PolicyPopulation(world.observation_size, world.action_feature_size, hidden_size=8)
    for island_id in decisions:
        population.add(island_id)
    action = Action("use_item", {"item": "tool"})
    choices = population.act(
        {identifier: decision.observation for identifier, decision in decisions.items()},
        {identifier: [decision.action_features[decision.candidates.index(action)]] for identifier, decision in decisions.items()},
    )
    assert all(index == 0 for index in choices.values())
    outcome = world.finish_turn({identifier: action for identifier in choices})
    following = world.preview_next_observations()
    metrics = population.learn(outcome.rewards, following, set(), outcome.learning_strengths)
    assert metrics
    for identifier, island in world.islands.items():
        assert island.inventory["tool"] == 0
        assert world.item_bonuses(island)["productivity"] == 0.5
        assert following[identifier] != decisions[identifier].observation


def test_neighbor_history_run_preserves_recurrent_learning_and_births_on_resume(tmp_path: Path) -> None:
    """Continue real neighbor-limited recurrent learners exactly across history and birth boundaries."""
    import torch

    from island_community.config import load_config
    from island_community.policies import PolicyPopulation
    from island_community.runner import run
    from island_community.storage import load_checkpoint, save_checkpoint
    from island_community.world import World

    configuration = load_config(Path(__file__).parents[1] / 'scenarios' / 'neighbors_50.json')
    configuration['initialization']['population'] = 20
    configuration['policy']['hidden_size'] = 8
    configuration['events']['probability_per_turn'] = 0
    world = World(configuration)
    for island in world.islands.values():
        island.next_birth_turn = 1000
    world.islands[0].next_birth_turn = 2
    world.islands[0].inventory['grain'] = world.islands[0].inventory['water'] = 100
    settings = dict(configuration['policy'])
    capacity = settings.pop('initial_capacity')
    population = PolicyPopulation(world.observation_size, world.action_feature_size, capacity=capacity,
        seed=configuration['seed'], initial_action_biases=world.initial_action_biases,
        mutation_std=configuration['reproduction']['policy_mutation_std'], **configuration['exploration'], **settings)
    for identifier in world.islands:
        population.add(identifier)
    initial = tmp_path / 'initial.pt'
    save_checkpoint(initial, configuration, world.state_dict(), population.state_dict())
    whole_path = run(tmp_path / 'whole', turns=4, resume=initial, checkpoint_every=2)
    first_path = run(tmp_path / 'first', turns=2, resume=initial, checkpoint_every=2)
    second_path = run(tmp_path / 'second', turns=2, resume=first_path / 'checkpoint.pt', checkpoint_every=2)
    whole = load_checkpoint(whole_path / 'checkpoint.pt')
    resumed = load_checkpoint(second_path / 'checkpoint.pt')
    assert whole['world'] == resumed['world']
    assert whole['world']['total_births'] >= 1
    assert whole['policies']['configuration']['observation_size'] == 1024
    assert torch.equal(whole['policies']['hidden'], resumed['policies']['hidden'])
    assert torch.count_nonzero(whole['policies']['hidden']).item() > 0
    for name in whole['policies']['parameters']:
        assert torch.equal(whole['policies']['parameters'][name], resumed['policies']['parameters'][name])
    for island in whole['world']['islands']:
        assert len(island['recent_history']) == min(4 - (2 if island['parent_id'] is not None else 0), island['age'])
    records = [json.loads(line) for line in (whole_path / 'experiences.jsonl').read_text().splitlines()]
    split = [json.loads(line) for path in (first_path, second_path) for line in (path / 'experiences.jsonl').read_text().splitlines()]
    assert records == split
    assert all(len(record['observation']) == 1024 for record in records)
    assert all(record['schema_version'] == 2 and record['quantity_index'] < record['quantity_candidate_count'] for record in records)
    assert any(record['quantity_candidate_count'] > 4 for record in records)
    assert any(quantity not in (1, 4) for record in records for side in ('offer', 'request') for quantity in record['action']['args'].get(side, {}).values())
    for record in records:
        turn = record['turn']
        whole_scores = json.loads((whole_path / 'decisions' / f'{turn}.json').read_text())
        split_path = first_path if turn <= 2 else second_path
        assert whole_scores == json.loads((split_path / 'decisions' / f'{turn}.json').read_text())
        stages = whole_scores['islands'][str(record['island_id'])]
        assert len(stages['action']['options']) == sum(record['candidate_counts'].values())
        quantities = stages['quantity']
        assert len(quantities['options']) == record['quantity_candidate_count']
        assert quantities['options'][quantities['selected_index']]['action'] == record['action']
