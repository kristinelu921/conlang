"""Check bounded communication, producer access, and private history without learned execution."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from island_community.config import load_config, validate_config
from island_community.world import Action, World


@pytest.fixture
def configuration() -> dict[str, Any]:
    """Use the neighborhood scenario with deterministic exogenous rule fixtures."""
    settings = load_config(Path(__file__).parents[1] / 'scenarios' / 'neighbors_50.json')
    settings['events']['probability_per_turn'] = 0
    settings['reproduction']['enabled'] = False
    return settings


@pytest.mark.parametrize('seed', [0, 1, 2, 17, 42, 91])
@pytest.mark.parametrize('population', [20, 50, 100])
def test_founders_have_five_mutual_neighbors_and_both_producers(configuration: dict[str, Any], seed: int, population: int) -> None:
    """Cover every founder's resource needs without making any producer an unlimited hub."""
    configuration['seed'] = seed
    configuration['initialization']['population'] = population
    world = World(configuration)
    for identifier in world.islands:
        neighbors = world.neighbors(identifier)
        assert len(neighbors) == 5
        assert identifier not in neighbors
        assert not world.missing_neighbor_resources(identifier)
        assert all(identifier in world.neighbors(neighbor) for neighbor in neighbors)
    assert len(world.snapshot()['channels']) == population * 5 // 2
    restored = World.from_state_dict(configuration, json.loads(json.dumps(world.state_dict())))
    assert restored.state_dict() == world.state_dict()


def test_messages_and_action_menus_cannot_reach_outside_neighbors(configuration: dict[str, Any]) -> None:
    """Scope public broadcasts and targeted actions at both observation and execution boundaries."""
    world = World(configuration)
    decisions = world.start_turn()
    sender = 0
    receiver = world.neighbors(sender)[0]
    outsider = next(identifier for identifier in world.islands if identifier != sender and identifier not in world.neighbors(sender))
    assert all(action.args.get('target', receiver) in world.neighbors(sender) for action in decisions[sender].candidates)
    before = world.state_dict()
    for kind in ('private_request', 'propose_trade', 'propose_service', 'propose_teaching', 'propose_transfer'):
        assert not world._execute(sender, Action(kind, {'target': outsider, 'item': 'water', 'offer': {'water': 1}, 'request': {'grain': 1}}))
    assert world.state_dict() == before
    assert world._execute(sender, Action('public_request', {'item': 'water'}))
    assert world.messages[-1]['recipients'] == world.neighbors(sender)
    assert not world._visible_messages(receiver)
    world.finish_turn({})
    world.start_turn()
    assert world._visible_messages(receiver)
    outsider_observation = world.observe(outsider)
    assert not world._visible_messages(outsider)
    world.messages[-1]['item'] = 'grain'
    assert world.observe(outsider) == outsider_observation
    assert world._execute(sender, Action('private_request', {'target': receiver, 'item': 'grain'}))


def test_neighbors_depend_on_public_producer_types_not_private_stock(configuration: dict[str, Any]) -> None:
    """Prevent rewiring from revealing other islands' inventories or learned histories."""
    first = World(configuration)
    second = World(configuration)
    second.islands[0].inventory['water'] = 100000
    second.islands[1].recent_history = [{'private': 'not part of topology'}]
    first._rebuild_neighbors()
    second._rebuild_neighbors()
    assert first._neighbors == second._neighbors


def test_population_changes_repair_access_and_report_unavailable_producers(configuration: dict[str, Any]) -> None:
    """Keep degrees bounded through deaths, births, and eventual loss of a producer type."""
    world = World(configuration)
    world.start_turn()
    dead = next(identifier for identifier, island in world.islands.items() if island.specialization == ['grain'])
    world.islands[dead].health = 0
    world.islands[dead].inventory['water'] = 0
    outcome = world.finish_turn({})
    assert dead in outcome.deaths
    assert all(dead not in world.neighbors(identifier) for identifier in world.islands)
    assert all(len(world.neighbors(identifier)) <= 5 and not world.missing_neighbor_resources(identifier) for identifier in world.islands)
    world.config['reproduction']['enabled'] = True
    for island in world.islands.values():
        island.next_birth_turn = 1000
    parent = next(iter(world.islands.values()))
    parent.next_birth_turn = world.turn
    parent.inventory['grain'] = parent.inventory['water'] = 100
    world.start_turn()
    outcome = world.finish_turn({})
    assert len(outcome.births) == 1
    child = next(iter(outcome.births))
    assert not world.islands[child].recent_history
    assert world.neighbors(child) and not world.missing_neighbor_resources(child)
    assert all(len(world.neighbors(identifier)) <= 5 for identifier in world.islands)
    for identifier, island in list(world.islands.items()):
        if island.specialization == ['water']:
            world._die(identifier)
    world._rebuild_neighbors()
    assert all('water' in world.missing_neighbor_resources(identifier) for identifier in world.islands)
    assert all(len(world.neighbors(identifier)) <= 5 for identifier in world.islands)
    assert world.events[-1]['missing_resources']


def test_rewiring_cancels_old_offers_but_keeps_accepted_cargo(configuration: dict[str, Any]) -> None:
    """Prevent new deals over removed contacts without destroying legally dispatched shipments."""
    world = World(configuration)
    world.start_turn()
    source = 0
    target = world.neighbors(source)[0]
    world.islands[source].inventory['grain'] = 100
    world.islands[target].inventory['water'] = 100
    assert world._propose(source, Action('propose_trade', {'target': target, 'offer': {'grain': 1}, 'request': {'water': 1}}))
    world.finish_turn({})
    world.start_turn()
    proposal = world.proposals[0]
    assert world._execute(target, Action('accept', {'proposal': proposal.id}))
    world.finish_turn({})
    # A removed contact remains a physical route for already accepted cargo only.
    world._neighbors[source].remove(target)
    world._neighbors[target].remove(source)
    assert not world._propose(source, Action('propose_trade', {'target': target, 'offer': {'grain': 1}, 'request': {'water': 1}}))
    for _ in range(2):
        world.start_turn()
        world.finish_turn({})
    assert all(shipment.status == 'delivered' for shipment in world.shipments.values())
    # Inject a pending offer from the formerly valid contact to exercise repair cancellation.
    pending = copy.deepcopy(proposal)
    pending.id = 100
    pending.status = 'pending'
    world.proposals[pending.id] = pending
    world._die(target)
    world._rebuild_neighbors()
    assert pending.status == 'cancelled'


def test_history_is_private_bounded_chronological_and_restorable(configuration: dict[str, Any]) -> None:
    """Expose only eight own completed outcomes, with padding and exact checkpoint continuation."""
    configuration['initialization']['population'] = 20
    world = World(configuration)
    width = world.action_feature_size + 5
    assert world.observation_size == 112 + 8 * width == 1024
    assert world.observe(0)[-8 * width:] == [0.0] * (8 * width)
    for _ in range(10):
        world.start_turn()
        world.finish_turn({0: Action('public_request', {'item': 'grain'})})
    history = world.islands[0].recent_history
    assert [experience['turn'] for experience in history] == list(range(3, 11))
    assert all(experience['action']['kind'] == 'public_request' for experience in history)
    assert all(experience['action']['kind'] == 'wait' for experience in world.islands[1].recent_history)
    assert history[-1]['reward'] == world.events[-1]['rewards'][0]
    observation = world.observe(0)
    assert all(observation[112 + index * width] == 1 for index in range(8))
    restored = World.from_state_dict(configuration, json.loads(json.dumps(world.state_dict())))
    assert restored.observe(0) == observation
    world.islands[1].recent_history[-1]['reward'] += 123
    assert world.observe(0) == observation
    restored.islands[0].recent_history[-1]['reward'] += 1
    assert restored.observe(0) != observation
    snapshot = world.snapshot()
    assert snapshot['islands'][0]['recent_history'] == history


@pytest.mark.parametrize('mutation', [
    lambda settings: settings['transport'].update(maximum_neighbors=6),
    lambda settings: settings['transport'].update(maximum_neighbors=1),
    lambda settings: settings['transport'].update(required_neighbor_resources=['grain']),
    lambda settings: settings['allocation'].update(scope='global', recipient_rule='uniform_eligible'),
    lambda settings: settings['allocation'].update(turns_per_minute=2),
    lambda settings: settings['allocation']['resource_weights'].update(grain=0.5),
    lambda settings: settings['memory'].update(history_turns=-1),
    lambda settings: settings['memory'].update(history_turns=True),
    lambda settings: settings['memory'].update(history_turns=33),
    lambda settings: settings['reproduction'].update(channels='connect_all'),
])
def test_invalid_neighbor_and_history_settings_are_rejected(configuration: dict[str, Any], mutation: Any) -> None:
    """Reject impossible founding producer coverage and ambiguous unbounded settings."""
    mutation(configuration)
    with pytest.raises(ValueError):
        validate_config(configuration)


def test_birth_rewiring_cancels_pending_offers_on_removed_contacts(configuration: dict[str, Any]) -> None:
    """Release reservations for offers whose formerly valid contacts were rewired for a newborn."""
    configuration['reproduction']['enabled'] = True
    world = World(configuration)
    world.start_turn()
    for island in world.islands.values():
        island.inventory['grain'] = island.inventory['water'] = 100
        island.next_birth_turn = 1000
        for target in world.neighbors(island.id):
            assert world._propose(island.id, Action('propose_trade', {'target': target, 'offer': {'grain': 1}, 'request': {'water': 1}}))
    world.islands[0].next_birth_turn = 0
    outcome = world.finish_turn({})
    assert len(outcome.births) == 1
    cancelled = [event for event in world.events if event['type'] == 'proposal_cancelled' and event.get('reason') == 'no_longer_neighbors']
    assert cancelled
    for event in cancelled:
        proposal = world.proposals[event['proposal']]
        assert proposal.status == 'cancelled'
        assert proposal.acceptor not in world.neighbors(proposal.proposer)
    assert all(proposal.acceptor in world.neighbors(proposal.proposer) for proposal in world.proposals.values() if proposal.status == 'pending')


def test_terminal_history_contains_final_outcome_and_zero_history_preserves_small_input(configuration: dict[str, Any]) -> None:
    """Archive the final own reward and allow history to be disabled without stale input fields."""
    world = World(configuration)
    identifier = next(identifier for identifier, island in world.islands.items() if island.specialization == ['grain'])
    world.islands[identifier].health = 1
    world.islands[identifier].inventory['water'] = 0
    world.start_turn()
    outcome = world.finish_turn({})
    assert identifier in outcome.deaths
    history = world.deceased[identifier]['recent_history']
    assert history[-1]['reward'] == outcome.rewards[identifier]
    assert history[-1]['health'] == 0
    configuration['memory']['history_turns'] = 0
    without_history = World(configuration)
    assert without_history.observation_size == len(without_history.observe(0)) == 112
    assert 'recent_history' not in without_history.snapshot()['islands'][0]
