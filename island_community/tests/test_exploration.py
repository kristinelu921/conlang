"""Remote neural checks for balanced action sampling and per-island exploration."""

import math

import pytest
import torch

from island_community.policies import PolicyPopulation


pytestmark = pytest.mark.neural


@pytest.fixture
def population() -> PolicyPopulation:
    """Create two independently annealing policies with unequal candidate menus."""
    policies = PolicyPopulation(
        3, 2, hidden_size=8, capacity=2, seed=17, mutation_std=0,
        initial_temperature=2, final_temperature=1, initial_entropy_coefficient=0.05,
        anneal_updates=2, balance_action_kinds=True,
    )
    policies.add(11)
    policies.add(29)
    return policies


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0, 10.0])
def test_equal_scores_give_equal_kind_mass_despite_a_hundred_trade_variants(temperature: float) -> None:
    """Verify the balanced prior and entropy optimum across sampling temperatures."""
    scores = torch.zeros((1, 103))
    valid = torch.tensor([[True] * 102 + [False]])
    counts = torch.tensor([[math.log(100)] * 100 + [0.0] * 3])
    logits = PolicyPopulation._behavior_logits(scores, valid, torch.tensor([temperature]), counts)
    probabilities = logits.softmax(-1)[0]
    assert probabilities[:100].sum().item() == pytest.approx(1 / 3)
    assert probabilities[100].item() == pytest.approx(1 / 3)
    assert probabilities[101].item() == pytest.approx(1 / 3)
    assert probabilities[102].item() == 0
    bonus = -(probabilities * (logits.log_softmax(-1)[0] + counts[0])).sum()
    assert bonus.item() == pytest.approx(math.log(3))


def test_temperature_flattens_preferences_without_unmasking_padding() -> None:
    """Verify higher temperature spreads probability only across valid candidates."""
    scores = torch.tensor([[0.0, 4.0, 100.0]])
    valid = torch.tensor([[True, True, False]])
    counts = torch.tensor([[math.log(2), math.log(2), 0.0]])
    cold = PolicyPopulation._behavior_logits(scores, valid, torch.tensor([1.0]), counts).softmax(-1)
    warm = PolicyPopulation._behavior_logits(scores, valid, torch.tensor([2.0]), counts).softmax(-1)
    assert 0.5 < warm[0, 1] < cold[0, 1]
    assert warm[0, 2] == cold[0, 2] == 0


def test_duplicate_variants_do_not_change_kind_mass_or_exploration_gradient() -> None:
    """Ensure menu duplication cannot bias either action mass or entropy gradients."""
    results = []
    for repetitions in (1, 100):
        scores = torch.tensor([0.3, -0.4], requires_grad=True)
        expanded = torch.cat((scores[:1].expand(repetitions), scores[1:]))[None, :]
        counts = torch.tensor([[math.log(repetitions)] * repetitions + [0.0]])
        logits = PolicyPopulation._behavior_logits(expanded, torch.ones_like(expanded, dtype=torch.bool), torch.tensor([2.0]), counts)
        log_probabilities = logits.log_softmax(-1)
        probabilities = log_probabilities.exp()
        bonus = -(probabilities * (log_probabilities + counts)).sum()
        gradients = torch.autograd.grad(bonus, scores)[0]
        results.append((probabilities[0, :repetitions].sum(), bonus, gradients))
    for single, duplicated in zip(*results):
        torch.testing.assert_close(single, duplicated)


def test_actor_and_entropy_gradients_match_the_distribution_used_for_sampling(population: PolicyPopulation) -> None:
    """Independently derive the loss from scores and compare every resulting Adam moment."""
    choices = {11: [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]}
    population.act({11: [0.2, 0.4, 0.8]}, choices, {11: ["trade", "trade", "manufacture"]})
    pending = population._pending
    assert pending is not None
    parameters = {name: value.index_select(0, pending.slots) for name, value in population._parameters.items()}
    scores, values, _ = population._forward(parameters, pending.observations, pending.candidates, pending.valid_mask, pending.previous_hidden)
    probabilities = (scores / 2).exp() / torch.tensor([[2.0, 2.0, 1.0]])
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    advantage = 1.25 - values
    actor_loss = -probabilities[0, pending.actions[0]].log() * advantage.detach()[0]
    bonus = -(probabilities * (probabilities.log() + torch.tensor([[math.log(2), math.log(2), 0.0]]))).sum()
    expected_loss = actor_loss + population.value_coefficient * 0.5 * advantage.square().sum() - 0.05 * bonus
    gradients = torch.autograd.grad(expected_loss, tuple(parameters.values()))
    norm = torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()
    clipping = (population.gradient_clip / (norm + 1e-6)).clamp(max=1)
    # Settings may change after acting; the pending transition must retain its sampled distribution.
    population.initial_temperature = 7
    population.initial_entropy_coefficient = 0.8
    metrics = population.learn({11: 1.25}, {}, {11})
    assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-6)
    assert metrics["temperature"] == 2
    assert metrics["entropy_coefficient"] == pytest.approx(0.05)
    for name, gradient in zip(parameters, gradients):
        expected_moment = (1 - population.adam_beta1) * clipping * gradient[0]
        torch.testing.assert_close(population._first_moments[name][population._island_slots[11]], expected_moment, atol=2e-7, rtol=2e-5)


def test_annealing_follows_only_own_completed_updates_and_resets_for_children(population: PolicyPopulation) -> None:
    """Check independent schedules, skipped updates, endpoints, and newborn reset."""
    observation = {11: [0.2, 0.4, 0.8], 29: [0.9, -0.1, 0.3]}
    choices = {11: [[1.0, 0.0]], 29: [[0.0, 1.0]]}
    groups = {11: ["wait"], 29: ["wait"]}
    for expected_temperature, expected_entropy in ((2, 0.05), (1.5, 0.03), (1, 0.01), (1, 0.01)):
        population.act(observation, choices, groups)
        pending = population._pending
        assert pending is not None
        torch.testing.assert_close(pending.temperatures, torch.tensor([expected_temperature, 2.0]))
        torch.testing.assert_close(pending.entropy_coefficients, torch.tensor([expected_entropy, 0.05]))
        population.learn({11: 1, 29: 1}, observation, set(), strengths={11: 1, 29: 0})
    population.add(88, parent_id=11)
    child_temperature, child_entropy = population._exploration_schedule(torch.tensor([population._island_slots[88]]))
    assert child_temperature.item() == 2
    assert child_entropy.item() == pytest.approx(0.05)
    population.anneal_updates = 0
    temperatures, entropy = population._exploration_schedule(torch.tensor([0, 1, 2]))
    torch.testing.assert_close(temperatures, torch.ones(3))
    torch.testing.assert_close(entropy, torch.full((3,), 0.01))


@pytest.mark.parametrize("use_vmap", [True, False])
def test_exploration_checkpoint_and_loop_preserve_pending_and_future_decisions(population: PolicyPopulation, use_vmap: bool) -> None:
    """Compare restored learning and later decisions through both execution modes."""
    observation = {11: [0.2, 0.4, 0.8], 29: [0.9, -0.1, 0.3]}
    choices = {11: [[1.0, 0.0]], 29: [[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]}
    groups = {11: ["wait"], 29: ["trade", "trade", "manufacture"]}
    population.act(observation, choices, groups)
    restored = PolicyPopulation.from_state_dict(population.state_dict(), use_vmap=use_vmap)
    for _ in range(4):
        metrics = population.learn({11: 1, 29: -0.25}, observation, set())
        restored_metrics = restored.learn({11: 1, 29: -0.25}, observation, set())
        assert metrics == pytest.approx(restored_metrics, abs=2e-6)
        for name, value in population._parameters.items():
            torch.testing.assert_close(value, restored._parameters[name], atol=2e-7, rtol=2e-5)
        assert population.act(observation, choices, groups) == restored.act(observation, choices, groups)


def test_legacy_pending_checkpoint_retains_flat_temperature_one_policy() -> None:
    """Migrate a version-one checkpoint without changing its next update or action."""
    original = PolicyPopulation(3, 2, hidden_size=8, capacity=1, seed=17)
    original.add(11)
    observation = {11: [0.2, 0.4, 0.8]}
    choices = {11: [[1.0, 0.0], [0.0, 1.0]]}
    original.act(observation, choices)
    state = original.state_dict()
    state["version"] = 1
    for field in ("initial_temperature", "final_temperature", "initial_entropy_coefficient", "anneal_updates", "balance_action_kinds"):
        del state["configuration"][field]
    for field in ("temperatures", "entropy_coefficients", "log_candidate_counts"):
        del state["pending"][field]
    restored = PolicyPopulation.from_state_dict(state)
    assert not restored.balance_action_kinds
    assert original.learn({11: 1}, observation, set()) == restored.learn({11: 1}, observation, set())
    assert original.act(observation, choices) == restored.act(observation, choices)
    for name, value in original._parameters.items():
        assert torch.equal(value, restored._parameters[name])


@pytest.mark.parametrize("groups", [None, {}, {11: []}, {11: [""]}, {11: [1]}, {11: ["wait", "trade"]}])
def test_balancing_rejects_missing_or_misaligned_groups(population: PolicyPopulation, groups: dict[int, list[str]] | None) -> None:
    with pytest.raises(ValueError, match="label"):
        population.act({11: [0.2, 0.4, 0.8]}, {11: [[1.0, 0.0]]}, groups)
    assert population._pending is None
