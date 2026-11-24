"""Remote-only neural tests for independent policy storage and learning."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from island_community.policies import PolicyPopulation
from island_community.config import load_config
from island_community.world import World


pytestmark = pytest.mark.neural


def make_population(use_vmap: bool = True, capacity: int = 4, mutation_std: float = 0.0) -> PolicyPopulation:
    """Create a tiny population with distinct identities and reproducible initial weights."""
    population = PolicyPopulation(3, 2, hidden_size=8, capacity=capacity, seed=17, use_vmap=use_vmap, mutation_std=mutation_std)
    population.add(11)
    population.add(29)
    return population


def observations() -> dict[int, list[float]]:
    return {11: [0.2, 0.4, 0.8], 29: [0.9, -0.1, 0.3]}


def candidates() -> dict[int, list[list[float]]]:
    return {11: [[1.0, 0.0], [0.0, 1.0]], 29: [[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]}


def next_observations() -> dict[int, list[float]]:
    return {11: [0.3, 0.5, 0.7], 29: [0.7, 0.1, 0.4]}


def island_parameters(population: PolicyPopulation, island_id: int) -> dict[str, Tensor]:
    """Snapshot one island's parameter slice without sharing its backing storage."""
    slot = population._island_slots[island_id]
    return {name: value[slot].detach().clone() for name, value in population._parameters.items()}


def assert_parameters_match(first: dict[str, Tensor], second: dict[str, Tensor], exact: bool = False) -> None:
    """Compare complete parameter sets using exact or floating-point-aware tolerances."""
    assert first.keys() == second.keys()
    for name in first:
        if exact:
            assert torch.equal(first[name], second[name]), name
        else:
            torch.testing.assert_close(
                first[name], second[name], atol=2e-7, rtol=2e-5,
                msg=lambda message: f"{name}: {message}",
            )


def test_vmap_matches_loop_logits_actions_and_independent_updates() -> None:
    vectorized = make_population()
    sequential = PolicyPopulation.from_state_dict(vectorized.state_dict(), use_vmap=False)
    for _ in range(3):
        assert vectorized.act(observations(), candidates()) == sequential.act(observations(), candidates())
        assert vectorized._pending is not None
        pending = vectorized._pending
        parameters = {name: value.index_select(0, pending.slots) for name, value in vectorized._parameters.items()}
        with torch.no_grad():
            vectorized_outputs = vectorized._forward(parameters, pending.observations, pending.candidates, pending.valid_mask, pending.previous_hidden)
            sequential_outputs = sequential._forward(parameters, pending.observations, pending.candidates, pending.valid_mask, pending.previous_hidden)
        for vectorized_output, sequential_output in zip(vectorized_outputs, sequential_outputs):
            torch.testing.assert_close(vectorized_output, sequential_output, atol=2e-7, rtol=2e-5)
        vectorized.learn({11: 1.0, 29: -0.25}, next_observations(), set())
        sequential.learn({11: 1.0, 29: -0.25}, next_observations(), set())
        assert_parameters_match(vectorized._parameters, sequential._parameters)
        assert_parameters_match(vectorized._first_moments, sequential._first_moments)
        assert_parameters_match(vectorized._second_moments, sequential._second_moments)


def test_gru256_preset_supports_independent_learning_and_exact_continuation() -> None:
    """Exercise the wider preset's actual feature sizes, optimizer, and checkpoint state."""
    scenario_directory = Path(__file__).parents[1] / "scenarios"
    configuration = load_config(scenario_directory / "neighbors_50_gru256.json")
    baseline = load_config(scenario_directory / "neighbors_50.json")
    assert configuration["policy"]["hidden_size"] == 256
    baseline["policy"]["hidden_size"] = 256
    assert configuration == baseline
    world = World(configuration)
    population = PolicyPopulation(
        world.observation_size, world.action_feature_size,
        hidden_size=configuration["policy"]["hidden_size"], capacity=2, seed=42,
    )
    population.add(11)
    population.add(29)
    assert population.memory_report()["parameters_per_agent"] == 751_470
    assert population._hidden.shape == (2, 256)
    inputs = {identifier: [0.1] * world.observation_size for identifier in (11, 29)}
    options = {identifier: [[value] * world.action_feature_size for value in (0.1, 0.2)] for identifier in inputs}
    initial = island_parameters(population, 11)
    population.act(inputs, options, capture_scores=True)
    population.choose_quantities(options)
    restored = PolicyPopulation.from_state_dict(population.state_dict())
    assert population.learn({11: 1.0, 29: -1.0}, inputs, set()) == restored.learn({11: 1.0, 29: -1.0}, inputs, set())
    assert_parameters_match(population._parameters, restored._parameters, exact=True)
    assert_parameters_match(population._first_moments, restored._first_moments, exact=True)
    assert any(not torch.equal(initial[name], value) for name, value in island_parameters(population, 11).items())
    assert not torch.equal(population._hidden[0], population._hidden[1])


def test_quantity_selection_learns_both_choices_and_resumes_without_extra_memory_step() -> None:
    """Compare conditional learning with the loop and an exact pending checkpoint continuation."""
    vectorized = make_population()
    sequential = PolicyPopulation.from_state_dict(vectorized.state_dict(), use_vmap=False)
    for _ in range(3):
        assert vectorized.act(observations(), candidates()) == sequential.act(observations(), candidates())
        hidden = vectorized._hidden.clone()
        assert vectorized.choose_quantities(candidates()) == sequential.choose_quantities(candidates())
        assert torch.equal(hidden, vectorized._hidden)
        restored = PolicyPopulation.from_state_dict(vectorized.state_dict())
        assert vectorized._pending is not None
        assert vectorized._pending.quantity_valid_mask is not None
        assert vectorized._pending.quantity_valid_mask.sum(-1).tolist() == [2, 3]
        results = vectorized.learn({11: 1.0, 29: -0.25}, next_observations(), set())
        sequential.learn({11: 1.0, 29: -0.25}, next_observations(), set())
        assert results == restored.learn({11: 1.0, 29: -0.25}, next_observations(), set())
        assert results["quantity_entropy"] > 0
        assert_parameters_match(vectorized._parameters, sequential._parameters)
        assert_parameters_match(vectorized._parameters, restored._parameters, exact=True)
        assert_parameters_match(vectorized._first_moments, restored._first_moments, exact=True)


def test_conditional_action_probability_contributes_to_actor_loss() -> None:
    """Verify the actor's loss uses the product of template and conditional probabilities."""
    population = make_population()
    population.act(observations(), candidates())
    population.choose_quantities(candidates())
    pending = population._pending
    assert pending is not None and pending.quantity_candidates is not None and pending.quantity_valid_mask is not None and pending.quantity_actions is not None
    parameters = {name: value.index_select(0, pending.slots) for name, value in population._parameters.items()}
    with torch.no_grad():
        logits, values, _ = population._forward(parameters, pending.observations, pending.candidates, pending.valid_mask, pending.previous_hidden)
        quantity_logits, _, _ = population._forward(parameters, pending.observations, pending.quantity_candidates, pending.quantity_valid_mask, pending.previous_hidden)
        selected = logits.log_softmax(-1).gather(1, pending.actions[:, None]).squeeze(1)
        conditional = quantity_logits.log_softmax(-1).gather(1, pending.quantity_actions[:, None]).squeeze(1)
        rewards = torch.tensor([1.0, -0.25])
        expected = (-(selected + conditional) * (rewards - values)).mean().item()
    metrics = population.learn({11: 1.0, 29: -0.25}, {}, {11, 29})
    assert metrics["actor_loss"] == pytest.approx(expected)


def test_captured_logits_explain_sampling_without_changing_actions_or_learning() -> None:
    """Check temperature/count correction, masked options, and exact diagnostic-free continuation."""
    import math

    recorded = make_population()
    recorded.initial_temperature = recorded.final_temperature = 2.0
    recorded.balance_action_kinds = True
    baseline = PolicyPopulation.from_state_dict(recorded.state_dict())
    groups = {11: ["trade", "wait"], 29: ["trade", "trade", "wait"]}
    for _ in range(3):
        first = recorded.act(observations(), candidates(), groups, capture_scores=True)
        assert first == baseline.act(observations(), candidates(), groups)
        second = recorded.choose_quantities(candidates())
        assert second == baseline.choose_quantities(candidates())
        assert torch.equal(recorded._hidden, baseline._hidden)
        assert torch.equal(recorded._sampling_generator.get_state(), baseline._sampling_generator.get_state())
        for identifier, stages in recorded.last_decision_scores.items():
            for stage_name, stage in stages.items():
                assert stage["temperature"] == 2.0
                assert stage["selected_index"] == (first if stage_name == "action" else second)[identifier]
                options = stage["options"]
                assert len(options) == len(candidates()[identifier])
                assert sum(option["probability"] for option in options) == pytest.approx(1.0)
                maximum = max(option["adjusted_logit"] for option in options)
                denominator = sum(math.exp(option["adjusted_logit"] - maximum) for option in options)
                for option in options:
                    assert option["adjusted_logit"] == pytest.approx(option["raw_logit"] / 2 - option["group_log_count"], abs=1e-7)
                    assert option["probability"] == pytest.approx(math.exp(option["adjusted_logit"] - maximum) / denominator)
                if stage_name == "quantity":
                    assert all(option["group_log_count"] == 0 for option in options)
        assert recorded.last_decision_scores[29]["action"]["options"][0]["group_log_count"] == pytest.approx(math.log(2))
        recorded.learn({11: 1.0, 29: -0.25}, next_observations(), set())
        baseline.learn({11: 1.0, 29: -0.25}, next_observations(), set())
        assert_parameters_match(recorded._parameters, baseline._parameters, exact=True)
        assert_parameters_match(recorded._first_moments, baseline._first_moments, exact=True)
    recorded.act(observations(), candidates(), groups)
    assert recorded.last_decision_scores == {}


def test_single_quantity_options_leave_learning_unchanged_and_reject_invalid_sequences() -> None:
    """Singleton refinements add no loss, entropy, or hidden-state change."""
    population = make_population()
    with pytest.raises(RuntimeError):
        population.choose_quantities(candidates())
    baseline = PolicyPopulation.from_state_dict(population.state_dict())
    population.act(observations(), candidates())
    baseline.act(observations(), candidates())
    with pytest.raises(ValueError):
        population.choose_quantities({11: []})
    assert population.choose_quantities({11: [[1.0, 0.0]], 29: [[0.0, 1.0]]}) == {11: 0, 29: 0}
    with pytest.raises(RuntimeError):
        population.choose_quantities(candidates())
    population.learn({11: 1.0, 29: -0.25}, next_observations(), set())
    baseline.learn({11: 1.0, 29: -0.25}, next_observations(), set())
    assert_parameters_match(population._parameters, baseline._parameters, exact=True)


def test_population_does_not_mix_gradients_or_gradient_clipping_between_islands() -> None:
    combined = make_population()
    isolated = PolicyPopulation.from_state_dict(combined.state_dict())
    isolated.remove(29)
    combined.act(observations(), candidates())
    isolated.act({11: observations()[11]}, {11: candidates()[11]})
    combined.learn({11: 2.0, 29: 1_000_000.0}, next_observations(), set())
    isolated.learn({11: 2.0}, {11: next_observations()[11]}, set())
    assert_parameters_match(island_parameters(combined, 11), island_parameters(isolated, 11))


def test_strength_zero_preserves_weights_moments_and_optimizer_clock() -> None:
    population = make_population()
    initial = island_parameters(population, 29)
    population.act(observations(), candidates())
    metrics = population.learn({11: 1.0, 29: 2.0}, next_observations(), set(), strengths={29: 0.0})
    assert metrics["updated_agents"] == 1.0
    assert_parameters_match(initial, island_parameters(population, 29), exact=True)
    slot = population._island_slots[29]
    assert population._update_steps[slot].item() == 0
    for moments in (population._first_moments, population._second_moments):
        assert all(torch.count_nonzero(moment[slot]).item() == 0 for moment in moments.values())
    assert population._update_steps[population._island_slots[11]].item() == 1


def test_strength_scales_parameter_update_instead_of_canceling_inside_adam() -> None:
    full_strength = make_population()
    half_strength = PolicyPopulation.from_state_dict(full_strength.state_dict())
    before = island_parameters(full_strength, 11)
    full_strength.act(observations(), candidates())
    half_strength.act(observations(), candidates())
    full_strength.learn({11: 1.0, 29: 1.0}, next_observations(), set())
    half_strength.learn({11: 1.0, 29: 1.0}, next_observations(), set(), strengths={11: 0.5})
    after_full = island_parameters(full_strength, 11)
    after_half = island_parameters(half_strength, 11)
    for name in before:
        torch.testing.assert_close(after_half[name] - before[name], 0.5 * (after_full[name] - before[name]), atol=3e-8, rtol=0.01)


def test_all_zero_strengths_skip_every_optimizer_without_losing_the_next_turn() -> None:
    population = make_population()
    before = population.state_dict()
    population.act(observations(), candidates())
    metrics = population.learn({11: 1.0, 29: 2.0}, next_observations(), set(), strengths={11: 0.0, 29: 0.0})
    assert metrics["updated_agents"] == 0.0
    assert_parameters_match(before["parameters"], population._parameters, exact=True)
    assert torch.count_nonzero(population._update_steps).item() == 0
    population.act(next_observations(), candidates())


def test_per_island_adam_matches_torch_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    population = PolicyPopulation(
        3, 2, hidden_size=8, seed=17, adam_beta1=0.8, adam_beta2=0.95, adam_epsilon=1e-6,
    )
    population.add(11)
    population.add(29)
    reference_parameters = {
        name: parameter.requires_grad_(True) for name, parameter in island_parameters(population, 11).items()
    }
    reference_optimizer = torch.optim.Adam(
        list(reference_parameters.values()), lr=population.learning_rate,
        betas=(population.adam_beta1, population.adam_beta2), eps=population.adam_epsilon,
    )
    original_grad = torch.autograd.grad
    captured_gradients: list[tuple[Tensor, ...]] = []

    def capture_gradients(*arguments: Any, **keyword_arguments: Any) -> tuple[Tensor, ...]:
        """Give the independent reference optimizer the gradients the population actually used."""
        gradients = original_grad(*arguments, **keyword_arguments)
        captured_gradients.append(tuple(gradient.detach().clone() for gradient in gradients))
        return gradients

    monkeypatch.setattr(torch.autograd, "grad", capture_gradients)
    for _ in range(2):
        population.act(observations(), candidates())
        population.learn({11: 2.0, 29: -5.0}, next_observations(), set())
        for parameter, gradient in zip(reference_parameters.values(), captured_gradients[-1]):
            parameter.grad = gradient[0].clone()
        torch.nn.utils.clip_grad_norm_(list(reference_parameters.values()), population.gradient_clip)
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        assert_parameters_match(island_parameters(population, 11), reference_parameters)
    restored = PolicyPopulation.from_state_dict(population.state_dict())
    assert (restored.adam_beta1, restored.adam_beta2, restored.adam_epsilon) == (0.8, 0.95, 1e-6)


def test_skipping_an_island_does_not_advance_its_adam_bias_correction() -> None:
    staggered = make_population()
    control = PolicyPopulation.from_state_dict(staggered.state_dict())
    staggered.act({29: observations()[29]}, {29: candidates()[29]})
    staggered.learn({29: 2.0}, {29: next_observations()[29]}, set())
    single_choice = {11: [[1.0, 0.0]]}
    staggered.act({11: observations()[11]}, single_choice)
    control.act({11: observations()[11]}, single_choice)
    staggered.learn({11: 1.0}, {11: next_observations()[11]}, set())
    control.learn({11: 1.0}, {11: next_observations()[11]}, set())
    assert_parameters_match(island_parameters(staggered, 11), island_parameters(control, 11), exact=True)


def test_child_inherits_weights_but_not_optimizer_or_recurrent_memory() -> None:
    population = make_population(capacity=2)
    population.act(observations(), candidates())
    population.learn({11: 1.0, 29: 1.0}, next_observations(), set())
    population.add(88, parent_id=11)
    assert_parameters_match(island_parameters(population, 11), island_parameters(population, 88), exact=True)
    child_slot = population._island_slots[88]
    assert population._update_steps[child_slot].item() == 0
    assert torch.count_nonzero(population._hidden[child_slot]).item() == 0
    for moments in (population._first_moments, population._second_moments):
        assert all(torch.count_nonzero(moment[child_slot]).item() == 0 for moment in moments.values())


def test_birth_mutation_changes_only_the_child() -> None:
    population = make_population(mutation_std=0.01)
    parent_before = island_parameters(population, 11)
    population.add(88, parent_id=11)
    assert_parameters_match(parent_before, island_parameters(population, 11), exact=True)
    child = island_parameters(population, 88)
    assert any(not torch.equal(parent_before[name], child[name]) for name in parent_before)


def test_growth_during_pending_transition_preserves_parent_learning_and_child_state() -> None:
    population = make_population(capacity=2)
    population.act(observations(), candidates())
    population.add(88, parent_id=11)
    child_before = island_parameters(population, 88)
    assert population.capacity == 4
    population.learn({11: 1.0, 29: 1.0}, next_observations(), set())
    assert_parameters_match(child_before, island_parameters(population, 88), exact=True)
    assert population._update_steps[population._island_slots[88]].item() == 0
    assert population._update_steps[population._island_slots[11]].item() == 1


def test_removed_slots_are_scrubbed_and_permanent_ids_are_not_reused() -> None:
    population = make_population(capacity=2)
    population.act(observations(), candidates())
    population.learn({11: 1.0, 29: 1.0}, next_observations(), {29})
    previous_slot = population._island_slots[29]
    population.remove(29)
    assert all(torch.count_nonzero(parameter[previous_slot]).item() == 0 for parameter in population._parameters.values())
    with pytest.raises(ValueError, match="already been used"):
        population.add(29)
    population.add(100, parent_id=11)
    assert population._island_slots[100] == previous_slot
    assert population._update_steps[previous_slot].item() == 0
    assert torch.count_nonzero(population._hidden[previous_slot]).item() == 0


def test_padded_candidates_are_never_selected() -> None:
    population = make_population()
    choices = {11: [[0.0, 0.0]], 29: [[float(index), 0.0] for index in range(7)]}
    for _ in range(20):
        selected = population.act(observations(), choices)
        assert selected[11] == 0
        assert 0 <= selected[29] < 7
        population.discard_transition()


def test_pending_transition_cannot_be_overwritten_or_removed() -> None:
    population = make_population()
    population.act(observations(), candidates())
    with pytest.raises(RuntimeError, match="previous transition"):
        population.act(observations(), candidates())
    with pytest.raises(RuntimeError, match="pending transition"):
        population.remove(11)
    population.discard_transition()
    population.remove(11)
    assert population.island_ids == (29,)


def test_checkpoint_restores_pending_learning_sampling_mutation_and_slot_history() -> None:
    population = make_population(mutation_std=0.01)
    population.act(observations(), candidates())
    serialized = io.BytesIO()
    torch.save(population.state_dict(), serialized)
    serialized.seek(0)
    restored = PolicyPopulation.from_state_dict(torch.load(serialized, weights_only=True))
    first_metrics = population.learn({11: 1.0, 29: -0.5}, next_observations(), {29})
    restored_metrics = restored.learn({11: 1.0, 29: -0.5}, next_observations(), {29})
    assert first_metrics == restored_metrics
    for current in (population, restored):
        current.remove(29)
        current.add(90, parent_id=11)
    assert_parameters_match(population._parameters, restored._parameters, exact=True)
    current_observations = {11: observations()[11], 90: observations()[29]}
    current_candidates = {11: candidates()[11], 90: candidates()[29]}
    assert population.act(current_observations, current_candidates) == restored.act(current_observations, current_candidates)
    assert torch.equal(population._hidden, restored._hidden)
    assert population.state_dict()["seen_ids"] == restored.state_dict()["seen_ids"]


def test_checkpoint_is_detached_from_future_population_mutations() -> None:
    population = make_population()
    checkpoint = population.state_dict()
    before = {name: value.clone() for name, value in checkpoint["parameters"].items()}
    population.act(observations(), candidates())
    population.learn({11: 1.0, 29: 1.0}, next_observations(), set())
    assert_parameters_match(checkpoint["parameters"], before, exact=True)


@pytest.mark.parametrize("field", ["parameters", "first_moments", "second_moments"])
@pytest.mark.parametrize("mismatch", ["extra", "missing"])
def test_checkpoint_rejects_parameter_or_optimizer_names_from_another_architecture(field: str, mismatch: str) -> None:
    population = make_population()
    checkpoint = population.state_dict()
    if mismatch == "extra":
        checkpoint[field]["action_bias.bias"] = torch.zeros((population.capacity, 1))
        expected_name = "action_bias.bias"
    else:
        expected_name = "action_bias.weight"
        del checkpoint[field][expected_name]
    with pytest.raises(ValueError, match="do not match the architecture") as raised:
        PolicyPopulation.from_state_dict(checkpoint)
    assert field in str(raised.value)
    assert expected_name in str(raised.value)


def test_terminal_transition_does_not_require_or_bootstrap_next_observation() -> None:
    population = make_population()
    population.act(observations(), candidates())
    metrics = population.learn({11: 1.0, 29: 0.0}, {}, {11, 29})
    assert metrics["updated_agents"] == 2.0
    population.remove(11)
    population.remove(29)
    assert population.island_ids == ()


def test_memory_accounting_counts_allocated_capacity_and_pending_tensors() -> None:
    population = make_population()
    report = population.memory_report()
    assert report["living_agents"] == 2
    assert report["weights_bytes"] == report["parameters_per_agent"] * report["capacity"] * 4
    assert report["optimizer_bytes"] == 2 * report["weights_bytes"] + 8 * report["capacity"]
    assert report["pending_bytes"] == 0
    population.act(observations(), candidates())
    assert population.memory_report()["pending_bytes"] > 0
    population.discard_transition()
    assert population.memory_report()["resident_tensor_bytes"] == report["resident_tensor_bytes"]


@pytest.mark.parametrize("invalid_strength", [-1.0, float("nan"), float("inf")])
def test_invalid_learning_strength_is_rejected_without_consuming_transition(invalid_strength: float) -> None:
    population = make_population()
    population.act(observations(), candidates())
    before = island_parameters(population, 11)
    with pytest.raises(ValueError, match="strengths"):
        population.learn({11: 1.0, 29: 1.0}, next_observations(), set(), strengths={11: invalid_strength})
    assert_parameters_match(before, island_parameters(population, 11), exact=True)
    assert population._pending is not None
