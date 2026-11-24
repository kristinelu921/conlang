"""Independent recurrent island policies stored in one expandable tensor population."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call, vmap


class _IslandPolicy(nn.Module):
    """Score variable action candidates using an island's recurrent observation state."""

    def __init__(self, observation_size: int, action_feature_size: int, hidden_size: int) -> None:
        super().__init__()
        self.observation_encoder = nn.Linear(observation_size, hidden_size, device="meta")
        self.input_gates = nn.Linear(hidden_size, 3 * hidden_size, device="meta")
        self.hidden_gates = nn.Linear(hidden_size, 3 * hidden_size, device="meta")
        self.action_encoder = nn.Linear(action_feature_size, hidden_size, device="meta")
        self.action_query = nn.Linear(hidden_size, hidden_size, device="meta")
        # A shared scalar logit offset cannot change the policy but amplifies rounding noise through Adam.
        self.action_bias = nn.Linear(action_feature_size, 1, bias=False, device="meta")
        self.value_head = nn.Linear(hidden_size, 1, device="meta")
        self.score_scale = math.sqrt(hidden_size)

    def forward(
        self, observation: Tensor, candidate_features: Tensor, valid_mask: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return masked action scores, state value, and the next recurrent memory."""
        encoded_observation = torch.tanh(self.observation_encoder(observation))
        input_reset, input_update, input_candidate = self.input_gates(encoded_observation).chunk(3, -1)
        hidden_reset, hidden_update, hidden_candidate = self.hidden_gates(hidden).chunk(3, -1)
        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        candidate_hidden = torch.tanh(input_candidate + reset_gate * hidden_candidate)
        next_hidden = (1 - update_gate) * candidate_hidden + update_gate * hidden
        action_embeddings = torch.tanh(self.action_encoder(candidate_features))
        action_query = self.action_query(next_hidden)
        logits = (action_embeddings * action_query).sum(-1) / self.score_scale
        logits = logits + self.action_bias(candidate_features).squeeze(-1)
        logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
        return logits, self.value_head(next_hidden).squeeze(-1), next_hidden


@dataclass
class _PendingTransition:
    island_ids: tuple[int, ...]
    slots: Tensor
    observations: Tensor
    candidates: Tensor
    valid_mask: Tensor
    previous_hidden: Tensor
    next_hidden: Tensor
    actions: Tensor
    temperatures: Tensor
    entropy_coefficients: Tensor
    log_candidate_counts: Tensor
    quantity_candidates: Tensor | None = None
    quantity_valid_mask: Tensor | None = None
    quantity_actions: Tensor | None = None


class PolicyPopulation:
    """Own separate island weights, recurrent state, random streams, and Adam histories."""

    def __init__(
        self,
        observation_size: int,
        action_feature_size: int,
        hidden_size: int = 64,
        capacity: int = 32,
        seed: int = 0,
        device: str = "cpu",
        use_vmap: bool = True,
        learning_rate: float = 0.0003,
        discount: float = 0.99,
        entropy_coefficient: float = 0.01,
        value_coefficient: float = 0.5,
        gradient_clip: float = 1.0,
        mutation_std: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        initial_temperature: float = 1.0,
        final_temperature: float = 1.0,
        initial_entropy_coefficient: float | None = None,
        anneal_updates: int = 0,
        balance_action_kinds: bool = False,
        initial_action_biases: dict[int, float] | None = None,
    ) -> None:
        """Allocate resident population storage without allocating a model per island."""
        if min(observation_size, action_feature_size, hidden_size, capacity) < 1:
            raise ValueError("Observation, action, hidden, and capacity sizes must be positive.")
        if initial_entropy_coefficient is None:
            initial_entropy_coefficient = entropy_coefficient
        coefficients = (
            learning_rate, discount, entropy_coefficient, value_coefficient, gradient_clip, mutation_std,
            adam_beta1, adam_beta2, adam_epsilon,
            initial_temperature, final_temperature, initial_entropy_coefficient,
        )
        if not all(math.isfinite(coefficient) for coefficient in coefficients):
            raise ValueError("Learning coefficients must be finite.")
        if learning_rate <= 0 or gradient_clip <= 0 or not 0 <= discount <= 1:
            raise ValueError("Learning rate and gradient clip must be positive; discount must lie in [0, 1].")
        if min(entropy_coefficient, value_coefficient, mutation_std) < 0:
            raise ValueError("Entropy, value, and mutation coefficients must be nonnegative.")
        if not 0 <= adam_beta1 < 1 or not 0 <= adam_beta2 < 1 or adam_epsilon <= 0:
            raise ValueError("Adam betas must lie in [0, 1), and epsilon must be positive.")
        if min(initial_temperature, final_temperature) <= 0 or initial_entropy_coefficient < 0:
            raise ValueError("Temperatures must be positive and entropy coefficients nonnegative.")
        if type(anneal_updates) is not int or anneal_updates < 0 or type(balance_action_kinds) is not bool:
            raise ValueError("Annealing needs a nonnegative integer duration and balancing must be boolean.")
        initial_action_biases = dict(initial_action_biases or {})
        if any(type(index) is not int or not 0 <= index < action_feature_size
               or type(value) not in (int, float) or not math.isfinite(value)
               for index, value in initial_action_biases.items()):
            raise ValueError("Initial action biases require valid feature indices and finite weights.")
        self.observation_size = observation_size
        self.action_feature_size = action_feature_size
        self.hidden_size = hidden_size
        self.capacity = capacity
        self.seed = seed
        self.device = torch.device(device)
        self.use_vmap = use_vmap
        self.learning_rate = learning_rate
        self.discount = discount
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.gradient_clip = gradient_clip
        self.mutation_std = mutation_std
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.initial_entropy_coefficient = initial_entropy_coefficient
        self.anneal_updates = anneal_updates
        self.balance_action_kinds = balance_action_kinds
        self.initial_action_biases = initial_action_biases
        self._template = _IslandPolicy(observation_size, action_feature_size, hidden_size)
        self._parameters = {
            name: torch.zeros((capacity, *parameter.shape), dtype=torch.float32, device=self.device, requires_grad=True)
            for name, parameter in self._template.named_parameters()
        }
        self._first_moments = {name: torch.zeros_like(parameter) for name, parameter in self._parameters.items()}
        self._second_moments = {name: torch.zeros_like(parameter) for name, parameter in self._parameters.items()}
        self._update_steps = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self._hidden = torch.zeros((capacity, hidden_size), dtype=torch.float32, device=self.device)
        self._island_slots: dict[int, int] = {}
        self._seen_ids: set[int] = set()
        self._free_slots = list(reversed(range(capacity)))
        self._initialization_generator = torch.Generator(device="cpu").manual_seed(seed)
        self._sampling_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        self._pending: _PendingTransition | None = None
        self.last_decision_scores: dict[int, dict[str, Any]] = {}

    @property
    def island_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._island_slots))

    def _grow(self) -> None:
        """Double storage while keeping island identities and pending transitions stable."""
        previous_capacity = self.capacity
        self.capacity *= 2
        for name, parameter in self._parameters.items():
            empty_parameters = torch.zeros_like(parameter)
            self._parameters[name] = torch.cat((parameter.detach(), empty_parameters)).requires_grad_(True)
            self._first_moments[name] = torch.cat((self._first_moments[name], empty_parameters))
            self._second_moments[name] = torch.cat((self._second_moments[name], empty_parameters))
        self._update_steps = torch.cat((self._update_steps, torch.zeros_like(self._update_steps)))
        self._hidden = torch.cat((self._hidden, torch.zeros_like(self._hidden)))
        self._free_slots.extend(reversed(range(previous_capacity, self.capacity)))

    def add(self, island_id: int, parent_id: int | None = None) -> None:
        """Create fresh learning state with independent initialization or mutated inheritance."""
        if island_id in self._seen_ids:
            raise ValueError(f"Island ID {island_id} has already been used.")
        if parent_id is not None and parent_id not in self._island_slots:
            raise KeyError(f"Parent island {parent_id} is not alive.")
        if not self._free_slots:
            self._grow()
        slot = self._free_slots.pop()
        with torch.no_grad():
            for name, parameter in self._parameters.items():
                if parent_id is None:
                    weight_name = name.rsplit(".", 1)[0] + ".weight"
                    fan_in = self._parameters[weight_name].shape[-1]
                    bound = 1 / math.sqrt(fan_in)
                    initialized = torch.empty(parameter.shape[1:], dtype=torch.float32, device="cpu").uniform_(
                        -bound, bound, generator=self._initialization_generator
                    )
                    parameter[slot].copy_(initialized)
                else:
                    parameter[slot].copy_(parameter[self._island_slots[parent_id]])
                    if self.mutation_std:
                        mutation = torch.randn(parameter.shape[1:], dtype=torch.float32, device="cpu", generator=self._initialization_generator)
                        parameter[slot].add_(mutation.to(self.device), alpha=self.mutation_std)
                self._first_moments[name][slot].zero_()
                self._second_moments[name][slot].zero_()
            if parent_id is None:
                for index, value in self.initial_action_biases.items():
                    self._parameters["action_bias.weight"][slot, 0, index].add_(value)
            self._hidden[slot].zero_()
            self._update_steps[slot] = 0
        self._island_slots[island_id] = slot
        self._seen_ids.add(island_id)

    def remove(self, island_id: int) -> None:
        """Release a deceased island's storage after its final transition has been handled."""
        if self._pending is not None and island_id in self._pending.island_ids:
            raise RuntimeError("Learn or discard the pending transition before removing its island.")
        slot = self._island_slots.pop(island_id)
        with torch.no_grad():
            for name, parameter in self._parameters.items():
                parameter[slot].zero_()
                self._first_moments[name][slot].zero_()
                self._second_moments[name][slot].zero_()
            self._hidden[slot].zero_()
            self._update_steps[slot] = 0
        self._free_slots.append(slot)

    def _forward_one(
        self,
        parameters: dict[str, Tensor],
        observation: Tensor,
        candidates: Tensor,
        valid_mask: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return functional_call(self._template, parameters, (observation, candidates, valid_mask, hidden), strict=True)

    def _forward(
        self,
        parameters: dict[str, Tensor],
        observations: Tensor,
        candidates: Tensor,
        valid_mask: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate independent parameter slices through vmap or the equivalent Python loop."""
        if self.use_vmap:
            return vmap(self._forward_one)(parameters, observations, candidates, valid_mask, hidden)
        results = [
            self._forward_one(
                {name: parameter[index] for name, parameter in parameters.items()},
                observations[index], candidates[index], valid_mask[index], hidden[index],
            )
            for index in range(len(observations))
        ]
        logits, values, next_hidden = zip(*results)
        return torch.stack(logits), torch.stack(values), torch.stack(next_hidden)

    def _exploration_schedule(self, slots: Tensor) -> tuple[Tensor, Tensor]:
        """Anneal each island's exploration using its own completed optimizer updates."""
        steps = self._update_steps.index_select(0, slots).to(torch.float32)
        progress = (steps / self.anneal_updates).clamp(max=1) if self.anneal_updates else torch.ones_like(steps)
        temperatures = self.initial_temperature + progress * (self.final_temperature - self.initial_temperature)
        entropy_coefficients = self.initial_entropy_coefficient + progress * (self.entropy_coefficient - self.initial_entropy_coefficient)
        return temperatures, entropy_coefficients

    @staticmethod
    def _behavior_logits(logits: Tensor, valid_mask: Tensor, temperatures: Tensor, log_candidate_counts: Tensor) -> Tensor:
        """Temper scores before correcting action counts, preserving a balanced prior at every temperature."""
        adjusted = logits / temperatures[:, None] - log_candidate_counts
        return adjusted.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)

    def act(
        self, observations: dict[int, list[float]], candidates: dict[int, list[list[float]]],
        candidate_groups: dict[int, list[str]] | None = None,
        capture_scores: bool = False,
    ) -> dict[int, int]:
        """Choose valid candidates together and cache each island's own transition inputs."""
        if self._pending is not None:
            raise RuntimeError("Learn or discard the previous transition before acting again.")
        self.last_decision_scores = {}
        if observations.keys() != candidates.keys():
            raise ValueError("Observations and candidates must contain the same island IDs.")
        if not observations:
            return {}
        island_ids = tuple(sorted(observations))
        if not observations.keys() <= self._island_slots.keys():
            raise KeyError("Every acting island must first be added to the policy population.")
        if any(len(observations[island_id]) != self.observation_size for island_id in island_ids):
            raise ValueError("Observation width does not match the policy configuration.")
        if any(not candidates[island_id] for island_id in island_ids):
            raise ValueError("Every island needs at least one valid action candidate.")
        if any(len(candidate) != self.action_feature_size for choices in candidates.values() for candidate in choices):
            raise ValueError("Candidate feature width does not match the policy configuration.")
        maximum_candidates = max(len(candidates[island_id]) for island_id in island_ids)
        slots = torch.tensor([self._island_slots[island_id] for island_id in island_ids], dtype=torch.long, device=self.device)
        observation_tensor = torch.tensor([observations[island_id] for island_id in island_ids], dtype=torch.float32, device=self.device)
        candidate_tensor = torch.tensor([
            candidates[island_id] + [[0.0] * self.action_feature_size] * (maximum_candidates - len(candidates[island_id]))
            for island_id in island_ids
        ], dtype=torch.float32, device=self.device)
        valid_mask = torch.tensor([
            [index < len(candidates[island_id]) for index in range(maximum_candidates)] for island_id in island_ids
        ], device=self.device)
        if not torch.isfinite(observation_tensor).all() or not torch.isfinite(candidate_tensor).all():
            raise ValueError("Observations and candidate features must be finite.")
        log_candidate_counts = torch.zeros_like(valid_mask, dtype=torch.float32)
        if self.balance_action_kinds:
            if candidate_groups is None or candidate_groups.keys() != observations.keys():
                raise ValueError("Balanced exploration requires an action-kind label for every candidate.")
            for index, island_id in enumerate(island_ids):
                groups = candidate_groups[island_id]
                if len(groups) != len(candidates[island_id]) or any(not isinstance(group, str) or not group for group in groups):
                    raise ValueError("Action-kind labels must be nonempty strings aligned with the candidates.")
                counts = Counter(groups)
                log_candidate_counts[index, :len(groups)] = torch.tensor(
                    [math.log(counts[group]) for group in groups], dtype=torch.float32, device=self.device,
                )
        temperatures, entropy_coefficients = self._exploration_schedule(slots)
        with torch.no_grad():
            previous_hidden = self._hidden.index_select(0, slots)
            parameters = {name: parameter.index_select(0, slots) for name, parameter in self._parameters.items()}
            raw_logits, _, next_hidden = self._forward(parameters, observation_tensor, candidate_tensor, valid_mask, previous_hidden)
            logits = self._behavior_logits(raw_logits, valid_mask, temperatures, log_candidate_counts)
            probabilities = logits.softmax(-1)
            cumulative_probabilities = probabilities.cumsum(-1)
            cumulative_probabilities /= cumulative_probabilities[:, -1:].clone()
            uniforms = (1 - torch.rand((len(island_ids), 1), dtype=torch.float32, device="cpu", generator=self._sampling_generator)).to(self.device)
            actions = (cumulative_probabilities < uniforms).sum(-1)
            self._hidden.index_copy_(0, slots, next_hidden)
            if capture_scores:
                self._capture_scores("action", island_ids, raw_logits, logits, probabilities, valid_mask, actions, temperatures, log_candidate_counts)
        self._pending = _PendingTransition(
            island_ids, slots, observation_tensor, candidate_tensor, valid_mask, previous_hidden, next_hidden, actions,
            temperatures, entropy_coefficients, log_candidate_counts,
        )
        return dict(zip(island_ids, actions.cpu().tolist()))

    def _capture_scores(
        self, stage: str, island_ids: tuple[int, ...], raw_logits: Tensor, adjusted_logits: Tensor,
        probabilities: Tensor, valid_mask: Tensor, actions: Tensor, temperatures: Tensor, log_candidate_counts: Tensor,
    ) -> None:
        """Detach the actual sampling scores for valid options without advancing any random stream."""
        counts = valid_mask.sum(-1).cpu().tolist()
        raw_values = raw_logits.detach().cpu().tolist()
        adjusted_values = adjusted_logits.detach().cpu().tolist()
        probability_values = probabilities.detach().cpu().tolist()
        group_counts = log_candidate_counts.detach().cpu().tolist()
        selected_indices = actions.cpu().tolist()
        temperature_values = temperatures.cpu().tolist()
        for row, island_id in enumerate(island_ids):
            self.last_decision_scores.setdefault(island_id, {})[stage] = {
                "temperature": temperature_values[row], "selected_index": selected_indices[row],
                "options": [{"raw_logit": raw_values[row][index], "adjusted_logit": adjusted_values[row][index],
                    "probability": probability_values[row][index], "group_log_count": group_counts[row][index]}
                    for index in range(counts[row])],
            }

    def discard_transition(self) -> None:
        """Keep the recurrent state advanced by acting, without learning from that action."""
        self._pending = None

    def choose_quantities(self, candidates: dict[int, list[list[float]]]) -> dict[int, int]:
        """Choose conditional trade quantities using the same recurrent state without advancing it twice."""
        pending = self._pending
        if pending is None or pending.quantity_candidates is not None:
            raise RuntimeError("Quantity selection requires an unrefined pending action.")
        if set(candidates) != set(pending.island_ids):
            raise ValueError("Quantity choices must contain exactly the pending island IDs.")
        if any(not choices or any(len(candidate) != self.action_feature_size for candidate in choices) for choices in candidates.values()):
            raise ValueError("Quantity choices need nonempty candidates of the configured width.")
        maximum_candidates = max(len(choices) for choices in candidates.values())
        candidate_tensor = torch.tensor([
            candidates[island_id] + [[0.0] * self.action_feature_size] * (maximum_candidates - len(candidates[island_id]))
            for island_id in pending.island_ids
        ], dtype=torch.float32, device=self.device)
        valid_mask = torch.tensor([
            [index < len(candidates[island_id]) for index in range(maximum_candidates)]
            for island_id in pending.island_ids
        ], device=self.device)
        if not torch.isfinite(candidate_tensor).all():
            raise ValueError("Quantity features must be finite.")
        with torch.no_grad():
            parameters = {name: parameter.index_select(0, pending.slots) for name, parameter in self._parameters.items()}
            raw_logits, _, _ = self._forward(parameters, pending.observations, candidate_tensor, valid_mask, pending.previous_hidden)
            group_counts = torch.zeros_like(raw_logits)
            logits = self._behavior_logits(raw_logits, valid_mask, pending.temperatures, group_counts)
            probabilities = logits.softmax(-1)
            cumulative_probabilities = probabilities.cumsum(-1)
            cumulative_probabilities /= cumulative_probabilities[:, -1:].clone()
            uniforms = (1 - torch.rand((len(pending.island_ids), 1), generator=self._sampling_generator)).to(self.device)
            actions = (cumulative_probabilities < uniforms).sum(-1)
            if self.last_decision_scores:
                self._capture_scores("quantity", pending.island_ids, raw_logits, logits, probabilities, valid_mask, actions, pending.temperatures, group_counts)
        pending.quantity_candidates = candidate_tensor
        pending.quantity_valid_mask = valid_mask
        pending.quantity_actions = actions
        return dict(zip(pending.island_ids, actions.cpu().tolist()))

    def learn(
        self,
        rewards: dict[int, float],
        next_observations: dict[int, list[float]],
        terminated: set[int],
        strengths: dict[int, float] | None = None,
    ) -> dict[str, float]:
        """Apply separate one-step actor-critic updates, with per-island learning-rate strength."""
        pending = self._pending
        if pending is None:
            if rewards:
                raise RuntimeError("There is no pending action to learn from.")
            return {"updated_agents": 0.0}
        if set(rewards) != set(pending.island_ids):
            raise ValueError("Rewards must contain exactly the islands in the pending transition.")
        if not terminated <= set(pending.island_ids):
            raise ValueError("Terminated IDs must belong to the pending transition.")
        if any(island_id not in next_observations for island_id in pending.island_ids if island_id not in terminated):
            raise ValueError("Every surviving island needs its next observation.")
        strength_values = [1.0 if strengths is None else strengths.get(island_id, 1.0) for island_id in pending.island_ids]
        if any(not math.isfinite(value) or value < 0 for value in strength_values):
            raise ValueError("Learning strengths must be finite and nonnegative.")
        if any(not math.isfinite(reward) for reward in rewards.values()):
            raise ValueError("Rewards must be finite.")
        next_values_list = [
            [0.0] * self.observation_size if island_id in terminated else next_observations[island_id]
            for island_id in pending.island_ids
        ]
        if any(len(observation) != self.observation_size for observation in next_values_list):
            raise ValueError("Next observation width does not match the policy configuration.")
        next_observation_tensor = torch.tensor(next_values_list, dtype=torch.float32, device=self.device)
        if not torch.isfinite(next_observation_tensor).all():
            raise ValueError("Next observations must be finite.")
        parameters = {name: parameter.index_select(0, pending.slots) for name, parameter in self._parameters.items()}
        count = len(pending.island_ids)
        with torch.no_grad():
            _, bootstrap_values, _ = self._forward(
                parameters, next_observation_tensor,
                torch.zeros((count, 1, self.action_feature_size), dtype=torch.float32, device=self.device),
                torch.ones((count, 1), dtype=torch.bool, device=self.device), pending.next_hidden,
            )
            survives = torch.tensor([island_id not in terminated for island_id in pending.island_ids], device=self.device)
            reward_tensor = torch.tensor([rewards[island_id] for island_id in pending.island_ids], dtype=torch.float32, device=self.device)
            targets = reward_tensor + self.discount * bootstrap_values * survives
        logits, values, _ = self._forward(
            parameters, pending.observations, pending.candidates, pending.valid_mask, pending.previous_hidden
        )
        logits = self._behavior_logits(logits, pending.valid_mask, pending.temperatures, pending.log_candidate_counts)
        log_probabilities = logits.log_softmax(-1)
        chosen_log_probabilities = log_probabilities.gather(1, pending.actions[:, None]).squeeze(1)
        entropy = -(log_probabilities.exp() * log_probabilities).sum(-1)
        # Relative to the balanced prior, entropy cannot reward a kind just for having more variants.
        exploration_bonus = entropy - (log_probabilities.exp() * pending.log_candidate_counts).sum(-1)
        quantity_entropy = torch.zeros_like(entropy)
        if pending.quantity_candidates is not None:
            assert pending.quantity_valid_mask is not None and pending.quantity_actions is not None
            quantity_logits, _, _ = self._forward(
                parameters, pending.observations, pending.quantity_candidates,
                pending.quantity_valid_mask, pending.previous_hidden,
            )
            quantity_logits = self._behavior_logits(
                quantity_logits, pending.quantity_valid_mask, pending.temperatures, torch.zeros_like(quantity_logits),
            )
            quantity_log_probabilities = quantity_logits.log_softmax(-1)
            chosen_log_probabilities = chosen_log_probabilities + quantity_log_probabilities.gather(1, pending.quantity_actions[:, None]).squeeze(1)
            quantity_entropy = -(quantity_log_probabilities.exp() * quantity_log_probabilities).sum(-1)
            # Normalize the conditional bonus so a larger quantity range does not dominate exploration.
            maximum_entropy = pending.quantity_valid_mask.sum(-1).float().log().clamp(min=1)
            exploration_bonus = exploration_bonus + quantity_entropy / maximum_entropy
        advantages = targets - values
        # Asynchronous Methods for Deep Reinforcement Learning: https://arxiv.org/abs/1602.01783 (one-step, synchronous updates).
        actor_losses = -chosen_log_probabilities * advantages.detach()
        value_losses = 0.5 * advantages.square()
        losses = actor_losses + self.value_coefficient * value_losses - pending.entropy_coefficients * exploration_bonus
        # Summing keeps each island's gradient independent of the number of living islands.
        gradients = torch.autograd.grad(losses.sum(), tuple(parameters.values()))
        gradient_norms = torch.stack([gradient.detach().reshape(count, -1).square().sum(-1) for gradient in gradients]).sum(0).sqrt()
        if not torch.isfinite(losses).all() or not torch.isfinite(gradient_norms).all():
            raise FloatingPointError("Nonfinite policy loss or gradient; no optimizer state was updated.")
        clipping = (self.gradient_clip / (gradient_norms + 1e-6)).clamp(max=1)
        strength_tensor = torch.tensor(strength_values, dtype=torch.float32, device=self.device)
        if not torch.isfinite(strength_tensor).all():
            raise ValueError("Learning strengths must fit in a float32 tensor.")
        active_indices = (strength_tensor > 0).nonzero(as_tuple=True)[0]
        active_slots = pending.slots.index_select(0, active_indices)
        with torch.no_grad():
            next_steps = self._update_steps.index_select(0, active_slots) + 1
            for (name, parameter), gradient in zip(self._parameters.items(), gradients):
                broadcast_shape = (-1,) + (1,) * (gradient.ndim - 1)
                clipped_gradient = gradient * clipping.reshape(broadcast_shape)
                active_gradient = clipped_gradient.index_select(0, active_indices)
                first_moment = self.adam_beta1 * self._first_moments[name].index_select(0, active_slots) + (1 - self.adam_beta1) * active_gradient
                second_moment = self.adam_beta2 * self._second_moments[name].index_select(0, active_slots) + (1 - self.adam_beta2) * active_gradient.square()
                first_correction = (1 - self.adam_beta1 ** next_steps).reshape(broadcast_shape)
                second_correction = (1 - self.adam_beta2 ** next_steps).reshape(broadcast_shape)
                normalized_update = (first_moment / first_correction) / ((second_moment / second_correction).sqrt() + self.adam_epsilon)
                rate = self.learning_rate * strength_tensor.index_select(0, active_indices).reshape(broadcast_shape)
                parameter.index_copy_(0, active_slots, parameter.index_select(0, active_slots) - rate * normalized_update)
                self._first_moments[name].index_copy_(0, active_slots, first_moment)
                self._second_moments[name].index_copy_(0, active_slots, second_moment)
            self._update_steps.index_copy_(0, active_slots, next_steps)
        self._pending = None
        return {
            "loss": losses.detach().mean().item(),
            "actor_loss": actor_losses.detach().mean().item(),
            "value_loss": value_losses.detach().mean().item(),
            "entropy": entropy.detach().mean().item(),
            "exploration_bonus": exploration_bonus.detach().mean().item(),
            "temperature": pending.temperatures.mean().item(),
            "entropy_coefficient": pending.entropy_coefficients.mean().item(),
            "candidate_count": pending.valid_mask.sum(-1).float().mean().item(),
            "mean_reward": reward_tensor.float().mean().item(),
            "gradient_norm": gradient_norms.detach().mean().item(),
            "updated_agents": float(len(active_indices)),
            **({"quantity_entropy": quantity_entropy.detach().mean().item()} if pending.quantity_candidates is not None else {}),
        }

    def state_dict(self) -> dict[str, Any]:
        """Snapshot parameters, learning state, identity history, random streams, and pending work."""
        pending = None
        if self._pending is not None:
            pending = {
                name: value.detach().cpu().clone() if isinstance(value, Tensor) else value
                for name, value in vars(self._pending).items()
            }
        return {
            "version": 3,
            "configuration": {
                "observation_size": self.observation_size, "action_feature_size": self.action_feature_size,
                "hidden_size": self.hidden_size, "capacity": self.capacity, "seed": self.seed,
                "use_vmap": self.use_vmap, "learning_rate": self.learning_rate, "discount": self.discount,
                "entropy_coefficient": self.entropy_coefficient, "value_coefficient": self.value_coefficient,
                "gradient_clip": self.gradient_clip, "mutation_std": self.mutation_std,
                "adam_beta1": self.adam_beta1, "adam_beta2": self.adam_beta2, "adam_epsilon": self.adam_epsilon,
                "initial_temperature": self.initial_temperature, "final_temperature": self.final_temperature,
                "initial_entropy_coefficient": self.initial_entropy_coefficient,
                "anneal_updates": self.anneal_updates, "balance_action_kinds": self.balance_action_kinds,
                "initial_action_biases": dict(self.initial_action_biases),
            },
            "parameters": {name: value.detach().cpu().clone() for name, value in self._parameters.items()},
            "first_moments": {name: value.cpu().clone() for name, value in self._first_moments.items()},
            "second_moments": {name: value.cpu().clone() for name, value in self._second_moments.items()},
            "update_steps": self._update_steps.cpu().clone(),
            "hidden": self._hidden.cpu().clone(),
            "island_slots": dict(self._island_slots), "seen_ids": sorted(self._seen_ids),
            "free_slots": list(self._free_slots),
            "initialization_rng": self._initialization_generator.get_state().clone(),
            "sampling_rng": self._sampling_generator.get_state().clone(),
            "pending": pending,
        }

    @classmethod
    def from_state_dict(
        cls, state: dict[str, Any], device: str = "cpu", use_vmap: bool | None = None
    ) -> PolicyPopulation:
        """Restore a complete checkpoint without resetting births, sampling, or optimizer clocks."""
        if state["version"] not in (1, 2, 3):
            raise ValueError("Unsupported policy checkpoint version.")
        configuration = dict(state["configuration"])
        if use_vmap is not None:
            configuration["use_vmap"] = use_vmap
        population = cls(**configuration, device=device)
        expected_parameter_names = set(population._parameters)
        for field in ("parameters", "first_moments", "second_moments"):
            saved_parameter_names = set(state[field])
            if saved_parameter_names != expected_parameter_names:
                missing = sorted(expected_parameter_names - saved_parameter_names)
                unexpected = sorted(saved_parameter_names - expected_parameter_names)
                raise ValueError(
                    f"Policy checkpoint {field} do not match the architecture: "
                    f"missing={missing}, unexpected={unexpected}."
                )
        with torch.no_grad():
            for name in population._parameters:
                population._parameters[name].copy_(state["parameters"][name])
                population._first_moments[name].copy_(state["first_moments"][name])
                population._second_moments[name].copy_(state["second_moments"][name])
            population._update_steps.copy_(state["update_steps"])
            population._hidden.copy_(state["hidden"])
        population._island_slots = dict(state["island_slots"])
        population._seen_ids = set(state["seen_ids"])
        population._free_slots = list(state["free_slots"])
        population._initialization_generator.set_state(state["initialization_rng"].cpu())
        population._sampling_generator.set_state(state["sampling_rng"].cpu())
        if state["pending"] is not None:
            pending = dict(state["pending"])
            pending["island_ids"] = tuple(pending["island_ids"])
            for name in pending.keys() - {"island_ids"}:
                if pending[name] is not None:
                    pending[name] = pending[name].to(population.device)
            if state["version"] == 1:
                pending["temperatures"] = torch.ones(len(pending["island_ids"]), device=population.device)
                pending["entropy_coefficients"] = torch.full_like(pending["temperatures"], population.entropy_coefficient)
                pending["log_candidate_counts"] = torch.zeros_like(pending["valid_mask"], dtype=torch.float32)
            population._pending = _PendingTransition(**pending)
        return population

    def memory_report(self) -> dict[str, int]:
        """Report resident tensor bytes separately from transient inference and training work."""
        parameters_per_island = sum(parameter[0].numel() for parameter in self._parameters.values())
        weights_bytes = sum(parameter.numel() * parameter.element_size() for parameter in self._parameters.values())
        recurrent_bytes = self._hidden.numel() * self._hidden.element_size()
        counter_bytes = self._update_steps.numel() * self._update_steps.element_size()
        pending_bytes = 0
        if self._pending is not None:
            pending_bytes = sum(
                value.numel() * value.element_size() for value in vars(self._pending).values() if isinstance(value, Tensor)
            )
        return {
            "living_agents": len(self._island_slots), "capacity": self.capacity,
            "parameters_per_agent": parameters_per_island, "weights_bytes": weights_bytes,
            "optimizer_bytes": 2 * weights_bytes + counter_bytes, "recurrent_bytes": recurrent_bytes,
            "pending_bytes": pending_bytes,
            "resident_tensor_bytes": 3 * weights_bytes + counter_bytes + recurrent_bytes + pending_bytes,
        }
