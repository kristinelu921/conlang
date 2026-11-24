"""Shared Qwen inference with private island histories and auditable constrained actions."""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import time
from typing import Any

import torch
from torch import Tensor, nn

from island_community.language_policy import TokenChoiceTrie, decision_messages, parse_choice, private_context
from island_community.world import Action, Decision, TurnResult, World


def chunked_qwen_attention(
    module: nn.Module, query: Tensor, key: Tensor, value: Tensor, attention_mask: Tensor | None,
    scaling: float, dropout: float = 0.0, **keywords: Any,
) -> tuple[Tensor, Tensor | None]:
    """Bound eager attention memory without dropping keys, context, or gradient contributions."""
    from torch.utils.checkpoint import checkpoint
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

    if query.shape[-2] <= 512:
        return eager_attention_forward(module, query, key, value, attention_mask,
                                       scaling=scaling, dropout=dropout, **keywords)
    if dropout != 0:
        raise ValueError("Chunked Qwen attention requires the pinned model's zero attention dropout")
    if attention_mask is not None and attention_mask.shape[-2] != query.shape[-2]:
        raise ValueError("Expected the eager causal mask to have one row per query token")

    def attend(query_block: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None) -> Tensor:
        """Compute a full-key attention block whose intermediates can be recomputed on backward."""
        return eager_attention_forward(module, query_block, keys, values, mask, scaling=scaling, dropout=0.0)[0]

    blocks = []
    needs_gradients = torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (query, key, value))
    for start in range(0, query.shape[-2], 512):
        query_block = query[..., start:start + 512, :]
        mask = None if attention_mask is None else attention_mask[..., start:start + 512, :]
        if needs_gradients:
            # Training Deep Nets with Sublinear Memory Cost, https://arxiv.org/abs/1604.06174
            result = checkpoint(attend, query_block, key, value, mask,
                                use_reentrant=False, preserve_rng_state=False)
        else:
            result = attend(query_block, key, value, mask)
        blocks.append(result)
    return torch.cat(blocks, dim=1), None


class IslandAdapters(nn.Module):
    """Keep a frozen projection and independently selected low-rank residuals for each island."""

    def __init__(self, projection: nn.Linear, rank: int) -> None:
        super().__init__()
        self.base = projection
        self.rank = rank
        self.input_factors = nn.ParameterDict()
        self.output_factors = nn.ParameterDict()
        self.active_island: str | None = None

    def forward(self, values: Tensor) -> Tensor:
        """Add only the active island's residual to the shared frozen projection."""
        result = self.base(values)
        if self.active_island is not None:
            # LoRA: Low-Rank Adaptation of Large Language Models, https://arxiv.org/abs/2106.09685
            result = result + (values @ self.input_factors[self.active_island].T) @ self.output_factors[self.active_island].T
        return result


class QwenPopulation:
    """Own one immutable language model and isolated per-island experience and optional adapters."""

    def __init__(self, settings: dict[str, Any], seed: int, observation_size: int, device: str = "cpu") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers

        self.settings = copy.deepcopy(settings)
        self.seed = seed
        self.observation_size = observation_size
        if device.split(":", 1)[0] == "tpu":
            import torch_tpu  # Register the native PyTorch device before constructing it.
            from transformers import AttentionInterface
            from transformers.masking_utils import AttentionMaskInterface, eager_mask

            torch.tpu.distributed.set_tpu_launch_env(nnodes=1, nproc_per_node=1)
            AttentionInterface.register("island_chunked_eager", chunked_qwen_attention)
            AttentionMaskInterface.register("island_chunked_eager", eager_mask)

        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda", "tpu"}:
            raise ValueError("Qwen supports authorized CPU, CUDA, or native TPU execution")
        torch.set_num_threads(settings["threads"])
        self.tokenizer = AutoTokenizer.from_pretrained(settings["model_id"], revision=settings["revision"], trust_remote_code=False)
        # Native SDPA fails with a VMEM bounds error for the full Qwen shape on v5e.
        attention_implementation = "island_chunked_eager" if self.device.type == "tpu" else "sdpa"
        self.model = AutoModelForCausalLM.from_pretrained(
            settings["model_id"], revision=settings["revision"], trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.float32, attn_implementation=attention_implementation,
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        self.base_parameters = list(self.model.parameters())
        self.base_checksum = self.frozen_checksum()
        self.adapter_modules: list[IslandAdapters] = []
        if settings["mode"] != "frozen":
            for layer in self.model.model.layers[-settings["adapter_layers"]:]:
                for name in ("q_proj", "v_proj"):
                    projection = getattr(layer.self_attn, name)
                    assert isinstance(projection, nn.Linear)
                    adapter = IslandAdapters(projection, settings["adapter_rank"])
                    setattr(layer.self_attn, name, adapter)
                    self.adapter_modules.append(adapter)
        self.histories: dict[int, list[dict[str, Any]]] = {}
        self.optimizers: dict[int, torch.optim.Adam] = {}
        self.critics: dict[int, nn.Module] = {}
        self.critic_optimizers: dict[int, torch.optim.Adam] = {}
        self.updates: dict[int, int] = {}
        self.pending: dict[int, list[dict[str, Any]]] = {}
        self.last_decision_scores: dict[int, dict[str, Any]] = {}
        self.runtime = {"transformers_version": transformers.__version__, "dtype": "float32",
                        "attention_implementation": attention_implementation,
                        "base_parameters": sum(parameter.numel() for parameter in self.base_parameters),
                        "base_sha256_before": self.base_checksum, "model_id": settings["model_id"],
                        "revision": settings["revision"], "mode": settings["mode"]}
        self.runtime["policy_support"] = "recorded_generation_tokens"
        if self.device.type == "tpu":
            self.runtime.update(torch_tpu_version=version("torch-tpu"), generation_cache="static",
                                generation_cache_tokens=settings["max_context_tokens"], prefill_chunk_tokens=512,
                                generation_auto_compile=False, attention_query_chunk_tokens=512,
                                attention_gradient_checkpointing=True)

    def frozen_checksum(self) -> str:
        """Hash all shared pretrained parameters to detect any unintended weight change."""
        digest = hashlib.sha256()
        for parameter in self.base_parameters:
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def add(self, island_id: int, parent_id: int | None = None) -> None:
        """Start a private history and, when enabled, independent inherited adapter and optimizer state."""
        identifier = str(island_id)
        self.histories[island_id] = []
        self.updates[island_id] = 0
        parameters = []
        initialization_device = torch.device("cpu") if self.device.type == "tpu" else self.device
        generator = torch.Generator(device=initialization_device).manual_seed(self.seed + island_id * 1009)
        for module in self.adapter_modules:
            if parent_id is None:
                input_factor = (torch.randn(module.rank, module.base.in_features, generator=generator,
                                            device=initialization_device) / module.base.in_features ** 0.5).to(self.device)
                output_factor = torch.zeros(module.base.out_features, module.rank, device=self.device)
            else:
                input_factor = module.input_factors[str(parent_id)].detach().clone()
                output_factor = module.output_factors[str(parent_id)].detach().clone()
            module.input_factors[identifier] = nn.Parameter(input_factor)
            module.output_factors[identifier] = nn.Parameter(output_factor)
            parameters.extend([module.input_factors[identifier], module.output_factors[identifier]])
        if parameters:
            self.optimizers[island_id] = torch.optim.Adam(parameters, lr=self.settings["learning_rate"])
        if self.settings["mode"] == "value_adapter":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + island_id * 1009)
                critic = nn.Sequential(nn.Linear(self.observation_size, self.settings["value_hidden_size"]),
                                       nn.Tanh(), nn.Linear(self.settings["value_hidden_size"], 1)).to(self.device)
            if parent_id is not None:
                critic.load_state_dict(self.critics[parent_id].state_dict())
            self.critics[island_id] = critic
            self.critic_optimizers[island_id] = torch.optim.Adam(critic.parameters(), lr=self.settings["value_learning_rate"])

    def _activate(self, island_id: int) -> None:
        """Select exactly one island's trainable parameters while keeping every other adapter frozen."""
        for module in self.adapter_modules:
            module.active_island = str(island_id)
            for identifier, parameter in module.input_factors.items():
                parameter.requires_grad_(identifier == str(island_id))
                module.output_factors[identifier].requires_grad_(identifier == str(island_id))

    def _choose(self, messages: list[dict[str, str]], count: int, seed: int) -> tuple[int, dict[str, Any], dict[str, Any]]:
        """Generate bounded reasoning followed by an exact JSON choice using a token prefix grammar."""
        from transformers.generation.utils import GenerateDecoderOnlyOutput

        started = time.monotonic()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        responses = [f'{{"choice":{index}}}' for index in range(count)]
        sequences = [self.tokenizer.encode(response, add_special_tokens=False) for response in responses]
        for response, sequence in zip(responses, sequences):
            if self.tokenizer.decode(sequence, clean_up_tokenization_spaces=False) != response:
                raise ValueError("Tokenizer does not round-trip the canonical action grammar")
        grammar = TokenChoiceTrie(sequences, self.tokenizer.eos_token_id)
        maximum = len(prompt_ids) + self.settings["max_reasoning_tokens"] + grammar.maximum_length + 8
        if maximum > self.settings["max_context_tokens"]:
            raise ValueError(f"Decision needs {maximum} tokens, exceeding the configured context limit; no state was truncated")
        end_thinking = self.tokenizer.encode("</think>", add_special_tokens=False)
        if len(end_thinking) != 1:
            raise ValueError("Expected Qwen's single end-of-thinking token")
        sampling = dict(do_sample=True, temperature=self.settings["temperature"],
                        top_p=self.settings["top_p"], top_k=self.settings["top_k"],
                        pad_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.0,
                        return_dict_in_generate=True, output_scores=True)
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        generation_device = torch.device("cpu") if self.device.type == "tpu" else self.device
        cache_arguments = {}
        if self.device.type == "tpu":
            from transformers import StaticCache

            sampling["prefill_chunk_size"] = 512
            sampling["disable_compile"] = True
            cache_arguments["past_key_values"] = StaticCache(
                config=self.model.config, max_cache_len=self.settings["max_context_tokens"],
            )
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            inputs = torch.tensor([prompt_ids], device=generation_device)
            reasoning_output = self.model.generate(
                inputs, attention_mask=torch.ones_like(inputs), **sampling, **cache_arguments,
                max_new_tokens=self.settings["max_reasoning_tokens"], eos_token_id=end_thinking[0],
            )
            assert isinstance(reasoning_output, GenerateDecoderOnlyOutput)
            assert isinstance(reasoning_output.scores, tuple)
            reasoning_ids = reasoning_output.sequences[0, len(prompt_ids):].tolist()
            reasoning_allowed = [torch.isfinite(scores[0]).nonzero().flatten().tolist()
                                 for scores in reasoning_output.scores]
            closed = bool(reasoning_ids and reasoning_ids[-1] == end_thinking[0])
            separator = self.tokenizer.encode("\n" if closed else "\n</think>\n", add_special_tokens=False)
            answer_prefix = [*prompt_ids, *reasoning_ids, *separator]
            answer_input = torch.tensor([answer_prefix], device=generation_device)

            def allowed_tokens(batch_id: int, tokens: Tensor) -> list[int]:
                """Constrain only generated answer tokens, leaving the full private prompt intact."""
                return grammar.allowed(tokens[len(answer_prefix):].tolist())

            answer_output = self.model.generate(
                answer_input, attention_mask=torch.ones_like(answer_input), **sampling,
                max_new_tokens=grammar.maximum_length, eos_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=allowed_tokens,
                past_key_values=reasoning_output.past_key_values,
            )
            assert isinstance(answer_output, GenerateDecoderOnlyOutput)
            assert isinstance(answer_output.scores, tuple)
            answer_ids = answer_output.sequences[0, len(answer_prefix):].tolist()
            answer_allowed = [torch.isfinite(scores[0]).nonzero().flatten().tolist()
                              for scores in answer_output.scores]
        if not answer_ids or answer_ids[-1] != self.tokenizer.eos_token_id:
            raise ValueError("Constrained response did not terminate; refusing to substitute an action")
        response = self.tokenizer.decode(answer_ids[:-1], clean_up_tokenization_spaces=False)
        choice = parse_choice(response, count)
        for token, allowed in zip([*reasoning_ids, *answer_ids], [*reasoning_allowed, *answer_allowed]):
            if token not in allowed:
                raise ValueError("Generated token is outside its recorded sampling support")
        record = {
            "format": "qwen_choice_v1", "messages": messages, "rendered_prompt": prompt,
            "raw_reasoning": self.tokenizer.decode(reasoning_ids, skip_special_tokens=False),
            "reasoning_truncated": not closed, "raw_response": response, "selected_index": choice,
            "option_count": count, "prompt_tokens": len(prompt_ids), "reasoning_tokens": len(reasoning_ids),
            "answer_tokens": len(answer_ids), "seconds": time.monotonic() - started,
            "temperature": self.settings["temperature"], "sampling_seed": seed,
        }
        trajectory = {
            "tokens": [*answer_prefix, *answer_ids], "prompt_length": len(prompt_ids),
            "reasoning_length": len(reasoning_ids), "answer_start": len(answer_prefix),
            "reasoning_allowed": reasoning_allowed, "answer_allowed": answer_allowed,
        }
        return choice, record, trajectory

    def act(self, world: World, decisions: dict[int, Decision]) -> tuple[dict[int, Action], dict[int, int], dict[int, int], dict[int, list[Action]]]:
        """Choose all actions against the same unmodified pre-action world snapshot."""
        actions: dict[int, Action] = {}
        choices: dict[int, int] = {}
        quantity_choices: dict[int, int] = {}
        quantity_candidates: dict[int, list[Action]] = {}
        self.pending = {}
        self.last_decision_scores = {}
        contexts = {identifier: private_context(world, identifier, self.histories[identifier]) for identifier in decisions}
        for identifier, decision in decisions.items():
            self._activate(identifier)
            seed = self.seed + world.turn * 1_000_003 + identifier * 101
            choice, record, trajectory = self._choose(
                decision_messages(world, contexts[identifier], decision.candidates, "action"), len(decision.candidates), seed,
            )
            choices[identifier] = choice
            template = decision.candidates[choice]
            candidates = world.quantity_choices(identifier, template)
            if not candidates:
                raise ValueError("Selected action has no legal quantity realization")
            quantity_candidates[identifier] = candidates
            record["options"] = [{"action": asdict(candidate)} for candidate in decision.candidates]
            stages = {"action": record}
            trajectories = [trajectory]
            quantity_index = 0
            if len(candidates) > 1:
                quantity_index, quantity_record, quantity_trajectory = self._choose(
                    decision_messages(world, contexts[identifier], candidates, "quantity"), len(candidates), seed + 1,
                )
                quantity_record["options"] = [{"action": asdict(candidate)} for candidate in candidates]
                stages["quantity"] = quantity_record
                trajectories.append(quantity_trajectory)
            quantity_choices[identifier] = quantity_index
            actions[identifier] = candidates[quantity_index]
            stages["execution"] = {"requested_action": asdict(actions[identifier]), "submitted_action": asdict(actions[identifier])}
            self.last_decision_scores[identifier] = stages
            self.pending[identifier] = trajectories
            print(f"Turn {world.turn}, island {identifier}: {actions[identifier].kind}", flush=True)
        return actions, choices, quantity_choices, quantity_candidates

    def remember(self, world: World, actions: dict[int, Action], outcome: TurnResult) -> None:
        """Feed exact execution outcomes and own rewards back into bounded private history."""
        for identifier, action in actions.items():
            results = outcome.action_results[identifier]
            if len(results) != 1 or results[0]["action"] != asdict(action):
                raise ValueError("Submitted language-model action does not match the world's execution record")
            self.last_decision_scores[identifier]["execution"].update(results[0])
            island = world.islands.get(identifier)
            self.histories[identifier].append({
                "turn": world.turn, "action": asdict(action), "executed": results[0]["executed"],
                "reward": outcome.rewards[identifier], "reward_components": outcome.reward_components.get(identifier, {}),
                "health_after": island.health if island else None,
                "inventory_after": dict(island.inventory) if island else None,
            })
            limit = world.history_turns
            self.histories[identifier] = self.histories[identifier][-limit:] if limit else []

    def _trajectory_log_probability(self, trajectory: dict[str, Any]) -> Tensor:
        """Differentiate token probabilities on the exact support used during generation."""
        tokens = torch.tensor([trajectory["tokens"]], device=self.device)
        prompt_length = trajectory["prompt_length"]
        predictions = tokens.shape[1] - prompt_length
        output = self.model(input_ids=tokens[:, :-1], attention_mask=torch.ones_like(tokens[:, :-1]),
                            use_cache=False, logits_to_keep=predictions)
        logits = output.logits[0]
        if logits.shape[0] != predictions:
            raise ValueError("Qwen returned an unexpected teacher-forcing logit shape")
        positions = [(prompt_length + index, allowed)
                     for index, allowed in enumerate(trajectory["reasoning_allowed"])]
        positions.extend((trajectory["answer_start"] + index, allowed)
                         for index, allowed in enumerate(trajectory["answer_allowed"]))
        log_probabilities = []
        for position, allowed in positions:
            # Cached decoding and full-sequence replay can straddle a top-k/top-p numerical boundary.
            indices = torch.tensor(allowed, dtype=torch.long, device=self.device)
            scores = logits[position - prompt_length].index_select(0, indices) / self.settings["temperature"]
            selected_index = allowed.index(trajectory["tokens"][position])
            log_probabilities.append(torch.log_softmax(scores, dim=-1)[selected_index])
        result = torch.stack(log_probabilities).sum()
        if not torch.isfinite(result):
            raise ValueError("Sampled token is outside the recomputed policy support")
        return result

    def learn(self, decisions: dict[int, Decision], outcome: TurnResult, next_observations: dict[int, list[float]]) -> dict[str, float]:
        """Update only each acting island's adapter using its own reward and optional TD value baseline."""
        if self.settings["mode"] == "frozen":
            raise ValueError("Frozen Qwen must never enter the update path")
        total_actor_loss = total_value_loss = 0.0
        for identifier, decision in decisions.items():
            self._activate(identifier)
            reward = self.settings["reward_scale"] * outcome.rewards[identifier]
            advantage = torch.tensor(reward, device=self.device)
            value_record: dict[str, float] = {}
            if identifier in self.critics:
                critic = self.critics[identifier]
                current = torch.tensor(decision.observation, dtype=torch.float32, device=self.device)
                value = critic(current).squeeze()
                with torch.no_grad():
                    next_value = torch.zeros((), device=self.device)
                    if identifier not in outcome.deaths:
                        next_value = critic(torch.tensor(next_observations[identifier], dtype=torch.float32,
                                                         device=self.device)).squeeze()
                    target = reward + self.settings["discount"] * next_value
                advantage = (target - value).detach()
                value_loss = 0.5 * (target - value).square()
                critic_optimizer = self.critic_optimizers[identifier]
                critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), self.settings["gradient_clip"], error_if_nonfinite=True)
                critic_optimizer.step()
                total_value_loss += float(value_loss.detach())
                value_record = {"value_scaled": float(value.detach()), "target_scaled": float(target),
                                "next_value_scaled": float(next_value), "value_loss": float(value_loss.detach())}
            optimizer = self.optimizers[identifier]
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            before = [parameter.detach().clone() for parameter in parameters]
            optimizer.zero_grad(set_to_none=True)
            actor_loss_total = 0.0
            # Simple statistical gradient-following algorithms for connectionist RL, https://doi.org/10.1007/BF00992696
            for trajectory in self.pending[identifier]:
                actor_loss = -advantage * self._trajectory_log_probability(trajectory)
                actor_loss.backward()
                actor_loss_total += float(actor_loss.detach())
            gradient_norm = nn.utils.clip_grad_norm_(parameters, self.settings["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            difference = sum(float((parameter.detach() - original).square().sum()) for parameter, original in zip(parameters, before)) ** 0.5
            self.updates[identifier] += 1
            self.last_decision_scores[identifier]["learning"] = {
                "mode": self.settings["mode"], "reward": outcome.rewards[identifier],
                "reward_scale": self.settings["reward_scale"], "advantage_scaled": float(advantage),
                "actor_loss": actor_loss_total, "gradient_norm_before_clip": float(gradient_norm),
                "adapter_change_l2": difference, "update": self.updates[identifier], **value_record,
            }
            total_actor_loss += actor_loss_total
        self.pending = {}
        return {"updates": float(len(decisions)), "actor_loss": total_actor_loss / len(decisions),
                "value_loss": total_value_loss / len(decisions)}

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint private continuation state without duplicating the immutable pretrained model."""
        return {"backend": "qwen", "settings": self.settings, "seed": self.seed,
                "observation_size": self.observation_size, "histories": copy.deepcopy(self.histories),
                "updates": dict(self.updates), "runtime": self.runtime,
                "adapters": [{"input": {key: value.detach().cpu() for key, value in module.input_factors.items()},
                              "output": {key: value.detach().cpu() for key, value in module.output_factors.items()}}
                             for module in self.adapter_modules],
                "optimizers": {key: value.state_dict() for key, value in self.optimizers.items()},
                "critics": {key: value.state_dict() for key, value in self.critics.items()},
                "critic_optimizers": {key: value.state_dict() for key, value in self.critic_optimizers.items()}}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], device: str = "cpu") -> QwenPopulation:
        """Restore isolated histories, adapters, and optimizer moments from a pinned-model checkpoint."""
        population = cls(state["settings"], state["seed"], state["observation_size"], device)
        if state["runtime"]["base_sha256_before"] != population.base_checksum:
            raise ValueError("The resumed pretrained weights do not match the recorded checksum")
        for identifier in state["histories"]:
            population.add(identifier)
        population.histories = copy.deepcopy(state["histories"])
        population.updates = dict(state["updates"])
        with torch.no_grad():
            for module, saved in zip(population.adapter_modules, state["adapters"]):
                for identifier in saved["input"]:
                    module.input_factors[identifier].copy_(saved["input"][identifier])
                    module.output_factors[identifier].copy_(saved["output"][identifier])
        for identifier, optimizer in population.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][identifier])
        for identifier, critic in population.critics.items():
            critic.load_state_dict(state["critics"][identifier])
            population.critic_optimizers[identifier].load_state_dict(state["critic_optimizers"][identifier])
        return population

    def memory_report(self) -> dict[str, Any]:
        """Report shared base storage separately from independent island state."""
        return {**self.runtime, "base_weight_bytes": sum(parameter.numel() * parameter.element_size() for parameter in self.base_parameters),
                "independent_adapters": len(self.optimizers), "total_updates": sum(self.updates.values())}
