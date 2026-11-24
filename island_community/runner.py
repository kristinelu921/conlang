"""Coordinate CPU world turns with a resident population of independent policies."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from island_community.config import load_config, validate_config
from island_community.storage import RunRecorder, load_checkpoint, save_checkpoint, write_json_atomic


def run(
    output: Path,
    turns: int = 100,
    config_path: Path | None = None,
    resume: Path | None = None,
    device: str = "cpu",
    use_vmap: bool | None = None,
    passive: bool = False,
    checkpoint_every: int = 10,
    command: list[str] | None = None,
) -> Path:
    """Execute a bounded run segment on authorized compute and preserve full continuation state."""
    from island_community.policies import PolicyPopulation
    from island_community.world import Action, World

    language_population = None
    if turns < 1 or checkpoint_every < 1:
        raise ValueError("turns and checkpoint_every must be positive")
    if resume is not None and config_path is not None:
        raise ValueError("A resumed run uses its checkpoint's exact configuration; omit --config")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}. Choose a new run segment.")
    if resume is not None:
        checkpoint = load_checkpoint(resume)
        configuration = checkpoint["config"]
        validate_config(configuration)
        world = World.from_state_dict(configuration, checkpoint["world"])
        if checkpoint["policies"] is not None and checkpoint["policies"].get("backend") == "qwen":
            from island_community.qwen_backend import QwenPopulation

            language_population = QwenPopulation.from_state_dict(checkpoint["policies"], device=device)
            population = None
        else:
            population = None if checkpoint["policies"] is None else PolicyPopulation.from_state_dict(
                checkpoint["policies"], device=device, use_vmap=use_vmap
            )
        if passive and (population is not None or language_population is not None):
            raise ValueError("Cannot replace a learned checkpoint with a passive policy on resume")
    else:
        configuration = load_config(config_path)
        world = World(configuration)
        if passive:
            population = None
        elif "language_model" in configuration:
            from island_community.qwen_backend import QwenPopulation

            population = None
            language_population = QwenPopulation(configuration["language_model"], configuration["seed"], world.observation_size, device)
            for island_id in sorted(world.islands):
                language_population.add(island_id)
        else:
            settings = dict(configuration["policy"])
            capacity = settings.pop("initial_capacity")
            population = PolicyPopulation(
                observation_size=world.observation_size,
                action_feature_size=world.action_feature_size,
                capacity=capacity,
                seed=configuration["seed"],
                device=device,
                use_vmap=True if use_vmap is None else use_vmap,
                mutation_std=configuration["reproduction"]["policy_mutation_std"],
                initial_action_biases=world.initial_action_biases,
                **configuration.get("exploration", {}),
                **settings,
            )
            for island_id in sorted(world.islands):
                population.add(island_id)
    recorder = RunRecorder(output, configuration, command or [sys.executable, *sys.argv], resumed_from=resume)
    if resume is not None:
        recorder.event_count = len(world.events)
    import torch

    recorder.metadata.update(
        policy=("qwen_" + configuration["language_model"]["mode"]) if language_population is not None
               else ("passive" if population is None else "independent_actor_critic"),
        device=device,
        use_vmap=population.use_vmap if population is not None else None,
        python_version=sys.version,
        torch_version=str(torch.__version__),
        requested_turns=turns,
        checkpoint_every=checkpoint_every,
        policy_scores_format="qwen_choices_v1" if language_population is not None else ("per_turn_candidates_v1" if population is not None else None),
    )
    if language_population is not None:
        recorder.metadata["language_model"] = language_population.runtime
    write_json_atomic(output / "manifest.json", recorder.metadata)

    def record(
        learning: dict[str, float], experiences: list[dict[str, Any]] | None = None,
        policy_scores: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        """Save the world and optional decision scores from the completed turn."""
        snapshot = world.snapshot()
        snapshot["maximum_health"] = configuration["survival"]["maximum_health"]
        if population is not None:
            snapshot["policy_memory"] = population.memory_report()
        if language_population is not None:
            snapshot["policy_memory"] = language_population.memory_report()
        recorder.record(snapshot, world.events, learning, experiences, policy_scores)

    def checkpoint() -> None:
        save_checkpoint(
            output / "checkpoint.pt", configuration, world.state_dict(),
            language_population.state_dict() if language_population is not None else (None if population is None else population.state_dict()),
        )

    completed_turns = 0
    try:
        record({})
        checkpoint()
        while completed_turns < turns and world.islands:
            decisions = world.start_turn()
            quantity_choices: dict[int, int] = {}
            quantity_candidates: dict[int, list[Action]] = {}
            policy_scores: dict[int, dict[str, Any]] = {}
            if language_population is not None:
                actions, choices, quantity_choices, quantity_candidates = language_population.act(world, decisions)
                policy_scores = language_population.last_decision_scores
            elif population is None:
                choices = {island_id: 0 for island_id in decisions}
                actions = {island_id: Action("wait", {}) for island_id in decisions}
            else:
                choices = population.act(
                    {island_id: decision.observation for island_id, decision in decisions.items()},
                    {island_id: decision.action_features for island_id, decision in decisions.items()},
                    candidate_groups={island_id: [action.kind for action in decision.candidates] for island_id, decision in decisions.items()},
                    capture_scores=True,
                )
                actions = {island_id: decisions[island_id].candidates[index] for island_id, index in choices.items()}
                if configuration["actions"].get("trade_quantity_mode") == "all_feasible":
                    quantity_candidates = {island_id: world.quantity_choices(island_id, action) for island_id, action in actions.items()}
                    quantity_choices = population.choose_quantities({
                        island_id: [world._action_features(island_id, candidate) for candidate in candidates]
                        for island_id, candidates in quantity_candidates.items()
                    })
                    actions = {island_id: quantity_candidates[island_id][index] for island_id, index in quantity_choices.items()}
                policy_scores = population.last_decision_scores
                for island_id, stages in policy_scores.items():
                    for stage, scores in stages.items():
                        candidates = decisions[island_id].candidates if stage == "action" else quantity_candidates[island_id]
                        for option, candidate in zip(scores["options"], candidates):
                            option["action"] = asdict(candidate)
            action_features = {island_id: world._action_features(island_id, action) for island_id, action in actions.items()}
            outcome = world.finish_turn(actions)
            next_observations = world.preview_next_observations() if world.islands else {}
            experiences = [
                {
                    "schema_version": 2 if quantity_choices else 1,
                    "island_id": island_id,
                    "turn": world.turn,
                    "observation": decision.observation,
                    "action": asdict(actions[island_id]),
                    "candidate_index": choices[island_id],
                    "candidate_counts": dict(Counter(action.kind for action in decision.candidates)),
                    "action_features": action_features[island_id],
                    **({"quantity_index": quantity_choices[island_id], "quantity_candidate_count": len(quantity_candidates[island_id])} if quantity_choices else {}),
                    "reward": outcome.rewards[island_id],
                    **({"reward_components": outcome.reward_components[island_id]} if island_id in outcome.reward_components else {}),
                    "terminal": island_id in outcome.deaths,
                    "next_observation": next_observations.get(island_id),
                    **({"action_results": outcome.action_results[island_id]} if language_population is not None else {}),
                }
                for island_id, decision in decisions.items()
            ]
            learning: dict[str, float] = {}
            if language_population is not None:
                language_population.remember(world, actions, outcome)
                for child_id, parent_id in outcome.births.items():
                    language_population.add(child_id, parent_id)
                if configuration["language_model"]["mode"] != "frozen":
                    learning = language_population.learn(decisions, outcome, next_observations)
                else:
                    learning = {"updates": 0.0}
            if population is not None:
                for child_id, parent_id in outcome.births.items():
                    population.add(child_id, parent_id)
                learning = population.learn(
                    {island_id: outcome.rewards[island_id] for island_id in decisions},
                    next_observations,
                    set(outcome.deaths),
                    outcome.learning_strengths,
                )
                for island_id in outcome.deaths:
                    population.remove(island_id)
            completed_turns += 1
            record(learning, experiences, policy_scores)
            if completed_turns % checkpoint_every == 0:
                checkpoint()
                print(f"Turn {world.turn}: {len(world.islands)} islands; checkpoint saved", flush=True)
        if language_population is not None:
            checksum = language_population.frozen_checksum()
            language_population.runtime["base_sha256_after"] = checksum
            if checksum != language_population.base_checksum:
                raise ValueError("Shared Qwen base weights changed during the run")
        checkpoint()
        recorder.finish("extinct" if not world.islands else "completed", world.turn)
    except BaseException as error:
        # Preserve the last complete checkpoint rather than publishing a partially resolved turn.
        recorder.finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", world.turn, str(error))
        raise
    print(f"Run recorded at {output.resolve()}", flush=True)
    return output
