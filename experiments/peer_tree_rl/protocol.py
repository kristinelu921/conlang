"""Conversation transitions and outcome-only credit for a two-peer pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import re
from typing import Callable, Protocol


SYSTEM_PROMPT = """You and another equally capable math agent are working together.
You both receive the complete original problem. Help each other by asking focused
questions, dividing the problem into useful subproblems, answering questions, or
checking a specific claim. You choose what context to send to your partner.
You may also submit the final answer when ready. Your partner sees only your
message and context, plus the original problem. There is no shared memory.
Return exactly one JSON object, without markdown or additional text:
{"action":"message","message":"your question or reply","context":"relevant context"}
or {"action":"submit","answer":"the final integer"}.
Do not submit a partial answer. A submission ends the conversation. The final
answer must be an integer. You have at most eight individual messages together."""

SINGLE_PROMPT = """Solve the original math problem. You may work through the
solution in your response. End with exactly one final integer in \\boxed{...}."""


@dataclass(frozen=True)
class Problem:
    identifier: str
    problem: str
    answer: int
    split: str


@dataclass(frozen=True)
class State:
    problem: str
    turn: int = 0
    incoming_message: str = ""
    incoming_context: str = ""

    @property
    def agent(self) -> int:
        return self.turn % 2

    def messages(self) -> list[dict[str, str]]:
        """Expose the problem and selected incoming content without a hidden transcript."""
        content = json.dumps({"original_problem": self.problem, "message_number": self.turn + 1,
                              "incoming_message": self.incoming_message,
                              "incoming_context": self.incoming_context}, ensure_ascii=False)
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


@dataclass(frozen=True)
class Generation:
    prompt_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    log_probabilities: tuple[float, ...]
    text: str
    ended: bool


@dataclass(frozen=True)
class Transition:
    next_state: State | None
    submission: int | None
    reason: str


@dataclass
class TrainingItem:
    identifier: str
    parent: str | None
    agent: int
    state: State
    generation: Generation
    successes: float
    observations: int
    weight: float


class Generator(Protocol):
    def generate(self, states: list[State], seeds: list[int], agent: int) -> list[Generation]: ...


def parse_integer(value: str) -> int | None:
    """Accept a single integer, never an incidental number inside a longer answer."""
    return int(value.strip()) if re.fullmatch(r"[+-]?\d+", value.strip()) else None


def boxed_integer(solution: str) -> int | None:
    """Extract the final balanced boxed integer for grading without exposing the solution."""
    start = solution.rfind("\\boxed{")
    if start < 0:
        return None
    start += len("\\boxed{")
    depth = 1
    for index in range(start, len(solution)):
        depth += (solution[index] == "{") - (solution[index] == "}")
        if depth == 0:
            return parse_integer(solution[start:index])
    return None


def question_fingerprint(problem: str) -> str:
    """Hash normalized problem text for exact overlap audits across splits."""
    normalized = re.sub(r"\s+", "", problem).casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def transition(state: State, generation: Generation, maximum_messages: int) -> Transition:
    """Validate one action and terminate only on submission, invalid output, or turn exhaustion."""
    if not generation.ended:
        return Transition(None, None, "token_limit")
    try:
        action = json.loads(generation.text.strip())
    except (json.JSONDecodeError, ValueError):
        return Transition(None, None, "invalid_json")
    if not isinstance(action, dict):
        return Transition(None, None, "invalid_schema")
    if action.get("action") == "submit":
        if set(action) != {"action", "answer"} or not isinstance(action["answer"], str):
            return Transition(None, None, "invalid_schema")
        answer = parse_integer(action["answer"])
        return Transition(None, answer, "submitted" if answer is not None else "invalid_answer")
    if (action.get("action") != "message" or set(action) != {"action", "message", "context"}
            or not isinstance(action["message"], str) or not isinstance(action["context"], str)
            or not action["message"].strip()):
        return Transition(None, None, "invalid_schema")
    if state.turn + 1 >= maximum_messages:
        return Transition(None, None, "turn_limit")
    return Transition(State(state.problem, state.turn + 1, action["message"], action["context"]),
                      None, "message")


def outcome_weight(successes: float, observations: int, baseline: float, scale: float) -> float:
    """Preserve absolute positive and negative credit even when all branches agree."""
    if observations < 1 or not 0 <= successes <= observations:
        raise ValueError("Outcome counts must describe completed independent continuations.")
    return scale * (successes / observations - baseline)


def sample_seed(seed: int, *coordinates: object) -> int:
    """Derive stable independent sampling keys without depending on Python's hash seed."""
    digest = hashlib.sha256(json.dumps([seed, *coordinates]).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def collect_tree(generator: Generator, problem: Problem, settings: dict, tree_index: int,
                 check_time: Callable[[], None]) -> tuple[list[TrainingItem], dict]:
    """Branch one selected node and back up completed outcomes without multiplying prefix credit."""
    # Optima https://arxiv.org/abs/2410.08115 and MALT https://arxiv.org/abs/2412.01928; retain graded outcomes for policy gradients.
    count = settings["candidate_actions"]
    continuations = settings["continuations_per_candidate"]
    root = State(problem.problem)
    branch_turn = settings["branch_turns"][tree_index % len(settings["branch_turns"]) ]
    records: list[TrainingItem] = []
    events: list[dict] = []
    prefix: list[tuple[State, Generation]] = []
    root_seed = sample_seed(settings["seed"], "train", tree_index, problem.identifier)

    for prefix_index in range(branch_turn):
        check_time()
        generation = generator.generate([root], [sample_seed(root_seed, "prefix", prefix_index)], root.agent)[0]
        result = transition(root, generation, settings["maximum_messages"])
        events.append({"kind": "prefix", "state": asdict(root), "generation": asdict(generation),
                       "transition": asdict(result)})
        if result.next_state is None:
            reward = float(result.submission == problem.answer) if result.submission is not None else 0.0
            records.append(TrainingItem("prefix-terminal", None, root.agent, root, generation, reward, 1,
                outcome_weight(reward, 1, settings["reward_baseline"], settings["reward_scale"])))
            return records, {"events": events, "leaf_rewards": [reward], "branch_turn": branch_turn,
                             "branched": False, "problem_id": problem.identifier}
        prefix.append((root, generation))
        root = result.next_state

    check_time()
    candidates = generator.generate([root] * count,
        [sample_seed(root_seed, "candidate", candidate) for candidate in range(count)], root.agent)
    paths: list[dict] = []
    candidate_rewards: list[list[float]] = [[] for _ in candidates]
    candidate_reasons: list[list[str]] = [[] for _ in candidates]
    for candidate_index, generation in enumerate(candidates):
        result = transition(root, generation, settings["maximum_messages"])
        events.append({"kind": "candidate", "candidate": candidate_index, "state": asdict(root),
                       "generation": asdict(generation), "transition": asdict(result)})
        if result.next_state is None:
            reward = float(result.submission == problem.answer) if result.submission is not None else 0.0
            candidate_rewards[candidate_index].append(reward)
            candidate_reasons[candidate_index].append(result.reason)
        else:
            for continuation in range(continuations):
                paths.append({"candidate": candidate_index, "continuation": continuation,
                              "state": result.next_state, "items": [], "done": False})

    while any(not path["done"] for path in paths):
        check_time()
        active = [path for path in paths if not path["done"]]
        agent = active[0]["state"].agent
        assert all(path["state"].agent == agent for path in active)
        generations = generator.generate([path["state"] for path in active],
            [sample_seed(root_seed, "continuation", path["candidate"], path["continuation"],
                         path["state"].turn) for path in active], agent)
        for path, generation in zip(active, generations, strict=True):
            current = path["state"]
            result = transition(current, generation, settings["maximum_messages"])
            identifier = f"candidate-{path['candidate']}/continuation-{path['continuation']}/turn-{current.turn}"
            parent = path["items"][-1][0] if path["items"] else f"candidate-{path['candidate']}"
            path["items"].append((identifier, parent, current, generation))
            events.append({"kind": "continuation", "id": identifier, "parent": parent,
                           "state": asdict(current), "generation": asdict(generation),
                           "transition": asdict(result)})
            if result.next_state is not None:
                path["state"] = result.next_state
                continue
            reward = float(result.submission == problem.answer) if result.submission is not None else 0.0
            candidate_rewards[path["candidate"]].append(reward)
            candidate_reasons[path["candidate"]].append(result.reason)
            for identifier, parent, state, output in path["items"]:
                records.append(TrainingItem(identifier, parent, state.agent, state, output, reward, 1,
                    outcome_weight(reward, 1, settings["reward_baseline"], settings["reward_scale"])))
            path["done"] = True

    for candidate_index, generation in enumerate(candidates):
        rewards = candidate_rewards[candidate_index]
        records.append(TrainingItem(f"candidate-{candidate_index}", "prefix" if prefix else None,
            root.agent, root, generation, sum(rewards), len(rewards),
            outcome_weight(sum(rewards), len(rewards), settings["reward_baseline"], settings["reward_scale"])))
    # Equal candidate weighting avoids overweighting nonterminal actions merely because they have more leaves.
    prefix_value = sum(sum(rewards) / len(rewards) for rewards in candidate_rewards) / count
    for index, (state, generation) in enumerate(prefix):
        records.append(TrainingItem(f"prefix-{index}", None, state.agent, state, generation, prefix_value, 1,
            outcome_weight(prefix_value, 1, settings["reward_baseline"], settings["reward_scale"])))
    return records, {"problem_id": problem.identifier, "branch_turn": branch_turn, "branched": True,
                     "candidate_rewards": candidate_rewards, "candidate_reasons": candidate_reasons,
                     "leaf_rewards": [reward for rewards in candidate_rewards for reward in rewards],
                     "candidate_mean_success": prefix_value, "events": events}


def completion_mask(prompt_length: int, completion_length: int, width: int) -> list[float]:
    """Mark generated targets including EOS while excluding every prompt and padding token."""
    if prompt_length < 1 or completion_length < 1 or prompt_length + completion_length > width:
        raise ValueError("The complete prompt and response must fit without truncation.")
    return [float(prompt_length <= index < prompt_length + completion_length) for index in range(width)]


def validate_settings(settings: dict) -> None:
    """Reject changes that violate the frozen pilot's execution or scientific assumptions."""
    if settings["model"] != "Qwen/Qwen3-1.7B" or settings["agents"] != 2:
        raise ValueError("This pilot supports the agreed two Qwen3-1.7B peers only.")
    if settings["enable_thinking"] or settings["shared_scratchpad"] or settings["top_p"] != 1.0:
        raise ValueError("Thinking, scratchpad, and sampling truncation are disabled.")
    if settings["branch_turns"] != [0, 1] or settings["maximum_messages"] != 8:
        raise ValueError("The pilot fixes branch positions and conversation length.")
    if settings["candidate_actions"] != 4 or settings["continuations_per_candidate"] < 1:
        raise ValueError("Four candidates and at least one continuation are required.")
    if not 0 < settings["temperature"] or not math.isfinite(settings["temperature"]):
        raise ValueError("The collection and replay policy must have positive finite temperature.")
    for key in ("model_revision", "training_dataset_revision", "evaluation_dataset_revision"):
        if not isinstance(settings[key], str) or not re.fullmatch(r"[0-9a-f]{40}", settings[key]):
            raise ValueError(f"An immutable revision is required: {key}")
