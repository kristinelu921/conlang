"""Build leakage-controlled GSM8K prompts and rank matched validation results."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Protocol
import unicodedata


class Tokenizer(Protocol):
    """Describe the tokenizer operations used by the prompt protocol."""

    eos_token_id: int

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class Question:
    source_index: int
    question: str
    answer: str


@dataclass(frozen=True)
class PromptExample:
    query_index: int
    demonstration_indices: tuple[int, ...]
    prompt: str
    prompt_tokens: tuple[int, ...]
    input_tokens: tuple[int, ...]
    loss_mask: tuple[int, ...]
    sampling_attempts: int


@dataclass(frozen=True)
class DemonstrationPool:
    questions: tuple[Question, ...]
    normalized_questions: frozenset[str]
    question_by_source: dict[int, str]


@dataclass(frozen=True)
class ValidationResult:
    trial_id: str
    epoch: int
    examples: int
    zero_shot_correct: int
    five_shot_correct: int
    zero_shot_nll: float
    five_shot_nll: float
    protocol_sha256: str

    def __post_init__(self) -> None:
        """Reject invalid counts, missing identities and nonfinite validation loss."""
        if not self.trial_id or self.epoch < 0 or self.examples <= 0:
            raise ValueError("Validation identity, epoch and example count must be valid.")
        if not all(0 <= count <= self.examples for count in
                   (self.zero_shot_correct, self.five_shot_correct)):
            raise ValueError("Correct counts must lie within the validation partition.")
        if not all(math.isfinite(loss) and loss > 0 for loss in
                   (self.zero_shot_nll, self.five_shot_nll)):
            raise ValueError("Validation NLL must be finite and positive.")
        if len(self.protocol_sha256) != 64 or any(character not in "0123456789abcdef"
                                                for character in self.protocol_sha256):
            raise ValueError("Validation requires a complete prompt-protocol SHA-256.")


def normalized_question(question: str) -> str:
    """Normalize Unicode, case and whitespace for demonstration duplicate exclusion."""
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def deterministic_seed(seed: int, query_index: int, epoch: int, purpose: str) -> int:
    """Separate per-query random streams without process-dependent Python hashes."""
    payload = json.dumps([seed, query_index, epoch, purpose], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "big")


def validate_partition(training: list[Question], validation: list[Question],
                       total_examples: int) -> None:
    """Require complete source-index coverage and no normalized question overlap."""
    training_indices = {row.source_index for row in training}
    validation_indices = {row.source_index for row in validation}
    if len(training_indices) != len(training) or len(validation_indices) != len(validation):
        raise ValueError("Duplicate source indices in a partition.")
    if not training or not validation or training_indices & validation_indices:
        raise ValueError("Training and validation must be nonempty and disjoint.")
    if training_indices | validation_indices != set(range(total_examples)):
        raise ValueError("Partition source indices do not cover the original training split.")
    training_questions = {normalized_question(row.question) for row in training}
    if training_questions & {normalized_question(row.question) for row in validation}:
        raise ValueError("Normalized questions overlap training and validation.")


def demonstration_pool(training: list[Question]) -> DemonstrationPool:
    """Validate and deduplicate the training demonstration pool once per dataset partition."""
    eligible_by_question: dict[str, Question] = {}
    question_by_source: dict[int, str] = {}
    for row in sorted(training, key=lambda item: item.source_index):
        if row.source_index in question_by_source:
            raise ValueError("Demonstration pool has duplicate source indices.")
        normalized = normalized_question(row.question)
        question_by_source[row.source_index] = normalized
        eligible_by_question.setdefault(normalized, row)
    return DemonstrationPool(tuple(eligible_by_question.values()),
                             frozenset(eligible_by_question), question_by_source)


def mixed_query_indices(source_indices: list[int], fraction: float,
                        seed: int, epoch: int) -> frozenset[int]:
    """Choose an exact-size deterministic five-shot subset without changing query coverage."""
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("Optimization source indices must be unique.")
    if not math.isfinite(fraction) or not 0 <= fraction <= 1 or epoch < 0:
        raise ValueError("Mixture fraction must lie in [0, 1] and epoch must be nonnegative.")
    generator = random.Random(deterministic_seed(seed, -1, epoch, "prompt-mixture"))
    return frozenset(generator.sample(sorted(source_indices), math.floor(len(source_indices) * fraction)))


def prepare_prompt(query: Question, training_pool: DemonstrationPool, tokenizer: Tokenizer,
                   shots: int, seed: int, epoch: int, context_length: int,
                   answer_token_reserve: int = 640,
                   maximum_sampling_attempts: int = 10000) -> PromptExample:
    """Sample training-only demonstrations without letting the query answer choose its prompt."""
    if shots not in (0, 5) or maximum_sampling_attempts <= 0 or epoch < 0:
        raise ValueError("Only zero/five-shot prompts and positive sampling limits are supported.")
    if not 256 <= answer_token_reserve < context_length:
        raise ValueError("Reserve must cover 256 generated tokens and fit within model context.")
    query_normalized = normalized_question(query.question)
    if (query.source_index in training_pool.question_by_source
            and training_pool.question_by_source[query.source_index] != query_normalized):
        raise ValueError("Query source index refers to a different question in the training pool.")
    eligible_count = len(training_pool.questions) - int(query_normalized in training_pool.normalized_questions)
    if eligible_count < shots:
        raise ValueError("Not enough distinct training questions for the required shot count.")
    generator = random.Random(deterministic_seed(seed, query.source_index, epoch, "demonstrations"))
    # The Flan Collection (https://arxiv.org/abs/2301.13688) motivates mixing prompt settings, not assuming transfer.
    for attempt in range(1, maximum_sampling_attempts + 1):
        demonstrations = generator.sample(training_pool.questions, shots)
        if any(row.source_index == query.source_index or normalized_question(row.question) == query_normalized
               for row in demonstrations):
            continue
        segments = ["Question: " + row.question + "\nAnswer: " + row.answer for row in demonstrations]
        segments.append("Question: " + query.question + "\nAnswer:")
        prompt = "\n\n".join(segments)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        padded_prompt_width = math.ceil(len(prompt_tokens) / 128) * 128
        if padded_prompt_width + answer_token_reserve <= context_length:
            break
        if shots == 0:
            raise ValueError("Zero-shot query exceeds context; truncation is disabled.")
    else:
        raise ValueError("No complete five-shot prompt fits within the sampling limit.")
    complete_tokens = tokenizer.encode(prompt + " " + query.answer, add_special_tokens=False)
    if complete_tokens[:len(prompt_tokens)] != prompt_tokens:
        raise ValueError("Tokenization changes across the query prompt/answer boundary.")
    input_tokens = complete_tokens + [tokenizer.eos_token_id]
    if len(input_tokens) - len(prompt_tokens) > answer_token_reserve:
        raise ValueError("Query answer exceeds its fixed reserve; do not resample using the answer.")
    if math.ceil(len(input_tokens) / 128) * 128 > context_length:
        raise ValueError("Padded answer sequence exceeds context; truncation is disabled.")
    loss_mask = [0] * len(prompt_tokens) + [1] * (len(input_tokens) - len(prompt_tokens))
    return PromptExample(query.source_index, tuple(row.source_index for row in demonstrations),
                         prompt, tuple(prompt_tokens), tuple(input_tokens), tuple(loss_mask), attempt)


def protocol_digest(zero_shot: list[PromptExample], five_shot: list[PromptExample],
                    targets: list[str], generation_tokens: int = 256) -> str:
    """Bind selection to identical ordered questions, targets, prompts and decode settings."""
    zero_indices = [example.query_index for example in zero_shot]
    if (not zero_indices or len(set(zero_indices)) != len(zero_indices)
            or zero_indices != [example.query_index for example in five_shot]
            or len(zero_indices) != len(targets)):
        raise ValueError("Both protocols and targets must cover the same unique ordered queries.")
    if generation_tokens != 256:
        raise ValueError("The declared validation protocol fixes generation at 256 tokens.")
    if any(example.demonstration_indices for example in zero_shot):
        raise ValueError("Zero-shot protocol contains demonstrations.")
    if any(len(example.demonstration_indices) != 5 for example in five_shot):
        raise ValueError("Five-shot protocol must retain exactly five demonstrations.")
    payload = {"indices": zero_indices, "targets": targets,
               "zero_shot_prompt_tokens": [example.prompt_tokens for example in zero_shot],
               "five_shot_prompt_tokens": [example.prompt_tokens for example in five_shot],
               "demonstrations": [example.demonstration_indices for example in five_shot],
               "generation_tokens": generation_tokens, "do_sample": False,
               "num_beams": 1, "stop": ["Question:", "</s>", "<|im_end|>", "tokenizer EOS"],
               "scoring": "pinned GSM8K strict/flexible extraction v3.0"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def balanced_selection_key(candidate: ValidationResult, reference: ValidationResult,
                           maximum_relative_nll: float = 1.10) -> tuple[int, int, float, int, str] | None:
    """Return a lower-is-better rank only for candidates passing both accuracy and loss guards."""
    if candidate.examples != reference.examples or candidate.protocol_sha256 != reference.protocol_sha256:
        raise ValueError("Candidate and reference validation protocols differ.")
    if not math.isfinite(maximum_relative_nll) or maximum_relative_nll < 1:
        raise ValueError("The relative NLL guard must be finite and at least one.")
    relative_loss = max(candidate.zero_shot_nll / reference.zero_shot_nll,
                        candidate.five_shot_nll / reference.five_shot_nll)
    if (candidate.zero_shot_correct < reference.zero_shot_correct
            or candidate.five_shot_correct < reference.five_shot_correct
            or relative_loss > maximum_relative_nll):
        return None
    return (-min(candidate.zero_shot_correct, candidate.five_shot_correct),
            -(candidate.zero_shot_correct + candidate.five_shot_correct),
            relative_loss, candidate.epoch, candidate.trial_id)


@dataclass(frozen=True)
class ComputeBudget:
    started_unix_seconds: float
    authorized_seconds: int

    def __post_init__(self) -> None:
        """Reject absent approval and invalid budget clocks before model initialization."""
        if not math.isfinite(self.started_unix_seconds) or self.started_unix_seconds <= 0:
            raise ValueError("Budget requires a valid persisted starting timestamp.")
        if self.authorized_seconds <= 0:
            raise ValueError("An explicit positive additional compute budget is required.")

    def remaining(self, now_unix_seconds: float) -> float:
        """Return remaining authorized wall time without resetting it across resumptions."""
        if not math.isfinite(now_unix_seconds) or now_unix_seconds < self.started_unix_seconds:
            raise ValueError("Budget clock is invalid or moved behind its recorded start.")
        return max(0.0, self.started_unix_seconds + self.authorized_seconds - now_unix_seconds)

    def admits(self, now_unix_seconds: float, estimated_seconds: float,
               reserve_seconds: float) -> bool:
        """Admit work only if its conservative duration and reserved work fit before the deadline."""
        if not math.isfinite(estimated_seconds) or estimated_seconds <= 0:
            raise ValueError("Admission requires a positive finite runtime estimate.")
        if not math.isfinite(reserve_seconds) or reserve_seconds < 0:
            raise ValueError("Reserved time must be finite and nonnegative.")
        return estimated_seconds + reserve_seconds <= self.remaining(now_unix_seconds)
