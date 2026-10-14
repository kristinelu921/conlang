"""Profile linked-facts training and report where each step spends its time.

The metrics phase is the extra backward work that separates the three model
gradients before the optimizer backward. Run a short profile with:

python speed.py --dataset-dir datasets/linked_facts_v1
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

from control_model import build_control_model
from control_train import collate_control_examples
from models import CommunicationOutput, TextTokenizer, build_model
from train import adamw_optimizer, collate_examples, load_examples, resolve_split_path
from training_metrics import communication_metrics, control_metrics, populated_gradient_norm

PHASE_NAMES = (
    "collate",
    "host_to_device",
    "zero_grad",
    "forward",
    "metrics",
    "backward",
    "clip",
    "optimizer",
)


@dataclass
class PhaseClock:
    """Accumulate per-phase seconds for a measured training loop."""

    seconds: dict[str, float]
    recording: bool = False
    _phase_name: str = ""
    _started_at: float = 0.0

    def begin(self, phase_name: str) -> None:
        """Start timing one named phase."""
        self._phase_name = phase_name
        self._started_at = time.perf_counter()

    def end(self, device: torch.device) -> None:
        """Stop the current phase and keep the elapsed time when recording."""
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if self.recording:
            self.seconds[self._phase_name] += time.perf_counter() - self._started_at


@dataclass(frozen=True)
class TrainingProfile:
    """Mean phase times and the operator table for one profiled model."""

    model_name: str
    include_metrics: bool
    measured_steps: int
    examples_per_step: int
    tokens_per_step: float
    parameter_count: int
    phase_seconds: tuple[tuple[str, float], ...]
    operator_table: str


def profile_communication_training(
    model: nn.Module,
    tokenizer: TextTokenizer,
    examples: Sequence[Mapping[str, str]],
    optimizer: AdamW,
    batch_size: int,
    measured_steps: int,
    warmup_steps: int,
    device: torch.device,
    include_metrics: bool,
    operator_limit: int,
) -> TrainingProfile:
    """Time the communication training step, including metric logging."""

    def collate(selected: Sequence[Mapping[str, str]]) -> Any:
        return collate_examples(selected, tokenizer)

    def forward(batch: Any) -> CommunicationOutput:
        return model(
            batch.agent_1_token_ids,
            batch.agent_2_token_ids,
            batch.question_token_ids,
            batch.answer_token_ids,
        )

    def collect(output: CommunicationOutput, batch: Any) -> Any:
        return communication_metrics(model, output, batch.answer_token_ids)

    def token_count(batch: Any) -> int:
        return _non_padding_count(
            (
                batch.agent_1_token_ids,
                batch.agent_2_token_ids,
                batch.question_token_ids,
                batch.answer_token_ids,
            ),
            tokenizer.pad_id,
        )

    return _profile_training(
        model_name="communication",
        model=model,
        optimizer=optimizer,
        examples=examples,
        collate=collate,
        forward=forward,
        collect=collect if include_metrics else None,
        token_count=token_count,
        batch_size=batch_size,
        measured_steps=measured_steps,
        warmup_steps=warmup_steps,
        device=device,
        include_metrics=include_metrics,
        operator_limit=operator_limit,
    )


def profile_control_training(
    model: nn.Module,
    tokenizer: TextTokenizer,
    examples: Sequence[Mapping[str, str]],
    optimizer: AdamW,
    batch_size: int,
    measured_steps: int,
    warmup_steps: int,
    device: torch.device,
    include_metrics: bool,
    operator_limit: int,
) -> TrainingProfile:
    """Time the control training step, including metric logging."""

    def collate(selected: Sequence[Mapping[str, str]]) -> Any:
        return collate_control_examples(selected, tokenizer)

    def forward(batch: Any) -> CommunicationOutput:
        return model(batch.context_token_ids, batch.answer_token_ids)

    def collect(output: CommunicationOutput, batch: Any) -> Any:
        return control_metrics(model, output, batch.answer_token_ids)

    def token_count(batch: Any) -> int:
        return _non_padding_count((batch.context_token_ids, batch.answer_token_ids), tokenizer.pad_id)

    return _profile_training(
        model_name="control",
        model=model,
        optimizer=optimizer,
        examples=examples,
        collate=collate,
        forward=forward,
        collect=collect if include_metrics else None,
        token_count=token_count,
        batch_size=batch_size,
        measured_steps=measured_steps,
        warmup_steps=warmup_steps,
        device=device,
        include_metrics=include_metrics,
        operator_limit=operator_limit,
    )


def format_training_profile(profile: TrainingProfile) -> str:
    """Render phase times as milliseconds per step and their share of the step."""
    total_seconds = sum(seconds for _, seconds in profile.phase_seconds)
    metric_note = "metric logging on" if profile.include_metrics else "metric logging off"
    lines = [
        f"{profile.model_name} training profile",
        (
            f"batch {profile.examples_per_step}, measured steps {profile.measured_steps}, "
            f"parameters {profile.parameter_count}, tokens/step {profile.tokens_per_step:.0f}, {metric_note}"
        ),
        "",
        f"{'phase':<16}{'ms/step':>12}{'share':>10}",
    ]
    slowest_name = ""
    slowest_seconds = -1.0
    for phase_name, seconds in profile.phase_seconds:
        share = 0.0 if total_seconds == 0 else 100.0 * seconds / total_seconds
        lines.append(f"{phase_name:<16}{seconds * 1000:>12.2f}{share:>9.1f}%")
        if seconds > slowest_seconds:
            slowest_name = phase_name
            slowest_seconds = seconds
    lines.append(f"{'total':<16}{total_seconds * 1000:>12.2f}{100.0:>9.1f}%")
    slowest_share = 0.0 if total_seconds == 0 else 100.0 * slowest_seconds / total_seconds
    lines.append(f"slowest: {slowest_name} ({slowest_share:.1f}% of the step)")
    if profile.operator_table:
        lines.extend(["", "slowest operators", profile.operator_table.rstrip()])
    return "\n".join(lines)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line settings for a training profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("communication", "control"), default="communication")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit-examples", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--operators", type=int, default=15)
    parser.add_argument("--no-metrics", action="store_true")
    return parser.parse_args(arguments)


def profile_from_arguments(arguments: Sequence[str] | None = None) -> TrainingProfile:
    """Load the dataset and profile the selected trainer."""
    parsed = parse_arguments(arguments)
    if min(parsed.batch_size, parsed.steps) < 1 or parsed.warmup < 0 or parsed.operators < 0:
        raise ValueError("batch size and steps must be positive, and warmup and operators must be non-negative")
    device = torch.device(parsed.device)
    examples = load_examples(resolve_split_path(parsed.dataset_dir, "train"), parsed.limit_examples)
    if parsed.model == "communication":
        model, tokenizer = build_model()
        profile = profile_communication_training
    else:
        model, tokenizer = build_control_model()
        profile = profile_control_training
    model.to(device)
    optimizer = adamw_optimizer(model, parsed.learning_rate, parsed.weight_decay)
    return profile(
        model,
        tokenizer,
        examples,
        optimizer,
        parsed.batch_size,
        parsed.steps,
        parsed.warmup,
        device,
        not parsed.no_metrics,
        parsed.operators,
    )


def _profile_training(
    model_name: str,
    model: nn.Module,
    optimizer: AdamW,
    examples: Sequence[Mapping[str, str]],
    collate: Callable[[Sequence[Mapping[str, str]]], Any],
    forward: Callable[[Any], CommunicationOutput],
    collect: Callable[[CommunicationOutput, Any], Any] | None,
    token_count: Callable[[Any], int],
    batch_size: int,
    measured_steps: int,
    warmup_steps: int,
    device: torch.device,
    include_metrics: bool,
    operator_limit: int,
) -> TrainingProfile:
    """Time repeated training steps and optionally record the slowest operators."""
    if measured_steps < 1 or warmup_steps < 0 or batch_size < 1:
        raise ValueError("measured steps and batch size must be positive")
    if not examples:
        raise ValueError("profiling needs at least one example")
    model.train()
    clock = PhaseClock(seconds={phase_name: 0.0 for phase_name in PHASE_NAMES})
    for index in range(warmup_steps):
        _run_profiled_step(
            clock, model, optimizer, examples, collate, forward, collect, token_count, batch_size, index, device
        )
    clock.recording = True
    token_total = 0
    for index in range(measured_steps):
        token_total += _run_profiled_step(
            clock,
            model,
            optimizer,
            examples,
            collate,
            forward,
            collect,
            token_count,
            batch_size,
            warmup_steps + index,
            device,
        )
    clock.recording = False
    phase_seconds = tuple((phase_name, clock.seconds[phase_name] / measured_steps) for phase_name in PHASE_NAMES)
    operator_table = ""
    if operator_limit > 0:
        operator_table = _operator_table(
            lambda: _run_profiled_step(
                clock,
                model,
                optimizer,
                examples,
                collate,
                forward,
                collect,
                token_count,
                batch_size,
                warmup_steps + measured_steps,
                device,
            ),
            device,
            operator_limit,
        )
    return TrainingProfile(
        model_name=model_name,
        include_metrics=include_metrics,
        measured_steps=measured_steps,
        examples_per_step=batch_size,
        tokens_per_step=token_total / measured_steps,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        phase_seconds=phase_seconds,
        operator_table=operator_table,
    )


def _run_profiled_step(
    clock: PhaseClock,
    model: nn.Module,
    optimizer: AdamW,
    examples: Sequence[Mapping[str, str]],
    collate: Callable[[Sequence[Mapping[str, str]]], Any],
    forward: Callable[[Any], CommunicationOutput],
    collect: Callable[[CommunicationOutput, Any], Any] | None,
    token_count: Callable[[Any], int],
    batch_size: int,
    index: int,
    device: torch.device,
) -> int:
    """Run one training step and return its non-padding token count."""
    selected = _select_examples(examples, batch_size, index)
    clock.begin("collate")
    batch = collate(selected)
    clock.end(device)
    clock.begin("host_to_device")
    batch = batch.to(device)
    clock.end(device)
    clock.begin("zero_grad")
    optimizer.zero_grad(set_to_none=True)
    clock.end(device)
    clock.begin("forward")
    output = forward(batch)
    clock.end(device)
    clock.begin("metrics")
    collected = None if collect is None else collect(output, batch)
    clock.end(device)
    clock.begin("backward")
    output.loss.backward()
    if collected is not None:
        collected["gradients/global"] = populated_gradient_norm(list(model.parameters()))
    clock.end(device)
    clock.begin("clip")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    clock.end(device)
    clock.begin("optimizer")
    optimizer.step()
    clock.end(device)
    return token_count(batch)


def _select_examples(examples: Sequence[Mapping[str, str]], batch_size: int, index: int) -> list[Mapping[str, str]]:
    """Cycle the example list into one batch for a profile step."""
    count = len(examples)
    start = (index * batch_size) % count
    return [examples[(start + offset) % count] for offset in range(batch_size)]


def _non_padding_count(tensors: Sequence[torch.Tensor], padding_id: int) -> int:
    """Count tokens that the model actually reads."""
    return sum(int(tensor.ne(padding_id).sum()) for tensor in tensors)


def _operator_table(run_step: Callable[[], int], device: torch.device, operator_limit: int) -> str:
    """Return the slowest operators from one profiled training step."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    with torch.profiler.profile(activities=activities, record_shapes=False) as profiler:
        run_step()
    return profiler.key_averages().table(sort_by=sort_by, row_limit=operator_limit)


def main(arguments: Sequence[str] | None = None) -> int:
    """Profile a trainer from the command line and print the phase table."""
    print(format_training_profile(profile_from_arguments(arguments)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
