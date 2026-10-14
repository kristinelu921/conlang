"""Train the full-context control transformer.

The objective is the joint log probability of the answer tokens. Decoupled Weight
Decay Regularization: https://arxiv.org/abs/1711.05101
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from control_model import ControlModel, build_control_model, join_full_context
from models import TextTokenizer, collate_token_ids
from regional_data import resolve_training_dataset
from train import TrainConfig, adamw_optimizer, load_examples, resolve_split_path
from training_metrics import Metrics, WandbRun, add_wandb_arguments, control_metrics, log_backward


@dataclass
class ControlBatch:
    """One padded batch of full-context question examples."""

    answers: list[str]
    answer_token_ids: torch.Tensor
    context_token_ids: torch.Tensor

    def to(self, device: torch.device) -> ControlBatch:
        """Move the batch tensors onto the training device."""
        return ControlBatch(
            answers=self.answers,
            answer_token_ids=self.answer_token_ids.to(device),
            context_token_ids=self.context_token_ids.to(device),
        )


def collate_control_examples(examples: Sequence[Mapping[str, str]], tokenizer: TextTokenizer) -> ControlBatch:
    """Pad full-context examples for the control transformer."""
    answers = [example["answer"] for example in examples]
    contexts = [
        join_full_context(example["agent_1_input"], example["agent_2_input"], example["question"]) for example in examples
    ]
    return ControlBatch(
        answers=answers,
        answer_token_ids=collate_token_ids(
            [tokenizer.encode_answer(answer) for answer in answers],
            tokenizer.pad_id,
            padded_length=tokenizer.max_answer_length,
        ),
        context_token_ids=collate_token_ids([tokenizer.encode_text(context) for context in contexts], tokenizer.pad_id),
    )


def training_step(
    model: ControlModel,
    optimizer: torch.optim.AdamW,
    batch: ControlBatch,
    max_gradient_norm: float,
    metrics_logger: Callable[[Metrics], None] | None = None,
) -> float:
    """Take one gradient step on the joint answer log probability."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch.context_token_ids, batch.answer_token_ids)
    log_backward(metrics_logger, control_metrics, model, output, batch.answer_token_ids)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
    optimizer.step()
    return float(output.loss.detach())


def iter_epoch_batches(
    examples: Sequence[Mapping[str, str]],
    tokenizer: TextTokenizer,
    batch_size: int,
    random_generator: random.Random,
    device: torch.device,
) -> Iterator[ControlBatch]:
    """Yield one shuffled epoch of control batches."""
    order = list(range(len(examples)))
    random_generator.shuffle(order)
    for start in range(0, len(order), batch_size):
        chosen = [examples[index] for index in order[start : start + batch_size]]
        yield collate_control_examples(chosen, tokenizer).to(device)


@torch.no_grad()
def evaluate_exact_match(
    model: ControlModel,
    examples: Sequence[Mapping[str, str]],
    tokenizer: TextTokenizer,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Measure greedy exact match and mean joint answer log probability."""
    model.eval()
    correct = 0
    total = 0
    joint_log_probability_sum = 0.0
    for start in range(0, len(examples), batch_size):
        batch = collate_control_examples(examples[start : start + batch_size], tokenizer).to(device)
        output = model(batch.context_token_ids, batch.answer_token_ids)
        generated = model.greedy_answer_token_ids(batch.context_token_ids)
        joint_log_probability_sum += float(output.joint_log_probability.sum())
        for row, answer in zip(generated, batch.answers, strict=True):
            correct += int(tokenizer.decode_answer(row.tolist()) == answer)
        total += len(batch.answers)
    return {
        "exact_match": correct / total,
        "mean_joint_log_probability": joint_log_probability_sum / total,
    }


def train(config: TrainConfig) -> dict[str, float]:
    """Train the control transformer and return the last evaluation metrics."""
    config.validate()
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    random_generator = random.Random(config.seed)
    dataset_dir = resolve_training_dataset(config.dataset_dir, config.dataset_prefix)
    train_examples = load_examples(resolve_split_path(dataset_dir, "train"), config.limit_examples)
    validation_path = None
    for suffix in (".jsonl.gz", ".jsonl"):
        candidate = dataset_dir / f"validation{suffix}"
        if candidate.exists():
            validation_path = candidate
    validation_examples = [] if validation_path is None else load_examples(validation_path, config.eval_examples)
    model, tokenizer = build_control_model()
    model.to(device)
    optimizer = adamw_optimizer(model, config.learning_rate, config.weight_decay)
    metrics = {"train_loss": 0.0, "step": 0.0}
    step = 0
    with WandbRun(config, "control") as run:
        while step < config.steps:
            for batch in iter_epoch_batches(train_examples, tokenizer, config.batch_size, random_generator, device):
                loss = training_step(model, optimizer, batch, config.max_gradient_norm, run.logger(step + 1))
                step += 1
                if step % config.eval_every == 0 or step == config.steps:
                    metrics = {"train_loss": loss, "step": float(step)}
                    if validation_examples:
                        metrics.update(
                            evaluate_exact_match(model, validation_examples, tokenizer, config.batch_size, device)
                        )
                        print(
                            f"step={step} train_loss={loss:.4f} "
                            f"exact_match={metrics['exact_match']:.4f} "
                            f"mean_joint_log_probability={metrics['mean_joint_log_probability']:.4f}"
                        )
                    else:
                        print(f"step={step} train_loss={loss:.4f}")
                    run.log_eval(step, metrics)
                if step >= config.steps:
                    break
    if config.output_path is not None:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_name": "control", "model": model.state_dict(), "steps": config.steps}, config.output_path)
    return metrics


def parse_arguments(arguments: Sequence[str] | None = None) -> TrainConfig:
    """Parse command-line settings for a control-model training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, help="Local dataset directory. Omit on a TPU to stage the regional bucket copy.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-examples", type=int)
    add_wandb_arguments(parser)
    parsed = parser.parse_args(arguments)
    return TrainConfig(
        dataset_dir=parsed.dataset_dir,
        batch_size=parsed.batch_size,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        max_gradient_norm=parsed.max_gradient_norm,
        steps=parsed.steps,
        eval_every=parsed.eval_every,
        eval_examples=parsed.eval_examples,
        seed=parsed.seed,
        device=parsed.device,
        output_path=parsed.output,
        limit_examples=parsed.limit_examples,
        wandb_project=parsed.wandb_project,
        wandb_enabled=not parsed.disable_wandb,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Train the control model from the command line."""
    train(parse_arguments(arguments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
