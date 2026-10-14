"""Weights & Biases metrics for the linked-facts trainers."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from control_model import ControlModel
from models import CommunicationModel, CommunicationOutput

MetricValue = float | torch.Tensor
Metrics = dict[str, MetricValue]


def add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared Weights & Biases command-line flags."""
    parser.add_argument("--wandb-project", default="conlangs")
    parser.add_argument("--disable-wandb", action="store_true")


def serializable_config(config: Any) -> dict[str, Any]:
    """Convert a dataclass config into plain values for Weights & Biases."""
    values = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def root_mean_square(values: torch.Tensor) -> float:
    """Return the root mean square of one activation tensor."""
    flat = values.detach().float()
    return float(flat.pow(2).mean().sqrt())


def parameter_norm(parameters: Sequence[nn.Parameter]) -> float:
    """Return the L2 norm of a parameter list."""
    squares = [parameter.detach().float().pow(2).sum() for parameter in parameters]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


def gradient_norm(gradients: Sequence[torch.Tensor | None]) -> float:
    """Return the L2 norm of a gradient list, skipping unused parameters."""
    squares = [gradient.detach().float().pow(2).sum() for gradient in gradients if gradient is not None]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


def populated_gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    """Return the L2 norm of gradients already stored on parameters."""
    return gradient_norm([parameter.grad for parameter in parameters])


def parameters_excluding(module: nn.Module, excluded_prefix: str) -> list[nn.Parameter]:
    """Return trainable parameters whose names do not start with the prefix."""
    return [
        parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and not name.startswith(excluded_prefix)
    ]


def isolated_path_gradients(
    loss: torch.Tensor,
    upstream: torch.Tensor,
    parameters: Sequence[nn.Parameter],
) -> tuple[torch.Tensor | None, ...]:
    """Return parameter gradients that flow through one upstream activation."""
    if not parameters:
        return ()
    upstream_gradient = torch.autograd.grad(loss, upstream, retain_graph=True)[0]
    return torch.autograd.grad(
        upstream,
        list(parameters),
        grad_outputs=upstream_gradient,
        retain_graph=True,
        allow_unused=True,
    )


def direct_gradient_norm(loss: torch.Tensor, parameters: Sequence[nn.Parameter]) -> float:
    """Return a parameter gradient norm without freeing the autograd graph."""
    if not parameters:
        return 0.0
    gradients = torch.autograd.grad(loss, list(parameters), retain_graph=True, allow_unused=True)
    return gradient_norm(gradients)


def output_distribution_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    padding_id: int,
) -> Metrics:
    """Summarize the answer-token distribution at each position."""
    probabilities = logits.detach().float().softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    target_probability = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    predicted = logits.detach().argmax(dim=-1)
    metrics: Metrics = {}
    active_predictions: list[torch.Tensor] = []
    for position in range(probabilities.shape[1]):
        active = targets[:, position].ne(padding_id)
        if not bool(active.any()):
            continue
        metrics[f"output/position_{position}/entropy"] = float(entropy[:, position][active].mean())
        metrics[f"output/position_{position}/target_probability"] = float(
            target_probability[:, position][active].mean()
        )
        metrics[f"output/position_{position}/mean_distribution"] = probabilities[:, position][active].mean(dim=0)
        active_predictions.append(predicted[:, position][active].float())
    if active_predictions:
        metrics["output/predicted_token_ids"] = torch.cat(active_predictions)
    mask = targets.ne(padding_id)
    denominator = mask.sum().clamp_min(1)
    metrics["output/mean_entropy"] = float((entropy * mask).sum() / denominator)
    metrics["output/mean_target_probability"] = float((target_probability * mask).sum() / denominator)
    return metrics


def four_token_metrics(prefix: str, states: torch.Tensor) -> Metrics:
    """Log the value distribution and RMS of each of four intermediate vectors."""
    metrics: Metrics = {f"{prefix}/rms": root_mean_square(states)}
    for index in range(min(4, states.shape[1])):
        token = states[:, index]
        metrics[f"{prefix}/token_{index}/values"] = token.detach()
        metrics[f"{prefix}/token_{index}/rms"] = root_mean_square(token)
    return metrics


def communication_metrics(
    model: CommunicationModel,
    output: CommunicationOutput,
    answer_token_ids: torch.Tensor,
) -> Metrics:
    """Collect loss, magnitudes, token distributions, and the three model gradients."""
    assert output.message_1 is not None
    assert output.message_2 is not None
    assert output.answer_hidden is not None
    assert output.receiver_memory is not None
    sender_parameters = parameters_excluding(model.sender, "token_embedding")
    receiver_parameters = parameters_excluding(model.receiver, "token_embedding")
    embedding = model.token_embedding.weight
    agent_1_gradients = isolated_path_gradients(output.loss, output.message_1, [*sender_parameters, embedding])
    agent_2_gradients = isolated_path_gradients(output.loss, output.message_2, [*sender_parameters, embedding])
    receiver_gradients = torch.autograd.grad(
        output.loss, receiver_parameters, retain_graph=True, allow_unused=True
    )
    embedding_gradient = torch.autograd.grad(output.loss, embedding, retain_graph=True)[0]
    assert agent_1_gradients[-1] is not None
    assert agent_2_gradients[-1] is not None
    receiver_embedding_gradient = embedding_gradient - agent_1_gradients[-1] - agent_2_gradients[-1]
    metrics: Metrics = {
        "loss": float(output.loss.detach()),
        "joint_log_probability": float(output.joint_log_probability.detach().mean()),
        "weights/global": parameter_norm(list(model.parameters())),
        "weights/sender": parameter_norm(sender_parameters),
        "weights/receiver": parameter_norm(receiver_parameters),
        "weights/token_embedding": parameter_norm([embedding]),
        "gradients/agent_1_sender": gradient_norm(agent_1_gradients),
        "gradients/agent_2_sender": gradient_norm(agent_2_gradients),
        "gradients/receiver": gradient_norm([*receiver_gradients, receiver_embedding_gradient]),
        "gradients/token_embedding": gradient_norm([embedding_gradient]),
        "activations/agent_1_sender_rms": root_mean_square(output.message_1),
        "activations/agent_2_sender_rms": root_mean_square(output.message_2),
        "activations/receiver_answer_rms": root_mean_square(output.answer_hidden),
        "activations/receiver_memory_rms": root_mean_square(output.receiver_memory),
    }
    metrics.update(output_distribution_metrics(output.logits, answer_token_ids, model.config.padding_id))
    metrics.update(four_token_metrics("messages/agent_1", output.message_1))
    metrics.update(four_token_metrics("messages/agent_2", output.message_2))
    metrics.update(four_token_metrics("receiver/answer", output.answer_hidden))
    return metrics


def control_metrics(model: ControlModel, output: CommunicationOutput, answer_token_ids: torch.Tensor) -> Metrics:
    """Collect control loss, three module gradients, and the four answer states."""
    assert output.answer_hidden is not None
    assert output.receiver_memory is not None
    encoder_parameters = list(model.encoder.parameters()) + list(model.position_embedding.parameters())
    decoder_parameters = list(model.decoder.parameters()) + list(model.answer_position_embedding.parameters())
    output_parameters = list(model.output_projection.parameters())
    embedding = model.token_embedding.weight
    encoder_gradients = isolated_path_gradients(output.loss, output.receiver_memory, [*encoder_parameters, embedding])
    decoder_gradients = torch.autograd.grad(output.loss, decoder_parameters, retain_graph=True, allow_unused=True)
    embedding_gradient = torch.autograd.grad(output.loss, embedding, retain_graph=True)[0]
    assert encoder_gradients[-1] is not None
    decoder_embedding_gradient = embedding_gradient - encoder_gradients[-1]
    metrics: Metrics = {
        "loss": float(output.loss.detach()),
        "joint_log_probability": float(output.joint_log_probability.detach().mean()),
        "weights/global": parameter_norm(list(model.parameters())),
        "weights/encoder": parameter_norm(encoder_parameters),
        "weights/decoder": parameter_norm(decoder_parameters),
        "weights/output_projection": parameter_norm(output_parameters),
        "weights/token_embedding": parameter_norm([embedding]),
        "gradients/encoder": gradient_norm(encoder_gradients),
        "gradients/decoder": gradient_norm([*decoder_gradients, decoder_embedding_gradient]),
        "gradients/output_projection": direct_gradient_norm(output.loss, output_parameters),
        "gradients/token_embedding": gradient_norm([embedding_gradient]),
        "activations/encoder_rms": root_mean_square(output.receiver_memory),
        "activations/answer_rms": root_mean_square(output.answer_hidden),
    }
    metrics.update(output_distribution_metrics(output.logits, answer_token_ids, model.config.padding_id))
    metrics.update(four_token_metrics("decoder/answer", output.answer_hidden))
    return metrics


def wandb_payload(metrics: Metrics) -> dict[str, Any]:
    """Turn activation tensors into histograms and leave scalars unchanged."""
    import wandb

    payload: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            payload[key] = wandb.Histogram(value.detach().float().flatten().cpu().numpy())
        else:
            payload[key] = value
    return payload


def log_metrics(run: Any, metrics: Metrics, step: int) -> None:
    """Send one step of scalars and histograms to a Weights & Biases run."""
    run.log(wandb_payload(metrics), step=step)


def step_logger(run: Any, step: int) -> Callable[[Metrics], None]:
    """Return a logger that records one training step at a fixed index."""

    def log_step(metrics: Metrics) -> None:
        """Send this step's metrics to the open Weights & Biases run."""
        log_metrics(run, metrics, step)

    return log_step


def start_wandb(project: str, job_name: str, config: dict[str, Any], enabled: bool) -> Any | None:
    """Open a Weights & Biases run, or return None when logging is disabled."""
    if not enabled:
        return None
    import wandb

    return wandb.init(project=project, name=job_name, job_type=job_name, config=config)


def finish_wandb(run: Any | None) -> None:
    """Close a Weights & Biases run when one was opened."""
    if run is not None:
        run.finish()


def log_backward(
    logger: Callable[[Metrics], None] | None,
    collect: Callable[..., Metrics],
    model: nn.Module,
    output: CommunicationOutput,
    answer_token_ids: torch.Tensor,
) -> None:
    """Backpropagate and, when a logger is set, record the pre-clip metrics."""
    metrics = collect(model, output, answer_token_ids) if logger is not None else None
    output.loss.backward()
    if metrics is None or logger is None:
        return
    metrics["gradients/global"] = populated_gradient_norm(list(model.parameters()))
    logger(metrics)


class WandbRun:
    """One Weights & Biases run that stays inert when logging is disabled."""

    def __init__(self, config: Any, job_name: str) -> None:
        """Open the run named by the training config."""
        self._run = start_wandb(config.wandb_project, job_name, serializable_config(config), config.wandb_enabled)

    def __enter__(self) -> WandbRun:
        """Return this run for a with-block."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: Any) -> None:
        """Close the run when the training block ends."""
        finish_wandb(self._run)

    def logger(self, step: int) -> Callable[[Metrics], None] | None:
        """Return a one-step logger, or None when logging is disabled."""
        if self._run is None:
            return None
        return step_logger(self._run, step)

    def log_eval(self, step: int, metrics: Mapping[str, float]) -> None:
        """Log evaluation scalars under the eval prefix."""
        if self._run is not None:
            log_metrics(self._run, {f"eval/{key}": value for key, value in metrics.items()}, step)
