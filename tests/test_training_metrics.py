"""Tests for linked-facts training metrics."""

from __future__ import annotations

import math

import torch

from control_model import build_control_model, join_full_context
from control_train import collate_control_examples, training_step as control_training_step
from models import TextTokenizer, build_model
from train import adamw_optimizer, collate_examples, parse_arguments, training_step
from training_metrics import (
    WandbRun,
    communication_metrics,
    control_metrics,
    isolated_path_gradients,
    parameters_excluding,
    wandb_payload,
)


def _examples() -> list[dict[str, str]]:
    """Return two question examples with different passages."""
    return [
        {
            "agent_1_input": "Feren keeps a buffalo.",
            "agent_2_input": "The buffalo rests on a rocker.",
            "question": "Which animal?",
            "answer": "buffalo",
        },
        {
            "agent_1_input": "Mesa owns a rocker.",
            "agent_2_input": "The rocker stands in the chapel.",
            "question": "Which room?",
            "answer": "chapel",
        },
    ]


def test_communication_metrics_cover_three_models_and_four_tokens() -> None:
    """Each sender path, the receiver, and their four vectors should be logged."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    model, tokenizer = build_model(tokenizer, hidden_size=32, num_layers=1, num_heads=4)
    model.train()
    batch = collate_examples(_examples(), tokenizer)
    output = model(
        batch.agent_1_token_ids,
        batch.agent_2_token_ids,
        batch.question_token_ids,
        batch.answer_token_ids,
    )
    metrics = communication_metrics(model, output, batch.answer_token_ids)

    assert output.message_1 is not None and output.message_1.shape == (2, 4, 32)
    assert output.message_2 is not None and output.message_2.shape == (2, 4, 32)
    for key in (
        "loss",
        "gradients/agent_1_sender",
        "gradients/agent_2_sender",
        "gradients/receiver",
        "weights/sender",
        "weights/receiver",
        "activations/agent_1_sender_rms",
        "activations/receiver_answer_rms",
        "output/position_0/mean_distribution",
    ):
        assert key in metrics
    for model_name in ("messages/agent_1", "messages/agent_2", "receiver/answer"):
        for index in range(4):
            values = metrics[f"{model_name}/token_{index}/values"]
            assert isinstance(values, torch.Tensor)
            assert values.shape == (2, 32)
            assert math.isfinite(float(metrics[f"{model_name}/token_{index}/rms"]))
    assert metrics["output/position_0/mean_distribution"].shape == (tokenizer.vocab_size,)
    assert all(math.isfinite(float(metrics[key])) for key in ("loss", "gradients/agent_1_sender", "gradients/receiver"))


def test_sender_path_gradients_sum_to_the_shared_sender() -> None:
    """The two sender paths should partition the shared sender weights."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    model, tokenizer = build_model(tokenizer, hidden_size=32, num_layers=1, num_heads=4)
    model.train()
    batch = collate_examples(_examples(), tokenizer)
    output = model(
        batch.agent_1_token_ids,
        batch.agent_2_token_ids,
        batch.question_token_ids,
        batch.answer_token_ids,
    )
    assert output.message_1 is not None and output.message_2 is not None
    sender_parameters = parameters_excluding(model.sender, "token_embedding")
    agent_1 = isolated_path_gradients(output.loss, output.message_1, sender_parameters)
    agent_2 = isolated_path_gradients(output.loss, output.message_2, sender_parameters)
    combined = torch.autograd.grad(output.loss, sender_parameters, retain_graph=True)
    for left, right, total in zip(agent_1, agent_2, combined, strict=True):
        assert left is not None and right is not None and total is not None
        assert torch.allclose(left + right, total, atol=1e-4, rtol=1e-4)


def test_training_step_logs_preclip_global_gradient() -> None:
    """A logged step should keep a finite global gradient and still update weights."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    model, tokenizer = build_model(tokenizer, hidden_size=32, num_layers=1, num_heads=4)
    batch = collate_examples(_examples(), tokenizer)
    before = model.sender.query_embedding.weight.detach().clone()
    recorded: list[dict[str, float | torch.Tensor]] = []
    loss = training_step(
        model,
        adamw_optimizer(model, learning_rate=3e-4, weight_decay=0.01),
        batch,
        max_gradient_norm=1.0,
        metrics_logger=recorded.append,
    )

    assert math.isfinite(loss)
    assert len(recorded) == 1
    assert math.isfinite(float(recorded[0]["gradients/global"]))
    assert not torch.equal(before, model.sender.query_embedding.weight)


def test_control_metrics_log_three_modules_and_four_answer_states() -> None:
    """The control run should split encoder, decoder, and output gradients."""
    tokenizer = TextTokenizer(["buffalo", "chapel"], max_answer_length=4)
    model, tokenizer = build_control_model(tokenizer, hidden_size=32, num_layers=1, num_heads=4)
    examples = [
        {**example, "context": join_full_context(example["agent_1_input"], example["agent_2_input"], example["question"])}
        for example in _examples()
    ]
    batch = collate_control_examples(examples, tokenizer)
    recorded: list[dict[str, float | torch.Tensor]] = []
    control_training_step(
        model,
        adamw_optimizer(model, learning_rate=3e-4, weight_decay=0.0),
        batch,
        max_gradient_norm=1.0,
        metrics_logger=recorded.append,
    )
    metrics = recorded[0]

    for key in ("gradients/encoder", "gradients/decoder", "gradients/output_projection", "activations/answer_rms"):
        assert math.isfinite(float(metrics[key]))
    for index in range(4):
        values = metrics[f"decoder/answer/token_{index}/values"]
        assert isinstance(values, torch.Tensor)
        assert values.shape[0] == 2
    direct = control_metrics(model, model(batch.context_token_ids, batch.answer_token_ids), batch.answer_token_ids)
    assert "output/position_0/entropy" in direct


def test_wandb_payload_turns_token_values_into_histograms() -> None:
    """Tensor metrics should become histograms and scalars should stay scalars."""
    payload = wandb_payload({"loss": 1.5, "messages/agent_1/token_0/values": torch.zeros(2, 4)})

    assert payload["loss"] == 1.5
    assert payload["messages/agent_1/token_0/values"].__class__.__name__ == "Histogram"


def test_disabled_wandb_stays_closed() -> None:
    """The disable flag should keep a run from opening."""
    config = parse_arguments(["--dataset-dir", "datasets/linked_facts_v1", "--disable-wandb"])

    assert config.wandb_enabled is False
    assert config.wandb_project == "conlangs"
    with WandbRun(config, "communication") as run:
        assert run.logger(1) is None
        run.log_eval(1, {"exact_match": 0.0})
