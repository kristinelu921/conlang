"""Tests for the training-step profiler."""

from __future__ import annotations

import torch

from models import TextTokenizer, build_model
from speed import format_training_profile, profile_communication_training
from train import adamw_optimizer


def _examples() -> list[dict[str, str]]:
    """Return two short question examples for a profiled step."""
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


def test_profile_separates_metric_logging_from_the_training_step() -> None:
    """Metric logging should show up as its own phase and disappear when disabled."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    model, tokenizer = build_model(tokenizer, hidden_size=32, num_layers=1, num_heads=4)
    examples = _examples()
    device = torch.device("cpu")
    logged = profile_communication_training(
        model,
        tokenizer,
        examples,
        adamw_optimizer(model, learning_rate=3e-4, weight_decay=0.0),
        batch_size=2,
        measured_steps=1,
        warmup_steps=0,
        device=device,
        include_metrics=True,
        operator_limit=5,
    )
    plain = profile_communication_training(
        model,
        tokenizer,
        examples,
        adamw_optimizer(model, learning_rate=3e-4, weight_decay=0.0),
        batch_size=2,
        measured_steps=1,
        warmup_steps=0,
        device=device,
        include_metrics=False,
        operator_limit=0,
    )
    logged_seconds = dict(logged.phase_seconds)
    plain_seconds = dict(plain.phase_seconds)

    assert set(logged_seconds) == {
        "collate",
        "host_to_device",
        "zero_grad",
        "forward",
        "metrics",
        "backward",
        "clip",
        "optimizer",
    }
    assert logged_seconds["forward"] > 0
    assert logged_seconds["backward"] > 0
    assert logged_seconds["metrics"] > plain_seconds["metrics"]
    assert logged.tokens_per_step > 0
    report = format_training_profile(logged)
    assert "slowest:" in report
    assert "metrics" in report
    assert "CPU" in logged.operator_table
    assert plain.operator_table == ""
