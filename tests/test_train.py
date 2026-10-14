"""Tests for the linked-facts training step."""

from __future__ import annotations

import math

import torch

from models import TextTokenizer, build_model
from train import adamw_optimizer, collate_examples, training_step


def _examples() -> list[dict[str, str]]:
    """Return two question examples that share no communication bottleneck."""
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


def test_adamw_decays_matrices_and_skips_vectors() -> None:
    """Weight decay should apply to matrices and skip biases and layer-norm scales."""
    model, _ = build_model(TextTokenizer(["buffalo", "chapel"]), num_layers=1)
    optimizer = adamw_optimizer(model, learning_rate=3e-4, weight_decay=0.01)
    decayed, undecayed = optimizer.param_groups

    assert decayed["weight_decay"] == 0.01
    assert undecayed["weight_decay"] == 0.0
    assert all(parameter.ndim >= 2 for parameter in decayed["params"])
    assert all(parameter.ndim < 2 for parameter in undecayed["params"])


def test_training_step_updates_the_communication_model() -> None:
    """One joint-logprob step should change a sender weight and stay finite."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    model, tokenizer = build_model(tokenizer, num_layers=1)
    batch = collate_examples(_examples(), tokenizer)
    query_weight = model.sender.query_embedding.weight.detach().clone()
    optimizer = adamw_optimizer(model, learning_rate=3e-4, weight_decay=0.01)

    loss = training_step(model, optimizer, batch, max_gradient_norm=1.0)

    assert math.isfinite(loss)
    assert not torch.equal(model.sender.query_embedding.weight, query_weight)
