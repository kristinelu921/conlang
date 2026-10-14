"""Tests for the full-context control transformer."""

from __future__ import annotations

import math

import torch

from control_model import build_control_model, join_full_context
from control_train import collate_control_examples, training_step
from models import TextTokenizer, collate_token_ids
from train import adamw_optimizer


def test_control_model_reads_both_passages_and_the_question() -> None:
    """The control transformer scores the answer from one full character context."""
    tokenizer = TextTokenizer(["buffalo"], max_answer_length=4)
    model, tokenizer = build_control_model(tokenizer, hidden_size=128, num_layers=1, num_heads=4)
    model.eval()
    context = join_full_context("Feren keeps a buffalo.", "The buffalo rests on a rocker.", "Which animal?")
    context_ids = collate_token_ids([tokenizer.encode_text(context)], tokenizer.pad_id)
    answer_ids = collate_token_ids([tokenizer.encode_answer("buffalo")], tokenizer.pad_id, padded_length=4)
    output = model(context_ids, answer_ids)

    assert "Feren keeps a buffalo." in context
    assert "Which animal?" in context
    assert output.logits.shape == (1, 4, tokenizer.vocab_size)
    assert torch.allclose(output.loss, -output.joint_log_probability.mean())
    assert model.greedy_answer_token_ids(context_ids).shape[1] <= 4
    assert model.config.hidden_size == 128


def test_default_control_model_is_twice_the_narrow_width() -> None:
    """The default control model uses hidden size 192 and six heads."""
    model, _ = build_control_model(TextTokenizer(["buffalo"]))

    assert model.config.hidden_size == 192
    assert model.config.num_heads == 6


def test_control_training_step_updates_the_output_projection() -> None:
    """One joint-logprob step should change the control transformer."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    model, tokenizer = build_control_model(tokenizer, num_layers=1)
    examples = [
        {
            "agent_1_input": "Feren keeps a buffalo.",
            "agent_2_input": "The buffalo rests on a rocker.",
            "question": "Which animal?",
            "answer": "buffalo",
        }
    ]
    batch = collate_control_examples(examples, tokenizer)
    output_weight = model.output_projection.weight.detach().clone()
    optimizer = adamw_optimizer(model, learning_rate=3e-4, weight_decay=0.01)

    loss = training_step(model, optimizer, batch, max_gradient_norm=1.0)

    assert math.isfinite(loss)
    assert not torch.equal(model.output_projection.weight, output_weight)
