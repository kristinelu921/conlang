"""Tests for the linked-facts answer tokenizer."""

from __future__ import annotations

from linked_facts.generate_dataset import GenerationConfig, generate_world
from tokenizer import build_task_tokenizer


def test_dataset_answers_fit_in_four_tokens() -> None:
    """Every generated answer word should be one token plus the end token."""
    tokenizer = build_task_tokenizer()
    world = generate_world("train", 0, GenerationConfig())

    for question in world["questions"]:
        encoded = tokenizer.encode_answer(question["answer"])
        assert len(encoded) <= 4
        assert encoded[-1] == tokenizer.eos_id
        assert tokenizer.decode_answer(encoded) == question["answer"]
