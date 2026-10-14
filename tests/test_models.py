"""Tests for the linked-facts sender and receiver."""

from __future__ import annotations

import torch

from linked_facts.generate_dataset import GenerationConfig, generate_world
from control_model import build_control_model
from models import (
    CommunicationModel,
    TextTokenizer,
    build_model,
    build_task_tokenizer,
    collate_token_ids,
    count_owned_parameters,
    joint_answer_log_probability,
)


def _tiny_model() -> tuple[CommunicationModel, TextTokenizer]:
    """Build a tiny vocabulary model for shape and objective checks."""
    tokenizer = TextTokenizer(["buffalo", "chapel"], max_answer_length=4)
    model, tokenizer = build_model(
        tokenizer,
        hidden_size=128,
        message_length=4,
        max_answer_length=4,
        num_layers=1,
        num_heads=4,
    )
    model.eval()
    return model, tokenizer


def _batch_from_strings(
    tokenizer: TextTokenizer,
    passages: list[str],
    questions: list[str],
    answers: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate character passages, questions, and padded answer targets."""
    text = collate_token_ids([tokenizer.encode_text(passage) for passage in passages], tokenizer.pad_id)
    question = collate_token_ids([tokenizer.encode_text(item) for item in questions], tokenizer.pad_id)
    answer = collate_token_ids(
        [tokenizer.encode_answer(item) for item in answers],
        tokenizer.pad_id,
        padded_length=tokenizer.max_answer_length,
    )
    return text, question, answer


def test_sender_returns_four_hidden_states_and_receiver_sees_eight() -> None:
    """Each agent message is four vectors, and the receiver reads both."""
    model, tokenizer = _tiny_model()
    agent_text, question, answer = _batch_from_strings(
        tokenizer,
        ["Feren keeps a buffalo.", "The rocker stands in the chapel."],
        ["Which animal?", "Which room?"],
        ["buffalo", "chapel"],
    )
    messages = model.encode_messages(agent_text, agent_text)
    output = model(agent_text, agent_text, question, answer)

    assert messages.shape == (2, 8, 128)
    assert output.logits.shape == (2, 4, tokenizer.vocab_size)
    assert output.joint_log_probability.shape == (2,)
    assert torch.allclose(output.loss, -output.joint_log_probability.mean())


def test_joint_log_probability_sums_only_answer_tokens() -> None:
    """Padding positions should not change the sequence joint log probability."""
    vocab_size = 6
    logits = torch.zeros(1, 4, vocab_size)
    logits[0, 0, 4] = 4.0
    logits[0, 1, 2] = 3.0
    targets = torch.tensor([[4, 2, 0, 0]])

    joint = joint_answer_log_probability(logits, targets, padding_id=0)
    expected = torch.log_softmax(logits[0, 0], dim=-1)[4] + torch.log_softmax(logits[0, 1], dim=-1)[2]

    assert torch.allclose(joint, expected.reshape(1))


def test_sender_weights_are_shared_and_use_library_initialization() -> None:
    """Both agents should share one sender, with embedding and Xavier defaults."""
    model, _ = _tiny_model()
    content_embedding = model.token_embedding.weight[1:]
    attention_weight = model.sender.transformer.layers[0].self_attn.in_proj_weight

    assert model.sender.token_embedding is model.receiver.token_embedding
    assert model.token_embedding.padding_idx == 0
    assert torch.count_nonzero(model.token_embedding.weight[0]) == 0
    assert 0.9 < content_embedding.std().item() < 1.1
    assert attention_weight.std().item() < 0.2


def test_joint_objective_trains_the_shared_sender() -> None:
    """Answer log probability should send a gradient through both messages."""
    model, tokenizer = _tiny_model()
    model.train()
    agent_text, question, answer = _batch_from_strings(
        tokenizer,
        ["buffalo", "chapel"],
        ["Which animal?", "Which room?"],
        ["buffalo", "chapel"],
    )
    output = model(agent_text, agent_text, question, answer)
    output.loss.backward()

    assert model.sender.query_embedding.weight.grad is not None
    assert torch.count_nonzero(model.sender.query_embedding.weight.grad) > 0


def test_greedy_decoding_stops_within_four_tokens() -> None:
    """The receiver should emit at most four tokens and can stop at EOS."""
    model, tokenizer = _tiny_model()
    agent_text, question, _ = _batch_from_strings(
        tokenizer,
        ["buffalo"],
        ["Which animal?"],
        ["buffalo"],
    )
    generated = model.greedy_answer_token_ids(agent_text, agent_text, question)

    assert generated.shape[0] == 1
    assert generated.shape[1] <= 4
    decoded = tokenizer.decode_answer(generated[0].tolist())
    assert isinstance(decoded, str)


def test_dataset_answers_fit_in_four_tokens() -> None:
    """Every generated answer word should be one token plus the end token."""
    tokenizer = build_task_tokenizer()
    world = generate_world("train", 0, GenerationConfig())

    for question in world["questions"]:
        encoded = tokenizer.encode_answer(question["answer"])
        assert len(encoded) <= 4
        assert encoded[-1] == tokenizer.eos_id
        assert tokenizer.decode_answer(encoded) == question["answer"]


def test_sender_and_receiver_parameter_counts() -> None:
    """Record the parameter counts of the shared sender and the receiver."""
    communication_model, _ = build_model()
    embedding_count = count_owned_parameters(communication_model.token_embedding)
    sender_count = count_owned_parameters(communication_model.sender, communication_model.token_embedding)
    receiver_count = count_owned_parameters(communication_model.receiver, communication_model.token_embedding)
    control_model, _ = build_control_model()

    assert embedding_count == 280_448
    assert sender_count == 463_104
    assert receiver_count == 1_226_255
    assert embedding_count + sender_count + receiver_count == 1_969_807
    assert count_owned_parameters(control_model) == 3_117_583
