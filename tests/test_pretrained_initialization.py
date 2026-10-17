"""Tests for deterministic T5-Efficient-Tiny parameter transfer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from pretrained_initialization import initialize_from_t5_efficient_tiny
from tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, TextTokenizer


def _write_fake_t5_artifacts(
    model_directory: Path,
    task_tokenizer: TextTokenizer,
) -> tuple[str, str, np.ndarray, dict[tuple[str, ...], Any]]:
    """Write compact checkpoint and tokenizer fixtures with recognizable values."""
    source_vocabulary = {PAD_TOKEN: 0, "</s>": 1, UNK_TOKEN: 2}
    next_token_id = 3
    for task_token in task_tokenizer.id_to_token:
        if task_token in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN}:
            continue
        source_vocabulary[task_token] = next_token_id
        next_token_id += 1
    for query_index in range(4):
        source_vocabulary[f"<extra_id_{query_index}>"] = next_token_id
        next_token_id += 1
    source_tokenizer = Tokenizer(WordLevel(source_vocabulary, unk_token=UNK_TOKEN))
    tokenizer_path = model_directory / "tokenizer.json"
    source_tokenizer.save(str(tokenizer_path))

    source_embedding = np.arange(next_token_id * 256, dtype=np.float32).reshape(
        next_token_id,
        256,
    )
    square_kernel = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    feedforward_in = np.arange(256 * 1024, dtype=np.float32).reshape(256, 1024)
    feedforward_out = np.arange(1024 * 256, dtype=np.float32).reshape(1024, 256)
    encoder_norm = np.linspace(1.0, 2.0, 256, dtype=np.float32)
    decoder_norm = np.linspace(2.0, 3.0, 256, dtype=np.float32)
    source_parameters = {
        ("shared", "embedding"): source_embedding,
        ("encoder", "final_layer_norm", "weight"): encoder_norm,
        ("decoder", "final_layer_norm", "weight"): decoder_norm,
        ("encoder", "block", "0", "layer", "0", "SelfAttention", "q", "kernel"): square_kernel,
        (
            "encoder",
            "block",
            "0",
            "layer",
            "0",
            "SelfAttention",
            "relative_attention_bias",
            "embedding",
        ): np.arange(32 * 4, dtype=np.float32).reshape(32, 4),
        ("encoder", "block", "0", "layer", "0", "layer_norm", "weight"): encoder_norm,
        (
            "encoder",
            "block",
            "0",
            "layer",
            "1",
            "DenseReluDense",
            "wi",
            "kernel",
        ): feedforward_in,
        (
            "encoder",
            "block",
            "0",
            "layer",
            "1",
            "DenseReluDense",
            "wo",
            "kernel",
        ): feedforward_out,
        ("encoder", "block", "0", "layer", "1", "layer_norm", "weight"): encoder_norm,
        ("decoder", "block", "0", "layer", "0", "SelfAttention", "q", "kernel"): square_kernel,
        (
            "decoder",
            "block",
            "0",
            "layer",
            "0",
            "SelfAttention",
            "relative_attention_bias",
            "embedding",
        ): np.arange(32 * 4, dtype=np.float32).reshape(32, 4),
        ("decoder", "block", "0", "layer", "0", "layer_norm", "weight"): decoder_norm,
        (
            "decoder",
            "block",
            "0",
            "layer",
            "1",
            "EncDecAttention",
            "q",
            "kernel",
        ): square_kernel,
        ("decoder", "block", "0", "layer", "1", "layer_norm", "weight"): decoder_norm,
        (
            "decoder",
            "block",
            "0",
            "layer",
            "2",
            "DenseReluDense",
            "wi",
            "kernel",
        ): feedforward_in,
        (
            "decoder",
            "block",
            "0",
            "layer",
            "2",
            "DenseReluDense",
            "wo",
            "kernel",
        ): feedforward_out,
        ("decoder", "block", "0", "layer", "2", "layer_norm", "weight"): decoder_norm,
    }
    checkpoint_path = model_directory / "flax_model.msgpack"
    checkpoint_path.write_bytes(serialization.msgpack_serialize(unflatten_dict(source_parameters)))
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    return checkpoint_sha256, tokenizer_sha256, source_embedding, source_parameters


def test_t5_initialization_assigns_every_target_leaf(tmp_path: Path) -> None:
    """All standard and communication-specific parameters should receive explicit values."""
    task_tokenizer = TextTokenizer(["buffalo"])
    checkpoint_sha256, tokenizer_sha256, source_embedding, source_parameters = (
        _write_fake_t5_artifacts(tmp_path, task_tokenizer)
    )
    target_parameters = {
        "token_embedding": {
            "embedding": np.zeros((task_tokenizer.vocab_size, 256), dtype=np.float32),
        },
        "sender": {
            "query_embedding": np.zeros((4, 256), dtype=np.float32),
            "message_projection": {
                "kernel": np.zeros((256, 256), dtype=np.float32),
                "bias": np.ones((256,), dtype=np.float32),
            },
            "layer_0": {
                "relative_attention_bias": np.zeros((4, 32), dtype=np.float32),
                "self_attention": {
                    "query": {
                        "kernel": np.zeros((256, 4, 64), dtype=np.float32),
                        "bias": np.ones((4, 64), dtype=np.float32),
                    },
                },
                "attention_norm": {"scale": np.zeros((256,), dtype=np.float32)},
                "feedforward_in": {"kernel": np.zeros((256, 1024), dtype=np.float32)},
                "feedforward_out": {"kernel": np.zeros((1024, 256), dtype=np.float32)},
                "feedforward_norm": {"scale": np.zeros((256,), dtype=np.float32)},
            },
        },
        "receiver": {
            "query_key_value_encoder": {
                "layer_0": {
                    "self_attention": {
                        "query": {"kernel": np.zeros((256, 4, 64), dtype=np.float32)},
                    },
                },
            },
            "query_message_block": {
                "question_message_attention": {
                    "query": {"kernel": np.zeros((256, 4, 64), dtype=np.float32)},
                },
                "feedforward_norm": {"scale": np.zeros((256,), dtype=np.float32)},
            },
            "decoder_layer_0": {
                "self_relative_attention_bias": np.zeros((4, 32), dtype=np.float32),
                "cross_relative_attention_bias": np.ones((4, 32), dtype=np.float32),
                "self_attention": {
                    "query": {"kernel": np.zeros((256, 4, 64), dtype=np.float32)},
                },
                "cross_attention": {
                    "query": {"kernel": np.zeros((256, 4, 64), dtype=np.float32)},
                },
                "cross_attention_norm": {"scale": np.zeros((256,), dtype=np.float32)},
                "feedforward_in": {"kernel": np.zeros((256, 1024), dtype=np.float32)},
            },
            "output_hidden_projection": {
                "kernel": np.zeros((256, 256), dtype=np.float32),
                "bias": np.ones((256,), dtype=np.float32),
            },
            "output_projection": {
                "kernel": np.zeros((256, task_tokenizer.vocab_size), dtype=np.float32),
                "bias": np.ones((task_tokenizer.vocab_size,), dtype=np.float32),
            },
        },
    }

    initialized_parameters, summary = initialize_from_t5_efficient_tiny(
        target_parameters,
        task_tokenizer,
        tmp_path,
        checkpoint_sha256,
        tokenizer_sha256,
    )
    flat_parameters = flatten_dict(initialized_parameters)
    source_query_kernel = source_parameters[
        ("encoder", "block", "0", "layer", "0", "SelfAttention", "q", "kernel")
    ]
    buffalo_id = task_tokenizer.token_to_id["buffalo"]
    fake_source_tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    source_buffalo_id = fake_source_tokenizer.token_to_id("buffalo")
    assert source_buffalo_id is not None
    source_embedding_rms = np.sqrt(np.mean(np.square(source_embedding)))

    assert summary.target_leaves == len(flat_parameters)
    assert (
        summary.checkpoint_leaves
        + summary.checkpoint_derived_leaves
        + summary.explicit_fallback_leaves
        == summary.target_leaves
    )
    assert (
        summary.checkpoint_parameters
        + summary.checkpoint_derived_parameters
        + summary.explicit_fallback_parameters
        == summary.target_parameters
    )
    assert np.array_equal(
        flat_parameters[("sender", "layer_0", "self_attention", "query", "kernel")],
        source_query_kernel.reshape(256, 4, 64),
    )
    assert np.all(flat_parameters[("sender", "layer_0", "self_attention", "query", "bias")] == 0)
    assert np.array_equal(
        flat_parameters[("token_embedding", "embedding")][buffalo_id],
        source_embedding[source_buffalo_id] / source_embedding_rms,
    )
    assert np.array_equal(
        flat_parameters[("receiver", "output_projection", "kernel")],
        flat_parameters[("token_embedding", "embedding")].T / np.sqrt(256),
    )
    assert np.array_equal(
        np.diag(flat_parameters[("sender", "message_projection", "kernel")]),
        source_parameters[("encoder", "final_layer_norm", "weight")],
    )
    assert np.all(flat_parameters[("receiver", "decoder_layer_0", "cross_relative_attention_bias")] == 0)
    assert np.all(flat_parameters[("receiver", "output_hidden_projection", "bias")] == 0)


def test_t5_initialization_rejects_artifact_hash_mismatch(tmp_path: Path) -> None:
    """A local checkpoint with the wrong digest should fail before deserialization."""
    task_tokenizer = TextTokenizer(["buffalo"])
    _write_fake_t5_artifacts(tmp_path, task_tokenizer)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        initialize_from_t5_efficient_tiny(
            {},
            task_tokenizer,
            tmp_path,
            "0" * 64,
            hashlib.sha256((tmp_path / "tokenizer.json").read_bytes()).hexdigest(),
        )
