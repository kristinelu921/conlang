"""Load a maximal T5-Efficient-Tiny warm start into the communication model."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict
from tokenizers import Tokenizer

from tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, TextTokenizer


T5_EFFICIENT_TINY_INITIALIZATION = "google_t5_efficient_tiny"
T5_EFFICIENT_TINY_CHECKPOINT_FILENAME = "flax_model.msgpack"
T5_EFFICIENT_TINY_TOKENIZER_FILENAME = "tokenizer.json"
# T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.
# https://arxiv.org/abs/1910.10683
T5_MODEL_DIMENSION = 256


@dataclass(frozen=True)
class PretrainedInitializationSummary:
    """Count how every target parameter leaf received its initial value."""

    target_leaves: int
    checkpoint_leaves: int
    checkpoint_derived_leaves: int
    explicit_fallback_leaves: int
    target_parameters: int
    checkpoint_parameters: int
    checkpoint_derived_parameters: int
    explicit_fallback_parameters: int


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one local pretrained-model artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact(path: Path, expected_sha256: str) -> None:
    """Require one pretrained-model artifact to match its configured digest."""
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("pretrained artifact SHA-256 must be 64 lowercase hexadecimal characters")
    if not path.is_file():
        raise FileNotFoundError(f"pretrained artifact does not exist: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"pretrained artifact SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )


def verify_t5_efficient_tiny_artifacts(
    model_directory: Path,
    checkpoint_sha256: str,
    tokenizer_sha256: str,
) -> None:
    """Verify both pinned T5 artifacts before creating run state or opening JAX."""
    _verify_artifact(
        model_directory / T5_EFFICIENT_TINY_CHECKPOINT_FILENAME,
        checkpoint_sha256,
    )
    _verify_artifact(
        model_directory / T5_EFFICIENT_TINY_TOKENIZER_FILENAME,
        tokenizer_sha256,
    )


def _source_array(
    source_parameters: Mapping[tuple[str, ...], Any],
    path: tuple[str, ...],
) -> np.ndarray:
    """Read one required array from the flattened T5 checkpoint."""
    if path not in source_parameters:
        raise ValueError(f"T5 checkpoint is missing {'/'.join(path)}")
    return np.asarray(source_parameters[path], dtype=np.float32)


def _reshape_attention_kernel(
    source: np.ndarray,
    target: np.ndarray,
    compensate_query_scaling: bool,
) -> np.ndarray:
    """Reshape a T5 attention kernel and preserve its query-scaling convention."""
    if source.size != target.size:
        raise ValueError(
            f"attention kernel size mismatch: source {source.shape}, target {target.shape}"
        )
    reshaped = source.reshape(target.shape)
    if compensate_query_scaling:
        reshaped = reshaped * np.sqrt(target.shape[-1])
    return reshaped


def _attention_parameter(
    source_parameters: Mapping[tuple[str, ...], Any],
    source_attention_path: tuple[str, ...],
    attention_part: str,
    parameter_name: str,
    target: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Map one attention leaf while preserving native T5 query scaling."""
    if parameter_name == "bias":
        return np.zeros_like(target), "fallback"
    source_names = {"query": "q", "key": "k", "value": "v", "out": "o"}
    source = _source_array(
        source_parameters,
        (*source_attention_path, source_names[attention_part], "kernel"),
    )
    compensate_query_scaling = attention_part == "query"
    category = "derived" if compensate_query_scaling else "checkpoint"
    return _reshape_attention_kernel(source, target, compensate_query_scaling), category


def _encoder_parameter(
    source_parameters: Mapping[tuple[str, ...], Any],
    layer_index: int,
    suffix: tuple[str, ...],
    target: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Map one communication encoder leaf from the matching T5 encoder block."""
    source_prefix = ("encoder", "block", str(layer_index), "layer")
    if suffix == ("relative_attention_bias",):
        source = _source_array(
            source_parameters,
            (
                "encoder",
                "block",
                "0",
                "layer",
                "0",
                "SelfAttention",
                "relative_attention_bias",
                "embedding",
            ),
        )
        return source.T, "checkpoint"
    if suffix[0] == "self_attention":
        attention_part, parameter_name = suffix[1:]
        return _attention_parameter(
            source_parameters,
            (*source_prefix, "0", "SelfAttention"),
            attention_part,
            parameter_name,
            target,
        )
    if suffix[0] == "attention_norm":
        if suffix[1] == "bias":
            return np.zeros_like(target), "fallback"
        return (
            _source_array(source_parameters, (*source_prefix, "0", "layer_norm", "weight")),
            "checkpoint",
        )
    if suffix[0] == "feedforward_in":
        if suffix[1] == "bias":
            return np.zeros_like(target), "fallback"
        return (
            _source_array(
                source_parameters,
                (*source_prefix, "1", "DenseReluDense", "wi", "kernel"),
            ),
            "checkpoint",
        )
    if suffix[0] == "feedforward_out":
        if suffix[1] == "bias":
            return np.zeros_like(target), "fallback"
        return (
            _source_array(
                source_parameters,
                (*source_prefix, "1", "DenseReluDense", "wo", "kernel"),
            ),
            "checkpoint",
        )
    if suffix[0] == "feedforward_norm":
        if suffix[1] == "bias":
            return np.zeros_like(target), "fallback"
        return (
            _source_array(source_parameters, (*source_prefix, "1", "layer_norm", "weight")),
            "checkpoint",
        )
    raise ValueError(f"unsupported encoder parameter {'/'.join(suffix)}")


def _decoder_parameter(
    source_parameters: Mapping[tuple[str, ...], Any],
    layer_index: int,
    suffix: tuple[str, ...],
    target: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Map one communication decoder leaf from the matching T5 decoder block."""
    source_prefix = ("decoder", "block", str(layer_index), "layer")
    if suffix == ("self_relative_attention_bias",):
        source = _source_array(
            source_parameters,
            (
                "decoder",
                "block",
                "0",
                "layer",
                "0",
                "SelfAttention",
                "relative_attention_bias",
                "embedding",
            ),
        )
        return source.T, "checkpoint"
    if suffix == ("cross_relative_attention_bias",):
        return np.zeros_like(target), "fallback"
    if suffix[0] in {"self_attention", "cross_attention"}:
        target_attention_name, attention_part, parameter_name = suffix
        source_layer = "0" if target_attention_name == "self_attention" else "1"
        source_attention_name = (
            "SelfAttention" if target_attention_name == "self_attention" else "EncDecAttention"
        )
        return _attention_parameter(
            source_parameters,
            (*source_prefix, source_layer, source_attention_name),
            attention_part,
            parameter_name,
            target,
        )
    norm_layers = {
        "self_attention_norm": "0",
        "cross_attention_norm": "1",
        "feedforward_norm": "2",
    }
    if suffix[0] in norm_layers:
        if suffix[1] == "bias":
            return np.zeros_like(target), "fallback"
        return (
            _source_array(
                source_parameters,
                (*source_prefix, norm_layers[suffix[0]], "layer_norm", "weight"),
            ),
            "checkpoint",
        )
    if suffix[0] in {"feedforward_in", "feedforward_out"}:
        if suffix[1] == "bias":
            return np.zeros_like(target), "fallback"
        source_projection = "wi" if suffix[0] == "feedforward_in" else "wo"
        return (
            _source_array(
                source_parameters,
                (*source_prefix, "2", "DenseReluDense", source_projection, "kernel"),
            ),
            "checkpoint",
        )
    raise ValueError(f"unsupported decoder parameter {'/'.join(suffix)}")


def _question_message_parameter(
    source_parameters: Mapping[tuple[str, ...], Any],
    suffix: tuple[str, ...],
    target: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Seed question-to-message attention from T5 decoder cross-attention block zero."""
    source_prefix = ("decoder", "block", "0", "layer")
    if suffix[0] == "question_message_attention":
        attention_part, parameter_name = suffix[1:]
        return _attention_parameter(
            source_parameters,
            (*source_prefix, "1", "EncDecAttention"),
            attention_part,
            parameter_name,
            target,
        )
    if suffix[0] == "attention_norm":
        if suffix[1] == "bias":
            return np.zeros_like(target), "fallback"
        return (
            _source_array(source_parameters, (*source_prefix, "1", "layer_norm", "weight")),
            "checkpoint",
        )
    if suffix[0] in {"feedforward_in", "feedforward_out"}:
        if suffix[1] == "bias":
            return np.zeros_like(target), "fallback"
        source_projection = "wi" if suffix[0] == "feedforward_in" else "wo"
        return (
            _source_array(
                source_parameters,
                (*source_prefix, "2", "DenseReluDense", source_projection, "kernel"),
            ),
            "checkpoint",
        )
    if suffix[0] == "feedforward_norm":
        if suffix[1] == "bias":
            return np.zeros_like(target), "fallback"
        return (
            _source_array(source_parameters, (*source_prefix, "2", "layer_norm", "weight")),
            "checkpoint",
        )
    raise ValueError(f"unsupported question-message parameter {'/'.join(suffix)}")


def _task_token_source_ids(
    task_tokenizer: TextTokenizer,
    source_tokenizer: Tokenizer,
) -> list[list[int]]:
    """Map each task token to one or more T5 SentencePiece token IDs."""
    special_source_ids = {
        PAD_TOKEN: 0,
        BOS_TOKEN: 0,
        EOS_TOKEN: 1,
        UNK_TOKEN: 2,
    }
    source_ids: list[list[int]] = []
    for task_token in task_tokenizer.id_to_token:
        if task_token in special_source_ids:
            source_ids.append([special_source_ids[task_token]])
            continue
        encoded_ids = source_tokenizer.encode(task_token, add_special_tokens=False).ids
        if not encoded_ids:
            raise ValueError(f"T5 tokenizer produced no pieces for task token {task_token!r}")
        source_ids.append(encoded_ids)
    return source_ids


def _task_embeddings(
    source_embedding: np.ndarray,
    token_source_ids: Sequence[Sequence[int]],
) -> np.ndarray:
    """Average T5 subtoken rows into one embedding for every task token."""
    embeddings = []
    for source_ids in token_source_ids:
        if min(source_ids) < 0 or max(source_ids) >= source_embedding.shape[0]:
            raise ValueError("T5 tokenizer emitted an ID outside the checkpoint embedding table")
        embeddings.append(np.mean(source_embedding[np.asarray(source_ids)], axis=0))
    return np.stack(embeddings).astype(np.float32)


def _initialized_parameter(
    path: tuple[str, ...],
    target: np.ndarray,
    source_parameters: Mapping[tuple[str, ...], Any],
    task_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return the explicit pretrained or fallback value for one target leaf."""
    if path == ("token_embedding", "embedding"):
        return task_embeddings, "derived"
    if path == ("sender", "query_embedding"):
        return query_embeddings, "derived"
    if path[:2] == ("sender", "message_projection"):
        if path[2] == "bias":
            return np.zeros_like(target), "fallback"
        encoder_norm = _source_array(
            source_parameters,
            ("encoder", "final_layer_norm", "weight"),
        )
        return np.diag(encoder_norm), "derived"
    if path[0] == "sender" and path[1].startswith("layer_"):
        layer_index = int(path[1].removeprefix("layer_"))
        return _encoder_parameter(source_parameters, layer_index, path[2:], target)
    if path[:2] == ("receiver", "query_key_value_encoder"):
        layer_index = int(path[2].removeprefix("layer_"))
        return _encoder_parameter(source_parameters, layer_index, path[3:], target)
    if path[:2] == ("receiver", "query_message_block"):
        return _question_message_parameter(source_parameters, path[2:], target)
    if path[0] == "receiver" and path[1].startswith("decoder_layer_"):
        layer_index = int(path[1].removeprefix("decoder_layer_"))
        return _decoder_parameter(source_parameters, layer_index, path[2:], target)
    if path[:2] == ("receiver", "output_hidden_projection"):
        if path[2] == "bias":
            return np.zeros_like(target), "fallback"
        decoder_norm = _source_array(
            source_parameters,
            ("decoder", "final_layer_norm", "weight"),
        )
        return np.diag(decoder_norm), "derived"
    if path[:2] == ("receiver", "output_projection"):
        if path[2] == "bias":
            return np.zeros_like(target), "fallback"
        return task_embeddings.T / np.sqrt(T5_MODEL_DIMENSION), "derived"
    raise ValueError(f"T5 initialization has no rule for target parameter {'/'.join(path)}")


def initialize_from_t5_efficient_tiny(
    initial_parameters: Mapping[str, Any],
    task_tokenizer: TextTokenizer,
    model_directory: Path,
    checkpoint_sha256: str,
    tokenizer_sha256: str,
) -> tuple[dict[str, Any], PretrainedInitializationSummary]:
    """Initialize every model leaf from T5-Efficient-Tiny or an explicit adapter fallback."""
    checkpoint_path = model_directory / T5_EFFICIENT_TINY_CHECKPOINT_FILENAME
    tokenizer_path = model_directory / T5_EFFICIENT_TINY_TOKENIZER_FILENAME
    verify_t5_efficient_tiny_artifacts(
        model_directory,
        checkpoint_sha256,
        tokenizer_sha256,
    )

    restored_checkpoint = serialization.msgpack_restore(checkpoint_path.read_bytes())
    assert isinstance(restored_checkpoint, dict)
    source_parameters = flatten_dict(restored_checkpoint)
    source_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    source_embedding = _source_array(source_parameters, ("shared", "embedding"))
    source_embedding_rms = float(np.sqrt(np.mean(np.square(source_embedding))))
    if not np.isfinite(source_embedding_rms) or source_embedding_rms <= 0:
        raise ValueError("T5 checkpoint embedding RMS must be finite and positive")
    normalized_source_embedding = source_embedding / source_embedding_rms
    token_source_ids = _task_token_source_ids(task_tokenizer, source_tokenizer)
    task_embeddings = _task_embeddings(normalized_source_embedding, token_source_ids)
    query_source_ids = []
    for query_index in range(4):
        source_id = source_tokenizer.token_to_id(f"<extra_id_{query_index}>")
        if source_id is None:
            raise ValueError(f"T5 tokenizer is missing <extra_id_{query_index}>")
        query_source_ids.append(source_id)
    query_embeddings = normalized_source_embedding[np.asarray(query_source_ids)]

    target_parameters = flatten_dict(dict(initial_parameters))
    initialized_parameters: dict[tuple[str, ...], np.ndarray] = {}
    category_counts = {"checkpoint": 0, "derived": 0, "fallback": 0}
    category_parameter_counts = {"checkpoint": 0, "derived": 0, "fallback": 0}
    for path, initial_value in target_parameters.items():
        target = np.asarray(initial_value)
        initialized_value, category = _initialized_parameter(
            path,
            target,
            source_parameters,
            task_embeddings,
            query_embeddings,
        )
        if initialized_value.shape != target.shape:
            raise ValueError(
                f"T5 initialization shape mismatch for {'/'.join(path)}: "
                f"source {initialized_value.shape}, target {target.shape}"
            )
        initialized_parameters[path] = initialized_value.astype(target.dtype)
        category_counts[category] += 1
        category_parameter_counts[category] += target.size

    summary = PretrainedInitializationSummary(
        target_leaves=len(initialized_parameters),
        checkpoint_leaves=category_counts["checkpoint"],
        checkpoint_derived_leaves=category_counts["derived"],
        explicit_fallback_leaves=category_counts["fallback"],
        target_parameters=sum(category_parameter_counts.values()),
        checkpoint_parameters=category_parameter_counts["checkpoint"],
        checkpoint_derived_parameters=category_parameter_counts["derived"],
        explicit_fallback_parameters=category_parameter_counts["fallback"],
    )
    return unflatten_dict(initialized_parameters), summary
