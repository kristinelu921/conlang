"""Tests for the JAX communication-model training path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jax_models import JaxCommunicationModel
from tokenizer import TextTokenizer
from train import (
    TrainConfig,
    _initialize_state,
    _model_config,
    _optimizer_for_params,
    build_parallel_train_step,
    parse_arguments,
    shard_batch,
    tokenize_examples,
)


def _examples() -> list[dict[str, str]]:
    """Return two compact communication examples."""
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


def _small_config() -> TrainConfig:
    """Return a small CPU configuration for model tests."""
    return TrainConfig(
        batch_size=2,
        steps=2,
        eval_every=2,
        eval_examples=2,
        log_every=1,
        throughput_warmup_steps=1,
        wandb_enabled=False,
        hidden_size=16,
        max_text_length=40,
        max_question_length=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )


def test_tokenization_produces_static_training_shapes() -> None:
    """Every host payload should use the configured static sequence lengths."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = _small_config()

    tokenized = tokenize_examples(_examples(), tokenizer, config)

    assert tokenized.agent_1_token_ids.shape == (2, 40)
    assert tokenized.agent_2_token_ids.shape == (2, 40)
    assert tokenized.question_token_ids.shape == (2, 16)
    assert tokenized.answer_token_ids.shape == (2, 4)
    assert tokenized.agent_1_token_ids.dtype == np.int32


def test_jax_training_step_updates_parameters_and_stays_finite() -> None:
    """One pmapped JAX step should update model parameters with finite metrics."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = _small_config()
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    initial_state = _initialize_state(model, optax.identity(), batch, config.seed)
    optimizer = _optimizer_for_params(config, initial_state.params)
    state = initial_state.replace(tx=optimizer, opt_state=optimizer.init(initial_state.params))
    replicated_state = jax.device_put_replicated(state, jax.local_devices())
    before = jax.tree_util.tree_leaves(state.params)[0]

    updated_state, metrics = build_parallel_train_step(model)(
        replicated_state,
        shard_batch(batch, jax.local_device_count()),
        jax.random.split(jax.random.PRNGKey(1), jax.local_device_count()),
    )
    host_loss = float(jax.device_get(metrics["loss"][0]))
    after = jax.device_get(jax.tree_util.tree_leaves(updated_state.params)[0][0])

    assert np.isfinite(host_loss)
    assert not np.array_equal(jax.device_get(before), after)


def test_model_forward_uses_float32_loss_with_bfloat16_compute() -> None:
    """Mixed-precision TPU compute should retain a finite float32 objective."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = replace(_small_config(), precision="bfloat16")
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    parameter_key, dropout_key = jax.random.split(jax.random.PRNGKey(config.seed))
    variables = model.init(
        {"params": parameter_key, "dropout": dropout_key},
        jnp.asarray(batch.agent_1_token_ids),
        jnp.asarray(batch.agent_2_token_ids),
        jnp.asarray(batch.question_token_ids),
        jnp.asarray(batch.answer_token_ids),
        deterministic=False,
    )

    output = model.apply(
        variables,
        jnp.asarray(batch.agent_1_token_ids),
        jnp.asarray(batch.agent_2_token_ids),
        jnp.asarray(batch.question_token_ids),
        jnp.asarray(batch.answer_token_ids),
        deterministic=True,
    )

    assert output.logits.shape == (2, 4, tokenizer.vocab_size)
    assert output.loss.dtype == jnp.float32
    assert np.isfinite(float(output.loss))


def test_config_rejects_unknown_yaml_fields(tmp_path: Path) -> None:
    """A misspelled run setting should fail rather than be silently ignored."""
    config_path = tmp_path / "run.yml"
    config_path.write_text("batch_szie: 128\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown training config fields"):
        parse_arguments(["--config", str(config_path), "--disable-wandb"])


def test_tpu_config_requires_regional_cache_and_node_local_output() -> None:
    """TPU runs should fail before launch when locality paths are missing."""
    config = TrainConfig(device="tpu", throughput_warmup_steps=1, steps=2)

    with pytest.raises(ValueError, match="dataset_cache_dir"):
        config.validate()
