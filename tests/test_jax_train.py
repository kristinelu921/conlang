"""Tests for the JAX communication-model training path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from flax.traverse_util import flatten_dict

from jax_models import JaxCommunicationModel, _relative_position_bucket
from linked_facts.generate_dataset import iter_question_examples
from tokenizer import TextTokenizer
from train import (
    TrainConfig,
    _initialize_state,
    _model_config,
    _optimizer_for_params,
    build_parallel_train_step,
    parse_arguments,
    save_checkpoint,
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


def test_regional_training_split_fits_static_sequence_limits() -> None:
    """The committed dataset should fit the TPU graph without truncation or recompilation."""
    config = TrainConfig()
    maximum_lengths = {"agent_1_input": 0, "agent_2_input": 0, "question": 0}
    for example in iter_question_examples(Path("datasets/linked_facts_v1/train.jsonl.gz")):
        for field_name in maximum_lengths:
            maximum_lengths[field_name] = max(maximum_lengths[field_name], len(str(example[field_name])))

    assert maximum_lengths["agent_1_input"] <= config.max_text_length
    assert maximum_lengths["agent_2_input"] <= config.max_text_length
    assert maximum_lengths["question"] <= config.max_question_length


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


def test_eval_mode_disables_dropout() -> None:
    """A deterministic apply should be repeatable and should not require a dropout key."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = replace(_small_config(), dropout=0.5)
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    parameter_key, dropout_key = jax.random.split(jax.random.PRNGKey(0))
    variables = model.init(
        {"params": parameter_key, "dropout": dropout_key},
        jnp.asarray(batch.agent_1_token_ids),
        jnp.asarray(batch.agent_2_token_ids),
        jnp.asarray(batch.question_token_ids),
        jnp.asarray(batch.answer_token_ids),
        deterministic=False,
    )

    def evaluate() -> jax.Array:
        return model.apply(
            variables,
            jnp.asarray(batch.agent_1_token_ids),
            jnp.asarray(batch.agent_2_token_ids),
            jnp.asarray(batch.question_token_ids),
            jnp.asarray(batch.answer_token_ids),
            deterministic=True,
        ).logits

    first = jax.jit(evaluate)()
    second = jax.jit(evaluate)()
    assert jnp.array_equal(first, second)


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


def test_config_rejects_invalid_relative_attention_buckets() -> None:
    """Relative-position settings should fail before model initialization."""
    with pytest.raises(ValueError, match="even bucket count"):
        replace(TrainConfig(), relative_attention_num_buckets=31).validate()

    with pytest.raises(ValueError, match="max distance"):
        replace(
            TrainConfig(),
            relative_attention_num_buckets=32,
            relative_attention_max_distance=16,
        ).validate()


def test_tpu_config_requires_regional_cache_and_node_local_output() -> None:
    """TPU runs should fail before launch when locality paths are missing."""
    config = TrainConfig(device="tpu", throughput_warmup_steps=1, steps=2)

    with pytest.raises(ValueError, match="dataset_cache_dir"):
        config.validate()


def test_tpu_config_rejects_deceptive_nonlocal_paths() -> None:
    """A path containing an elf-data substring should not bypass locality checks."""
    config = TrainConfig(
        run_id="gate",
        device="tpu",
        dataset_cache_dir=Path("/tmp/home/user/elf-data/cache"),
        output_path=Path("/home/user/elf-runs/gate/checkpoint.msgpack"),
        throughput_warmup_steps=1,
        steps=2,
    )

    with pytest.raises(ValueError, match="dataset_cache_dir"):
        config.validate()

    traversal_config = replace(
        config,
        dataset_cache_dir=Path("/home/user/elf-data/../../../tmp/cache"),
    )
    with pytest.raises(ValueError, match="dataset_cache_dir"):
        traversal_config.validate()


def test_tpu_config_binds_output_directory_to_run_id() -> None:
    """Run identity and output ownership should not silently diverge."""
    config = TrainConfig(
        run_id="gate",
        device="tpu",
        dataset_cache_dir=Path("/home/user/elf-data/cache"),
        output_path=Path("/home/user/elf-runs/different/checkpoint.msgpack"),
        throughput_warmup_steps=1,
        steps=2,
    )

    with pytest.raises(ValueError, match="exactly match run_id"):
        config.validate()


def test_final_checkpoint_serializes_one_local_replica(tmp_path: Path) -> None:
    """The final checkpoint should serialize a pmapped state without array indexing."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = _small_config()
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    state = _initialize_state(model, optax.adam(1e-3), batch, config.seed)
    replicated_state = jax.device_put_replicated(state, jax.local_devices())
    output_path = tmp_path / "checkpoint.msgpack"

    save_checkpoint(replicated_state, 2, output_path)

    assert output_path.stat().st_size > 0
    assert not output_path.with_suffix(".msgpack.tmp").exists()


def test_relative_buckets_separate_nearby_offsets_and_clamp_far_ones() -> None:
    """Nearby offsets should keep their own buckets, and far offsets should share the last one."""
    nearby = _relative_position_bucket(jnp.array([-2, -1, 0, 1, 2]), True, 32, 128)
    far_positive = _relative_position_bucket(jnp.array([10_000]), True, 32, 128)
    far_negative = _relative_position_bucket(jnp.array([-10_000]), True, 32, 128)
    past = _relative_position_bucket(jnp.array([0, -1]), False, 32, 128)

    assert len(set(int(bucket) for bucket in nearby)) == 5
    assert int(far_positive[0]) == 31
    assert int(far_negative[0]) == 15
    assert int(past[0]) == 0
    assert int(past[1]) == 1


def test_model_uses_relative_bias_without_absolute_positions() -> None:
    """Order should come from relative attention bias, and token order should change the logits."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = _small_config()
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    variables = model.init(
        {"params": jax.random.PRNGKey(0)},
        jnp.asarray(batch.agent_1_token_ids),
        jnp.asarray(batch.agent_2_token_ids),
        jnp.asarray(batch.question_token_ids),
        jnp.asarray(batch.answer_token_ids),
        deterministic=True,
    )
    parameter_names = {"/".join(path) for path in flatten_dict(variables["params"])}
    swapped_agent_1 = batch.agent_1_token_ids.copy()
    swapped_agent_1[:, [0, 1]] = swapped_agent_1[:, [1, 0]]

    original = model.apply(
        variables,
        jnp.asarray(batch.agent_1_token_ids),
        jnp.asarray(batch.agent_2_token_ids),
        jnp.asarray(batch.question_token_ids),
        jnp.asarray(batch.answer_token_ids),
        deterministic=True,
    ).logits
    swapped = model.apply(
        variables,
        jnp.asarray(swapped_agent_1),
        jnp.asarray(batch.agent_2_token_ids),
        jnp.asarray(batch.question_token_ids),
        jnp.asarray(batch.answer_token_ids),
        deterministic=True,
    ).logits

    assert any(name.endswith("relative_attention_bias") for name in parameter_names)
    assert not any("position_embedding" in name for name in parameter_names)
    assert not jnp.allclose(original, swapped)
