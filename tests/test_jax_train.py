"""Tests for the JAX communication-model training path."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from flax.traverse_util import flatten_dict

from jax_models import (
    JaxCommunicationModel,
    _relative_position_bucket,
    apply_message_intervention,
)
from linked_facts.generate_dataset import iter_question_examples
from tokenizer import TextTokenizer
from train import (
    JaxExampleBatch,
    TrainConfig,
    _append_model_outputs,
    _as_wandb_log,
    _gate_activation_statistics,
    _host_training_record,
    _initialize_state,
    _model_config,
    _optimizer_for_params,
    _read_step_metrics,
    _selected_gradient_norm,
    _top_vocabulary_histogram,
    _validated_wrong_world_shift,
    build_parallel_evaluation_step,
    build_parallel_train_step,
    evaluate_exact_match,
    iter_epoch_batches,
    parse_arguments,
    save_checkpoint,
    shard_batch,
    train,
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


def test_epoch_batches_reject_a_dataset_smaller_than_the_batch_by_default() -> None:
    """Undersized training sets should require explicit replication."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = _small_config()
    tokenized = tokenize_examples(_examples()[:1], tokenizer, config)

    batch_iterator = iter_epoch_batches(tokenized, 4, random.Random(config.seed))

    with pytest.raises(ValueError, match="smaller than one full batch"):
        next(batch_iterator)


def test_epoch_batches_can_repeat_one_example_to_fill_a_device_batch() -> None:
    """Explicit replication should fill a fixed-shape batch from one unique example."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = _small_config()
    tokenized = tokenize_examples(_examples()[:1], tokenizer, config)

    batch_iterator = iter_epoch_batches(
        tokenized,
        4,
        random.Random(config.seed),
        repeat_examples_to_batch_size=True,
    )
    first_batch = next(batch_iterator)
    second_batch = next(batch_iterator)

    assert first_batch.answer_token_ids.shape == (4, config.max_answer_length)
    for batch_field in first_batch:
        assert np.array_equal(batch_field, np.repeat(batch_field[:1], 4, axis=0))
    for first_field, second_field in zip(first_batch, second_batch, strict=True):
        assert np.array_equal(first_field, second_field)


def test_epoch_batches_reject_an_empty_dataset_when_replication_is_enabled() -> None:
    """Replication should fail clearly when there is no source example."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = _small_config()
    tokenized = tokenize_examples([], tokenizer, config)

    batch_iterator = iter_epoch_batches(
        tokenized,
        4,
        random.Random(config.seed),
        repeat_examples_to_batch_size=True,
    )

    with pytest.raises(ValueError, match="training dataset is empty"):
        next(batch_iterator)


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


def test_config_rejects_nonpositive_output_hidden_size() -> None:
    """An enabled output hidden layer should have a positive width."""
    with pytest.raises(ValueError, match="output hidden size must be positive"):
        replace(TrainConfig(), output_hidden_size=0).validate()


def test_config_rejects_nonpositive_role_gate_channels() -> None:
    """Enabled role gates should have a positive bottleneck width."""
    with pytest.raises(ValueError, match="role gate channels must be positive"):
        replace(TrainConfig(), role_gate_channels=0).validate()


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


def test_throughput_timing_excludes_evaluation_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evaluation and W&B delays should not reduce the next training interval's throughput."""
    config = replace(_small_config(), steps=3, eval_every=2, wandb_enabled=True)
    clock = {"time": 0.0}
    metric_records: list[dict[str, float]] = []

    class FakeWandbRun:
        """Model the wall-time cost of one W&B enqueue."""

        def log(self, payload: dict[str, Any], step: int) -> None:
            del payload, step
            clock["time"] += 100.0

        def finish(self, exit_code: int) -> None:
            assert exit_code == 0

    def fake_train_step(collect_diagnostics: bool) -> Any:
        def run_step(
            state: object,
            batch: object,
            dropout_keys: object,
        ) -> tuple[object, dict[str, jax.Array]]:
            del batch, dropout_keys
            clock["time"] += 1.0
            metrics = {"loss": jnp.array([1.0]), "gradient_norm": jnp.array([1.0])}
            if collect_diagnostics:
                metrics["generated_token_ids"] = jnp.ones((1, 2, 4), dtype=jnp.int32)
            return state, metrics

        return run_step

    def fake_evaluation(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        clock["time"] += 100.0
        return {"exact_match": 0.5, "mean_joint_log_probability": -1.0}

    monkeypatch.setattr("train.resolve_training_dataset", lambda *args, **kwargs: Path("dataset"))
    monkeypatch.setattr("train.verify_dataset", lambda dataset_path: {"questions": 2})
    monkeypatch.setattr("train.resolve_split_path", lambda dataset_path, split: Path(split))
    monkeypatch.setattr("train.load_examples", lambda *args, **kwargs: _examples())
    monkeypatch.setattr(
        "train.build_parallel_train_step",
        lambda model, collect_path_gradients=True, collect_diagnostics=True: fake_train_step(
            collect_diagnostics
        ),
    )
    monkeypatch.setattr(
        "train.build_parallel_evaluation_step",
        lambda model, collect_message_interventions=False, wrong_world_shift=1: object(),
    )
    monkeypatch.setattr("train.evaluate_exact_match", fake_evaluation)
    monkeypatch.setattr("train._initialize_wandb", lambda current_config: FakeWandbRun())
    monkeypatch.setattr("train.time.perf_counter", lambda: clock["time"])
    monkeypatch.setattr(
        "train._append_metrics",
        lambda metrics_path, metrics: metric_records.append(dict(metrics)),
    )

    train(config)

    step_three_metrics = next(record for record in metric_records if record.get("step") == 3.0)
    assert step_three_metrics["examples_per_second"] == pytest.approx(2.0)


def test_outputs_follow_logging_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model-output artifacts should sample warmup, logging, and terminal steps."""
    config = replace(
        _small_config(),
        steps=5,
        eval_every=5,
        log_every=2,
        throughput_warmup_steps=1,
        output_path=tmp_path / "cadence" / "checkpoint.msgpack",
    )
    recorded_steps: list[int] = []

    def fake_step(collect_diagnostics: bool) -> Any:
        def run_step(
            state: object,
            batch: object,
            dropout_keys: object,
        ) -> tuple[object, dict[str, jax.Array]]:
            del batch, dropout_keys
            metrics = {"loss": jnp.array([1.0]), "gradient_norm": jnp.array([1.0])}
            if collect_diagnostics:
                metrics["generated_token_ids"] = jnp.ones((1, 2, 4), dtype=jnp.int32)
            return state, metrics

        return run_step

    monkeypatch.setattr("train.resolve_training_dataset", lambda *args, **kwargs: Path("dataset"))
    monkeypatch.setattr("train.verify_dataset", lambda dataset_path: {"questions": 2})
    monkeypatch.setattr("train.resolve_split_path", lambda dataset_path, split: Path(split))
    monkeypatch.setattr("train.load_examples", lambda *args, **kwargs: _examples())
    monkeypatch.setattr(
        "train.build_parallel_train_step",
        lambda model, collect_path_gradients=True, collect_diagnostics=True: fake_step(
            collect_diagnostics
        ),
    )
    monkeypatch.setattr(
        "train.build_parallel_evaluation_step",
        lambda model, collect_message_interventions=False, wrong_world_shift=1: object(),
    )
    monkeypatch.setattr(
        "train.evaluate_exact_match",
        lambda *args, **kwargs: {"exact_match": 0.5, "mean_joint_log_probability": -1.0},
    )
    monkeypatch.setattr(
        "train._append_model_outputs",
        lambda outputs_path, step, generated, targets, tokenizer: recorded_steps.append(step),
    )
    monkeypatch.setattr("train.save_checkpoint", lambda *args, **kwargs: None)

    train(config)

    assert recorded_steps == [1, 2, 4, 5]


def test_lightweight_train_step_skips_diagnostic_outputs() -> None:
    """Ordinary TPU steps should avoid generation and histogram diagnostics."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = _small_config()
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    initial_state = _initialize_state(model, optax.identity(), batch, config.seed)
    optimizer = _optimizer_for_params(config, initial_state.params)
    state = initial_state.replace(tx=optimizer, opt_state=optimizer.init(initial_state.params))
    replicated_state = jax.device_put_replicated(state, jax.local_devices())

    _updated_state, metrics = build_parallel_train_step(
        model,
        collect_path_gradients=False,
        collect_diagnostics=False,
    )(
        replicated_state,
        shard_batch(batch, jax.local_device_count()),
        jax.random.split(jax.random.PRNGKey(1), jax.local_device_count()),
    )

    assert set(metrics) == {"loss", "gradient_norm"}


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


def test_each_sender_emits_four_message_vectors() -> None:
    """Each agent should send four vectors, and the receiver should see all eight."""
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

    messages = model.apply(
        variables,
        jnp.asarray(batch.agent_1_token_ids),
        jnp.asarray(batch.agent_2_token_ids),
        deterministic=True,
        method=model.encode_messages,
    )

    assert config.message_length == 4
    assert messages.shape == (2, 8, config.hidden_size)


def test_question_conditioned_senders_require_and_use_the_question() -> None:
    """Conditioned sender messages should change when only the question changes."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = replace(_small_config(), sender_question_conditioning=True)
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
    repeated_agent_1 = jnp.repeat(jnp.asarray(batch.agent_1_token_ids[:1]), 2, axis=0)
    repeated_agent_2 = jnp.repeat(jnp.asarray(batch.agent_2_token_ids[:1]), 2, axis=0)

    messages = model.apply(
        variables,
        repeated_agent_1,
        repeated_agent_2,
        deterministic=True,
        question_token_ids=jnp.asarray(batch.question_token_ids),
        method=model.encode_messages,
    )

    assert messages.shape == (2, config.message_length * 2, config.hidden_size)
    assert not jnp.allclose(messages[0], messages[1])
    with pytest.raises(ValueError, match="question token IDs are required"):
        model.apply(
            variables,
            repeated_agent_1,
            repeated_agent_2,
            deterministic=True,
            method=model.encode_messages,
        )


def test_message_interventions_apply_the_requested_causal_controls() -> None:
    """Message controls should preserve shape while selecting the intended information."""
    messages = jnp.arange(24, dtype=jnp.float32).reshape(2, 6, 2)
    message_1 = messages[:, :3]
    message_2 = messages[:, 3:]

    assert jnp.array_equal(apply_message_intervention(messages, 3, "intact"), messages)
    assert jnp.array_equal(
        apply_message_intervention(messages, 3, "wrong_world", wrong_world_shift=1),
        jnp.roll(messages, shift=1, axis=0),
    )
    grouped_world_messages = jnp.repeat(jnp.arange(3), 2).reshape(6, 1, 1)
    wrong_world_messages = apply_message_intervention(
        grouped_world_messages,
        1,
        "wrong_world",
        wrong_world_shift=2,
    )
    assert jnp.all(wrong_world_messages != grouped_world_messages)
    assert jnp.array_equal(
        apply_message_intervention(messages, 3, "swap_senders"),
        jnp.concatenate((message_2, message_1), axis=1),
    )
    assert jnp.array_equal(
        apply_message_intervention(messages, 3, "agent_1_only"),
        jnp.concatenate((message_1, jnp.zeros_like(message_2)), axis=1),
    )
    assert jnp.array_equal(
        apply_message_intervention(messages, 3, "agent_2_only"),
        jnp.concatenate((jnp.zeros_like(message_1), message_2), axis=1),
    )
    assert jnp.count_nonzero(apply_message_intervention(messages, 3, "zero")) == 0
    with pytest.raises(ValueError, match="unknown message intervention"):
        apply_message_intervention(messages, 3, "unsupported")
    with pytest.raises(ValueError, match="another row"):
        apply_message_intervention(messages[:1], 3, "wrong_world")


def test_wrong_world_shift_requires_two_aligned_worlds_per_device() -> None:
    """Wrong-world evaluation should fail before any same-world pairing is possible."""
    assert _validated_wrong_world_shift(256, 4, 8) == 8
    with pytest.raises(ValueError, match="world-aligned"):
        _validated_wrong_world_shift(24, 4, 8)
    with pytest.raises(ValueError, match="at least two worlds"):
        _validated_wrong_world_shift(32, 4, 8)


def test_evaluation_reports_every_message_intervention() -> None:
    """Causal evaluation should emit intact and intervened metrics from one batch."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = _small_config()
    tokenized = tokenize_examples(_examples(), tokenizer, config)
    batch = tokenized.select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    state = _initialize_state(model, optax.identity(), batch, config.seed)
    replicated_params = jax.device_put_replicated(state.params, jax.local_devices())

    metrics = evaluate_exact_match(
        replicated_params,
        tokenized,
        config.batch_size,
        jax.local_device_count(),
        build_parallel_evaluation_step(
            model,
            collect_message_interventions=True,
            wrong_world_shift=1,
        ),
    )

    expected_prefixes = {
        "zero_message",
        "wrong_world_message",
        "swapped_sender",
        "agent_1_only",
        "agent_2_only",
    }
    assert metrics.keys() >= {"exact_match", "mean_joint_log_probability"}
    for prefix in expected_prefixes:
        assert 0.0 <= metrics[f"{prefix}_exact_match"] <= 1.0
        assert np.isfinite(metrics[f"{prefix}_mean_joint_log_probability"])


def test_output_hidden_size_is_independent_of_transformer_width() -> None:
    """The optional output MLP should widen only the answer head."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = replace(_small_config(), output_hidden_size=24)
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

    output = model.apply(
        variables,
        jnp.asarray(batch.agent_1_token_ids),
        jnp.asarray(batch.agent_2_token_ids),
        jnp.asarray(batch.question_token_ids),
        jnp.asarray(batch.answer_token_ids),
        deterministic=True,
    )
    receiver_parameters = variables["params"]["receiver"]

    assert output.message_1.shape[-1] == config.hidden_size
    assert output.receiver_activation.shape[-1] == config.output_hidden_size
    assert receiver_parameters["output_hidden_projection"]["kernel"].shape == (16, 24)
    assert receiver_parameters["output_projection"]["kernel"].shape == (
        24,
        tokenizer.vocab_size,
    )


def test_sender_and_receiver_can_share_one_encoder_stack() -> None:
    """The tying option should store one encoder used by both model roles."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = replace(_small_config(), tie_sender_receiver_encoder=True)
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

    output = model.apply(
        variables,
        jnp.asarray(batch.agent_1_token_ids),
        jnp.asarray(batch.agent_2_token_ids),
        jnp.asarray(batch.question_token_ids),
        jnp.asarray(batch.answer_token_ids),
        deterministic=True,
    )
    parameters = variables["params"]

    assert output.logits.shape == (2, config.max_answer_length, tokenizer.vocab_size)
    assert "shared_sender_receiver_encoder" in parameters
    assert "layer_0" in parameters["shared_sender_receiver_encoder"]
    assert "layer_0" not in parameters["sender"]
    assert "encoder_layer_0" not in parameters["receiver"]
    assert "decoder_layer_0" in parameters["receiver"]


def test_role_gates_are_separate_identity_initialized_adapters() -> None:
    """Sender and receiver should get distinct, initially closed residual gates."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = replace(
        _small_config(),
        tie_sender_receiver_encoder=True,
        role_gate_channels=4,
    )
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

    output = model.apply(
        variables,
        jnp.asarray(batch.agent_1_token_ids),
        jnp.asarray(batch.agent_2_token_ids),
        jnp.asarray(batch.question_token_ids),
        jnp.asarray(batch.answer_token_ids),
        deterministic=True,
    )
    sender_gate_parameters = variables["params"]["sender"]["role_gate"]
    receiver_gate_parameters = variables["params"]["receiver"]["role_gate"]

    assert output.sender_gate_1 is not None
    assert output.sender_gate_2 is not None
    assert output.receiver_gate is not None
    assert output.sender_gate_1.shape == (2, config.message_length, config.role_gate_channels)
    assert output.sender_gate_2.shape == output.sender_gate_1.shape
    assert output.receiver_gate.shape == (
        2,
        config.message_length * 2 + config.max_question_length,
        config.role_gate_channels,
    )
    assert sender_gate_parameters["candidate_projection"]["kernel"].shape == (16, 4)
    assert receiver_gate_parameters["candidate_projection"]["kernel"].shape == (16, 4)
    assert jnp.all(sender_gate_parameters["output_projection"]["kernel"] == 0)
    assert jnp.all(receiver_gate_parameters["output_projection"]["kernel"] == 0)
    expected_initial_gate = jax.nn.sigmoid(jnp.array(-2.0))
    assert jnp.allclose(output.sender_gate_1, expected_initial_gate)
    assert jnp.allclose(output.receiver_gate, expected_initial_gate)


def test_role_gate_diagnostics_are_finite_fractions() -> None:
    """Logged role-gate summaries should remain finite and bounded."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = replace(
        _small_config(),
        tie_sender_receiver_encoder=True,
        role_gate_channels=4,
    )
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    initial_state = _initialize_state(model, optax.identity(), batch, config.seed)
    replicated_state = jax.device_put_replicated(initial_state, jax.local_devices())

    _updated_state, device_metrics = build_parallel_train_step(model)(
        replicated_state,
        shard_batch(batch, jax.local_device_count()),
        jax.random.split(jax.random.PRNGKey(1), jax.local_device_count()),
    )
    host_metrics, _generated_token_ids = _read_step_metrics(device_metrics)

    for key in (
        "sender_gate_mean",
        "sender_gate_near_zero_fraction",
        "sender_gate_near_one_fraction",
        "receiver_gate_mean",
        "receiver_gate_near_zero_fraction",
        "receiver_gate_near_one_fraction",
    ):
        assert np.isfinite(host_metrics[key])
        assert 0.0 <= host_metrics[key] <= 1.0


def test_gate_statistics_ignore_inactive_positions() -> None:
    """Receiver gate summaries should exclude padded memory positions."""
    device_count = jax.local_device_count()
    valid = jnp.broadcast_to(
        jnp.array([[True, True, False, False]]),
        (device_count, 1, 4),
    )
    valid_values = jnp.array([[[[0.25, 0.25], [0.75, 0.75]]]])
    valid_values = jnp.broadcast_to(valid_values, (device_count, 1, 2, 2))
    low_padding = jnp.zeros((device_count, 1, 2, 2))
    high_padding = jnp.ones((device_count, 1, 2, 2))
    low_padded_gate = jnp.concatenate((valid_values, low_padding), axis=2)
    high_padded_gate = jnp.concatenate((valid_values, high_padding), axis=2)
    parallel_statistics = jax.pmap(_gate_activation_statistics, axis_name="devices")

    low_padded_statistics = parallel_statistics(low_padded_gate, valid)[0]
    high_padded_statistics = parallel_statistics(high_padded_gate, valid)[0]

    assert jnp.allclose(low_padded_statistics, jnp.array([0.5, 0.0, 0.0]))
    assert jnp.allclose(high_padded_statistics, low_padded_statistics)


def test_default_keeps_sender_and_receiver_encoders_separate() -> None:
    """The default model should preserve the existing independent parameter tree."""
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
    parameters = variables["params"]

    assert "shared_sender_receiver_encoder" not in parameters
    assert "layer_0" in parameters["sender"]
    assert "encoder_layer_0" in parameters["receiver"]


def test_tied_encoder_is_included_in_weight_and_path_diagnostics() -> None:
    """Diagnostics should include the shared weights in both model roles."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = replace(_small_config(), tie_sender_receiver_encoder=True)
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    initial_state = _initialize_state(model, optax.identity(), batch, config.seed)
    replicated_state = jax.device_put_replicated(initial_state, jax.local_devices())

    _updated_state, device_metrics = build_parallel_train_step(model)(
        replicated_state,
        shard_batch(batch, jax.local_device_count()),
        jax.random.split(jax.random.PRNGKey(1), jax.local_device_count()),
    )
    host_metrics, _generated_token_ids = _read_step_metrics(device_metrics)

    for key in (
        "weight_norm_shared_encoder",
        "gradient_norm_agent_1",
        "gradient_norm_agent_2",
        "gradient_norm_receiver",
    ):
        assert np.isfinite(host_metrics[key])
        assert host_metrics[key] > 0


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


def _component_gradient_slices(
    model: JaxCommunicationModel,
    params: Any,
    batch: JaxExampleBatch,
) -> tuple[Any, Any, Any, Any]:
    """Return full, agent-specific, and receiver-only gradient trees."""
    def gradients(stop_message_1: bool, stop_message_2: bool, stop_receiver_tokens: bool) -> Any:
        def loss_function(current_params: Any) -> jax.Array:
            return model.apply(
                {"params": current_params},
                jnp.asarray(batch.agent_1_token_ids),
                jnp.asarray(batch.agent_2_token_ids),
                jnp.asarray(batch.question_token_ids),
                jnp.asarray(batch.answer_token_ids),
                deterministic=True,
                stop_message_1=stop_message_1,
                stop_message_2=stop_message_2,
                stop_receiver_tokens=stop_receiver_tokens,
            ).loss

        return jax.grad(loss_function)(params)

    return (
        gradients(False, False, False),
        gradients(False, True, True),
        gradients(True, False, True),
        gradients(True, True, False),
    )


def test_component_gradients_partition_the_full_gradient() -> None:
    """Sender and receiver gradient slices should add back to the full gradient."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = _small_config()
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    params = model.init(
        {"params": jax.random.PRNGKey(0)},
        jnp.asarray(batch.agent_1_token_ids),
        jnp.asarray(batch.agent_2_token_ids),
        jnp.asarray(batch.question_token_ids),
        jnp.asarray(batch.answer_token_ids),
        deterministic=True,
    )["params"]
    full, agent_1, agent_2, receiver = _component_gradient_slices(model, params, batch)

    sender_sum = jax.tree_util.tree_map(lambda left, right: left + right, agent_1["sender"], agent_2["sender"])
    embedding_sum = jax.tree_util.tree_map(
        lambda first, second, third: first + second + third,
        agent_1["token_embedding"],
        agent_2["token_embedding"],
        receiver["token_embedding"],
    )
    assert all(
        jnp.allclose(left, right)
        for left, right in zip(
            jax.tree_util.tree_leaves(sender_sum),
            jax.tree_util.tree_leaves(full["sender"]),
            strict=True,
        )
    )
    assert all(
        jnp.allclose(left, right)
        for left, right in zip(
            jax.tree_util.tree_leaves(embedding_sum),
            jax.tree_util.tree_leaves(full["token_embedding"]),
            strict=True,
        )
    )
    assert all(
        jnp.allclose(left, right)
        for left, right in zip(
            jax.tree_util.tree_leaves(receiver["receiver"]),
            jax.tree_util.tree_leaves(full["receiver"]),
            strict=True,
        )
    )


def test_tied_component_gradients_isolate_each_shared_encoder_use() -> None:
    """Tied path diagnostics should partition the shared encoder gradient."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = replace(_small_config(), tie_sender_receiver_encoder=True)
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    params = model.init(
        {"params": jax.random.PRNGKey(0)},
        jnp.asarray(batch.agent_1_token_ids),
        jnp.asarray(batch.agent_2_token_ids),
        jnp.asarray(batch.question_token_ids),
        jnp.asarray(batch.answer_token_ids),
        deterministic=True,
    )["params"]

    full, agent_1, agent_2, receiver = _component_gradient_slices(model, params, batch)
    shared_encoder_part = "shared_sender_receiver_encoder"
    agent_1_shared = jax.tree_util.tree_map(
        lambda combined, receiver_only: combined - receiver_only,
        agent_1[shared_encoder_part],
        receiver[shared_encoder_part],
    )
    agent_2_shared = jax.tree_util.tree_map(
        lambda combined, receiver_only: combined - receiver_only,
        agent_2[shared_encoder_part],
        receiver[shared_encoder_part],
    )
    partitioned_shared = jax.tree_util.tree_map(
        lambda first, second, receiver_only: first + second + receiver_only,
        agent_1_shared,
        agent_2_shared,
        receiver[shared_encoder_part],
    )
    sender_parts = ("sender", "token_embedding", shared_encoder_part)
    isolated_agent_1 = {
        "sender": agent_1["sender"],
        "token_embedding": agent_1["token_embedding"],
        shared_encoder_part: agent_1_shared,
    }

    assert all(
        jnp.allclose(partitioned, complete)
        for partitioned, complete in zip(
            jax.tree_util.tree_leaves(partitioned_shared),
            jax.tree_util.tree_leaves(full[shared_encoder_part]),
            strict=True,
        )
    )
    assert jnp.allclose(
        _selected_gradient_norm(agent_1, sender_parts, receiver),
        optax.global_norm(isolated_agent_1),
    )


def test_step_saves_one_output_record_and_logs_requested_metrics(tmp_path: Path) -> None:
    """Each step should append its greedy answers, and the log should carry the requested metrics."""
    tokenizer = TextTokenizer(["buffalo", "chapel"])
    config = _small_config()
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    initial_state = _initialize_state(model, optax.identity(), batch, config.seed)
    optimizer = _optimizer_for_params(config, initial_state.params)
    state = initial_state.replace(tx=optimizer, opt_state=optimizer.init(initial_state.params))
    replicated_state = jax.device_put_replicated(state, jax.local_devices())
    _updated_state, device_metrics = build_parallel_train_step(model)(
        replicated_state,
        shard_batch(batch, jax.local_device_count()),
        jax.random.split(jax.random.PRNGKey(1), jax.local_device_count()),
    )
    host_metrics, generated_token_ids = _read_step_metrics(device_metrics)
    outputs_path = tmp_path / "outputs.jsonl"
    _append_model_outputs(outputs_path, 1, generated_token_ids, batch.answer_token_ids, tokenizer)
    record = _host_training_record(1, host_metrics, 2.0, generated_token_ids)
    wandb_payload = _as_wandb_log(record)
    saved = json.loads(outputs_path.read_text(encoding="utf-8"))

    assert generated_token_ids.shape == (2, 4)
    assert saved["step"] == 1
    assert len(saved["generated_token_ids"]) == 2
    assert len(saved["generated_token_ids"][0]) == 4
    assert 0.0 <= saved["second_position_eos_fraction"] <= 1.0
    for key in (
        "train_loss",
        "gradient_norm",
        "gradient_norm_agent_1",
        "gradient_norm_agent_2",
        "gradient_norm_receiver",
        "weight_norm",
        "weight_norm_embedding",
        "weight_norm_sender",
        "weight_norm_receiver",
        "activation_rms_message_1",
        "activation_rms_message_2",
        "activation_rms_receiver",
        "second_position_eos_fraction",
        "message_1_histogram",
        "message_2_histogram",
        "receiver_activation_histogram",
        "generated_vocabulary_histogram",
    ):
        assert key in record
    assert all(np.isfinite(record[key]) for key in (
        "train_loss",
        "gradient_norm",
        "gradient_norm_agent_1",
        "gradient_norm_agent_2",
        "gradient_norm_receiver",
        "activation_rms_message_1",
    ))
    assert "message_1_histogram" in wandb_payload
    assert "generated_vocabulary_histogram" in wandb_payload
    assert len(record["generated_vocabulary_histogram"]) <= 100
    assert len(record["generated_vocabulary_histogram_edges"]) == (
        len(record["generated_vocabulary_histogram"]) + 1
    )
    assert "generated_vocabulary_top_token_ids" not in wandb_payload
    assert "token_output_histograms" not in record
    assert not any(key.startswith("token_output_position_") for key in wandb_payload)


def test_read_step_metrics_requires_generated_token_ids() -> None:
    """Logged-step metrics should fail at the boundary without generated tokens."""
    device_metrics = {
        "loss": jnp.array([1.0]),
        "gradient_norm": jnp.array([1.0]),
    }

    with pytest.raises(AssertionError):
        _read_step_metrics(device_metrics)


def test_vocabulary_histogram_keeps_only_the_100_most_frequent_tokens() -> None:
    """The compact vocabulary chart should cap bins and deterministically break ties."""
    token_values = np.concatenate(
        (
            np.arange(101, dtype=np.int32),
            np.repeat(np.int32(100), 9),
        )
    ).reshape(1, -1)

    token_ids, token_counts, histogram_edges = _top_vocabulary_histogram(token_values)

    assert token_ids.shape == (100,)
    assert token_counts.shape == (100,)
    assert histogram_edges.shape == (101,)
    assert 99 not in token_ids
    assert token_ids[-1] == 100
    assert token_counts[-1] == 10
    assert np.all(np.diff(histogram_edges) > 0)


def test_wandb_vocabulary_histogram_rejects_more_than_100_bins() -> None:
    """The W&B boundary should fail closed before recreating a large histogram."""
    record = {
        "generated_vocabulary_histogram": [1] * 101,
        "generated_vocabulary_histogram_edges": list(range(102)),
    }

    with pytest.raises(ValueError, match="exceeds the 100-token limit"):
        _as_wandb_log(record)
