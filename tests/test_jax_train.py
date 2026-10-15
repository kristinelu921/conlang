"""Tests for the JAX communication-model training path."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    _append_model_outputs,
    _as_wandb_log,
    _host_training_record,
    _initialize_state,
    _model_config,
    _optimizer_for_params,
    _read_step_metrics,
    _top_vocabulary_histogram,
    build_parallel_train_step,
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

    def fake_train_step(
        state: object,
        batch: object,
        dropout_keys: object,
    ) -> tuple[object, dict[str, jax.Array]]:
        del batch, dropout_keys
        clock["time"] += 1.0
        return state, {"loss": jnp.array([1.0]), "gradient_norm": jnp.array([1.0])}

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
        lambda model, collect_path_gradients=True, collect_diagnostics=True: fake_train_step,
    )
    monkeypatch.setattr("train.build_parallel_evaluation_step", lambda model: object())
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
    monkeypatch.setattr("train.build_parallel_evaluation_step", lambda model: object())
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

    full = gradients(False, False, False)
    agent_1 = gradients(False, True, True)
    agent_2 = gradients(True, False, True)
    receiver = gradients(True, True, False)

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
    assert generated_token_ids is not None
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
