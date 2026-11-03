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
    _masked_region_average,
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
    _bridge_supervision_statistics,
    _gate_activation_statistics,
    _host_training_record,
    _initialize_state,
    _model_config,
    _optimizer_for_params,
    _read_step_metrics,
    _selected_gradient_norm,
    _top_vocabulary_histogram,
    _validated_world_block_size,
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
            "bridge": "buffalo",
            "person_side_agent": "agent_1",
        },
        {
            "agent_1_input": "The rocker stands in the chapel.",
            "agent_2_input": "Mesa owns a rocker.",
            "question": "Which room?",
            "answer": "chapel",
            "bridge": "rocker",
            "person_side_agent": "agent_2",
        },
    ]


def _example_tokenizer() -> TextTokenizer:
    """Return a tokenizer covering every token in the compact examples."""
    return TextTokenizer(["buffalo", "chapel", "rocker"])


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
    tokenizer = _example_tokenizer()
    config = _small_config()

    tokenized = tokenize_examples(_examples(), tokenizer, config)

    assert tokenized.agent_1_token_ids.shape == (2, 40)
    assert tokenized.agent_2_token_ids.shape == (2, 40)
    assert tokenized.question_token_ids.shape == (2, 16)
    assert tokenized.answer_token_ids.shape == (2, 4)
    assert tokenized.bridge_token_ids.shape == (2,)
    assert tokenized.person_side_agent_ids.shape == (2,)
    assert tokenized.agent_1_token_ids.dtype == np.int32
    assert tokenized.bridge_token_ids.dtype == np.int32


def test_epoch_batches_reject_a_dataset_smaller_than_the_batch_by_default() -> None:
    """Undersized training sets should require explicit replication."""
    tokenizer = _example_tokenizer()
    config = _small_config()
    tokenized = tokenize_examples(_examples()[:1], tokenizer, config)

    batch_iterator = iter_epoch_batches(tokenized, 4, random.Random(config.seed))

    with pytest.raises(ValueError, match="smaller than one full batch"):
        next(batch_iterator)


def test_epoch_batches_can_repeat_one_example_to_fill_a_device_batch() -> None:
    """Explicit replication should fill a fixed-shape batch from one unique example."""
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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


def test_sender_learning_rate_multiplier_scales_only_sender_updates() -> None:
    """The sender multiplier should scale post-Adam sender updates only."""
    params = {
        "sender": {"kernel": jnp.ones((2, 2))},
        "receiver": {"kernel": jnp.ones((2, 2))},
    }
    gradients = jax.tree_util.tree_map(jnp.ones_like, params)
    config = replace(
        _small_config(),
        learning_rate=1e-3,
        sender_learning_rate_multiplier=4.0,
        weight_decay=0.0,
        max_gradient_norm=100.0,
    )
    optimizer = _optimizer_for_params(config, params)

    updates, _optimizer_state = optimizer.update(
        gradients,
        optimizer.init(params),
        params,
    )

    sender_update_norm = float(optax.global_norm(updates["sender"]))
    receiver_update_norm = float(optax.global_norm(updates["receiver"]))
    assert sender_update_norm == pytest.approx(4.0 * receiver_update_norm)


def test_default_optimizer_matches_single_rate_adamw() -> None:
    """The default multiplier should preserve the original optimizer updates."""
    params = {
        "sender": {"kernel": jnp.ones((2, 2))},
        "receiver": {"kernel": jnp.ones((2, 2))},
    }
    gradients = jax.tree_util.tree_map(jnp.ones_like, params)
    config = replace(_small_config(), weight_decay=0.0, max_gradient_norm=100.0)
    optimizer = _optimizer_for_params(config, params)
    expected_optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adamw(config.learning_rate, weight_decay=0.0),
    )

    updates, _optimizer_state = optimizer.update(
        gradients,
        optimizer.init(params),
        params,
    )
    expected_updates, _expected_state = expected_optimizer.update(
        gradients,
        expected_optimizer.init(params),
        params,
    )

    assert all(
        jnp.array_equal(actual, expected)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(updates),
            jax.tree_util.tree_leaves(expected_updates),
            strict=True,
        )
    )


def test_bridge_supervision_updates_its_head_and_reports_metrics() -> None:
    """The optional bridge objective should train and expose its own metrics."""
    tokenizer = _example_tokenizer()
    config = replace(_small_config(), bridge_supervision_weight=1.0)
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    initial_state = _initialize_state(model, optax.identity(), batch, config.seed)
    optimizer = _optimizer_for_params(config, initial_state.params)
    state = initial_state.replace(tx=optimizer, opt_state=optimizer.init(initial_state.params))
    replicated_state = jax.device_put_replicated(state, jax.local_devices())
    before = state.params["bridge_output_projection"]["kernel"]

    updated_state, metrics = build_parallel_train_step(
        model,
        collect_path_gradients=False,
        collect_diagnostics=False,
    )(
        replicated_state,
        shard_batch(batch, jax.local_device_count()),
        jax.random.split(jax.random.PRNGKey(1), jax.local_device_count()),
    )
    after = updated_state.params["bridge_output_projection"]["kernel"][0]

    assert set(metrics) == {
        "loss",
        "gradient_norm",
        "answer_loss",
        "bridge_loss",
        "bridge_accuracy",
    }
    assert all(np.isfinite(float(jax.device_get(metrics[key][0]))) for key in metrics)
    assert not np.array_equal(jax.device_get(before), jax.device_get(after))


def test_bridge_supervision_selects_the_person_side_sender() -> None:
    """Bridge labels should score the sender holding the person-to-bridge fact."""
    tokenizer = _example_tokenizer()
    config = replace(_small_config(), bridge_supervision_weight=1.0)
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
    wrong_token_ids = (jnp.asarray(batch.bridge_token_ids) + 1) % tokenizer.vocab_size
    bridge_logits_1 = 10.0 * jax.nn.one_hot(
        jnp.asarray([batch.bridge_token_ids[0], wrong_token_ids[1]]),
        tokenizer.vocab_size,
    )
    bridge_logits_2 = 10.0 * jax.nn.one_hot(
        jnp.asarray([wrong_token_ids[0], batch.bridge_token_ids[1]]),
        tokenizer.vocab_size,
    )

    bridge_loss, bridge_correct = _bridge_supervision_statistics(
        output._replace(
            bridge_logits_1=bridge_logits_1,
            bridge_logits_2=bridge_logits_2,
        ),
        batch,
    )

    assert float(bridge_loss) < 0.01
    assert int(bridge_correct) == 2

    swapped_output = output._replace(
        bridge_logits_1=bridge_logits_2,
        bridge_logits_2=bridge_logits_1,
    )
    _unchanged_channel_loss, unchanged_channel_correct = _bridge_supervision_statistics(
        swapped_output,
        batch,
    )
    swapped_loss, swapped_correct = _bridge_supervision_statistics(
        swapped_output,
        batch,
        swap_sender_channels=True,
    )
    assert int(unchanged_channel_correct) == 0
    assert float(swapped_loss) < 0.01
    assert int(swapped_correct) == 2


def test_default_model_has_no_bridge_projection_parameters() -> None:
    """Disabling the auxiliary objective should preserve old checkpoint trees."""
    tokenizer = _example_tokenizer()
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
    output = model.apply(
        variables,
        jnp.asarray(batch.agent_1_token_ids),
        jnp.asarray(batch.agent_2_token_ids),
        jnp.asarray(batch.question_token_ids),
        jnp.asarray(batch.answer_token_ids),
        deterministic=True,
    )

    assert "bridge_output_projection" not in variables["params"]
    assert output.bridge_logits_1 is None
    assert output.bridge_logits_2 is None


def test_eval_mode_disables_dropout() -> None:
    """A deterministic apply should be repeatable and should not require a dropout key."""
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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


def test_config_rejects_nonpositive_sender_learning_rate_multiplier() -> None:
    """Sender update balancing should require a positive multiplier."""
    with pytest.raises(ValueError, match="learning rates and gradient clip"):
        replace(TrainConfig(), sender_learning_rate_multiplier=0.0).validate()


def test_config_rejects_nonpositive_role_gate_channels() -> None:
    """Enabled role gates should have a positive bottleneck width."""
    with pytest.raises(ValueError, match="role gate channels must be positive"):
        replace(TrainConfig(), role_gate_channels=0).validate()


def test_config_validates_sender_specific_dimensions() -> None:
    """Sender-only dimensions should be positive, divisible, and incompatible with tying."""
    with pytest.raises(ValueError, match="sender width, depth, heads"):
        replace(TrainConfig(), sender_num_layers=0).validate()
    with pytest.raises(ValueError, match="sender hidden size must be divisible"):
        replace(
            TrainConfig(),
            sender_hidden_size=18,
            sender_num_heads=4,
        ).validate()
    with pytest.raises(ValueError, match="incompatible with a tied encoder"):
        replace(
            TrainConfig(),
            sender_num_layers=3,
            tie_sender_receiver_encoder=True,
        ).validate()


def test_config_requires_one_question_path() -> None:
    """The question must reach either the senders or the receiver."""
    with pytest.raises(ValueError, match="question must be available"):
        replace(
            TrainConfig(),
            sender_question_conditioning=False,
            receiver_question_input=False,
        ).validate()


def test_query_only_receiver_rejects_question_leaks_and_bridge_labels() -> None:
    """Query-only attention should keep questions out of senders and auxiliary targets."""
    with pytest.raises(ValueError, match="requires receiver question input"):
        replace(
            TrainConfig(),
            sender_question_conditioning=True,
            receiver_question_input=False,
            receiver_question_query_only=True,
        ).validate()
    with pytest.raises(ValueError, match="requires question-blind senders"):
        replace(
            TrainConfig(),
            sender_question_conditioning=True,
            receiver_question_query_only=True,
        ).validate()
    with pytest.raises(ValueError, match="incompatible with bridge supervision"):
        replace(
            TrainConfig(),
            receiver_question_query_only=True,
            bridge_supervision_weight=1.0,
        ).validate()


def test_config_rejects_unsupported_communication_round_count() -> None:
    """Communication must use one direct round or two synchronous rounds."""
    with pytest.raises(ValueError, match="communication_rounds must be 1 or 2"):
        replace(TrainConfig(), communication_rounds=3).validate()


def test_config_rejects_negative_bridge_supervision_weight() -> None:
    """An auxiliary loss weight must not invert the bridge objective."""
    with pytest.raises(ValueError, match="bridge supervision weight cannot be negative"):
        replace(TrainConfig(), bridge_supervision_weight=-0.1).validate()


def test_config_rejects_incomplete_or_incompatible_pretrained_initialization() -> None:
    """The T5 warm start should fail unless artifacts and model geometry match exactly."""
    with pytest.raises(ValueError, match="pretrained artifacts require"):
        replace(TrainConfig(), pretrained_model_dir=Path("/tmp/model")).validate()

    pretrained_config = replace(
        TrainConfig(),
        hidden_size=256,
        output_hidden_size=256,
        sender_hidden_size=256,
        sender_num_layers=4,
        sender_num_heads=4,
        sender_feedforward_size=1024,
        sender_average_residual=True,
        receiver_question_query_only=True,
        num_layers=4,
        num_heads=4,
        pretrained_initialization="google_t5_efficient_tiny",
        pretrained_model_dir=Path("/tmp/model"),
        pretrained_checkpoint_sha256="0" * 64,
        pretrained_tokenizer_sha256="1" * 64,
    )
    pretrained_config.validate()

    with pytest.raises(ValueError, match="exact model geometry"):
        replace(pretrained_config, num_layers=2).validate()


def test_tpu_config_requires_node_local_pretrained_artifacts() -> None:
    """TPU warm starts should reject pretrained files outside node-local run storage."""
    config = TrainConfig(
        run_id="gate",
        device="tpu",
        dataset_cache_dir=Path("/home/user/elf-data/cache"),
        output_path=Path("/home/user/elf-runs/gate/checkpoint.msgpack"),
        throughput_warmup_steps=1,
        steps=2,
        hidden_size=256,
        output_hidden_size=256,
        sender_hidden_size=256,
        sender_num_layers=4,
        sender_num_heads=4,
        sender_feedforward_size=1024,
        sender_average_residual=True,
        receiver_question_query_only=True,
        num_layers=4,
        num_heads=4,
        pretrained_initialization="google_t5_efficient_tiny",
        pretrained_model_dir=Path("/tmp/model"),
        pretrained_checkpoint_sha256="0" * 64,
        pretrained_tokenizer_sha256="1" * 64,
    )

    with pytest.raises(ValueError, match="pretrained_model_dir"):
        config.validate()


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
        lambda model, collect_message_interventions=False, world_block_size=1: object(),
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
        lambda model, collect_message_interventions=False, world_block_size=1: object(),
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
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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


def test_masked_region_average_uses_equal_population_contiguous_regions() -> None:
    """The sender bypass should average valid tokens without including padding."""
    hidden = jnp.arange(1, 9, dtype=jnp.float32).reshape(1, 8, 1)
    valid = jnp.array([[True, True, True, True, True, True, False, False]])

    averages = _masked_region_average(hidden, valid, region_count=4)

    assert jnp.allclose(averages[0, :, 0], jnp.array([1.5, 3.0, 4.5, 6.0]))


def test_sender_specific_capacity_starts_from_the_average_residual() -> None:
    """A wider shared sender should retain receiver width and begin at regional averages."""
    tokenizer = _example_tokenizer()
    config = replace(
        _small_config(),
        sender_hidden_size=24,
        sender_num_layers=2,
        sender_num_heads=6,
        sender_feedforward_size=72,
        sender_average_residual=True,
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
    messages = model.apply(
        variables,
        jnp.asarray(batch.agent_1_token_ids),
        jnp.asarray(batch.agent_2_token_ids),
        deterministic=True,
        method=model.encode_messages,
    )
    sender_parameters = variables["params"]["sender"]
    embedding = variables["params"]["token_embedding"]["embedding"]
    expected_agent_1 = _masked_region_average(
        embedding[jnp.asarray(batch.agent_1_token_ids)],
        jnp.asarray(batch.agent_1_token_ids) != tokenizer.pad_id,
        config.message_length,
    )
    expected_agent_2 = _masked_region_average(
        embedding[jnp.asarray(batch.agent_2_token_ids)],
        jnp.asarray(batch.agent_2_token_ids) != tokenizer.pad_id,
        config.message_length,
    )

    assert messages.shape == (2, config.message_length * 2, config.hidden_size)
    assert sender_parameters["query_embedding"].shape == (config.message_length, 24)
    assert sender_parameters["input_projection"]["kernel"].shape == (16, 24)
    assert sender_parameters["layer_0"]["feedforward_in"]["kernel"].shape == (24, 72)
    assert "layer_1" in sender_parameters
    assert sender_parameters["message_projection"]["kernel"].shape == (24, 16)
    assert jnp.all(sender_parameters["message_projection"]["kernel"] == 0)
    assert jnp.allclose(messages[:, : config.message_length], expected_agent_1)
    assert jnp.allclose(messages[:, config.message_length :], expected_agent_2)


def test_question_conditioned_senders_require_and_use_the_question() -> None:
    """Conditioned sender messages should change when only the question changes."""
    tokenizer = _example_tokenizer()
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


@pytest.mark.parametrize("tie_sender_receiver_encoder", [False, True])
def test_two_round_exchange_uses_the_other_senders_first_message(
    tie_sender_receiver_encoder: bool,
) -> None:
    """Each second-round message should depend on the other sender's passage."""
    tokenizer = _example_tokenizer()
    config = replace(
        _small_config(),
        sender_question_conditioning=True,
        communication_rounds=2,
        tie_sender_receiver_encoder=tie_sender_receiver_encoder,
    )
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model_config = _model_config(config, tokenizer)
    model = JaxCommunicationModel(model_config)
    agent_1_token_ids = jnp.asarray(batch.agent_1_token_ids)
    agent_2_token_ids = jnp.asarray(batch.agent_2_token_ids)
    repeated_questions = jnp.repeat(jnp.asarray(batch.question_token_ids[:1]), 2, axis=0)
    variables = model.init(
        {"params": jax.random.PRNGKey(0)},
        agent_1_token_ids,
        agent_2_token_ids,
        repeated_questions,
        jnp.asarray(batch.answer_token_ids),
        deterministic=True,
    )
    single_round_model = JaxCommunicationModel(
        replace(model_config, communication_rounds=1)
    )
    single_round_variables = single_round_model.init(
        {"params": jax.random.PRNGKey(0)},
        agent_1_token_ids,
        agent_2_token_ids,
        repeated_questions,
        jnp.asarray(batch.answer_token_ids),
        deterministic=True,
    )

    repeated_agent_1 = jnp.repeat(agent_1_token_ids[:1], 2, axis=0)
    repeated_agent_2 = jnp.repeat(agent_2_token_ids[:1], 2, axis=0)
    messages_with_changed_agent_2 = model.apply(
        variables,
        repeated_agent_1,
        agent_2_token_ids,
        deterministic=True,
        question_token_ids=repeated_questions,
        method=model.encode_messages,
    )
    messages_with_changed_agent_1 = model.apply(
        variables,
        agent_1_token_ids,
        repeated_agent_2,
        deterministic=True,
        question_token_ids=repeated_questions,
        method=model.encode_messages,
    )
    single_round_messages = single_round_model.apply(
        single_round_variables,
        repeated_agent_1,
        agent_2_token_ids,
        deterministic=True,
        question_token_ids=repeated_questions,
        method=single_round_model.encode_messages,
    )

    assert messages_with_changed_agent_2.shape == (
        2,
        config.message_length * 2,
        config.hidden_size,
    )
    assert jnp.allclose(
        single_round_messages[0, : config.message_length],
        single_round_messages[1, : config.message_length],
    )
    assert not jnp.allclose(
        messages_with_changed_agent_2[0, : config.message_length],
        messages_with_changed_agent_2[1, : config.message_length],
    )
    assert not jnp.allclose(
        messages_with_changed_agent_1[0, config.message_length :],
        messages_with_changed_agent_1[1, config.message_length :],
    )
    assert jax.tree_util.tree_structure(variables["params"]) == jax.tree_util.tree_structure(
        single_round_variables["params"]
    )
    for two_round_parameter, single_round_parameter in zip(
        jax.tree_util.tree_leaves(variables["params"]),
        jax.tree_util.tree_leaves(single_round_variables["params"]),
        strict=True,
    ):
        assert jnp.array_equal(two_round_parameter, single_round_parameter)


def test_receiver_question_input_can_be_disabled() -> None:
    """A disabled receiver question path should make logits question-invariant."""
    tokenizer = _example_tokenizer()
    config = replace(
        _small_config(),
        sender_question_conditioning=True,
        receiver_question_input=False,
    )
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    agent_1_token_ids = jnp.asarray(batch.agent_1_token_ids)
    agent_2_token_ids = jnp.asarray(batch.agent_2_token_ids)
    question_token_ids = jnp.asarray(batch.question_token_ids)
    answer_token_ids = jnp.asarray(batch.answer_token_ids)
    variables = model.init(
        {"params": jax.random.PRNGKey(0)},
        agent_1_token_ids,
        agent_2_token_ids,
        question_token_ids,
        answer_token_ids,
        deterministic=True,
    )
    messages = model.apply(
        variables,
        agent_1_token_ids,
        agent_2_token_ids,
        deterministic=True,
        question_token_ids=question_token_ids,
        method=model.encode_messages,
    )
    beginning = jnp.full((2, 1), tokenizer.bos_id, dtype=jnp.int32)
    answer_input_ids = jnp.concatenate((beginning, answer_token_ids[:, :-1]), axis=1)
    reversed_questions = jnp.flip(question_token_ids, axis=0)

    logits = model.apply(
        variables,
        messages,
        question_token_ids,
        answer_input_ids,
        deterministic=True,
        method=model.answer_logits,
    )
    reversed_question_logits = model.apply(
        variables,
        messages,
        reversed_questions,
        answer_input_ids,
        deterministic=True,
        method=model.answer_logits,
    )
    receiver_question_model = JaxCommunicationModel(
        replace(_model_config(config, tokenizer), receiver_question_input=True)
    )
    active_question_logits = receiver_question_model.apply(
        variables,
        messages,
        question_token_ids,
        answer_input_ids,
        deterministic=True,
        method=receiver_question_model.answer_logits,
    )
    active_reversed_question_logits = receiver_question_model.apply(
        variables,
        messages,
        reversed_questions,
        answer_input_ids,
        deterministic=True,
        method=receiver_question_model.answer_logits,
    )

    assert jnp.allclose(logits, reversed_question_logits)
    assert not jnp.allclose(active_question_logits, active_reversed_question_logits)


def test_query_only_receiver_routes_question_information_through_message_values() -> None:
    """Questions should select message values without becoming decoder content."""
    tokenizer = _example_tokenizer()
    config = replace(_small_config(), receiver_question_query_only=True)
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    agent_1_token_ids = jnp.asarray(batch.agent_1_token_ids)
    agent_2_token_ids = jnp.asarray(batch.agent_2_token_ids)
    question_token_ids = jnp.asarray(batch.question_token_ids)
    answer_token_ids = jnp.asarray(batch.answer_token_ids)
    variables = model.init(
        {"params": jax.random.PRNGKey(0)},
        agent_1_token_ids,
        agent_2_token_ids,
        question_token_ids,
        answer_token_ids,
        deterministic=True,
    )
    repeated_agent_1 = jnp.repeat(agent_1_token_ids[:1], 2, axis=0)
    repeated_agent_2 = jnp.repeat(agent_2_token_ids[:1], 2, axis=0)
    sender_messages = model.apply(
        variables,
        repeated_agent_1,
        repeated_agent_2,
        deterministic=True,
        question_token_ids=question_token_ids,
        method=model.encode_messages,
    )
    beginning = jnp.full((2, 1), tokenizer.bos_id, dtype=jnp.int32)
    repeated_answer_token_ids = jnp.repeat(answer_token_ids[:1], 2, axis=0)
    answer_input_ids = jnp.concatenate(
        (beginning, repeated_answer_token_ids[:, :-1]),
        axis=1,
    )
    question_selected_logits = model.apply(
        variables,
        sender_messages,
        question_token_ids,
        answer_input_ids,
        deterministic=True,
        method=model.answer_logits,
    )
    repeated_message_value = jnp.mean(sender_messages[:1], axis=1, keepdims=True)
    uniform_messages = jnp.broadcast_to(repeated_message_value, sender_messages.shape)
    uniform_message_logits = model.apply(
        variables,
        uniform_messages,
        question_token_ids,
        answer_input_ids,
        deterministic=True,
        method=model.answer_logits,
    )

    def selected_answer_logit(current_messages: jax.Array) -> jax.Array:
        logits = model.apply(
            variables,
            current_messages,
            question_token_ids,
            answer_input_ids,
            deterministic=True,
            method=model.answer_logits,
        )
        return logits[0, 0, int(answer_token_ids[0, 0])]

    message_gradients = jax.grad(selected_answer_logit)(sender_messages)
    assert jnp.allclose(sender_messages[0], sender_messages[1])
    assert not jnp.allclose(question_selected_logits[0], question_selected_logits[1])
    assert jnp.allclose(uniform_message_logits[0], uniform_message_logits[1], atol=1e-5)
    assert float(jnp.linalg.norm(message_gradients)) > 0


@pytest.mark.parametrize("tie_sender_receiver_encoder", [False, True])
@pytest.mark.parametrize("communication_rounds", [1, 2])
def test_conditioned_wrong_world_control_keeps_the_target_question(
    tie_sender_receiver_encoder: bool,
    communication_rounds: int,
) -> None:
    """Wrong-world controls should rotate passages but retain each target question."""
    tokenizer = _example_tokenizer()
    config = replace(
        _small_config(),
        sender_question_conditioning=True,
        tie_sender_receiver_encoder=tie_sender_receiver_encoder,
        communication_rounds=communication_rounds,
    )
    batch = tokenize_examples(_examples(), tokenizer, config).select([0, 1])
    model = JaxCommunicationModel(_model_config(config, tokenizer))
    agent_1_token_ids = jnp.asarray(batch.agent_1_token_ids)
    agent_2_token_ids = jnp.asarray(batch.agent_2_token_ids)
    question_token_ids = jnp.asarray(batch.question_token_ids)
    answer_token_ids = jnp.asarray(batch.answer_token_ids)
    variables = model.init(
        {"params": jax.random.PRNGKey(0)},
        agent_1_token_ids,
        agent_2_token_ids,
        question_token_ids,
        answer_token_ids,
        deterministic=True,
    )

    wrong_world_output = model.apply(
        variables,
        agent_1_token_ids,
        agent_2_token_ids,
        question_token_ids,
        answer_token_ids,
        deterministic=True,
        message_intervention="wrong_world",
        world_block_size=1,
    )
    expected_messages = model.apply(
        variables,
        jnp.roll(agent_1_token_ids, shift=1, axis=0),
        jnp.roll(agent_2_token_ids, shift=1, axis=0),
        deterministic=True,
        question_token_ids=question_token_ids,
        method=model.encode_messages,
    )
    source_messages = model.apply(
        variables,
        agent_1_token_ids,
        agent_2_token_ids,
        deterministic=True,
        question_token_ids=question_token_ids,
        method=model.encode_messages,
    )
    actual_messages = jnp.concatenate(
        (wrong_world_output.message_1, wrong_world_output.message_2),
        axis=1,
    )

    assert jnp.allclose(actual_messages, expected_messages)
    assert not jnp.allclose(actual_messages, jnp.roll(source_messages, shift=1, axis=0))
    wrong_world_tokens = model.apply(
        variables,
        agent_1_token_ids,
        agent_2_token_ids,
        question_token_ids,
        message_intervention="wrong_world",
        world_block_size=1,
        method=model.greedy_answer_token_ids,
    )
    expected_tokens = model.apply(
        variables,
        jnp.roll(agent_1_token_ids, shift=1, axis=0),
        jnp.roll(agent_2_token_ids, shift=1, axis=0),
        question_token_ids,
        method=model.greedy_answer_token_ids,
    )
    assert jnp.array_equal(wrong_world_tokens, expected_tokens)


@pytest.mark.parametrize("communication_rounds", [1, 2])
def test_conditioned_wrong_question_control_keeps_passages_and_receiver_question(
    communication_rounds: int,
) -> None:
    """Wrong-question controls should change only sender-side question identity."""
    tokenizer = _example_tokenizer()
    config = replace(
        _small_config(),
        sender_question_conditioning=True,
        communication_rounds=communication_rounds,
    )
    base_batch = tokenize_examples(_examples(), tokenizer, config)
    batch = base_batch.select([0, 1, 1, 0])
    model_config = _model_config(config, tokenizer)
    model = JaxCommunicationModel(model_config)
    agent_1_token_ids = jnp.asarray(batch.agent_1_token_ids)
    agent_2_token_ids = jnp.asarray(batch.agent_2_token_ids)
    question_token_ids = jnp.asarray(batch.question_token_ids)
    answer_token_ids = jnp.asarray(batch.answer_token_ids)
    variables = model.init(
        {"params": jax.random.PRNGKey(0)},
        agent_1_token_ids,
        agent_2_token_ids,
        question_token_ids,
        answer_token_ids,
        deterministic=True,
    )
    wrong_question_token_ids = jnp.roll(
        question_token_ids.reshape(2, 2, question_token_ids.shape[1]),
        shift=1,
        axis=1,
    ).reshape(question_token_ids.shape)

    wrong_question_output = model.apply(
        variables,
        agent_1_token_ids,
        agent_2_token_ids,
        question_token_ids,
        answer_token_ids,
        deterministic=True,
        message_intervention="wrong_question",
        world_block_size=2,
    )
    expected_messages = model.apply(
        variables,
        agent_1_token_ids,
        agent_2_token_ids,
        deterministic=True,
        question_token_ids=wrong_question_token_ids,
        method=model.encode_messages,
    )
    actual_messages = jnp.concatenate(
        (wrong_question_output.message_1, wrong_question_output.message_2),
        axis=1,
    )
    beginning = jnp.full((4, 1), tokenizer.bos_id, dtype=jnp.int32)
    answer_input_ids = jnp.concatenate((beginning, answer_token_ids[:, :-1]), axis=1)
    expected_logits = model.apply(
        variables,
        expected_messages,
        question_token_ids,
        answer_input_ids,
        deterministic=True,
        method=model.answer_logits,
    )
    wrong_receiver_question_logits = model.apply(
        variables,
        expected_messages,
        wrong_question_token_ids,
        answer_input_ids,
        deterministic=True,
        method=model.answer_logits,
    )

    assert jnp.allclose(actual_messages, expected_messages)
    assert jnp.allclose(wrong_question_output.logits, expected_logits)
    assert not jnp.allclose(wrong_question_output.logits, wrong_receiver_question_logits)

    sender_only_model = JaxCommunicationModel(
        replace(model_config, receiver_question_input=False)
    )
    wrong_question_tokens = sender_only_model.apply(
        variables,
        agent_1_token_ids,
        agent_2_token_ids,
        question_token_ids,
        message_intervention="wrong_question",
        world_block_size=2,
        method=sender_only_model.greedy_answer_token_ids,
    )
    expected_tokens = sender_only_model.apply(
        variables,
        agent_1_token_ids,
        agent_2_token_ids,
        wrong_question_token_ids,
        method=sender_only_model.greedy_answer_token_ids,
    )
    assert jnp.array_equal(wrong_question_tokens, expected_tokens)


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


def test_world_block_size_requires_two_aligned_worlds_per_device() -> None:
    """Wrong-world evaluation should fail before any same-world pairing is possible."""
    assert _validated_world_block_size(256, 4, 8) == 8
    with pytest.raises(ValueError, match="world-aligned"):
        _validated_world_block_size(24, 4, 8)
    with pytest.raises(ValueError, match="at least two worlds"):
        _validated_world_block_size(32, 4, 8)


@pytest.mark.parametrize("sender_question_conditioning", [False, True])
def test_evaluation_reports_every_message_intervention(
    sender_question_conditioning: bool,
) -> None:
    """Causal evaluation should emit intact and intervened metrics from one batch."""
    tokenizer = _example_tokenizer()
    config = replace(
        _small_config(),
        batch_size=4,
        eval_examples=4,
        sender_question_conditioning=sender_question_conditioning,
    )
    source_examples = _examples()
    tokenized = tokenize_examples(
        [source_examples[index] for index in (0, 1, 1, 0)],
        tokenizer,
        config,
    )
    batch = tokenized.select([0, 1, 2, 3])
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
            world_block_size=2,
        ),
    )

    expected_prefixes = {
        "zero_message",
        "wrong_world_message",
        "swapped_sender",
        "agent_1_only",
        "agent_2_only",
    }
    if sender_question_conditioning:
        expected_prefixes.add("wrong_question_message")
    assert metrics.keys() >= {"exact_match", "mean_joint_log_probability"}
    for prefix in expected_prefixes:
        assert 0.0 <= metrics[f"{prefix}_exact_match"] <= 1.0
        assert np.isfinite(metrics[f"{prefix}_mean_joint_log_probability"])
    wrong_question_metric_names = {
        "wrong_question_message_exact_match",
        "wrong_question_message_mean_joint_log_probability",
    }
    assert wrong_question_metric_names.issubset(metrics) == sender_question_conditioning


def test_evaluation_reports_bridge_accuracy_for_every_intervention() -> None:
    """Bridge supervision should remain measurable under every causal control."""
    tokenizer = _example_tokenizer()
    config = replace(
        _small_config(),
        batch_size=4,
        eval_examples=4,
        sender_question_conditioning=True,
        bridge_supervision_weight=1.0,
    )
    source_examples = _examples()
    tokenized = tokenize_examples(
        [source_examples[index] for index in (0, 1, 1, 0)],
        tokenizer,
        config,
    )
    batch = tokenized.select([0, 1, 2, 3])
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
            world_block_size=2,
        ),
    )

    expected_bridge_metrics = {
        "bridge_exact_match",
        "zero_message_bridge_exact_match",
        "wrong_world_message_bridge_exact_match",
        "wrong_question_message_bridge_exact_match",
        "swapped_sender_bridge_exact_match",
        "agent_1_only_bridge_exact_match",
        "agent_2_only_bridge_exact_match",
    }
    assert metrics.keys() >= expected_bridge_metrics
    assert all(0.0 <= metrics[key] <= 1.0 for key in expected_bridge_metrics)


def test_output_hidden_size_is_independent_of_transformer_width() -> None:
    """The optional output MLP should widen only the answer head."""
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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
    tokenizer = _example_tokenizer()
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
        "update_norm",
        "update_norm_embedding",
        "update_norm_sender",
        "update_norm_receiver",
        "relative_update_norm_sender",
        "relative_update_norm_receiver",
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
        "update_norm_sender",
        "update_norm_receiver",
        "relative_update_norm_sender",
        "relative_update_norm_receiver",
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
