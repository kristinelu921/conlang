"""Flax implementation of the linked-facts communication model.

The sender and receiver follow the Transformer architecture from Attention Is
All You Need: https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.attention import dot_product_attention


Array = jax.Array


def apply_message_intervention(
    messages: Array,
    message_length: int,
    intervention: str,
    wrong_world_shift: int = 1,
) -> Array:
    """Apply one causal intervention to two concatenated sender messages."""
    if intervention == "intact":
        return messages
    if intervention == "zero":
        return jnp.zeros_like(messages)
    if intervention == "wrong_world":
        if wrong_world_shift < 1 or wrong_world_shift >= messages.shape[0]:
            raise ValueError("wrong-world shift must select another row in the local batch")
        return jnp.roll(messages, shift=wrong_world_shift, axis=0)
    message_1 = messages[:, :message_length]
    message_2 = messages[:, message_length:]
    if intervention == "swap_senders":
        return jnp.concatenate((message_2, message_1), axis=1)
    if intervention == "agent_1_only":
        return jnp.concatenate((message_1, jnp.zeros_like(message_2)), axis=1)
    if intervention == "agent_2_only":
        return jnp.concatenate((jnp.zeros_like(message_1), message_2), axis=1)
    raise ValueError(f"unknown message intervention: {intervention}")


def _prepare_sender_inputs_for_message_intervention(
    agent_1_token_ids: Array,
    agent_2_token_ids: Array,
    question_token_ids: Array,
    sender_question_conditioning: bool,
    intervention: str,
    world_block_size: int,
) -> tuple[Array, Array, Array, str]:
    """Prepare matched passage or question controls before sender encoding."""
    if intervention == "wrong_question":
        if not sender_question_conditioning:
            raise ValueError("wrong-question intervention requires conditioned senders")
        batch_size = question_token_ids.shape[0]
        if world_block_size < 2 or batch_size % world_block_size != 0:
            raise ValueError("wrong-question intervention requires aligned question blocks")
        grouped_questions = question_token_ids.reshape(
            batch_size // world_block_size,
            world_block_size,
            question_token_ids.shape[1],
        )
        wrong_questions = jnp.roll(grouped_questions, shift=1, axis=1).reshape(
            question_token_ids.shape
        )
        return agent_1_token_ids, agent_2_token_ids, wrong_questions, "intact"
    if not sender_question_conditioning or intervention != "wrong_world":
        return agent_1_token_ids, agent_2_token_ids, question_token_ids, intervention
    if world_block_size < 1 or world_block_size >= agent_1_token_ids.shape[0]:
        raise ValueError("wrong-world shift must select another row in the local batch")
    return (
        jnp.roll(agent_1_token_ids, shift=world_block_size, axis=0),
        jnp.roll(agent_2_token_ids, shift=world_block_size, axis=0),
        question_token_ids,
        "intact",
    )


@dataclass(frozen=True)
class JaxCommunicationModelConfig:
    """Sizes and numeric settings for the Flax communication model."""

    vocab_size: int
    hidden_size: int = 128
    output_hidden_size: int | None = None
    sender_hidden_size: int | None = None
    sender_num_layers: int | None = None
    sender_num_heads: int | None = None
    sender_feedforward_size: int | None = None
    sender_average_residual: bool = False
    tie_sender_receiver_encoder: bool = False
    role_gate_channels: int | None = None
    sender_question_conditioning: bool = False
    receiver_question_input: bool = True
    receiver_question_query_only: bool = False
    communication_rounds: int = 1
    bridge_supervision_weight: float = 0.0
    message_length: int = 4
    max_answer_length: int = 4
    max_text_length: int = 512
    max_question_length: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    padding_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dtype: jnp.dtype = jnp.float32

    @property
    def feedforward_size(self) -> int:
        """Return the Transformer feedforward width."""
        return self.hidden_size * 4

    @property
    def resolved_sender_hidden_size(self) -> int:
        """Return the sender width after applying its optional override."""
        return self.hidden_size if self.sender_hidden_size is None else self.sender_hidden_size

    @property
    def resolved_sender_num_layers(self) -> int:
        """Return the sender depth after applying its optional override."""
        return self.num_layers if self.sender_num_layers is None else self.sender_num_layers

    @property
    def resolved_sender_num_heads(self) -> int:
        """Return the sender head count after applying its optional override."""
        return self.num_heads if self.sender_num_heads is None else self.sender_num_heads

    @property
    def resolved_sender_feedforward_size(self) -> int:
        """Return the sender feed-forward width after applying its optional override."""
        if self.sender_feedforward_size is None:
            return self.resolved_sender_hidden_size * 4
        return self.sender_feedforward_size

    def validate(self) -> None:
        """Reject dimensions that the Transformer cannot use."""
        if self.vocab_size < 4:
            raise ValueError("vocab_size must include the special tokens")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        sender_sizes = (
            self.resolved_sender_hidden_size,
            self.resolved_sender_num_layers,
            self.resolved_sender_num_heads,
            self.resolved_sender_feedforward_size,
        )
        if min(sender_sizes) < 1:
            raise ValueError("sender width, depth, heads, and feed-forward size must be positive")
        if self.resolved_sender_hidden_size % self.resolved_sender_num_heads != 0:
            raise ValueError("sender hidden size must be divisible by sender attention heads")
        if self.tie_sender_receiver_encoder and sender_sizes != (
            self.hidden_size,
            self.num_layers,
            self.num_heads,
            self.feedforward_size,
        ):
            raise ValueError("sender-specific dimensions are incompatible with a tied encoder")
        if self.output_hidden_size is not None and self.output_hidden_size < 1:
            raise ValueError("output_hidden_size must be positive when configured")
        if self.role_gate_channels is not None and self.role_gate_channels < 1:
            raise ValueError("role_gate_channels must be positive when configured")
        if not self.receiver_question_input and not self.sender_question_conditioning:
            raise ValueError("the question must be available to either senders or receiver")
        if self.receiver_question_query_only and not self.receiver_question_input:
            raise ValueError("query-only receiver attention requires receiver question input")
        if self.receiver_question_query_only and self.sender_question_conditioning:
            raise ValueError("query-only receiver attention requires question-blind senders")
        if self.receiver_question_query_only and self.bridge_supervision_weight > 0:
            raise ValueError("query-only receiver attention is incompatible with bridge supervision")
        if self.communication_rounds not in {1, 2}:
            raise ValueError("communication_rounds must be 1 or 2")
        if self.bridge_supervision_weight < 0:
            raise ValueError("bridge supervision weight cannot be negative")
        lengths = (
            self.message_length,
            self.max_answer_length,
            self.max_text_length,
            self.max_question_length,
            self.num_layers,
            self.num_heads,
        )
        if min(lengths) < 1:
            raise ValueError("model lengths and depth must be positive")
        if (
            self.relative_attention_num_buckets < 4
            or self.relative_attention_num_buckets % 2 != 0
        ):
            raise ValueError("relative attention needs an even bucket count of at least 4")
        if self.relative_attention_max_distance <= self.relative_attention_num_buckets // 2:
            raise ValueError("relative attention max distance must exceed half the bucket count")


class JaxCommunicationOutput(NamedTuple):
    """Answer logits, probability, loss, and the activations used for metrics."""

    logits: Array
    joint_log_probability: Array
    loss: Array
    message_1: Array
    message_2: Array
    receiver_activation: Array
    sender_gate_1: Array | None
    sender_gate_2: Array | None
    receiver_gate: Array | None
    bridge_logits_1: Array | None
    bridge_logits_2: Array | None


def _attention_mask(query_valid: Array, key_valid: Array) -> Array:
    """Build the boolean attention mask expected by Flax attention."""
    return nn.make_attention_mask(query_valid, key_valid, dtype=jnp.bool_)


def _relative_position_bucket(
    relative_position: Array,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
) -> Array:
    """Map a key-minus-query offset into one T5 relative-position bucket."""
    # Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer: https://arxiv.org/abs/1910.10683
    relative_buckets = jnp.zeros_like(relative_position, dtype=jnp.int32)
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).astype(jnp.int32) * num_buckets
        relative_position = jnp.abs(relative_position)
    else:
        relative_position = -jnp.minimum(relative_position, 0)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    safe_position = jnp.maximum(relative_position, 1).astype(jnp.float32)
    relative_position_if_large = max_exact + (
        jnp.log(safe_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).astype(jnp.int32)
    relative_position_if_large = jnp.minimum(relative_position_if_large, num_buckets - 1)
    return relative_buckets + jnp.where(
        is_small,
        relative_position.astype(jnp.int32),
        relative_position_if_large,
    )


def _relative_attention_bias(
    module: nn.Module,
    query_length: int,
    key_length: int,
    config: JaxCommunicationModelConfig,
    bidirectional: bool,
    name: str,
    num_heads: int | None = None,
) -> Array:
    """Return a head-wise relative bias of shape (1, heads, query, key)."""
    resolved_num_heads = config.num_heads if num_heads is None else num_heads
    relative_embedding = module.param(
        name,
        nn.initializers.normal(stddev=1.0),
        (resolved_num_heads, config.relative_attention_num_buckets),
        jnp.float32,
    )
    query_positions = jnp.arange(query_length)
    key_positions = jnp.arange(key_length)
    relative_position = key_positions[None, :] - query_positions[:, None]
    buckets = _relative_position_bucket(
        relative_position,
        bidirectional,
        config.relative_attention_num_buckets,
        config.relative_attention_max_distance,
    )
    return relative_embedding[:, buckets][None]


def _attend(
    query: Array,
    key: Array,
    value: Array,
    *,
    bias: Array,
    mask: Array | None = None,
    dropout_rng: Array | None = None,
    dropout_rate: float = 0.0,
    broadcast_dropout: bool = True,
    deterministic: bool = False,
    dtype: Any = None,
    precision: Any = None,
    force_fp32_for_softmax: bool = False,
    qk_attn_weights_einsum: Any = None,
    attn_weights_value_einsum: Any = None,
) -> Array:
    """Add a relative-position bias, then run dot-product attention."""
    return dot_product_attention(
        query,
        key,
        value,
        bias=bias.astype(query.dtype),
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        broadcast_dropout=broadcast_dropout,
        deterministic=deterministic,
        dtype=dtype,
        precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax,
        qk_attn_weights_einsum=qk_attn_weights_einsum,
        attn_weights_value_einsum=attn_weights_value_einsum,
    )
class EncoderBlock(nn.Module):
    """Apply one post-norm Transformer encoder block."""

    config: JaxCommunicationModelConfig
    hidden_size: int | None = None
    num_heads: int | None = None
    feedforward_size: int | None = None

    @nn.compact
    def __call__(self, hidden: Array, valid: Array, deterministic: bool) -> Array:
        """Encode one padded sequence while masking invalid keys."""
        config = self.config
        hidden_size = config.hidden_size if self.hidden_size is None else self.hidden_size
        num_heads = config.num_heads if self.num_heads is None else self.num_heads
        feedforward_size = (
            config.feedforward_size if self.feedforward_size is None else self.feedforward_size
        )
        sequence_length = hidden.shape[1]
        relative_bias = _relative_attention_bias(
            self,
            sequence_length,
            sequence_length,
            config,
            bidirectional=True,
            name="relative_attention_bias",
            num_heads=num_heads,
        )
        attention = nn.MultiHeadDotProductAttention(
            num_heads=num_heads,
            qkv_features=hidden_size,
            out_features=hidden_size,
            dropout_rate=config.dropout,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            force_fp32_for_softmax=True,
            attention_fn=functools.partial(_attend, bias=relative_bias),
            name="self_attention",
        )(hidden, mask=_attention_mask(valid, valid), deterministic=deterministic)
        hidden = nn.LayerNorm(dtype=jnp.float32, name="attention_norm")(
            hidden
            + nn.Dropout(config.dropout, name="attention_dropout")(
                attention, deterministic=deterministic
            )
        ).astype(config.dtype)
        feedforward = nn.Dense(
            feedforward_size,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            name="feedforward_in",
        )(hidden)
        feedforward = jax.nn.relu(feedforward)
        feedforward = nn.Dropout(config.dropout, name="feedforward_dropout")(
            feedforward, deterministic=deterministic
        )
        feedforward = nn.Dense(
            hidden_size,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            name="feedforward_out",
        )(feedforward)
        return nn.LayerNorm(dtype=jnp.float32, name="feedforward_norm")(
            hidden
            + nn.Dropout(config.dropout, name="residual_dropout")(feedforward, deterministic=deterministic)
        ).astype(config.dtype)


class EncoderStack(nn.Module):
    """Apply one reusable stack of Transformer encoder blocks."""

    config: JaxCommunicationModelConfig
    hidden_size: int | None = None
    num_layers: int | None = None
    num_heads: int | None = None
    feedforward_size: int | None = None

    @nn.compact
    def __call__(self, hidden: Array, valid: Array, deterministic: bool) -> Array:
        """Encode a padded sequence through every configured layer."""
        num_layers = self.config.num_layers if self.num_layers is None else self.num_layers
        for layer_index in range(num_layers):
            hidden = EncoderBlock(
                self.config,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                feedforward_size=self.feedforward_size,
                name=f"layer_{layer_index}",
            )(
                hidden, valid, deterministic
            )
        return hidden


class ResidualRoleGate(nn.Module):
    """Apply a bottleneck gate whose residual path starts as the identity."""

    hidden_size: int
    gate_channels: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, hidden: Array) -> tuple[Array, Array]:
        """Return gated hidden states and their bottleneck gate activations."""
        candidate = nn.Dense(
            self.gate_channels,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            name="candidate_projection",
        )(hidden)
        candidate = jax.nn.relu(candidate)
        gate = nn.Dense(
            self.gate_channels,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.constant(-2.0),
            name="gate_projection",
        )(hidden)
        gate = jax.nn.sigmoid(gate)
        update = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.zeros_init(),
            name="output_projection",
        )(candidate * gate)
        return hidden + update, gate


class QuestionMessageAttentionBlock(nn.Module):
    """Use encoded questions only as queries over message keys and values."""

    config: JaxCommunicationModelConfig

    @nn.compact
    def __call__(
        self,
        question_queries: Array,
        message_memory: Array,
        question_valid: Array,
        message_valid: Array,
        deterministic: bool,
    ) -> Array:
        """Return question-indexed states whose content comes only from messages."""
        config = self.config
        attended_messages = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            qkv_features=config.hidden_size,
            out_features=config.hidden_size,
            dropout_rate=config.dropout,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            force_fp32_for_softmax=True,
            name="question_message_attention",
        )(
            question_queries,
            message_memory,
            mask=_attention_mask(question_valid, message_valid),
            deterministic=deterministic,
        )
        hidden = nn.LayerNorm(dtype=jnp.float32, name="attention_norm")(
            nn.Dropout(config.dropout, name="attention_dropout")(
                attended_messages,
                deterministic=deterministic,
            )
        ).astype(config.dtype)
        feedforward = nn.Dense(
            config.feedforward_size,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            name="feedforward_in",
        )(hidden)
        feedforward = jax.nn.relu(feedforward)
        feedforward = nn.Dropout(config.dropout, name="feedforward_dropout")(
            feedforward,
            deterministic=deterministic,
        )
        feedforward = nn.Dense(
            config.hidden_size,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            name="feedforward_out",
        )(feedforward)
        return nn.LayerNorm(dtype=jnp.float32, name="feedforward_norm")(
            hidden
            + nn.Dropout(config.dropout, name="residual_dropout")(
                feedforward,
                deterministic=deterministic,
            )
        ).astype(config.dtype)


class DecoderBlock(nn.Module):
    """Apply one post-norm Transformer decoder block."""

    config: JaxCommunicationModelConfig

    @nn.compact
    def __call__(
        self,
        hidden: Array,
        memory: Array,
        answer_valid: Array,
        memory_valid: Array,
        deterministic: bool,
    ) -> Array:
        """Decode one answer sequence with causal and padding masks."""
        config = self.config
        causal_mask = nn.make_causal_mask(answer_valid, dtype=jnp.bool_)
        padding_mask = _attention_mask(answer_valid, answer_valid)
        self_bias = _relative_attention_bias(
            self,
            hidden.shape[1],
            hidden.shape[1],
            config,
            bidirectional=False,
            name="self_relative_attention_bias",
        )
        self_attention = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            qkv_features=config.hidden_size,
            out_features=config.hidden_size,
            dropout_rate=config.dropout,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            force_fp32_for_softmax=True,
            attention_fn=functools.partial(_attend, bias=self_bias),
            name="self_attention",
        )(hidden, mask=nn.combine_masks(causal_mask, padding_mask), deterministic=deterministic)
        hidden = nn.LayerNorm(dtype=jnp.float32, name="self_attention_norm")(
            hidden
            + nn.Dropout(config.dropout, name="self_attention_dropout")(
                self_attention, deterministic=deterministic
            )
        ).astype(config.dtype)
        cross_bias = _relative_attention_bias(
            self,
            hidden.shape[1],
            memory.shape[1],
            config,
            bidirectional=True,
            name="cross_relative_attention_bias",
        )
        cross_attention = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            qkv_features=config.hidden_size,
            out_features=config.hidden_size,
            dropout_rate=config.dropout,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            force_fp32_for_softmax=True,
            attention_fn=functools.partial(_attend, bias=cross_bias),
            name="cross_attention",
        )(
            hidden,
            memory,
            mask=_attention_mask(answer_valid, memory_valid),
            deterministic=deterministic,
        )
        hidden = nn.LayerNorm(dtype=jnp.float32, name="cross_attention_norm")(
            hidden
            + nn.Dropout(config.dropout, name="cross_attention_dropout")(
                cross_attention, deterministic=deterministic
            )
        ).astype(config.dtype)
        feedforward = nn.Dense(
            config.feedforward_size,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            name="feedforward_in",
        )(hidden)
        feedforward = jax.nn.relu(feedforward)
        feedforward = nn.Dropout(config.dropout, name="feedforward_dropout")(
            feedforward, deterministic=deterministic
        )
        feedforward = nn.Dense(
            config.hidden_size,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            name="feedforward_out",
        )(feedforward)
        return nn.LayerNorm(dtype=jnp.float32, name="feedforward_norm")(
            hidden
            + nn.Dropout(config.dropout, name="residual_dropout")(feedforward, deterministic=deterministic)
        ).astype(config.dtype)


def _masked_region_average(
    hidden: Array,
    valid: Array,
    region_count: int,
) -> Array:
    """Average valid sequence states into contiguous equal-population regions."""
    valid_float = valid.astype(jnp.float32)
    valid_count = jnp.sum(valid_float, axis=1, keepdims=True)
    valid_rank = jnp.cumsum(valid, axis=1).astype(jnp.int32) - 1
    region_index = jnp.minimum(
        valid_rank * region_count // jnp.maximum(valid_count.astype(jnp.int32), 1),
        region_count - 1,
    )
    region_weight = jax.nn.one_hot(
        region_index,
        region_count,
        dtype=jnp.float32,
    ) * valid_float[:, :, None]
    region_sum = jnp.einsum(
        "blr,bld->brd",
        region_weight,
        hidden.astype(jnp.float32),
    )
    region_size = jnp.sum(region_weight, axis=1)[:, :, None]
    return (region_sum / jnp.maximum(region_size, 1.0)).astype(hidden.dtype)


def _rectangular_identity(
    _key: Array,
    shape: tuple[int, ...],
    dtype: Any = jnp.float32,
) -> Array:
    """Initialize one rectangular dense kernel with an identity-aligned prefix."""
    assert len(shape) == 2
    return jnp.eye(shape[0], shape[1], dtype=dtype)


class Sender(nn.Module):
    """Read one private passage, optionally question-prefixed, and return messages."""

    config: JaxCommunicationModelConfig
    shared_encoder: EncoderStack | None = None

    @nn.compact
    def __call__(
        self,
        text_embeddings: Array,
        token_ids: Array,
        deterministic: bool,
    ) -> tuple[Array, Array | None]:
        """Encode text into message vectors and optional sender gate activations."""
        config = self.config
        batch_size = token_ids.shape[0]
        sender_hidden_size = config.resolved_sender_hidden_size
        sender_num_layers = config.resolved_sender_num_layers
        sender_num_heads = config.resolved_sender_num_heads
        sender_feedforward_size = config.resolved_sender_feedforward_size
        query_embedding = self.param(
            "query_embedding",
            nn.initializers.normal(stddev=1.0),
            (config.message_length, sender_hidden_size),
            jnp.float32,
        ).astype(config.dtype)
        text_length = text_embeddings.shape[1]
        maximum_sender_input_length = config.max_text_length
        if config.sender_question_conditioning:
            maximum_sender_input_length += config.max_question_length
        if config.communication_rounds == 2:
            maximum_sender_input_length += config.message_length
        if text_length > maximum_sender_input_length:
            raise ValueError("text is longer than the configured maximum")
        average_residual = None
        if config.sender_average_residual:
            average_residual = _masked_region_average(
                text_embeddings,
                token_ids != config.padding_id,
                config.message_length,
            )
        if sender_hidden_size != config.hidden_size:
            text_embeddings = nn.Dense(
                sender_hidden_size,
                dtype=config.dtype,
                param_dtype=jnp.float32,
                kernel_init=(
                    _rectangular_identity
                    if config.sender_average_residual
                    else nn.initializers.lecun_normal()
                ),
                name="input_projection",
            )(text_embeddings)
        queries = jnp.broadcast_to(query_embedding[None], (batch_size, *query_embedding.shape))
        hidden = jnp.concatenate((queries, text_embeddings), axis=1)
        valid = jnp.concatenate(
            (
                jnp.ones((batch_size, config.message_length), dtype=jnp.bool_),
                token_ids != config.padding_id,
            ),
            axis=1,
        )
        if self.shared_encoder is None:
            for layer_index in range(sender_num_layers):
                hidden = EncoderBlock(
                    config,
                    hidden_size=sender_hidden_size,
                    num_heads=sender_num_heads,
                    feedforward_size=sender_feedforward_size,
                    name=f"layer_{layer_index}",
                )(
                    hidden, valid, deterministic
                )
        else:
            hidden = self.shared_encoder(hidden, valid, deterministic)
        messages = hidden[:, : config.message_length]
        if sender_hidden_size != config.hidden_size or config.sender_average_residual:
            messages = nn.Dense(
                config.hidden_size,
                dtype=config.dtype,
                param_dtype=jnp.float32,
                kernel_init=(
                    nn.initializers.zeros_init()
                    if config.sender_average_residual
                    else nn.initializers.lecun_normal()
                ),
                name="message_projection",
            )(messages)
        if average_residual is not None:
            messages = messages + average_residual
        gate = None
        if config.role_gate_channels is not None:
            messages, gate = ResidualRoleGate(
                config.hidden_size,
                config.role_gate_channels,
                config.dtype,
                name="role_gate",
            )(messages)
        return messages, gate


class Receiver(nn.Module):
    """Read two messages and optional question context, then score answer tokens."""

    config: JaxCommunicationModelConfig
    shared_encoder: EncoderStack | None = None

    @nn.compact
    def __call__(
        self,
        messages: Array,
        question_embeddings: Array,
        question_token_ids: Array,
        answer_embeddings: Array,
        answer_input_ids: Array,
        deterministic: bool,
    ) -> tuple[Array, Array, Array | None]:
        """Return answer logits, decoder state, and optional receiver gate activations."""
        config = self.config
        batch_size = messages.shape[0]
        message_count = config.message_length * 2
        question_length = question_embeddings.shape[1]
        if messages.shape[1] != message_count:
            raise ValueError("receiver expected two concatenated messages")
        if (
            question_length > config.max_question_length
            or answer_embeddings.shape[1] > config.max_answer_length
        ):
            raise ValueError("question or answer is longer than the configured maximum")
        message_valid = jnp.ones((batch_size, message_count), dtype=jnp.bool_)
        question_valid = question_token_ids != config.padding_id
        if config.receiver_question_query_only:
            if self.shared_encoder is None:
                query_key_value_encoder = EncoderStack(
                    config,
                    name="query_key_value_encoder",
                )
            else:
                query_key_value_encoder = self.shared_encoder
            message_memory = query_key_value_encoder(messages, message_valid, deterministic)
            question_queries = query_key_value_encoder(
                question_embeddings,
                question_valid,
                deterministic,
            )
            memory = QuestionMessageAttentionBlock(
                config,
                name="query_message_block",
            )(
                question_queries,
                message_memory,
                question_valid,
                message_valid,
                deterministic,
            )
            memory_valid = question_valid
        else:
            memory = jnp.concatenate((messages, question_embeddings), axis=1)
            memory_valid = jnp.concatenate((message_valid, question_valid), axis=1)
            if self.shared_encoder is None:
                for layer_index in range(config.num_layers):
                    memory = EncoderBlock(config, name=f"encoder_layer_{layer_index}")(
                        memory, memory_valid, deterministic
                    )
            else:
                memory = self.shared_encoder(memory, memory_valid, deterministic)
        gate = None
        if config.role_gate_channels is not None:
            memory, gate = ResidualRoleGate(
                config.hidden_size,
                config.role_gate_channels,
                config.dtype,
                name="role_gate",
            )(memory)
        hidden = answer_embeddings
        answer_valid = answer_input_ids != config.padding_id
        for layer_index in range(config.num_layers):
            hidden = DecoderBlock(config, name=f"decoder_layer_{layer_index}")(
                hidden,
                memory,
                answer_valid,
                memory_valid,
                deterministic,
            )
        output_hidden = hidden
        if config.output_hidden_size is not None:
            output_hidden = nn.Dense(
                config.output_hidden_size,
                dtype=config.dtype,
                param_dtype=jnp.float32,
                name="output_hidden_projection",
            )(output_hidden)
            output_hidden = jax.nn.relu(output_hidden)
        logits = nn.Dense(
            config.vocab_size,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            name="output_projection",
        )(output_hidden)
        return logits, output_hidden, gate


class JaxCommunicationModel(nn.Module):
    """Shared Flax sender plus the receiver for the communication game."""

    config: JaxCommunicationModelConfig

    def setup(self) -> None:
        """Create shared embedding, sender, and receiver modules."""
        self.config.validate()
        self.token_embedding = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=1.0),
            dtype=self.config.dtype,
            param_dtype=jnp.float32,
            name="token_embedding",
        )
        shared_encoder = None
        if self.config.tie_sender_receiver_encoder:
            shared_encoder = EncoderStack(
                self.config,
                name="shared_sender_receiver_encoder",
            )
        self.sender = Sender(self.config, shared_encoder=shared_encoder, name="sender")
        self.receiver = Receiver(self.config, shared_encoder=shared_encoder, name="receiver")
        self.bridge_output_projection = (
            nn.Dense(
                self.config.vocab_size,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                name="bridge_output_projection",
            )
            if self.config.bridge_supervision_weight > 0
            else None
        )

    def encode_messages(
        self,
        agent_1_token_ids: Array,
        agent_2_token_ids: Array,
        deterministic: bool,
        question_token_ids: Array | None = None,
    ) -> Array:
        """Encode both private passages, optionally conditioned on the question."""
        messages, _sender_gate_1, _sender_gate_2 = self._encode_messages_with_gates(
            agent_1_token_ids,
            agent_2_token_ids,
            deterministic,
            question_token_ids,
        )
        return messages

    def _encode_messages_with_gates(
        self,
        agent_1_token_ids: Array,
        agent_2_token_ids: Array,
        deterministic: bool,
        question_token_ids: Array | None = None,
    ) -> tuple[Array, Array | None, Array | None]:
        """Encode both passages with optional question context and preserve gate telemetry."""
        agent_1_embeddings = self.token_embedding(agent_1_token_ids)
        agent_2_embeddings = self.token_embedding(agent_2_token_ids)
        if self.config.sender_question_conditioning:
            if question_token_ids is None:
                raise ValueError("question token IDs are required for conditioned senders")
            question_embeddings = self.token_embedding(question_token_ids)
            agent_1_embeddings = jnp.concatenate((question_embeddings, agent_1_embeddings), axis=1)
            agent_2_embeddings = jnp.concatenate((question_embeddings, agent_2_embeddings), axis=1)
            agent_1_token_ids = jnp.concatenate((question_token_ids, agent_1_token_ids), axis=1)
            agent_2_token_ids = jnp.concatenate((question_token_ids, agent_2_token_ids), axis=1)
        message_1, sender_gate_1 = self.sender(
            agent_1_embeddings,
            agent_1_token_ids,
            deterministic,
        )
        message_2, sender_gate_2 = self.sender(
            agent_2_embeddings,
            agent_2_token_ids,
            deterministic,
        )
        if self.config.communication_rounds == 2:
            round_1_message_1 = message_1
            round_1_message_2 = message_2
            # Sender consumes token IDs only as a padding mask; these placeholders
            # mark the incoming continuous message positions as valid.
            incoming_message_token_ids = jnp.full(
                (agent_1_token_ids.shape[0], self.config.message_length),
                self.config.padding_id + 1,
                dtype=agent_1_token_ids.dtype,
            )
            message_1, sender_gate_1 = self.sender(
                jnp.concatenate((round_1_message_2, agent_1_embeddings), axis=1),
                jnp.concatenate((incoming_message_token_ids, agent_1_token_ids), axis=1),
                deterministic,
            )
            message_2, sender_gate_2 = self.sender(
                jnp.concatenate((round_1_message_1, agent_2_embeddings), axis=1),
                jnp.concatenate((incoming_message_token_ids, agent_2_token_ids), axis=1),
                deterministic,
            )
        return jnp.concatenate((message_1, message_2), axis=1), sender_gate_1, sender_gate_2

    def answer_logits(
        self,
        messages: Array,
        question_token_ids: Array,
        answer_input_ids: Array,
        deterministic: bool,
    ) -> Array:
        """Score answer inputs from messages and the configured question context."""
        question_embeddings, receiver_question_token_ids = self._receiver_question_inputs(
            question_token_ids
        )
        logits, _receiver_activation, _receiver_gate = self.receiver(
            messages,
            question_embeddings,
            receiver_question_token_ids,
            self.token_embedding(answer_input_ids),
            answer_input_ids,
            deterministic,
        )
        return logits

    def _receiver_question_inputs(self, question_token_ids: Array) -> tuple[Array, Array]:
        """Return active question inputs or masked placeholders for the receiver."""
        question_embeddings = self.token_embedding(question_token_ids)
        if self.config.receiver_question_input:
            return question_embeddings, question_token_ids
        return (
            jnp.zeros_like(question_embeddings),
            jnp.full_like(question_token_ids, self.config.padding_id),
        )

    def __call__(
        self,
        agent_1_token_ids: Array,
        agent_2_token_ids: Array,
        question_token_ids: Array,
        answer_token_ids: Array,
        deterministic: bool = False,
        *,
        stop_message_1: bool = False,
        stop_message_2: bool = False,
        stop_receiver_tokens: bool = False,
        message_intervention: str = "intact",
        world_block_size: int = 1,
    ) -> JaxCommunicationOutput:
        """Teacher-force answers and return their joint log probability."""
        batch_size = answer_token_ids.shape[0]
        beginning = jnp.full((batch_size, 1), self.config.bos_id, dtype=jnp.int32)
        answer_input_ids = jnp.concatenate((beginning, answer_token_ids[:, :-1]), axis=1)
        (
            agent_1_token_ids,
            agent_2_token_ids,
            sender_question_token_ids,
            encoded_message_intervention,
        ) = _prepare_sender_inputs_for_message_intervention(
            agent_1_token_ids,
            agent_2_token_ids,
            question_token_ids,
            self.config.sender_question_conditioning,
            message_intervention,
            world_block_size,
        )
        messages, sender_gate_1, sender_gate_2 = self._encode_messages_with_gates(
            agent_1_token_ids,
            agent_2_token_ids,
            deterministic,
            sender_question_token_ids,
        )
        message_1 = messages[:, : self.config.message_length]
        message_2 = messages[:, self.config.message_length :]
        if stop_message_1:
            message_1 = jax.lax.stop_gradient(message_1)
        if stop_message_2:
            message_2 = jax.lax.stop_gradient(message_2)
        messages = apply_message_intervention(
            jnp.concatenate((message_1, message_2), axis=1),
            self.config.message_length,
            encoded_message_intervention,
            world_block_size,
        )
        message_1 = messages[:, : self.config.message_length]
        message_2 = messages[:, self.config.message_length :]
        bridge_logits_1 = None
        bridge_logits_2 = None
        if self.bridge_output_projection is not None:
            bridge_logits_1 = self.bridge_output_projection(
                jnp.mean(message_1.astype(jnp.float32), axis=1)
            )
            bridge_logits_2 = self.bridge_output_projection(
                jnp.mean(message_2.astype(jnp.float32), axis=1)
            )
        question_embeddings, receiver_question_token_ids = self._receiver_question_inputs(
            question_token_ids
        )
        answer_embeddings = self.token_embedding(answer_input_ids)
        if stop_receiver_tokens:
            question_embeddings = jax.lax.stop_gradient(question_embeddings)
            answer_embeddings = jax.lax.stop_gradient(answer_embeddings)
        logits, receiver_activation, receiver_gate = self.receiver(
            jnp.concatenate((message_1, message_2), axis=1),
            question_embeddings,
            receiver_question_token_ids,
            answer_embeddings,
            answer_input_ids,
            deterministic,
        )
        log_probabilities = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        token_log_probabilities = jnp.take_along_axis(
            log_probabilities, answer_token_ids[..., None], axis=-1
        ).squeeze(-1)
        active = answer_token_ids != self.config.padding_id
        joint_log_probability = (token_log_probabilities * active).sum(axis=-1)
        return JaxCommunicationOutput(
            logits=logits,
            joint_log_probability=joint_log_probability,
            loss=-joint_log_probability.mean(),
            message_1=message_1,
            message_2=message_2,
            receiver_activation=receiver_activation,
            sender_gate_1=sender_gate_1,
            sender_gate_2=sender_gate_2,
            receiver_gate=receiver_gate,
            bridge_logits_1=bridge_logits_1,
            bridge_logits_2=bridge_logits_2,
        )

    def greedy_answer_token_ids(
        self,
        agent_1_token_ids: Array,
        agent_2_token_ids: Array,
        question_token_ids: Array,
        message_intervention: str = "intact",
        world_block_size: int = 1,
    ) -> Array:
        """Generate a fixed-shape greedy answer without data-dependent control flow."""
        batch_size = agent_1_token_ids.shape[0]
        generated = jnp.full(
            (batch_size, self.config.max_answer_length),
            self.config.padding_id,
            dtype=jnp.int32,
        )
        finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
        (
            agent_1_token_ids,
            agent_2_token_ids,
            sender_question_token_ids,
            encoded_message_intervention,
        ) = _prepare_sender_inputs_for_message_intervention(
            agent_1_token_ids,
            agent_2_token_ids,
            question_token_ids,
            self.config.sender_question_conditioning,
            message_intervention,
            world_block_size,
        )
        messages = self.encode_messages(
            agent_1_token_ids,
            agent_2_token_ids,
            deterministic=True,
            question_token_ids=sender_question_token_ids,
        )
        messages = apply_message_intervention(
            messages,
            self.config.message_length,
            encoded_message_intervention,
            world_block_size,
        )
        beginning = jnp.full((batch_size, 1), self.config.bos_id, dtype=jnp.int32)
        for position in range(self.config.max_answer_length):
            answer_input_ids = jnp.concatenate((beginning, generated[:, :-1]), axis=1)
            logits = self.answer_logits(messages, question_token_ids, answer_input_ids, deterministic=True)
            next_token = logits[:, position].argmax(axis=-1).astype(jnp.int32)
            next_token = jnp.where(finished, self.config.padding_id, next_token)
            generated = generated.at[:, position].set(next_token)
            finished = jnp.logical_or(finished, next_token == self.config.eos_id)
        return generated
