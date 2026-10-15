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


@dataclass(frozen=True)
class JaxCommunicationModelConfig:
    """Sizes and numeric settings for the Flax communication model."""

    vocab_size: int
    hidden_size: int = 128
    output_hidden_size: int | None = None
    tie_sender_receiver_encoder: bool = False
    role_gate_channels: int | None = None
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

    def validate(self) -> None:
        """Reject dimensions that the Transformer cannot use."""
        if self.vocab_size < 4:
            raise ValueError("vocab_size must include the special tokens")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.output_hidden_size is not None and self.output_hidden_size < 1:
            raise ValueError("output_hidden_size must be positive when configured")
        if self.role_gate_channels is not None and self.role_gate_channels < 1:
            raise ValueError("role_gate_channels must be positive when configured")
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
) -> Array:
    """Return a head-wise relative bias of shape (1, heads, query, key)."""
    relative_embedding = module.param(
        name,
        nn.initializers.normal(stddev=1.0),
        (config.num_heads, config.relative_attention_num_buckets),
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

    @nn.compact
    def __call__(self, hidden: Array, valid: Array, deterministic: bool) -> Array:
        """Encode one padded sequence while masking invalid keys."""
        config = self.config
        sequence_length = hidden.shape[1]
        relative_bias = _relative_attention_bias(
            self,
            sequence_length,
            sequence_length,
            config,
            bidirectional=True,
            name="relative_attention_bias",
        )
        attention = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            qkv_features=config.hidden_size,
            out_features=config.hidden_size,
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


class EncoderStack(nn.Module):
    """Apply one reusable stack of Transformer encoder blocks."""

    config: JaxCommunicationModelConfig

    @nn.compact
    def __call__(self, hidden: Array, valid: Array, deterministic: bool) -> Array:
        """Encode a padded sequence through every configured layer."""
        for layer_index in range(self.config.num_layers):
            hidden = EncoderBlock(self.config, name=f"layer_{layer_index}")(
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


class Sender(nn.Module):
    """Read one private passage and return fixed-length message vectors."""

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
        query_embedding = self.param(
            "query_embedding",
            nn.initializers.normal(stddev=1.0),
            (config.message_length, config.hidden_size),
            jnp.float32,
        ).astype(config.dtype)
        text_length = text_embeddings.shape[1]
        if text_length > config.max_text_length:
            raise ValueError("text is longer than the configured maximum")
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
            for layer_index in range(config.num_layers):
                hidden = EncoderBlock(config, name=f"layer_{layer_index}")(
                    hidden, valid, deterministic
                )
        else:
            hidden = self.shared_encoder(hidden, valid, deterministic)
        messages = hidden[:, : config.message_length]
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
    """Read two messages and a question, then score answer tokens."""

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
        memory = jnp.concatenate((messages, question_embeddings), axis=1)
        memory_valid = jnp.concatenate(
            (
                jnp.ones((batch_size, message_count), dtype=jnp.bool_),
                question_token_ids != config.padding_id,
            ),
            axis=1,
        )
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

    def encode_messages(
        self,
        agent_1_token_ids: Array,
        agent_2_token_ids: Array,
        deterministic: bool,
    ) -> Array:
        """Encode both private passages with the shared sender."""
        messages, _sender_gate_1, _sender_gate_2 = self._encode_messages_with_gates(
            agent_1_token_ids,
            agent_2_token_ids,
            deterministic,
        )
        return messages

    def _encode_messages_with_gates(
        self,
        agent_1_token_ids: Array,
        agent_2_token_ids: Array,
        deterministic: bool,
    ) -> tuple[Array, Array | None, Array | None]:
        """Encode both private passages and preserve role-gate telemetry."""
        message_1, sender_gate_1 = self.sender(
            self.token_embedding(agent_1_token_ids),
            agent_1_token_ids,
            deterministic,
        )
        message_2, sender_gate_2 = self.sender(
            self.token_embedding(agent_2_token_ids),
            agent_2_token_ids,
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
        """Score answer inputs from encoded messages and a question."""
        logits, _receiver_activation, _receiver_gate = self.receiver(
            messages,
            self.token_embedding(question_token_ids),
            question_token_ids,
            self.token_embedding(answer_input_ids),
            answer_input_ids,
            deterministic,
        )
        return logits

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
    ) -> JaxCommunicationOutput:
        """Teacher-force answers and return their joint log probability."""
        batch_size = answer_token_ids.shape[0]
        beginning = jnp.full((batch_size, 1), self.config.bos_id, dtype=jnp.int32)
        answer_input_ids = jnp.concatenate((beginning, answer_token_ids[:, :-1]), axis=1)
        messages, sender_gate_1, sender_gate_2 = self._encode_messages_with_gates(
            agent_1_token_ids,
            agent_2_token_ids,
            deterministic,
        )
        message_1 = messages[:, : self.config.message_length]
        message_2 = messages[:, self.config.message_length :]
        if stop_message_1:
            message_1 = jax.lax.stop_gradient(message_1)
        if stop_message_2:
            message_2 = jax.lax.stop_gradient(message_2)
        question_embeddings = self.token_embedding(question_token_ids)
        answer_embeddings = self.token_embedding(answer_input_ids)
        if stop_receiver_tokens:
            question_embeddings = jax.lax.stop_gradient(question_embeddings)
            answer_embeddings = jax.lax.stop_gradient(answer_embeddings)
        logits, receiver_activation, receiver_gate = self.receiver(
            jnp.concatenate((message_1, message_2), axis=1),
            question_embeddings,
            question_token_ids,
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
        )

    def greedy_answer_token_ids(
        self,
        agent_1_token_ids: Array,
        agent_2_token_ids: Array,
        question_token_ids: Array,
    ) -> Array:
        """Generate a fixed-shape greedy answer without data-dependent control flow."""
        batch_size = agent_1_token_ids.shape[0]
        generated = jnp.full(
            (batch_size, self.config.max_answer_length),
            self.config.padding_id,
            dtype=jnp.int32,
        )
        finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
        messages = self.encode_messages(agent_1_token_ids, agent_2_token_ids, deterministic=True)
        beginning = jnp.full((batch_size, 1), self.config.bos_id, dtype=jnp.int32)
        for position in range(self.config.max_answer_length):
            answer_input_ids = jnp.concatenate((beginning, generated[:, :-1]), axis=1)
            logits = self.answer_logits(messages, question_token_ids, answer_input_ids, deterministic=True)
            next_token = logits[:, position].argmax(axis=-1).astype(jnp.int32)
            next_token = jnp.where(finished, self.config.padding_id, next_token)
            generated = generated.at[:, position].set(next_token)
            finished = jnp.logical_or(finished, next_token == self.config.eos_id)
        return generated
