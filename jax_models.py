"""Flax implementation of the linked-facts communication model.

The sender and receiver follow the Transformer architecture from Attention Is
All You Need: https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn


Array = jax.Array


@dataclass(frozen=True)
class JaxCommunicationModelConfig:
    """Sizes and numeric settings for the Flax communication model."""

    vocab_size: int
    hidden_size: int = 128
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


class JaxCommunicationOutput(NamedTuple):
    """Answer logits, per-example probability, and scalar loss."""

    logits: Array
    joint_log_probability: Array
    loss: Array


def _attention_mask(query_valid: Array, key_valid: Array) -> Array:
    """Build the boolean attention mask expected by Flax attention."""
    return nn.make_attention_mask(query_valid, key_valid, dtype=jnp.bool_)


class EncoderBlock(nn.Module):
    """Apply one post-norm Transformer encoder block."""

    config: JaxCommunicationModelConfig

    @nn.compact
    def __call__(self, hidden: Array, valid: Array, deterministic: bool) -> Array:
        """Encode one padded sequence while masking invalid keys."""
        config = self.config
        attention = nn.SelfAttention(
            num_heads=config.num_heads,
            qkv_features=config.hidden_size,
            out_features=config.hidden_size,
            dropout_rate=config.dropout,
            deterministic=deterministic,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            name="self_attention",
        )(hidden, mask=_attention_mask(valid, valid))
        hidden = nn.LayerNorm(dtype=jnp.float32, name="attention_norm")(
            hidden + nn.Dropout(config.dropout, deterministic=deterministic)(attention)
        ).astype(config.dtype)
        feedforward = nn.Dense(
            config.feedforward_size,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            name="feedforward_in",
        )(hidden)
        feedforward = jax.nn.relu(feedforward)
        feedforward = nn.Dropout(config.dropout, deterministic=deterministic)(feedforward)
        feedforward = nn.Dense(
            config.hidden_size,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            name="feedforward_out",
        )(feedforward)
        return nn.LayerNorm(dtype=jnp.float32, name="feedforward_norm")(
            hidden + nn.Dropout(config.dropout, deterministic=deterministic)(feedforward)
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
        self_attention = nn.SelfAttention(
            num_heads=config.num_heads,
            qkv_features=config.hidden_size,
            out_features=config.hidden_size,
            dropout_rate=config.dropout,
            deterministic=deterministic,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            name="self_attention",
        )(hidden, mask=nn.combine_masks(causal_mask, padding_mask))
        hidden = nn.LayerNorm(dtype=jnp.float32, name="self_attention_norm")(
            hidden + nn.Dropout(config.dropout, deterministic=deterministic)(self_attention)
        ).astype(config.dtype)
        cross_attention = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            qkv_features=config.hidden_size,
            out_features=config.hidden_size,
            dropout_rate=config.dropout,
            deterministic=deterministic,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            name="cross_attention",
        )(
            hidden,
            memory,
            mask=_attention_mask(answer_valid, memory_valid),
        )
        hidden = nn.LayerNorm(dtype=jnp.float32, name="cross_attention_norm")(
            hidden + nn.Dropout(config.dropout, deterministic=deterministic)(cross_attention)
        ).astype(config.dtype)
        feedforward = nn.Dense(
            config.feedforward_size,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            name="feedforward_in",
        )(hidden)
        feedforward = jax.nn.relu(feedforward)
        feedforward = nn.Dropout(config.dropout, deterministic=deterministic)(feedforward)
        feedforward = nn.Dense(
            config.hidden_size,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            name="feedforward_out",
        )(feedforward)
        return nn.LayerNorm(dtype=jnp.float32, name="feedforward_norm")(
            hidden + nn.Dropout(config.dropout, deterministic=deterministic)(feedforward)
        ).astype(config.dtype)


class Sender(nn.Module):
    """Read one private passage and return fixed-length message vectors."""

    config: JaxCommunicationModelConfig

    @nn.compact
    def __call__(self, text_embeddings: Array, token_ids: Array, deterministic: bool) -> Array:
        """Encode text into message vectors of shape batch by message by hidden."""
        config = self.config
        batch_size = token_ids.shape[0]
        query_embedding = self.param(
            "query_embedding",
            nn.initializers.normal(stddev=1.0),
            (config.message_length, config.hidden_size),
            jnp.float32,
        ).astype(config.dtype)
        position_embedding = self.param(
            "position_embedding",
            nn.initializers.normal(stddev=1.0),
            (config.message_length + config.max_text_length, config.hidden_size),
            jnp.float32,
        ).astype(config.dtype)
        queries = jnp.broadcast_to(query_embedding[None], (batch_size, *query_embedding.shape))
        queries = queries + position_embedding[None, : config.message_length]
        text = text_embeddings + position_embedding[None, config.message_length :]
        hidden = jnp.concatenate((queries, text), axis=1)
        valid = jnp.concatenate(
            (
                jnp.ones((batch_size, config.message_length), dtype=jnp.bool_),
                token_ids != config.padding_id,
            ),
            axis=1,
        )
        for layer_index in range(config.num_layers):
            hidden = EncoderBlock(config, name=f"layer_{layer_index}")(hidden, valid, deterministic)
        return hidden[:, : config.message_length]


class Receiver(nn.Module):
    """Read two messages and a question, then score answer tokens."""

    config: JaxCommunicationModelConfig

    @nn.compact
    def __call__(
        self,
        messages: Array,
        question_embeddings: Array,
        question_token_ids: Array,
        answer_embeddings: Array,
        answer_input_ids: Array,
        deterministic: bool,
    ) -> Array:
        """Return answer logits for fixed-shape message, question, and answer inputs."""
        config = self.config
        batch_size = messages.shape[0]
        message_count = config.message_length * 2
        encoder_position_embedding = self.param(
            "encoder_position_embedding",
            nn.initializers.normal(stddev=1.0),
            (message_count + config.max_question_length, config.hidden_size),
            jnp.float32,
        ).astype(config.dtype)
        answer_position_embedding = self.param(
            "answer_position_embedding",
            nn.initializers.normal(stddev=1.0),
            (config.max_answer_length, config.hidden_size),
            jnp.float32,
        ).astype(config.dtype)
        messages = messages + encoder_position_embedding[None, :message_count]
        question_hidden = question_embeddings + encoder_position_embedding[None, message_count:]
        memory = jnp.concatenate((messages, question_hidden), axis=1)
        memory_valid = jnp.concatenate(
            (
                jnp.ones((batch_size, message_count), dtype=jnp.bool_),
                question_token_ids != config.padding_id,
            ),
            axis=1,
        )
        for layer_index in range(config.num_layers):
            memory = EncoderBlock(config, name=f"encoder_layer_{layer_index}")(
                memory, memory_valid, deterministic
            )
        hidden = answer_embeddings + answer_position_embedding[None]
        answer_valid = answer_input_ids != config.padding_id
        for layer_index in range(config.num_layers):
            hidden = DecoderBlock(config, name=f"decoder_layer_{layer_index}")(
                hidden,
                memory,
                answer_valid,
                memory_valid,
                deterministic,
            )
        return nn.Dense(
            config.vocab_size,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            name="output_projection",
        )(hidden)


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
        self.sender = Sender(self.config, name="sender")
        self.receiver = Receiver(self.config, name="receiver")

    def encode_messages(
        self,
        agent_1_token_ids: Array,
        agent_2_token_ids: Array,
        deterministic: bool,
    ) -> Array:
        """Encode both private passages with the shared sender."""
        message_1 = self.sender(self.token_embedding(agent_1_token_ids), agent_1_token_ids, deterministic)
        message_2 = self.sender(self.token_embedding(agent_2_token_ids), agent_2_token_ids, deterministic)
        return jnp.concatenate((message_1, message_2), axis=1)

    def answer_logits(
        self,
        messages: Array,
        question_token_ids: Array,
        answer_input_ids: Array,
        deterministic: bool,
    ) -> Array:
        """Score answer inputs from encoded messages and a question."""
        return self.receiver(
            messages,
            self.token_embedding(question_token_ids),
            question_token_ids,
            self.token_embedding(answer_input_ids),
            answer_input_ids,
            deterministic,
        )

    def __call__(
        self,
        agent_1_token_ids: Array,
        agent_2_token_ids: Array,
        question_token_ids: Array,
        answer_token_ids: Array,
        deterministic: bool = False,
    ) -> JaxCommunicationOutput:
        """Teacher-force answers and return their joint log probability."""
        batch_size = answer_token_ids.shape[0]
        beginning = jnp.full((batch_size, 1), self.config.bos_id, dtype=jnp.int32)
        answer_input_ids = jnp.concatenate((beginning, answer_token_ids[:, :-1]), axis=1)
        messages = self.encode_messages(agent_1_token_ids, agent_2_token_ids, deterministic)
        logits = self.answer_logits(messages, question_token_ids, answer_input_ids, deterministic)
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
