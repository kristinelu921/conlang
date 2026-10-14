"""Full-context control transformer for the linked-facts task.

The control model reads both passages and the question as one sequence and
scores the answer. It uses the same joint log probability as the communication
model. Attention Is All You Need: https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models import (
    CommunicationModelConfig,
    CommunicationOutput,
    TextTokenizer,
    _causal_mask,
    _config_from_tokenizer,
    _decoder_layer,
    _encoder_layer,
    _positions,
    _transformer_encoder,
    build_task_tokenizer,
    generate_answer_token_ids,
    joint_answer_log_probability,
    shift_answer_inputs,
)


def join_full_context(agent_1_input: str, agent_2_input: str, question: str) -> str:
    """Join both private passages and the question into one character sequence."""
    return f"{agent_1_input} {agent_2_input} {question}"


class ControlModel(nn.Module):
    """Read both passages and the question, then score the answer tokens."""

    def __init__(self, config: CommunicationModelConfig) -> None:
        """Build one encoder-decoder over the full character context."""
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.padding_id)
        self.position_embedding = nn.Embedding(config.max_context_length, config.hidden_size)
        self.answer_position_embedding = nn.Embedding(config.max_answer_length, config.hidden_size)
        self.encoder = _transformer_encoder(_encoder_layer(config), config.num_layers)
        self.decoder = nn.TransformerDecoder(_decoder_layer(config), num_layers=config.num_layers)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, context_token_ids: torch.Tensor, answer_token_ids: torch.Tensor) -> CommunicationOutput:
        """Teacher-force the answer from the full context and return its joint log probability."""
        assert isinstance(context_token_ids, torch.Tensor)
        assert isinstance(answer_token_ids, torch.Tensor)
        if answer_token_ids.shape[1] > self.config.max_answer_length:
            raise ValueError("answer targets are longer than 4 tokens")
        answer_inputs = shift_answer_inputs(answer_token_ids, self.config.bos_id)
        logits, answer_hidden, encoder_memory = self._decode_with_states(context_token_ids, answer_inputs)
        joint_log_probability = joint_answer_log_probability(logits, answer_token_ids, self.config.padding_id)
        return CommunicationOutput(
            logits=logits,
            joint_log_probability=joint_log_probability,
            loss=-joint_log_probability.mean(),
            answer_hidden=answer_hidden,
            receiver_memory=encoder_memory,
        )

    def greedy_answer_token_ids(self, context_token_ids: torch.Tensor) -> torch.Tensor:
        """Generate at most four answer tokens from the full context."""
        return generate_answer_token_ids(
            lambda prefix: self._decode(context_token_ids, prefix),
            context_token_ids.shape[0],
            self.config.max_answer_length,
            self.config.bos_id,
            self.config.eos_id,
            self.config.padding_id,
            context_token_ids.device,
        )

    def _decode(self, context_token_ids: torch.Tensor, answer_input_ids: torch.Tensor) -> torch.Tensor:
        """Return answer logits for one full-context sequence."""
        logits, _, _ = self._decode_with_states(context_token_ids, answer_input_ids)
        return logits

    def _decode_with_states(
        self, context_token_ids: torch.Tensor, answer_input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return answer logits, decoder states, and encoder memory."""
        batch_size, context_length = context_token_ids.shape
        answer_length = answer_input_ids.shape[1]
        if context_length < 1 or context_length > self.config.max_context_length:
            raise ValueError("full context length is outside the position table")
        if answer_length < 1 or answer_length > self.config.max_answer_length:
            raise ValueError("answer decoder length is outside the configured maximum")
        device = context_token_ids.device
        context_hidden = self.token_embedding(context_token_ids) + self.position_embedding(
            _positions(batch_size, context_length, 0, device)
        )
        context_padding = context_token_ids.eq(self.config.padding_id)
        memory = self.encoder(context_hidden, src_key_padding_mask=context_padding)
        answer_hidden = self.token_embedding(answer_input_ids) + self.answer_position_embedding(
            _positions(batch_size, answer_length, 0, device)
        )
        decoded = self.decoder(
            answer_hidden,
            memory,
            tgt_mask=_causal_mask(answer_length, device),
            tgt_key_padding_mask=answer_input_ids.eq(self.config.padding_id),
            memory_key_padding_mask=context_padding,
        )
        return self.output_projection(decoded), decoded, memory


def build_control_model(tokenizer: TextTokenizer | None = None, **config_overrides: Any) -> tuple[ControlModel, TextTokenizer]:
    """Construct the control transformer at about twice the 128-wide parameter count."""
    task_tokenizer = build_task_tokenizer() if tokenizer is None else tokenizer
    # Width 192 and 6 heads keep 32-wide heads. A hidden size of 256 is about 3.1 times the parameters.
    control_config = {"hidden_size": 192, "num_heads": 6}
    control_config.update(config_overrides)
    return ControlModel(_config_from_tokenizer(task_tokenizer, control_config)), task_tokenizer
