"""Shared sender and receiver for the linked-facts communication game.

The sender is a small Transformer encoder. Both agents use it, and each pass
returns 4 vectors of size 128. The receiver reads those 8 vectors plus the
question and scores the answer one token at a time. Attention Is All You Need:
https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class CommunicationModelConfig:
    """Sizes for the shared sender and the receiver."""

    vocab_size: int
    hidden_size: int = 128
    message_length: int = 4
    max_answer_length: int = 4
    max_text_length: int = 512
    max_question_length: int = 128
    max_context_length: int = 1024
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    padding_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    @property
    def feedforward_size(self) -> int:
        """Return the Transformer feedforward width, four times the hidden size."""
        return self.hidden_size * 4

    def validate(self) -> None:
        """Reject sizes that a Transformer block cannot use."""
        if self.vocab_size < 4:
            raise ValueError("vocab_size must include the special tokens")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        lengths = (self.message_length, self.max_answer_length, self.max_context_length, self.num_layers, self.num_heads)
        if min(lengths) < 1:
            raise ValueError("model lengths and depth must be positive")


class TextTokenizer:
    """Character tokenizer for passages, with one token per possible answer word."""

    def __init__(self, answer_strings: Sequence[str], max_answer_length: int = 4) -> None:
        """Reserve specials, characters, and every generable answer string."""
        self.max_answer_length = max_answer_length
        characters = set(string.ascii_letters + string.digits + " .,;:?'\"!-()/")
        tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, *sorted(characters)]
        for answer in sorted(set(answer_strings)):
            if answer not in tokens:
                tokens.append(answer)
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.id_to_token = tokens
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens, including specials and answer words."""
        return len(self.id_to_token)

    def encode_text(self, text: str) -> list[int]:
        """Encode a passage or question as character ids."""
        return [self.token_to_id.get(character, self.unk_id) for character in text]

    def encode_answer(self, answer: str) -> list[int]:
        """Encode one answer word and an end token, in at most four tokens."""
        if answer not in self.token_to_id:
            raise KeyError(f"answer is not in the vocabulary: {answer}")
        token_ids = [self.token_to_id[answer], self.eos_id]
        if len(token_ids) > self.max_answer_length:
            raise ValueError("answer encoding exceeds max_answer_length")
        return token_ids

    def decode_answer(self, token_ids: Sequence[int]) -> str:
        """Join generated answer tokens, stopping at the end token."""
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id == self.eos_id:
                break
            if token_id in (self.pad_id, self.bos_id):
                continue
            pieces.append(self.id_to_token[token_id])
        return "".join(pieces)


def build_task_tokenizer() -> TextTokenizer:
    """Build a tokenizer covering every name and lexicon answer the generator can emit."""
    from linked_facts.generate_dataset import build_name_banks
    from linked_facts.relation_domains import LEXICONS

    name_banks = build_name_banks(1)
    answer_strings = [
        name
        for variant in name_banks.values()
        for role_names in variant.values()
        for name in role_names
    ]
    answer_strings.extend(
        word for variants in LEXICONS.values() for words in variants.values() for word in words
    )
    return TextTokenizer(answer_strings)


def collate_token_ids(
    sequences: Sequence[Sequence[int]],
    pad_id: int,
    padded_length: int | None = None,
) -> torch.Tensor:
    """Pad token-id sequences into a batch of shape (batch, length)."""
    if not sequences:
        raise ValueError("cannot collate an empty batch")
    length = max(len(sequence) for sequence in sequences) if padded_length is None else padded_length
    rows: list[list[int]] = []
    for sequence in sequences:
        if len(sequence) > length:
            raise ValueError("sequence is longer than the padded length")
        rows.append([*sequence, *([pad_id] * (length - len(sequence)))])
    return torch.tensor(rows, dtype=torch.long)


def joint_answer_log_probability(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    padding_id: int,
) -> torch.Tensor:
    """Return the summed answer-token log probability for each batch row."""
    assert isinstance(logits, torch.Tensor)
    assert isinstance(target_token_ids, torch.Tensor)
    batch_size, answer_length, vocab_size = logits.shape
    token_negative_log_probability = functional.cross_entropy(
        logits.reshape(batch_size * answer_length, vocab_size),
        target_token_ids.reshape(batch_size * answer_length),
        ignore_index=padding_id,
        reduction="none",
    )
    return -token_negative_log_probability.reshape(batch_size, answer_length).sum(dim=-1)


def _encoder_layer(config: CommunicationModelConfig) -> nn.TransformerEncoderLayer:
    """Build one encoder layer with the library's default initialization."""
    return nn.TransformerEncoderLayer(
        d_model=config.hidden_size,
        nhead=config.num_heads,
        dim_feedforward=config.feedforward_size,
        dropout=config.dropout,
        batch_first=True,
    )


def _decoder_layer(config: CommunicationModelConfig) -> nn.TransformerDecoderLayer:
    """Build one decoder layer with the library's default initialization."""
    return nn.TransformerDecoderLayer(
        d_model=config.hidden_size,
        nhead=config.num_heads,
        dim_feedforward=config.feedforward_size,
        dropout=config.dropout,
        batch_first=True,
    )


def _transformer_encoder(layer: nn.TransformerEncoderLayer, num_layers: int) -> nn.TransformerEncoder:
    """Stack encoder layers without the nested-tensor fast path."""
    return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Return a boolean mask that hides future answer tokens."""
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


def shift_answer_inputs(answer_token_ids: torch.Tensor, beginning_id: int) -> torch.Tensor:
    """Shift answer targets right and start the decoder at the beginning token."""
    batch_size, answer_length = answer_token_ids.shape
    beginning = torch.full((batch_size, 1), beginning_id, dtype=torch.long, device=answer_token_ids.device)
    if answer_length == 1:
        return beginning
    return torch.cat([beginning, answer_token_ids[:, :-1]], dim=1)


def generate_answer_token_ids(
    predict_logits: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
    max_answer_length: int,
    beginning_id: int,
    end_id: int,
    padding_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate at most max_answer_length tokens, stopping at the end token."""
    generated = torch.full((batch_size, 1), beginning_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    steps: list[torch.Tensor] = []
    for _ in range(max_answer_length):
        next_token = predict_logits(generated)[:, -1].argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, padding_id), next_token)
        steps.append(next_token)
        finished = finished | next_token.eq(end_id)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        if bool(finished.all()):
            break
    return torch.stack(steps, dim=1)


def count_owned_parameters(module: nn.Module, shared_module: nn.Module | None = None) -> int:
    """Count parameters in a module, skipping a shared submodule when given."""
    shared_ids = set() if shared_module is None else {id(parameter) for parameter in shared_module.parameters()}
    return sum(parameter.numel() for parameter in module.parameters() if id(parameter) not in shared_ids)


def _positions(batch_size: int, length: int, offset: int, device: torch.device) -> torch.Tensor:
    """Return position ids of shape (batch, length), shifted by offset."""
    positions = torch.arange(offset, offset + length, device=device)
    return positions.unsqueeze(0).expand(batch_size, -1)


class Sender(nn.Module):
    """Read one private passage and return four hidden states."""

    def __init__(self, config: CommunicationModelConfig, token_embedding: nn.Embedding) -> None:
        """Attach the shared text embedding and a four-slot readout."""
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = token_embedding
        self.position_embedding = nn.Embedding(config.message_length + config.max_text_length, config.hidden_size)
        self.query_embedding = nn.Embedding(config.message_length, config.hidden_size)
        self.transformer = _transformer_encoder(_encoder_layer(config), config.num_layers)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode text and return message vectors of shape (batch, 4, hidden)."""
        assert isinstance(token_ids, torch.Tensor)
        batch_size, text_length = token_ids.shape
        if text_length < 1 or text_length > self.config.max_text_length:
            raise ValueError("sender text length is outside the position table")
        device = token_ids.device
        query_ids = torch.arange(self.config.message_length, device=device).unsqueeze(0).expand(batch_size, -1)
        query_positions = _positions(batch_size, self.config.message_length, 0, device)
        queries = self.query_embedding(query_ids) + self.position_embedding(query_positions)
        text = self.token_embedding(token_ids) + self.position_embedding(
            _positions(batch_size, text_length, self.config.message_length, device)
        )
        hidden = torch.cat([queries, text], dim=1)
        query_padding = torch.zeros(batch_size, self.config.message_length, dtype=torch.bool, device=device)
        text_padding = token_ids.eq(self.config.padding_id)
        encoded = self.transformer(hidden, src_key_padding_mask=torch.cat([query_padding, text_padding], dim=1))
        return encoded[:, : self.config.message_length]


class Receiver(nn.Module):
    """Read eight message vectors plus a question and score answer tokens."""

    def __init__(self, config: CommunicationModelConfig, token_embedding: nn.Embedding) -> None:
        """Build the encoder-decoder that consumes messages and the question."""
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = token_embedding
        encoder_positions = config.message_length * 2 + config.max_question_length
        self.position_embedding = nn.Embedding(encoder_positions, config.hidden_size)
        self.answer_position_embedding = nn.Embedding(config.max_answer_length, config.hidden_size)
        self.encoder = _transformer_encoder(_encoder_layer(config), config.num_layers)
        self.decoder = nn.TransformerDecoder(_decoder_layer(config), num_layers=config.num_layers)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        messages: torch.Tensor,
        question_token_ids: torch.Tensor,
        answer_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return answer-token logits of shape (batch, answer length, vocab)."""
        logits, _, _ = self.forward_with_states(messages, question_token_ids, answer_input_ids)
        return logits

    def forward_with_states(
        self,
        messages: torch.Tensor,
        question_token_ids: torch.Tensor,
        answer_input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return answer logits, decoder states, and encoder memory."""
        assert isinstance(messages, torch.Tensor)
        assert isinstance(question_token_ids, torch.Tensor)
        assert isinstance(answer_input_ids, torch.Tensor)
        batch_size, message_length, hidden_size = messages.shape
        question_length = question_token_ids.shape[1]
        answer_length = answer_input_ids.shape[1]
        expected_messages = self.config.message_length * 2
        if message_length != expected_messages or hidden_size != self.config.hidden_size:
            raise ValueError("receiver expected eight message vectors of the hidden size")
        if question_length < 1 or question_length > self.config.max_question_length:
            raise ValueError("question length is outside the position table")
        if answer_length < 1 or answer_length > self.config.max_answer_length:
            raise ValueError("answer decoder length is outside the configured maximum")
        device = messages.device
        message_hidden = messages + self.position_embedding(_positions(batch_size, message_length, 0, device))
        question_hidden = self.token_embedding(question_token_ids) + self.position_embedding(
            _positions(batch_size, question_length, message_length, device)
        )
        encoder_hidden = torch.cat([message_hidden, question_hidden], dim=1)
        message_padding = torch.zeros(batch_size, message_length, dtype=torch.bool, device=device)
        encoder_padding = torch.cat([message_padding, question_token_ids.eq(self.config.padding_id)], dim=1)
        memory = self.encoder(encoder_hidden, src_key_padding_mask=encoder_padding)
        answer_hidden = self.token_embedding(answer_input_ids) + self.answer_position_embedding(
            _positions(batch_size, answer_length, 0, device)
        )
        decoded = self.decoder(
            answer_hidden,
            memory,
            tgt_mask=_causal_mask(answer_length, device),
            tgt_key_padding_mask=answer_input_ids.eq(self.config.padding_id),
            memory_key_padding_mask=encoder_padding,
        )
        return self.output_projection(decoded), decoded, memory


@dataclass(frozen=True)
class CommunicationOutput:
    """Receiver logits, joint answer log probability, and activations used for logging."""

    logits: torch.Tensor
    joint_log_probability: torch.Tensor
    loss: torch.Tensor
    message_1: torch.Tensor | None = None
    message_2: torch.Tensor | None = None
    answer_hidden: torch.Tensor | None = None
    receiver_memory: torch.Tensor | None = None


class CommunicationModel(nn.Module):
    """Shared sender for both agents, plus the receiver that scores the answer."""

    def __init__(self, config: CommunicationModelConfig) -> None:
        """Create the shared text embedding, sender, and receiver."""
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.padding_id)
        self.sender = Sender(config, self.token_embedding)
        self.receiver = Receiver(config, self.token_embedding)

    def encode_messages(self, agent_1_token_ids: torch.Tensor, agent_2_token_ids: torch.Tensor) -> torch.Tensor:
        """Run the shared sender on both passages and concatenate the eight vectors."""
        message_1 = self.sender(agent_1_token_ids)
        message_2 = self.sender(agent_2_token_ids)
        return torch.cat([message_1, message_2], dim=1)

    def forward(
        self,
        agent_1_token_ids: torch.Tensor,
        agent_2_token_ids: torch.Tensor,
        question_token_ids: torch.Tensor,
        answer_token_ids: torch.Tensor,
    ) -> CommunicationOutput:
        """Teacher-force the answer and return its joint log probability."""
        assert isinstance(answer_token_ids, torch.Tensor)
        if answer_token_ids.shape[1] > self.config.max_answer_length:
            raise ValueError("answer targets are longer than 4 tokens")
        message_1 = self.sender(agent_1_token_ids)
        message_2 = self.sender(agent_2_token_ids)
        messages = torch.cat([message_1, message_2], dim=1)
        answer_inputs = shift_answer_inputs(answer_token_ids, self.config.bos_id)
        logits, answer_hidden, receiver_memory = self.receiver.forward_with_states(
            messages, question_token_ids, answer_inputs
        )
        joint_log_probability = joint_answer_log_probability(logits, answer_token_ids, self.config.padding_id)
        return CommunicationOutput(
            logits=logits,
            joint_log_probability=joint_log_probability,
            loss=-joint_log_probability.mean(),
            message_1=message_1,
            message_2=message_2,
            answer_hidden=answer_hidden,
            receiver_memory=receiver_memory,
        )

    def greedy_answer_token_ids(
        self,
        agent_1_token_ids: torch.Tensor,
        agent_2_token_ids: torch.Tensor,
        question_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Generate at most four answer tokens, stopping at the end token."""
        messages = self.encode_messages(agent_1_token_ids, agent_2_token_ids)
        return generate_answer_token_ids(
            lambda prefix: self.receiver(messages, question_token_ids, prefix),
            messages.shape[0],
            self.config.max_answer_length,
            self.config.bos_id,
            self.config.eos_id,
            self.config.padding_id,
            messages.device,
        )


def _config_from_tokenizer(tokenizer: TextTokenizer, config_overrides: dict[str, Any]) -> CommunicationModelConfig:
    """Build a model config whose vocabulary matches the tokenizer."""
    config_values: dict[str, Any] = {
        "vocab_size": tokenizer.vocab_size,
        "padding_id": tokenizer.pad_id,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "max_answer_length": tokenizer.max_answer_length,
    }
    config_values.update(config_overrides)
    return CommunicationModelConfig(**config_values)


def build_model(tokenizer: TextTokenizer | None = None, **config_overrides: Any) -> tuple[CommunicationModel, TextTokenizer]:
    """Construct the communication model and the tokenizer it reads."""
    task_tokenizer = build_task_tokenizer() if tokenizer is None else tokenizer
    return CommunicationModel(_config_from_tokenizer(task_tokenizer, config_overrides)), task_tokenizer
