"""Character tokenizer for passages, with one token per answer word."""

from __future__ import annotations

import string
from collections.abc import Sequence


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


class TextTokenizer:
    """Map passages to characters and each answer word to one vocabulary item."""

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
    answer_strings.extend(word for variants in LEXICONS.values() for words in variants.values() for word in words)
    return TextTokenizer(answer_strings)
