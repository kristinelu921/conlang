"""Shared dataset loading for the linked-facts trainers."""

from __future__ import annotations

from pathlib import Path

from linked_facts.generate_dataset import iter_question_examples


def resolve_split_path(dataset_directory: Path, split: str) -> Path:
    """Find the gzip or plain JSONL file for one dataset split."""
    for suffix in (".jsonl.gz", ".jsonl"):
        candidate = dataset_directory / f"{split}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing {split} split under {dataset_directory}")


def load_examples(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """Load flattened question examples from one split file."""
    loaded: list[dict[str, str]] = []
    for example in iter_question_examples(path):
        loaded.append(
            {
                "agent_1_input": str(example["agent_1_input"]),
                "agent_2_input": str(example["agent_2_input"]),
                "question": str(example["question"]),
                "answer": str(example["answer"]),
                "bridge": str(example["bridge"]),
                "person_side_agent": str(example["person_side_agent"]),
            }
        )
        if limit is not None and len(loaded) >= limit:
            break
    if not loaded:
        raise ValueError(f"no question examples in {path}")
    return loaded
