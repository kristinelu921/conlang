"""Generate natural-language worlds for a shared-encoder communication game."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO, TypeVar

from linked_facts.relation_domains import DOMAINS_BY_ID, LEXICONS, RELATION_DOMAINS


SCHEMA_VERSION = 3

SampledItem = TypeVar("SampledItem")

SPLIT_SPECIFICATIONS = {
    "train": ("familiar", "familiar"),
    "validation": ("familiar", "familiar"),
    "test_familiar": ("familiar", "familiar"),
    "test_new_names": ("held_out", "familiar"),
    "test_new_wording": ("familiar", "held_out"),
    "test_new_names_and_wording": ("held_out", "held_out"),
}

PRESETS = {
    "smoke": {"train": 20, "validation": 4, "test": 4},
    "starter": {"train": 12_500, "validation": 625, "test": 625},
    "large": {"train": 125_000, "validation": 6_250, "test": 6_250},
}


@dataclass(frozen=True)
class GenerationConfig:
    """Configure deterministic linked-facts dataset generation."""

    seed: int = 1_729
    entities_per_world: int = 8
    train_worlds: int = 12_500
    validation_worlds: int = 625
    test_worlds_per_split: int = 625
    compression: str = "gzip"

    def world_count(self, split: str) -> int:
        """Return the configured number of worlds for one split."""
        if split == "train":
            return self.train_worlds
        if split == "validation":
            return self.validation_worlds
        return self.test_worlds_per_split

    def validate(self) -> None:
        """Reject invalid generation settings before creating files."""
        if not 2 <= self.entities_per_world <= 32:
            raise ValueError("entities_per_world must be between 2 and 32")
        if min(self.train_worlds, self.validation_worlds, self.test_worlds_per_split) < 1:
            raise ValueError("every split must contain at least one world")
        if self.compression not in {"gzip", "none"}:
            raise ValueError("compression must be 'gzip' or 'none'")


def _derive_seed(seed: int, *parts: object) -> int:
    """Derive a stable independent seed from a root seed and labels."""
    payload = ":".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _make_name_candidates() -> list[str]:
    """Create a large bank of pronounceable invented proper nouns."""
    beginnings = (
        "b",
        "d",
        "f",
        "g",
        "k",
        "l",
        "m",
        "n",
        "p",
        "r",
        "s",
        "t",
        "v",
        "z",
        "br",
        "dr",
        "gl",
        "kr",
        "pr",
        "tr",
    )
    middles = ("a", "e", "i", "o", "u", "ai", "ea", "io")
    endings = ("la", "len", "mi", "na", "nel", "ra", "ren", "ri", "sa", "sen", "ta", "va")
    return [f"{beginning}{middle}{ending}".capitalize() for beginning in beginnings for middle in middles for ending in endings]


def build_name_banks(seed: int) -> dict[str, dict[str, tuple[str, ...]]]:
    """Build disjoint familiar and held-out name banks for all entity roles."""
    candidates = _make_name_candidates()
    random.Random(_derive_seed(seed, "name-banks")).shuffle(candidates)
    bank_size = len(candidates) // 6
    role_names = ("persons", "groups", "locations")
    banks: dict[str, dict[str, tuple[str, ...]]] = {"familiar": {}, "held_out": {}}
    for role_index, role_name in enumerate(role_names):
        familiar_start = role_index * bank_size * 2
        held_out_start = familiar_start + bank_size
        banks["familiar"][role_name] = tuple(candidates[familiar_start:held_out_start])
        banks["held_out"][role_name] = tuple(candidates[held_out_start : held_out_start + bank_size])
    return banks


def _cycle_sample(items: Sequence[SampledItem], count: int, random_generator: random.Random) -> list[SampledItem]:
    """Sample items so each one is used once before any item repeats."""
    if count == 0:
        return []
    if not items:
        raise ValueError("cannot sample from an empty bank")
    pool = list(items)
    chosen: list[SampledItem] = []
    while len(chosen) < count:
        random_generator.shuffle(pool)
        chosen.extend(pool[: count - len(chosen)])
    return chosen


def _draw(pool: list[str], count: int, random_generator: random.Random) -> list[str]:
    """Draw distinct strings and remove them from a shared pool."""
    if count == 0:
        return []
    chosen = random_generator.sample(pool, count)
    chosen_set = set(chosen)
    pool[:] = [item for item in pool if item not in chosen_set]
    return chosen


def _render_fact_sentences(
    facts_by_domain: Mapping[str, Sequence[Mapping[str, str]]],
    side: str,
    wording_variant: str,
    random_generator: random.Random,
) -> dict[str, tuple[str, str]]:
    """Render one hop of every chain, keyed by person."""
    rendered: dict[str, tuple[str, str]] = {}
    for domain in RELATION_DOMAINS:
        facts = facts_by_domain[domain.domain_id]
        if side == "agent_1":
            templates = domain.agent_1_templates[wording_variant]
        else:
            templates = domain.agent_2_templates[wording_variant]
        chosen_templates = _cycle_sample(templates, len(facts), random_generator)
        for fact, (template_id, template) in zip(facts, chosen_templates, strict=True):
            if side == "agent_1":
                arguments = domain.agent_1_arguments(fact["person"], fact["middle"], fact["kind"])
            else:
                arguments = domain.agent_2_arguments(fact["middle"], fact["kind"], fact["answer"])
            rendered[fact["person"]] = (template.format(**arguments), template_id)
    return rendered


def _mix_sender_facts(
    person_side: Mapping[str, tuple[str, str]],
    answer_side: Mapping[str, tuple[str, str]],
    random_generator: random.Random,
) -> tuple[str, list[str], str, list[str], dict[str, str]]:
    """Give each sender a balanced mix of the two hops, split per person."""
    persons = list(person_side)
    random_generator.shuffle(persons)
    agent_1_person_side = set(persons[: len(persons) // 2])
    agent_1_facts: list[tuple[str, str]] = []
    agent_2_facts: list[tuple[str, str]] = []
    assignment: dict[str, str] = {}
    for person in persons:
        if person in agent_1_person_side:
            agent_1_facts.append(person_side[person])
            agent_2_facts.append(answer_side[person])
            assignment[person] = "agent_1"
        else:
            agent_1_facts.append(answer_side[person])
            agent_2_facts.append(person_side[person])
            assignment[person] = "agent_2"
    random_generator.shuffle(agent_1_facts)
    random_generator.shuffle(agent_2_facts)
    agent_1_input = " ".join(text for text, _ in agent_1_facts)
    agent_2_input = " ".join(text for text, _ in agent_2_facts)
    return (
        agent_1_input,
        [template_id for _, template_id in agent_1_facts],
        agent_2_input,
        [template_id for _, template_id in agent_2_facts],
        assignment,
    )


def generate_world(
    split: str,
    world_index: int,
    config: GenerationConfig,
    name_banks: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> dict[str, Any]:
    """Generate one world containing two private inputs and all linked questions."""
    config.validate()
    if split not in SPLIT_SPECIFICATIONS:
        raise ValueError(f"unknown split: {split}")
    names_variant, wording_variant = SPLIT_SPECIFICATIONS[split]
    available_name_banks = build_name_banks(config.seed) if name_banks is None else name_banks
    selected_banks = available_name_banks[names_variant]
    world_seed = _derive_seed(config.seed, split, world_index)
    random_generator = random.Random(world_seed)
    persons = random_generator.sample(tuple(selected_banks["persons"]), config.entities_per_world)
    domain_sequence = _cycle_sample([domain.domain_id for domain in RELATION_DOMAINS], len(persons), random_generator)
    remaining = {
        "groups": list(selected_banks["groups"]),
        "locations": list(selected_banks["locations"]),
    }
    for source, variants in LEXICONS.items():
        remaining[source] = list(variants[names_variant])
    domain_counts = Counter(domain_sequence)
    sampled_middles = {
        domain.domain_id: _draw(remaining[domain.middle_source], domain_counts[domain.domain_id], random_generator)
        for domain in RELATION_DOMAINS
    }
    sampled_answers = {
        domain.domain_id: _draw(remaining[domain.answer_source], domain_counts[domain.domain_id], random_generator)
        for domain in RELATION_DOMAINS
    }

    facts_by_domain: dict[str, list[dict[str, str]]] = {domain.domain_id: [] for domain in RELATION_DOMAINS}
    chains: dict[str, dict[str, str]] = {}
    domain_offsets = {domain.domain_id: 0 for domain in RELATION_DOMAINS}
    for person, domain_id in zip(persons, domain_sequence, strict=True):
        domain = DOMAINS_BY_ID[domain_id]
        offset = domain_offsets[domain_id]
        middle = sampled_middles[domain_id][offset]
        answer = sampled_answers[domain_id][offset]
        domain_offsets[domain_id] = offset + 1
        kind = random_generator.choice(domain.kinds) if domain.kinds else ""
        fact = {"person": person, "middle": middle, "kind": kind, "answer": answer, "domain": domain_id}
        facts_by_domain[domain_id].append(fact)
        chains[person] = fact

    person_side = _render_fact_sentences(facts_by_domain, "agent_1", wording_variant, random_generator)
    answer_side = _render_fact_sentences(facts_by_domain, "agent_2", wording_variant, random_generator)
    agent_1_input, agent_1_template_ids, agent_2_input, agent_2_template_ids, person_side_agent = _mix_sender_facts(
        person_side, answer_side, random_generator
    )

    question_persons = random_generator.sample(persons, len(persons))
    question_templates_by_person: dict[str, tuple[str, str]] = {}
    for domain in RELATION_DOMAINS:
        domain_persons = [person for person in question_persons if chains[person]["domain"] == domain.domain_id]
        chosen_question_templates = _cycle_sample(
            domain.question_templates[wording_variant], len(domain_persons), random_generator
        )
        for person, template in zip(domain_persons, chosen_question_templates, strict=True):
            question_templates_by_person[person] = template

    domain_answers = {
        domain.domain_id: [fact["answer"] for fact in facts_by_domain[domain.domain_id]]
        for domain in RELATION_DOMAINS
        if facts_by_domain[domain.domain_id]
    }
    questions: list[dict[str, Any]] = []
    for question_index, person in enumerate(question_persons):
        fact = chains[person]
        template_id, template = question_templates_by_person[person]
        answer = fact["answer"]
        questions.append(
            {
                "question_id": f"{split}-{world_index:08d}-q{question_index:02d}",
                "domain": fact["domain"],
                "question": template.format(**DOMAINS_BY_ID[fact["domain"]].question_arguments(person)),
                "answer": answer,
                "answer_index": domain_answers[fact["domain"]].index(answer),
                "template_id": template_id,
                "reasoning_path": [person, fact["middle"], answer],
            }
        )

    entities = {
        "persons": persons,
        "groups": [fact["middle"] for fact in facts_by_domain["membership"] + facts_by_domain["pledged_group"]],
        "locations": [fact["answer"] for fact in facts_by_domain["membership"] + facts_by_domain["pledged_group"]],
        "colors": [fact["middle"] for fact in facts_by_domain["color"]],
        "animals": [fact["answer"] for fact in facts_by_domain["color"]]
        + [fact["middle"] for fact in facts_by_domain["animal"]],
        "furniture": [fact["answer"] for fact in facts_by_domain["animal"]]
        + [fact["middle"] for fact in facts_by_domain["furniture"]],
        "rooms": [fact["answer"] for fact in facts_by_domain["furniture"]],
    }
    world = {
        "schema_version": SCHEMA_VERSION,
        "world_id": f"{split}-{world_index:08d}",
        "split": split,
        "generalization": {"names": names_variant, "wording": wording_variant},
        "agent_1_input": agent_1_input,
        "agent_2_input": agent_2_input,
        "questions": questions,
        "metadata": {
            "world_seed": world_seed,
            "entities": entities,
            "chains": chains,
            "domain_answers": domain_answers,
            "agent_1_template_ids": agent_1_template_ids,
            "agent_2_template_ids": agent_2_template_ids,
            "person_side_agent": person_side_agent,
        },
    }
    validate_world(world, config.entities_per_world)
    return world


def validate_world(world: Mapping[str, Any], expected_entities: int | None = None) -> None:
    """Validate answer chains and structural invariants for one generated world."""
    metadata = world["metadata"]
    assert isinstance(metadata, Mapping)
    chains = metadata["chains"]
    domain_answers = metadata["domain_answers"]
    questions = world["questions"]
    agent_1_input = world["agent_1_input"]
    agent_2_input = world["agent_2_input"]
    assert isinstance(chains, Mapping)
    assert isinstance(domain_answers, Mapping)
    assert isinstance(questions, Sequence)
    assert isinstance(agent_1_input, str)
    assert isinstance(agent_2_input, str)
    if expected_entities is not None and len(chains) != expected_entities:
        raise ValueError(f"expected {expected_entities} entities, found {len(chains)}")
    if len(questions) != len(chains):
        raise ValueError("each world must have one question per person")
    template_ids = metadata["agent_1_template_ids"]
    assert isinstance(template_ids, Sequence)
    agent_2_template_ids = metadata["agent_2_template_ids"]
    assert isinstance(agent_2_template_ids, Sequence)
    if len(template_ids) != len(chains) or len(agent_2_template_ids) != len(chains):
        raise ValueError("each person must contribute one fact to each sender")

    middles_by_domain: dict[str, set[str]] = {}
    answers_by_domain: dict[str, set[str]] = {}
    for person, chain in chains.items():
        assert isinstance(chain, Mapping)
        domain = DOMAINS_BY_ID[str(chain["domain"])]
        middle = str(chain["middle"])
        answer = str(chain["answer"])
        kind = str(chain["kind"])
        if middle in middles_by_domain.setdefault(domain.domain_id, set()):
            raise ValueError(f"domain {domain.domain_id} middle mappings must be bijections")
        if answer in answers_by_domain.setdefault(domain.domain_id, set()):
            raise ValueError(f"domain {domain.domain_id} answer mappings must be bijections")
        middles_by_domain[domain.domain_id].add(middle)
        answers_by_domain[domain.domain_id].add(answer)
        person_in_agent_1 = person in agent_1_input
        answer_in_agent_1 = answer in agent_1_input
        if person_in_agent_1 == (person in agent_2_input) or answer_in_agent_1 == (answer in agent_2_input):
            raise ValueError(f"each hop must reach exactly one sender for {person}")
        if person_in_agent_1 == answer_in_agent_1:
            raise ValueError(f"one sender holds both hops for {person}")
        if middle not in agent_1_input or middle not in agent_2_input:
            raise ValueError(f"middle missing from a sender for {person}")
        if bool(domain.kinds) != bool(kind):
            raise ValueError(f"kind does not match domain {domain.domain_id}")
        if kind and (kind not in domain.kinds or kind not in agent_1_input or kind not in agent_2_input):
            raise ValueError(f"group kind missing from inputs for {person}")
    person_side_counts = Counter(
        person in agent_1_input for person in chains
    )
    if max(person_side_counts.values()) - min(person_side_counts.values()) > 1:
        raise ValueError("person-side facts must be mixed across the two senders")
    observed_domains = Counter(str(chain["domain"]) for chain in chains.values())
    domain_counts = [observed_domains[domain.domain_id] for domain in RELATION_DOMAINS]
    if max(domain_counts) - min(domain_counts) > 1:
        raise ValueError("relation domains must be balanced across a world")

    observed_question_people: set[str] = set()
    for question in questions:
        assert isinstance(question, Mapping)
        reasoning_path = question["reasoning_path"]
        assert isinstance(reasoning_path, Sequence)
        person, middle, answer = reasoning_path
        chain = chains[person]
        assert isinstance(chain, Mapping)
        if chain["middle"] != middle or chain["answer"] != answer or chain["domain"] != question["domain"]:
            raise ValueError(f"invalid reasoning path in {question['question_id']}")
        answers = domain_answers[question["domain"]]
        assert isinstance(answers, Sequence)
        if question["answer"] != answer or question["answer_index"] != list(answers).index(answer):
            raise ValueError(f"invalid answer in {question['question_id']}")
        observed_question_people.add(str(person))
    if observed_question_people != set(chains):
        raise ValueError("questions must cover every person exactly once")


def _open_text(path: Path, mode: str) -> TextIO:
    """Open plain or gzip-compressed UTF-8 text based on the path suffix."""
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _write_split(
    output_directory: Path,
    split: str,
    config: GenerationConfig,
    name_banks: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, Any]:
    """Write one dataset split atomically and return its manifest entry."""
    suffix = ".jsonl.gz" if config.compression == "gzip" else ".jsonl"
    output_path = output_directory / f"{split}{suffix}"
    checksum = hashlib.sha256()
    world_count = config.world_count(split)
    temporary_file = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{split}-",
        suffix=suffix,
        dir=output_directory,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        with _open_text(temporary_path, "wt") as output_file:
            for world_index in range(world_count):
                world = generate_world(split, world_index, config, name_banks)
                serialized_world = json.dumps(world, ensure_ascii=False, separators=(",", ":")) + "\n"
                output_file.write(serialized_world)
                checksum.update(serialized_world.encode("utf-8"))
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return {
        "path": output_path.name,
        "worlds": world_count,
        "questions": world_count * config.entities_per_world,
        "uncompressed_sha256": checksum.hexdigest(),
        "bytes_on_disk": output_path.stat().st_size,
    }


def build_dataset(output_directory: Path, config: GenerationConfig) -> dict[str, Any]:
    """Generate every split and write a reproducibility manifest."""
    config.validate()
    output_directory.mkdir(parents=True, exist_ok=True)
    name_banks = build_name_banks(config.seed)
    split_entries = {
        split: _write_split(output_directory, split, config, name_banks)
        for split in SPLIT_SPECIFICATIONS
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "linked_facts_natural_language",
        "config": asdict(config),
        "splits": split_entries,
        "total_worlds": sum(entry["worlds"] for entry in split_entries.values()),
        "total_questions": sum(entry["questions"] for entry in split_entries.values()),
        "split_contract": {
            split: {"names": names_variant, "wording": wording_variant}
            for split, (names_variant, wording_variant) in SPLIT_SPECIFICATIONS.items()
        },
        "model_inputs": {
            "shared_encoder_call_1": "agent_1_input",
            "shared_encoder_call_2": "agent_2_input",
            "receiver_inputs": ["questions[].question", "message_1", "message_2"],
            "training_target": "questions[].answer",
            "optional_auxiliary_targets": [
                "questions[].reasoning_path[1]",
                "metadata.person_side_agent",
            ],
            "excluded_metadata": [
                "metadata except person_side_agent auxiliary label",
                "questions[].domain",
                "questions[].reasoning_path except bridge auxiliary label",
                "questions[].answer_index",
            ],
        },
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_dataset(output_directory: Path) -> dict[str, int]:
    """Verify every stored world, count, byte size, and content checksum."""
    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    manifest_splits = manifest["splits"]
    assert isinstance(manifest_splits, dict)
    seen_world_ids: set[str] = set()
    verified_worlds = 0
    verified_questions = 0
    verified_questions_per_world: int | None = None
    for split, entry in manifest_splits.items():
        assert isinstance(entry, dict)
        split_path = output_directory / entry["path"]
        if split_path.stat().st_size != entry["bytes_on_disk"]:
            raise ValueError(f"byte-size mismatch for {split_path.name}")
        checksum = hashlib.sha256()
        split_worlds = 0
        split_questions = 0
        with _open_text(split_path, "rt") as input_file:
            for line in input_file:
                checksum.update(line.encode("utf-8"))
                world = json.loads(line)
                assert isinstance(world, dict)
                if world["split"] != split:
                    raise ValueError(f"split mismatch in {world['world_id']}")
                if world["world_id"] in seen_world_ids:
                    raise ValueError(f"duplicate world id: {world['world_id']}")
                seen_world_ids.add(world["world_id"])
                validate_world(world)
                questions = world["questions"]
                assert isinstance(questions, list)
                current_questions_per_world = len(questions)
                if verified_questions_per_world is None:
                    verified_questions_per_world = current_questions_per_world
                elif current_questions_per_world != verified_questions_per_world:
                    raise ValueError("every dataset world must have the same question count")
                split_worlds += 1
                split_questions += current_questions_per_world
        if split_worlds != entry["worlds"] or split_questions != entry["questions"]:
            raise ValueError(f"count mismatch for {split_path.name}")
        if checksum.hexdigest() != entry["uncompressed_sha256"]:
            raise ValueError(f"checksum mismatch for {split_path.name}")
        verified_worlds += split_worlds
        verified_questions += split_questions
    if verified_worlds != manifest["total_worlds"] or verified_questions != manifest["total_questions"]:
        raise ValueError("dataset totals do not match the manifest")
    if verified_questions_per_world is None:
        raise ValueError("dataset contains no worlds")
    manifest_config = manifest["config"]
    assert isinstance(manifest_config, dict)
    if verified_questions_per_world != manifest_config["entities_per_world"]:
        raise ValueError("world question count does not match the generation configuration")
    return {
        "worlds": verified_worlds,
        "questions": verified_questions,
        "questions_per_world": verified_questions_per_world,
    }


def read_worlds(path: Path) -> Iterator[dict[str, Any]]:
    """Stream parsed world records from a plain or compressed JSONL split."""
    with _open_text(path, "rt") as input_file:
        for line in input_file:
            world = json.loads(line)
            assert isinstance(world, dict)
            yield world


def iter_question_examples(path: Path) -> Iterator[dict[str, Any]]:
    """Flatten world records into model-facing question examples."""
    for world in read_worlds(path):
        questions = world["questions"]
        metadata = world["metadata"]
        assert isinstance(questions, list)
        assert isinstance(metadata, dict)
        person_side_agents = metadata["person_side_agent"]
        assert isinstance(person_side_agents, dict)
        for question in questions:
            assert isinstance(question, dict)
            reasoning_path = question["reasoning_path"]
            assert isinstance(reasoning_path, list)
            person, bridge, _answer = reasoning_path
            yield {
                "world_id": world["world_id"],
                "question_id": question["question_id"],
                "agent_1_input": world["agent_1_input"],
                "agent_2_input": world["agent_2_input"],
                "question": question["question"],
                "answer": question["answer"],
                "bridge": bridge,
                "person_side_agent": person_side_agents[person],
            }


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line settings for dataset generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="starter")
    parser.add_argument("--seed", type=int, default=1_729)
    parser.add_argument("--entities-per-world", type=int, default=8)
    parser.add_argument("--train-worlds", type=int)
    parser.add_argument("--validation-worlds", type=int)
    parser.add_argument("--test-worlds-per-split", type=int)
    parser.add_argument("--compression", choices=("gzip", "none"), default="gzip")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Generate a configured linked-facts dataset from the command line."""
    parsed_arguments = parse_arguments(arguments)
    if parsed_arguments.verify_only:
        print(json.dumps(verify_dataset(parsed_arguments.output_dir), indent=2))
        return 0
    preset = PRESETS[parsed_arguments.preset]
    config = GenerationConfig(
        seed=parsed_arguments.seed,
        entities_per_world=parsed_arguments.entities_per_world,
        train_worlds=parsed_arguments.train_worlds or preset["train"],
        validation_worlds=parsed_arguments.validation_worlds or preset["validation"],
        test_worlds_per_split=parsed_arguments.test_worlds_per_split or preset["test"],
        compression=parsed_arguments.compression,
    )
    manifest = build_dataset(parsed_arguments.output_dir, config)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
