"""Tests for the natural-language linked-facts generator."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from string import Formatter

import pytest

from linked_facts.generate_dataset import (
    GenerationConfig,
    build_dataset,
    build_name_banks,
    generate_world,
    iter_question_examples,
    read_worlds,
    verify_dataset,
    validate_world,
)
from linked_facts.relation_domains import LEXICONS, MEMBERSHIP_KINDS, PLEDGED_GROUP_KINDS, RELATION_DOMAINS


def test_world_answers_follow_both_private_mappings() -> None:
    """Every target should equal the result of its two-hop reasoning path."""
    config = GenerationConfig(entities_per_world=6, train_worlds=1, validation_worlds=1, test_worlds_per_split=1)
    world = generate_world("train", 0, config)

    validate_world(world, expected_entities=6)
    chains = world["metadata"]["chains"]
    for question in world["questions"]:
        person, middle, answer = question["reasoning_path"]
        assert chains[person]["middle"] == middle
        assert chains[person]["answer"] == answer
        assert question["answer"] == answer
        assert (person in question["agent_1_input"]) != (person in question["agent_2_input"])
        assert (answer in question["agent_1_input"]) != (answer in question["agent_2_input"])
        assert (person in question["agent_1_input"]) != (answer in question["agent_1_input"])
    agent_1_people = [
        person
        for person, sender in world["metadata"]["person_side_agent"].items()
        if sender == "agent_1"
    ]
    assert len(agent_1_people) == 3
    assert len({question["answer"] for question in world["questions"]}) == 6


def test_each_question_contains_only_its_two_required_hops() -> None:
    """Each sender input should contain one required fact and no distractors."""
    config = GenerationConfig(
        entities_per_world=8,
        train_worlds=1,
        validation_worlds=1,
        test_worlds_per_split=1,
    )
    world = generate_world("train", 0, config)
    chains = world["metadata"]["chains"]

    for question in world["questions"]:
        person, middle, answer = question["reasoning_path"]
        agent_1_input = question["agent_1_input"]
        agent_2_input = question["agent_2_input"]
        combined_input = f"{agent_1_input} {agent_2_input}"
        assert (person in agent_1_input) != (person in agent_2_input)
        assert (answer in agent_1_input) != (answer in agent_2_input)
        assert middle in agent_1_input and middle in agent_2_input
        for other_person, other_chain in chains.items():
            if other_person == person:
                continue
            assert other_person not in combined_input
            assert other_chain["middle"] not in combined_input
            assert other_chain["answer"] not in combined_input


def test_world_generation_is_deterministic_and_split_stable() -> None:
    """A world identity should not depend on unrelated split sizes."""
    first_config = GenerationConfig(seed=73, train_worlds=2, validation_worlds=1, test_worlds_per_split=1)
    second_config = GenerationConfig(seed=73, train_worlds=99, validation_worlds=7, test_worlds_per_split=3)

    assert generate_world("train", 1, first_config) == generate_world("train", 1, second_config)
    assert generate_world("train", 1, first_config) != generate_world("train", 0, first_config)


def test_name_and_wording_generalization_axes_are_disjoint() -> None:
    """Held-out evaluations should change only their declared generation axis."""
    config = GenerationConfig(seed=81, entities_per_world=8, train_worlds=1, validation_worlds=1, test_worlds_per_split=1)
    name_banks = build_name_banks(config.seed)
    familiar_names = {name for role_names in name_banks["familiar"].values() for name in role_names}
    held_out_names = {name for role_names in name_banks["held_out"].values() for name in role_names}
    familiar_lexicon = {word for source in LEXICONS.values() for word in source["familiar"]}
    held_out_lexicon = {word for source in LEXICONS.values() for word in source["held_out"]}
    assert familiar_names.isdisjoint(held_out_names)
    assert familiar_lexicon.isdisjoint(held_out_lexicon)

    train_world = generate_world("train", 0, config, name_banks)
    new_name_world = generate_world("test_new_names", 0, config, name_banks)
    new_wording_world = generate_world("test_new_wording", 0, config, name_banks)

    def proper_names(world: dict[str, object]) -> set[str]:
        entities = world["metadata"]["entities"]
        assert isinstance(entities, dict)
        return set(entities["persons"]) | set(entities["groups"]) | set(entities["locations"])

    def lexicon_words(world: dict[str, object]) -> set[str]:
        entities = world["metadata"]["entities"]
        assert isinstance(entities, dict)
        return set(entities["colors"]) | set(entities["animals"]) | set(entities["furniture"]) | set(entities["rooms"])

    assert proper_names(train_world) <= familiar_names
    assert lexicon_words(train_world) <= familiar_lexicon
    assert proper_names(new_name_world) <= held_out_names
    assert lexicon_words(new_name_world) <= held_out_lexicon
    assert lexicon_words(new_wording_world) <= familiar_lexicon
    held_out_template_ids = [
        *[question["agent_1_template_id"] for question in new_wording_world["questions"]],
        *[question["agent_2_template_id"] for question in new_wording_world["questions"]],
        *[question["template_id"] for question in new_wording_world["questions"]],
    ]
    assert held_out_template_ids
    assert all("-h-" in template_id for template_id in held_out_template_ids)


def _template_fields(template: str) -> set[str]:
    """Return the named placeholders in one sentence template."""
    return {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name}


def _template_prefix(template_id: str, wording: str) -> str:
    """Return the domain prefix that precedes a familiar or held-out marker."""
    marker = "-f-" if wording == "familiar" else "-h-"
    head, separator, _ = template_id.partition(marker)
    assert separator == marker
    return head + "-"


def test_sentence_banks_keep_disjoint_syntactic_frames() -> None:
    """Each wording bank should use its own fully grounded sentence frames."""
    seen_templates: set[str] = set()
    placeholder_values = {
        "person": "Person",
        "group": "Group",
        "group_kind": "guild",
        "location": "Place",
        "color": "amber",
        "animal": "heron",
        "furniture": "daybed",
        "room": "library",
    }
    for domain in RELATION_DOMAINS:
        template_banks = (
            (domain.agent_1_templates, {placeholder for placeholder, _ in domain.agent_1_bindings}),
            (domain.agent_2_templates, {placeholder for placeholder, _ in domain.agent_2_bindings}),
            (domain.question_templates, {"person"}),
        )
        for bank, allowed_fields in template_banks:
            familiar_templates = [template for _, template in bank["familiar"]]
            held_out_templates = [template for _, template in bank["held_out"]]
            assert len(familiar_templates) == len(set(familiar_templates))
            assert len(held_out_templates) == len(set(held_out_templates))
            assert set(familiar_templates).isdisjoint(held_out_templates)
            for template in familiar_templates + held_out_templates:
                assert template not in seen_templates
                seen_templates.add(template)
                assert _template_fields(template) == allowed_fields
                assert "{" not in template.format(**placeholder_values)


def test_worlds_mix_colors_animals_furniture_and_other_groups() -> None:
    """A default world should include every relation, not only membership."""
    config = GenerationConfig(entities_per_world=8, train_worlds=1, validation_worlds=1, test_worlds_per_split=1)
    world = generate_world("train", 4, config)
    domains = {chain["domain"] for chain in world["metadata"]["chains"].values()}
    assert domains == {"membership", "color", "animal", "furniture", "pledged_group"}


def test_lexicon_tokens_do_not_contain_one_another() -> None:
    """Answer words must not hide inside names, kinds, or other lexicon words."""
    name_banks = build_name_banks(1)
    kinds = (*MEMBERSHIP_KINDS, *PLEDGED_GROUP_KINDS)
    for variant in ("familiar", "held_out"):
        tokens = [name for role_names in name_banks[variant].values() for name in role_names]
        tokens.extend(word for source in LEXICONS.values() for word in source[variant])
        tokens.extend(kinds)
        for index, left in enumerate(tokens):
            for right in tokens[index + 1 :]:
                assert left not in right and right not in left


def test_worlds_cycle_through_sentence_frames_before_repeating() -> None:
    """A world should spread its facts across every frame before reusing one."""
    config = GenerationConfig(entities_per_world=8, train_worlds=1, validation_worlds=1, test_worlds_per_split=1)
    world = generate_world("train", 3, config)
    assert len({question["agent_1_template_id"] for question in world["questions"]}) == 8
    assert len({question["agent_2_template_id"] for question in world["questions"]}) == 8
    assert len({question["template_id"] for question in world["questions"]}) == 8

    wide_config = GenerationConfig(entities_per_world=32, train_worlds=1, validation_worlds=1, test_worlds_per_split=1)
    for split, wording in (("train", "familiar"), ("test_new_wording", "held_out")):
        wide_world = generate_world(split, 0, wide_config)
        domain_counts = Counter(chain["domain"] for chain in wide_world["metadata"]["chains"].values())
        question_ids = [question["template_id"] for question in wide_world["questions"]]
        sender_template_ids = [
            *[question["agent_1_template_id"] for question in wide_world["questions"]],
            *[question["agent_2_template_id"] for question in wide_world["questions"]],
        ]
        for domain in RELATION_DOMAINS:
            fact_count = domain_counts[domain.domain_id]
            template_banks = (
                (sender_template_ids, domain.agent_1_templates[wording]),
                (sender_template_ids, domain.agent_2_templates[wording]),
                (question_ids, domain.question_templates[wording]),
            )
            for template_ids, bank in template_banks:
                prefix = _template_prefix(bank[0][0], wording)
                observed = Counter(template_id for template_id in template_ids if template_id.startswith(prefix))
                assert sum(observed.values()) == fact_count
                if fact_count >= len(bank):
                    assert set(observed) == {template_id for template_id, _ in bank}
                    assert max(observed.values()) - min(observed.values()) <= 1
                else:
                    assert len(observed) == fact_count


def test_invalid_answer_is_rejected() -> None:
    """Validation should catch a target that disagrees with its latent world."""
    config = GenerationConfig(entities_per_world=3, train_worlds=1, validation_worlds=1, test_worlds_per_split=1)
    world = generate_world("train", 0, config)
    world["questions"][0]["answer"] = "Incorrect"

    with pytest.raises(ValueError, match="invalid answer"):
        validate_world(world, expected_entities=3)


def test_dataset_builds_manifest_and_flattened_examples(tmp_path: Path) -> None:
    """A complete small build should be readable through both public iterators."""
    config = GenerationConfig(
        seed=19,
        entities_per_world=4,
        train_worlds=3,
        validation_worlds=2,
        test_worlds_per_split=1,
        compression="gzip",
    )
    manifest = build_dataset(tmp_path, config)

    saved_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest == manifest
    assert verify_dataset(tmp_path) == {
        "worlds": 9,
        "questions": 36,
        "questions_per_world": 4,
    }
    assert manifest["total_worlds"] == 9
    assert manifest["total_questions"] == 36
    train_path = tmp_path / manifest["splits"]["train"]["path"]
    assert len(list(read_worlds(train_path))) == 3
    examples = list(iter_question_examples(train_path))
    assert len(examples) == 12
    assert set(examples[0]) == {
        "world_id",
        "question_id",
        "agent_1_input",
        "agent_2_input",
        "question",
        "answer",
        "bridge",
        "person_side_agent",
    }
    first_world = next(read_worlds(train_path))
    first_question = first_world["questions"][0]
    first_person, first_bridge, _first_answer = first_question["reasoning_path"]
    assert examples[0]["bridge"] == first_bridge
    assert (
        examples[0]["person_side_agent"]
        == first_world["metadata"]["person_side_agent"][first_person]
    )
