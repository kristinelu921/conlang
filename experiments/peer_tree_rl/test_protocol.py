"""Remote-only protocol tests with scripted policies independent of model implementation."""

import json
from pathlib import Path
import unittest
from unittest.mock import Mock

if not Path(__file__).resolve().is_relative_to("/home/kristinelu/peer-tree"):
    raise unittest.SkipTest("Pilot tests execute only from the authorized remote source snapshot.")

from protocol import (Generation, Problem, State, answer_reward, boxed_integer, collect_tree,
                      outcome_weight, question_fingerprint, transition, validate_settings)


def generated(action: dict, ended: bool = True) -> Generation:
    return Generation((11, 12), (21, 22), (-0.2, -0.3), json.dumps(action), ended)


class ScriptedPolicy:
    def __init__(self, candidate_messages: bool = True, numeric_answers: bool = False) -> None:
        self.candidate_messages = candidate_messages
        self.numeric_answers = numeric_answers
        self.calls = []

    def generate(self, states: list[State], seeds: list[int], agent: int) -> list[Generation]:
        """Emit candidate questions and independently alternating correct/incorrect replies."""
        self.calls.append((states, seeds, agent))
        if len(states) == 4:
            if not self.candidate_messages:
                return [generated({"action": "submit", "answer": "42"}) for _ in states]
            return [generated({"action": "message", "message": f"question {index}", "context": "selected"})
                    for index in range(4)]
        if len(states) == 1:
            return [generated({"action": "message", "message": "prefix", "context": "private selection"})]
        return [generated({"action": "submit", "answer": (42 if index % 2 else 99) if self.numeric_answers
                          else ("42" if index % 2 else "99")})
                for index in range(len(states))]


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = json.loads((Path(__file__).parent / "pilot.json").read_text())
        self.problem = Problem("fixture", "Return the integer forty-two.", 42, "train")

    def test_selected_context_and_original_problem_only(self) -> None:
        original = State("full original problem", 0, "secret old message", "secret old context")
        following = transition(original, generated({"action": "message", "message": "new question",
                                                    "context": "chosen detail"}), 8).next_state
        self.assertIsNotNone(following)
        self.assertEqual(following.agent, 1)
        content = json.loads(following.messages()[1]["content"])
        self.assertEqual(content["original_problem"], "full original problem")
        self.assertEqual(content["incoming_context"], "chosen detail")
        self.assertNotIn("secret", json.dumps(content))

    def test_submission_at_last_turn_succeeds(self) -> None:
        outcome = transition(State("p", 7), generated({"action": "submit", "answer": "0042"}), 8)
        self.assertEqual(outcome.submission, 42)
        self.assertIsNone(outcome.next_state)

    def test_message_at_last_turn_exhausts_budget(self) -> None:
        outcome = transition(State("p", 7), generated({"action": "message", "message": "continue", "context": ""}), 8)
        self.assertEqual(outcome.reason, "turn_limit")

    def test_invalid_and_truncated_outputs_cannot_earn_submission_reward(self) -> None:
        for action in ({"action": "submit", "answer": "42 or 99"},
                       {"action": "submit", "answer": 42},
                       {"action": "submit", "answer": "42", "extra": "anything"}):
            self.assertIsNone(transition(State("p"), generated(action), 8).submission)
        self.assertIsNone(transition(State("p"), generated({"action": "submit", "answer": "42"}, False), 8).submission)

    def test_integer_reference_extraction_is_strict(self) -> None:
        self.assertEqual(boxed_integer("steps 18 and 99 then \\boxed{042}"), 42)
        self.assertIsNone(boxed_integer("\\boxed{\\frac{1}{2}}"))
        self.assertIsNone(boxed_integer("\\boxed{42} then \\boxed{x+1}"))
        self.assertIsNone(boxed_integer("missing box 42"))

    def test_integer_search_accepts_prose_and_numeric_json(self) -> None:
        """Accept complete signed integers across ordinary final-response formats."""
        for text, expected in (('{"action":"submit","answer":2}', 2),
                               ("The answer is 2.", 2), ("Answer: +002", 2),
                               ("The result is -42.", -42), ("There are 5,000 pairs.", 5000)):
            with self.subTest(text=text):
                self.assertEqual(answer_reward(text, expected, None, "integer_search"), 1)
                self.assertEqual(answer_reward(text, expected, None), 0)

    def test_integer_search_rejects_partial_numeric_matches(self) -> None:
        """Reject answers embedded inside larger numbers, decimals, or identifiers."""
        for text in ("20", "204", "-2", "2.5", "2,000", "2e3", "x2", "No answer"):
            with self.subTest(text=text):
                self.assertEqual(answer_reward(text, 2, None, "integer_search"), 0)
        self.assertEqual(answer_reward("42", -42, None, "integer_search"), 0)

    def test_integer_search_is_presence_not_semantic_verification(self) -> None:
        """Keep the loose matcher's incidental-mention limitation explicit."""
        self.assertEqual(answer_reward("It is not 42; my answer is 99.", 42, None, "integer_search"), 1)
        self.assertEqual(answer_reward("It is not 42; my answer is 99.", 42, 99), 0)
        with self.assertRaisesRegex(ValueError, "answer_matching"):
            answer_reward("42", 42, 42, "unknown")

    def test_integer_search_scores_terminal_candidates_and_prefixes(self) -> None:
        """Apply the selected reward to every direct terminal path, including truncation."""
        for tree_index in (0, 1):
            for ended in (False, True):
                for matching, expected in (("strict", 0), ("integer_search", 1)):
                    with self.subTest(tree_index=tree_index, ended=ended, matching=matching):
                        policy = Mock()
                        policy.generate.side_effect = lambda states, seeds, agent: [
                            generated({"action": "submit", "answer": 42}, ended) for _ in states]
                        items, report = collect_tree(policy, self.problem,
                            {**self.settings, "answer_matching": matching}, tree_index, lambda: None)
                        self.assertEqual(report["leaf_rewards"], [expected] * (4 if tree_index == 0 else 1))
                        self.assertTrue(all(item.weight == 2 * expected - 1 for item in items))

    def test_integer_search_backs_up_continuation_outcomes(self) -> None:
        """Propagate loose-scored child outcomes to their shared candidate messages."""
        items, report = collect_tree(ScriptedPolicy(numeric_answers=True), self.problem,
            {**self.settings, "answer_matching": "integer_search"}, 0, lambda: None)
        self.assertEqual(report["candidate_rewards"], [[0, 1]] * 4)
        candidates = [item for item in items if "/" not in item.identifier]
        self.assertTrue(all(item.weight == 0 and item.observations == 2 for item in candidates))

    def test_unanimous_outcomes_keep_signed_credit(self) -> None:
        self.assertEqual(outcome_weight(8, 8, 0.5, 2), 1)
        self.assertEqual(outcome_weight(0, 8, 0.5, 2), -1)
        self.assertEqual(outcome_weight(7, 8, 0.5, 2), 0.75)
        self.assertEqual(outcome_weight(4, 8, 0.5, 2), 0)

    def test_configuration_rejects_unimplemented_algorithm_changes(self) -> None:
        validate_settings(self.settings)
        for changes in ({"independent_lora_adapters": False}, {"lora_targets": ["o_proj"]},
                        {"update_epochs_per_tree": 2}, {"compute_dtype": "bfloat16"},
                        {"answer_matching": "unknown"}):
            with self.assertRaises(ValueError):
                validate_settings({**self.settings, **changes})
        validate_settings({**self.settings, "answer_matching": "integer_search"})

    def test_tree_has_independent_continuations_and_no_repeated_candidate(self) -> None:
        policy = ScriptedPolicy()
        items, report = collect_tree(policy, self.problem, self.settings, 0, lambda: None)
        self.assertEqual(report["candidate_rewards"], [[0, 1]] * 4)
        self.assertEqual(len(items), 12)
        candidates = [item for item in items if item.identifier.startswith("candidate-") and "/" not in item.identifier]
        self.assertEqual(len(candidates), 4)
        self.assertTrue(all(item.weight == 0 and item.observations == 2 for item in candidates))
        seeds = [seed for _, batch, _ in policy.calls for seed in batch]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(all(item.state.problem == self.problem.problem for item in items))

    def test_terminal_candidate_is_not_falsely_counted_as_two_rollouts(self) -> None:
        policy = ScriptedPolicy(candidate_messages=False)
        items, report = collect_tree(policy, self.problem, self.settings, 0, lambda: None)
        self.assertEqual(len(policy.calls), 1)
        self.assertEqual(len(items), 4)
        self.assertTrue(all(item.observations == 1 and item.weight == 1 for item in items))
        self.assertEqual(report["leaf_rewards"], [1] * 4)

    def test_shared_prefix_receives_one_averaged_credit(self) -> None:
        items, report = collect_tree(ScriptedPolicy(), self.problem, self.settings, 1, lambda: None)
        prefix = [item for item in items if item.identifier.startswith("prefix-")]
        self.assertEqual(len(prefix), 1)
        self.assertEqual(prefix[0].weight, 0)
        self.assertEqual(report["branch_turn"], 1)
        identifiers = {item.identifier for item in items}
        self.assertTrue(all(item.parent is None or item.parent in identifiers for item in items))
        candidates = [item for item in items if item.identifier.startswith("candidate-") and "/" not in item.identifier]
        self.assertTrue(all(item.parent == "prefix-0" for item in candidates))

    def test_overlap_hash_normalizes_whitespace_and_case(self) -> None:
        self.assertEqual(question_fingerprint("What is 2 + 2?"), question_fingerprint("what\n is 2+2?"))


if __name__ == "__main__":
    unittest.main()
