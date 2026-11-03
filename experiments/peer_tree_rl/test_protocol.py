"""Remote-only protocol tests with scripted policies independent of model implementation."""

import json
from pathlib import Path
import unittest

if not Path(__file__).resolve().is_relative_to("/home/kristinelu/peer-tree"):
    raise unittest.SkipTest("Pilot tests execute only from the authorized remote source snapshot.")

from protocol import (Generation, Problem, State, boxed_integer, collect_tree,
                      completion_mask, outcome_weight, question_fingerprint, transition)


def generated(action: dict, ended: bool = True) -> Generation:
    return Generation((11, 12), (21, 22), (-0.2, -0.3), json.dumps(action), ended)


class ScriptedPolicy:
    def __init__(self, candidate_messages: bool = True) -> None:
        self.candidate_messages = candidate_messages
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
        return [generated({"action": "submit", "answer": "42" if index % 2 else "99"})
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

    def test_unanimous_outcomes_keep_signed_credit(self) -> None:
        self.assertEqual(outcome_weight(8, 8, 0.5, 2), 1)
        self.assertEqual(outcome_weight(0, 8, 0.5, 2), -1)
        self.assertEqual(outcome_weight(7, 8, 0.5, 2), 0.75)
        self.assertEqual(outcome_weight(4, 8, 0.5, 2), 0)

    def test_prompt_and_padding_mask_excludes_non_actions(self) -> None:
        self.assertEqual(completion_mask(3, 2, 7), [0, 0, 0, 1, 1, 0, 0])
        with self.assertRaises(ValueError):
            completion_mask(5, 3, 7)

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

    def test_overlap_hash_normalizes_whitespace_and_case(self) -> None:
        self.assertEqual(question_fingerprint("What is 2 + 2?"), question_fingerprint("what\n is 2+2?"))


if __name__ == "__main__":
    unittest.main()
