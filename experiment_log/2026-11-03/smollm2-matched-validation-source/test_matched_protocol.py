"""Test prompt isolation, matched selection and budget enforcement without model loading."""

from dataclasses import replace
import unittest

from matched_protocol import (ComputeBudget, PromptExample, Question, ValidationResult,
                              balanced_selection_key, demonstration_pool, mixed_query_indices,
                              prepare_prompt, protocol_digest, validate_partition)


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Expose exact character boundaries for independent loss-mask expectations."""
        if add_special_tokens:
            raise ValueError("The protocol must explicitly disable automatic special tokens.")
        return [ord(character) for character in text]


class MatchedProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = [Question(index, f"Problem {index}?", f"Reasoning {index}.\n#### {index}")
                     for index in range(12)]
        self.query = Question(12, "How many remain?", "We have seven.\n#### 7")
        self.tokenizer = CharacterTokenizer()

    def prompt(self, shots: int = 5, query: Question | None = None,
               pool: list[Question] | None = None, **overrides: int) -> PromptExample:
        """Construct a prompt with explicit fixed test defaults."""
        settings = {"seed": 20261104, "epoch": 0, "context_length": 2048, **overrides}
        return prepare_prompt(self.query if query is None else query,
                              demonstration_pool(self.pool if pool is None else pool),
                              self.tokenizer, shots, **settings)

    def test_zero_shot_matches_existing_prompt_and_mask(self) -> None:
        example = self.prompt(0)
        expected_prompt = "Question: How many remain?\nAnswer:"
        expected_tokens = self.tokenizer.encode(expected_prompt + " " + self.query.answer) + [0]
        self.assertEqual(example.prompt, expected_prompt)
        self.assertEqual(example.input_tokens, tuple(expected_tokens))
        self.assertEqual(example.loss_mask, (0,) * len(expected_prompt) +
                         (1,) * (len(expected_tokens) - len(expected_prompt)))

    def test_five_demonstrations_are_unique_and_training_only(self) -> None:
        example = self.prompt()
        self.assertEqual(len(set(example.demonstration_indices)), 5)
        self.assertLessEqual(set(example.demonstration_indices), set(range(12)))
        self.assertNotIn(self.query.source_index, example.demonstration_indices)

    def test_demonstrations_and_query_prompt_have_no_loss(self) -> None:
        example = self.prompt()
        self.assertEqual(example.loss_mask[:len(example.prompt_tokens)], (0,) * len(example.prompt_tokens))
        self.assertEqual(example.loss_mask[len(example.prompt_tokens):], (1,) * (len(self.query.answer) + 2))
        self.assertEqual(example.input_tokens[-1], self.tokenizer.eos_token_id)

    def test_own_query_and_normalized_duplicate_excluded(self) -> None:
        duplicate = Question(90, " HOW  MANY REMAIN? ", "Poison answer")
        example = self.prompt(pool=self.pool + [self.query, duplicate])
        self.assertTrue(set(example.demonstration_indices).isdisjoint({12, 90}))

    def test_duplicate_demonstration_questions_are_deduplicated(self) -> None:
        duplicate = Question(90, " PROBLEM 0? ", "Different answer")
        example = self.prompt(pool=self.pool[:5] + [duplicate])
        self.assertEqual(set(example.demonstration_indices), set(range(5)))

    def test_demonstrations_do_not_depend_on_pool_input_order(self) -> None:
        self.assertEqual(self.prompt(), self.prompt(pool=list(reversed(self.pool))))

    def test_query_answer_cannot_choose_demonstrations(self) -> None:
        original = self.prompt()
        changed = self.prompt(query=replace(self.query, answer="Another target.\n#### 999"))
        self.assertEqual(original.prompt_tokens, changed.prompt_tokens)
        self.assertEqual(original.demonstration_indices, changed.demonstration_indices)
        self.assertEqual(original.sampling_attempts, changed.sampling_attempts)

    def test_overlong_answer_fails_instead_of_resampling(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not resample"):
            self.prompt(query=replace(self.query, answer="a" * 641))

    def test_context_overflow_never_truncates_or_reduces_shots(self) -> None:
        oversized = [replace(row, answer="a" * 1000) for row in self.pool]
        with self.assertRaisesRegex(ValueError, "No complete five-shot"):
            self.prompt(pool=oversized, maximum_sampling_attempts=3)
        with self.assertRaisesRegex(ValueError, "truncation is disabled"):
            self.prompt(0, query=replace(self.query, question="x" * 1800))

    def test_not_enough_distinct_demonstrations_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Not enough distinct"):
            self.prompt(pool=self.pool[:4])

    def test_duplicate_source_indices_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate source indices"):
            self.prompt(pool=self.pool + [self.pool[0]])

    def test_partition_checks_completeness_and_normalized_leakage(self) -> None:
        validate_partition(self.pool, [self.query], 13)
        with self.assertRaisesRegex(ValueError, "do not cover"):
            validate_partition(self.pool[:-1], [self.query], 13)
        with self.assertRaisesRegex(ValueError, "Normalized questions overlap"):
            validate_partition(self.pool, [replace(self.query, question=" PROBLEM 0? ")], 13)

    def test_exact_mixture_size_and_reproducibility(self) -> None:
        indices = list(range(13))
        selected = mixed_query_indices(indices, 0.5, 1234, 0)
        self.assertEqual(len(selected), 6)
        self.assertEqual(selected, mixed_query_indices(list(reversed(indices)), 0.5, 1234, 0))
        self.assertEqual(set(indices), set(selected) | (set(indices) - set(selected)))
        self.assertEqual(mixed_query_indices(indices, 0, 1234, 0), frozenset())
        self.assertEqual(mixed_query_indices(indices, 1, 1234, 0), frozenset(indices))

    def test_invalid_mixture_rejected(self) -> None:
        for fraction in [-0.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                mixed_query_indices([0, 1], fraction, 1, 0)
        with self.assertRaises(ValueError):
            mixed_query_indices([0, 0], 0.5, 1, 0)

    def test_protocol_digest_binds_targets_and_prompt_order(self) -> None:
        zero, five = [self.prompt(0)], [self.prompt()]
        digest = protocol_digest(zero, five, [self.query.answer])
        self.assertEqual(len(digest), 64)
        self.assertNotEqual(digest, protocol_digest(zero, five, ["#### 8"]))
        with self.assertRaises(ValueError):
            protocol_digest(zero, zero, [self.query.answer])
        with self.assertRaises(ValueError):
            protocol_digest(zero, [replace(five[0], query_index=99)], [self.query.answer])


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = ValidationResult("reference", 3, 747, 60, 40, 0.60, 0.70, "a" * 64)

    def test_balanced_accuracy_beats_one_format_specialization(self) -> None:
        balanced = replace(self.reference, trial_id="balanced", zero_shot_correct=70, five_shot_correct=55)
        specialized = replace(self.reference, trial_id="specialized", zero_shot_correct=100, five_shot_correct=41)
        self.assertLess(balanced_selection_key(balanced, self.reference),
                        balanced_selection_key(specialized, self.reference))

    def test_each_loss_and_accuracy_guard_is_required(self) -> None:
        for overrides in [{"zero_shot_nll": 0.67}, {"five_shot_nll": 0.78},
                          {"zero_shot_correct": 59}, {"five_shot_correct": 39}]:
            self.assertIsNone(balanced_selection_key(replace(self.reference, **overrides), self.reference))

    def test_ties_use_mean_accuracy_relative_loss_epoch_then_identity(self) -> None:
        first = replace(self.reference, zero_shot_correct=61)
        self.assertLess(balanced_selection_key(first, self.reference),
                        balanced_selection_key(self.reference, self.reference))
        lower_loss = replace(self.reference, zero_shot_nll=0.59, five_shot_nll=0.69)
        self.assertLess(balanced_selection_key(lower_loss, self.reference),
                        balanced_selection_key(self.reference, self.reference))
        earlier = replace(self.reference, epoch=2)
        self.assertLess(balanced_selection_key(earlier, self.reference),
                        balanced_selection_key(self.reference, self.reference))
        named_first = replace(self.reference, trial_id="aaa")
        self.assertLess(balanced_selection_key(named_first, self.reference),
                        balanced_selection_key(self.reference, self.reference))

    def test_mismatched_protocol_and_invalid_metrics_fail(self) -> None:
        for overrides in [{"protocol_sha256": "b" * 64}, {"examples": 748}]:
            with self.assertRaises(ValueError):
                balanced_selection_key(replace(self.reference, **overrides), self.reference)
        for overrides in [{"zero_shot_nll": float("nan")}, {"five_shot_nll": 0.0},
                          {"zero_shot_correct": 748}, {"protocol_sha256": ""}]:
            with self.assertRaises(ValueError):
                replace(self.reference, **overrides)


class BudgetTests(unittest.TestCase):
    def test_expiry_and_reserve_survive_reconstruction(self) -> None:
        budget = ComputeBudget(1000.0, 14400)
        self.assertEqual(budget.remaining(2000.0), 13400)
        self.assertTrue(budget.admits(2000.0, 9800, 3600))
        self.assertFalse(budget.admits(2000.0, 9801, 3600))
        reconstructed = ComputeBudget(1000.0, 14400)
        self.assertEqual(reconstructed.remaining(15401.0), 0)
        self.assertFalse(reconstructed.admits(15401.0, 1, 0))

    def test_missing_approval_and_invalid_time_fail(self) -> None:
        with self.assertRaises(ValueError):
            ComputeBudget(1000.0, 0)
        budget = ComputeBudget(1000.0, 14400)
        with self.assertRaises(ValueError):
            budget.remaining(999.0)
        with self.assertRaises(ValueError):
            budget.admits(1001.0, float("nan"), 0)
        with self.assertRaises(ValueError):
            budget.admits(1001.0, 1, -1)


if __name__ == "__main__":
    unittest.main()
