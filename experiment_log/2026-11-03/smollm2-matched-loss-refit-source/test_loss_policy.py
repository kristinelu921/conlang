"""Test selection isolation and admission bounds without loading an ML runtime."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from loss_policy import loss_recipe, required_seconds, require_primary_finished, training_lock


class LossPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selection = {'status': 'locked', 'test_used_for_selection': False,
            'selected': {'trial_id': 'balanced', 'epoch': 3},
            'minimum_zero_shot_nll': {'trial_id': 'minimum', 'epoch': 2,
                'configuration': {'learning_rate': 0.0001, 'seed': 1234}}}

    def test_recipe_uses_locked_minimum_without_mutating_selection(self) -> None:
        original = copy.deepcopy(self.selection)
        recipe = loss_recipe(self.selection)
        self.assertEqual(recipe['epochs'], 2)
        self.assertEqual(recipe['learning_rate'], 0.0001)
        self.assertTrue(recipe['final_fit'])
        self.assertEqual(self.selection, original)

    def test_rejects_unlocked_or_test_selected_recipe(self) -> None:
        for amendment in ({'status': 'running'}, {'test_used_for_selection': True}):
            with self.subTest(amendment=amendment), self.assertRaises(ValueError):
                loss_recipe({**self.selection, **amendment})

    def test_rejects_duplicate_refit(self) -> None:
        self.selection['selected'] = {'trial_id': 'minimum', 'epoch': 2}
        with self.assertRaises(ValueError):
            loss_recipe(self.selection)

    def test_admission_includes_worst_epoch_compile_evaluation_and_reporting(self) -> None:
        self.assertEqual(required_seconds(3, [100.0, 300.0, 120.0], 200.0), 3240.0)
        self.assertEqual(required_seconds(3, [100.0, 300.0], 1000.0), 3840.0)

    def test_rejects_missing_or_invalid_measurements(self) -> None:
        for epochs, durations, finalization in ((0, [1.0], 1.0), (3, [], 1.0),
                (3, [0.0], 1.0), (3, [1.0], -1.0)):
            with self.subTest(epochs=epochs, durations=durations), self.assertRaises(ValueError):
                required_seconds(epochs, durations, finalization)

    def test_requires_successful_primary_and_publication_and_no_live_process(self) -> None:
        for terminal, publication, alive in (('running', 'completed', False),
                ('failed', 'completed', False), ('completed', 'failed', False), ('completed', 'completed', True)):
            with self.subTest(terminal=terminal, publication=publication, alive=alive), self.assertRaises(PermissionError):
                require_primary_finished({'status': terminal}, {'study_terminal_status': publication}, alive)
        require_primary_finished({'status': 'completed'}, {'study_terminal_status': 'completed'}, False)

    def test_training_lock_excludes_duplicates_and_releases_after_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'accelerator.lock'
            with training_lock(path):
                with self.assertRaises(BlockingIOError), training_lock(path):
                    self.fail('Concurrent acquisition succeeded.')
            with training_lock(path):
                pass


if __name__ == '__main__':
    unittest.main()
