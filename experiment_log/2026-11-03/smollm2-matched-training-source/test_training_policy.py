"""Check objective patience, Pareto preservation and measured budget admission."""
import unittest
from training_policy import should_stop, nondominated, epoch_admission_seconds


class TrainingPolicyTests(unittest.TestCase):
    def test_joint_objective_patience(self) -> None:
        self.assertFalse(should_stop(2, 0, 3, 3))
        self.assertFalse(should_stop(5, 3, 3, 3))
        self.assertTrue(should_stop(6, 3, 3, 3))

    def test_frontier_preserves_accuracy_loss_tradeoff_and_ties(self) -> None:
        first = dict(zero_shot_correct=10, five_shot_correct=20, zero_shot_nll=.5, five_shot_nll=.6)
        second = dict(zero_shot_correct=20, five_shot_correct=20, zero_shot_nll=.7, five_shot_nll=.6)
        worse = dict(zero_shot_correct=9, five_shot_correct=19, zero_shot_nll=.8, five_shot_nll=.8)
        self.assertEqual(nondominated([first, second, worse, first.copy()]), [first, second, first])

    def test_admission_uses_slowest_measured_epoch(self) -> None:
        self.assertEqual(epoch_admission_seconds([]), 900)
        self.assertEqual(epoch_admission_seconds([100, 200, 150]), 330)


if __name__ == '__main__':
    unittest.main()
