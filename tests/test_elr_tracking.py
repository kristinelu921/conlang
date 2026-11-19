"""Verify paired metrics retain their seed, arm, and missing-angle semantics."""

from elr_probe.tracking import tracking_metrics


def test_tracking_preserves_arm_values_and_skips_undefined_ratio() -> None:
    record = {
        "seed": 1729, "step": 105,
        "baseline": {"loss": 3.5, "layer_0_q/kernel/elr": 0.01},
        "matched": {"loss": 3.6, "layer_0_q/kernel/elr": 0.01},
        "comparison": {"rotation_ratio": None, "loss_difference": 0.1},
    }
    assert tracking_metrics(record) == {
        "seed_1729/training_step": 105,
        "seed_1729/baseline/loss": 3.5,
        "seed_1729/baseline/layer_0_q/kernel/elr": 0.01,
        "seed_1729/matched/loss": 3.6,
        "seed_1729/matched/layer_0_q/kernel/elr": 0.01,
        "seed_1729/comparison/loss_difference": 0.1,
    }


def test_warmup_logs_without_a_matched_arm_and_separates_seeds() -> None:
    first = tracking_metrics({"seed": 1, "step": 0, "baseline": {"loss": 5.0}})
    second = tracking_metrics({"seed": 2, "step": 0, "baseline": {"loss": 6.0}})
    assert not first.keys() & second.keys()
    assert first["seed_1/baseline/loss"] == 5.0
