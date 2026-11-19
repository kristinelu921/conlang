"""Analytic spherical paths distinguish coherent turning from backtracking."""

import jax.numpy as jnp
import numpy as np

from elr_probe.dynamics import angular_distance, norm_multiplier, trajectory_diagnostics


def test_straight_geodesic_has_unit_alignment_and_efficiency() -> None:
    initial = jnp.array([[1.0, 0.0]])
    middle = jnp.array([[np.cos(0.2), np.sin(0.2)]])
    last = jnp.array([[np.cos(0.4), np.sin(0.4)]])
    state = {"previous_direction": jnp.zeros_like(initial), "previous_valid": jnp.array(False), "total_path": jnp.array(0.0), "controlled_path": jnp.array(0.0)}
    first_metrics, state = trajectory_diagnostics(initial, middle, initial, initial, state, jnp.array(True))
    assert float(first_metrics["successive_tangent_valid"]) == 0
    metrics, _ = trajectory_diagnostics(middle, last, initial, initial, state, jnp.array(True))
    np.testing.assert_allclose(metrics["path_length_radians"], 0.4, rtol=1e-5)
    np.testing.assert_allclose(metrics["displacement_from_init_radians"], 0.4, rtol=1e-5)
    np.testing.assert_allclose(metrics["successive_tangent_cosine"], 1, atol=1e-6)
    np.testing.assert_allclose(metrics["path_efficiency"], 1, atol=1e-6)


def test_backtracking_has_path_length_but_no_net_rotation() -> None:
    initial = jnp.array([[1.0, 0.0]])
    middle = 2 * jnp.array([[np.cos(0.3), np.sin(0.3)]])
    state = {"previous_direction": jnp.zeros_like(initial), "previous_valid": jnp.array(False), "total_path": jnp.array(0.0), "controlled_path": jnp.array(0.0)}
    _, state = trajectory_diagnostics(initial, middle, initial, initial, state, jnp.array(True))
    metrics, _ = trajectory_diagnostics(middle, 3 * initial, initial, initial, state, jnp.array(True))
    np.testing.assert_allclose(metrics["path_length_radians"], 0.6, rtol=1e-5)
    np.testing.assert_allclose(metrics["displacement_from_init_radians"], 0, atol=1e-6)
    np.testing.assert_allclose(metrics["successive_tangent_cosine"], -1, atol=1e-6)
    assert float(metrics["path_efficiency"]) < 1e-6


def test_stationary_and_antipodal_steps_are_marked_invalid_for_turning() -> None:
    initial = jnp.array([[1.0, 0.0]])
    state = {"previous_direction": jnp.zeros_like(initial), "previous_valid": jnp.array(False), "total_path": jnp.array(0.0), "controlled_path": jnp.array(0.0)}
    for after in (initial, -initial):
        metrics, following = trajectory_diagnostics(initial, after, initial, initial, state, jnp.array(False))
        assert float(metrics["successive_tangent_valid"]) == 0
        assert not bool(following["previous_valid"])
        assert all(np.all(np.isfinite(value)) for value in metrics.values())
        assert float(metrics["post_warmup_path_length_radians"]) == 0


def test_column_angles_reveal_hidden_neuron_rotation() -> None:
    first = jnp.array([[1.0, 1.0], [0.0, 0.0]])
    second = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    np.testing.assert_allclose(angular_distance(first, second, axis=0), [0, np.pi / 2], atol=1e-6)


def test_ramp_stops_at_original_endpoint_in_longer_run() -> None:
    assert norm_multiplier(100, 100, 600, 2.0) == 1.0
    assert norm_multiplier(350, 100, 600, 2.0) == 1.5
    assert norm_multiplier(600, 100, 600, 2.0) == 2.0
    assert norm_multiplier(6000, 100, 600, 2.0) == 2.0
