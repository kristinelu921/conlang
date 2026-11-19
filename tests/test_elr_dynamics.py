"""Analytic fixtures for ELR control; these do not run experiment workloads."""

import jax.numpy as jnp
import numpy as np

from elr_probe.dynamics import controlled_update, matrix_diagnostics


def test_rotation_matches_known_angle_and_ignores_radial_scaling() -> None:
    before = jnp.array([[1.0, 0.0]])
    angle = 1e-5
    after = 3 * jnp.array([[np.cos(angle), np.sin(angle)]])
    observed = matrix_diagnostics(before, after)
    np.testing.assert_allclose(observed["angle_radians"], angle, rtol=1e-5)
    np.testing.assert_allclose(observed["radial_relative_update"], 2, rtol=1e-6)


def test_projection_preserves_update_direction_and_hits_next_norm() -> None:
    parameter = jnp.array([[3.0, 4.0]])
    update = jnp.array([[-2.0, 1.0]])
    actual = controlled_update(parameter, update, jnp.array(0.5), jnp.array(10.0))
    expected_direction = np.array([[2.0, 4.5]])
    np.testing.assert_allclose(actual, 10 * expected_direction / np.linalg.norm(expected_direction), rtol=1e-6)


def test_equal_elr_can_have_different_rotations() -> None:
    parameter = jnp.array([[1.0, 0.0]])
    first = controlled_update(parameter, jnp.array([[0.0, 1.0]]), jnp.array(0.1), jnp.array(1.0))
    second = controlled_update(2 * parameter, jnp.array([[0.0, 2.0]]), jnp.array(0.2), jnp.array(2.0))
    first_angle = matrix_diagnostics(parameter, first)["angle_radians"]
    second_angle = matrix_diagnostics(2 * parameter, second)["angle_radians"]
    np.testing.assert_allclose(first_angle, np.arctan(0.1), rtol=1e-6)
    np.testing.assert_allclose(second_angle, np.arctan(0.2), rtol=1e-6)
    assert float(second_angle / first_angle) > 1.9


def test_equal_update_direction_and_matched_elr_preserve_rotation() -> None:
    parameter = jnp.array([[1.0, 2.0]])
    update = jnp.array([[-0.2, 0.3]])
    first = controlled_update(parameter, update, jnp.array(0.1), jnp.array(3.0))
    second = controlled_update(2 * parameter, update, jnp.array(0.2), jnp.array(9.0))
    np.testing.assert_allclose(
        matrix_diagnostics(parameter, first)["angle_radians"],
        matrix_diagnostics(2 * parameter, second)["angle_radians"],
        rtol=1e-5,
    )
