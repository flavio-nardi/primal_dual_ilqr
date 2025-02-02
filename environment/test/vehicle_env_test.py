import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from environment.vehicle_env import BicycleEnv, TrackReference


class TestBicycleEnv(unittest.TestCase):
    def setUp(self):
        """Set up test environment with common parameters."""
        self.dt = 0.1
        self.l_r = 1.0
        self.target_speed = 10.0
        self.ds = 0.1
        self.track_name = "Nuerburgring"
        self.env = BicycleEnv(
            dt=self.dt,
            l_r=self.l_r,
            target_speed=self.target_speed,
            ds=self.ds,
            track_name=self.track_name,
        )

    def test_env_initialization(self):
        """Test environment initialization and properties."""
        self.assertEqual(self.env.dt, 0.1)
        self.assertEqual(self.env.l_r, 1.0)
        self.assertEqual(self.env.target_speed, 10.0)
        self.assertEqual(self.env.get_wheel_base, self.l_r + self.env.L_F)

        # Check action space
        self.assertEqual(self.env.action_space.shape, (2,))
        self.assertTrue(jnp.all(self.env.action_space.low == -1.0))
        self.assertTrue(jnp.all(self.env.action_space.high == 1.0))

    def test_reset(self):
        """Test environment reset functionality."""
        initial_state = (0.0, 0.0, 0.0, 5.0)
        state, track_obs, info = self.env.reset(*initial_state)

        # Check state initialization
        self.assertTrue(jnp.allclose(state, jnp.array(initial_state)))

        # Check track observation
        self.assertIsInstance(track_obs, TrackReference)
        self.assertEqual(len(track_obs.x.shape), 1)  # Should be 1D array

        # Check info dict
        self.assertIsInstance(info, dict)

    def test_step(self):
        """Test environment step functionality."""
        # Reset to known state
        initial_state = (0.0, 0.0, 0.0, 5.0)
        state, _, _ = self.env.reset(*initial_state)

        # Take a step with zero action
        action = jnp.array([0.0, 0.0])
        next_state, track_obs, reward, done, info = self.env.step(action)

        # Check state update
        self.assertEqual(next_state.shape, (4,))
        self.assertTrue(jnp.all(jnp.isfinite(next_state)))

        # Check speed stays within bounds
        self.assertGreaterEqual(next_state[3], 0.0)
        self.assertLessEqual(next_state[3], self.env.MAX_SPEED)

        # Check basic physics (straight line motion with zero steering)
        self.assertGreater(
            next_state[0], state[0]
        )  # x position should increase
        self.assertAlmostEqual(next_state[1], state[1])  # y should stay same
        self.assertAlmostEqual(
            next_state[2], state[2]
        )  # heading should stay same

    def test_action_normalization(self):
        """Test action normalization function."""
        action = jnp.array([1.0, 1.0])  # Max normalized values
        delta, accel = self.env._unnormalize_actions(action)

        self.assertAlmostEqual(delta, self.env.MAX_STEER)
        self.assertAlmostEqual(accel, self.env.MAX_LONG_ACCEL)

    def test_kinematics(self):
        """Test kinematic model calculations."""
        # Test straight line motion
        x_dot, y_dot, psi_dot = self.env._update_kinematics(
            x=0.0, y=0.0, psi=0.0, v_x=1.0, delta=0.0
        )

        self.assertEqual(x_dot, 1.0)  # Forward motion
        self.assertAlmostEqual(y_dot, 0.0)  # No lateral motion
        self.assertAlmostEqual(psi_dot, 0.0)  # No rotation

    def test_track_reference(self):
        """Test track reference functionality."""
        track_ref = self.env.track_reference

        # Test basic properties
        self.assertIsInstance(track_ref, TrackReference)
        self.assertEqual(track_ref.x.shape, track_ref.y.shape)
        self.assertEqual(track_ref.psi.shape, track_ref.vx.shape)

        # Test reference line generation
        ref_line = track_ref.get_reference_line()
        self.assertEqual(ref_line.shape[1], 2)
        self.assertEqual(ref_line.shape[0], track_ref.x.shape[0])

    def test_gradients(self):
        """Test that gradients can be computed through the kinematic model."""

        def kinematic_fn(inputs):
            x, y, psi, v_x, delta = inputs
            x_dot, y_dot, psi_dot = self.env._update_kinematics(
                x, y, psi, v_x, delta
            )
            return jnp.array([x_dot, y_dot, psi_dot])

        inputs = jnp.array([0.0, 0.0, 0.0, 1.0, 0.1])

        # Check that gradients exist and are finite
        jac_fn = jax.jacfwd(kinematic_fn)
        jacobian = jac_fn(inputs)
        self.assertTrue(jnp.all(jnp.isfinite(jacobian)))


if __name__ == "__main__":
    unittest.main()
