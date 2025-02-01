import unittest

import jax.numpy as jnp
import numpy as np
from jax import grad, jit

from tests.single_track_bicycle_model import SingleTrackModel, VehicleParams


class TestSingleTrackModel(unittest.TestCase):
    def setUp(self):
        """Create a vehicle model instance for testing"""
        self.params = VehicleParams(
            m=1500.0,  # mass [kg]
            Izz=2500.0,  # moment of inertia [kg*m^2]
            a=1.0,  # distance CG to front axle [m]
            b=1.5,  # distance CG to rear axle [m]
            h_cm=0.5,  # CM height [m]
            Ca_f=100000.0,  # front cornering stiffness [N/rad]
            Ca_r=100000.0,  # rear cornering stiffness [N/rad]
            mu=1.0,  # friction coefficient [-]
        )
        self.model = SingleTrackModel(self.params)

    def test_wheelbase(self):
        """Test wheelbase calculation"""
        self.assertEqual(self.params.L, 2.5)

    def test_slip_angles_straight(self):
        """Test slip angles for straight driving"""
        state = jnp.array([10.0, 0.0, 0.0])  # [ux, uy, r]
        delta = 0.0
        alpha_f, alpha_r = self.model.compute_slip_angles(state, delta)

        # Both slip angles should be zero for straight driving
        np.testing.assert_almost_equal(alpha_f, 0.0, decimal=6)
        np.testing.assert_almost_equal(alpha_r, 0.0, decimal=6)

    def test_slip_angles_steering(self):
        """Test slip angles with steering input"""
        state = jnp.array([10.0, 0.0, 0.0])  # [ux, uy, r]
        delta = jnp.deg2rad(5.0)  # 5 degrees steering
        alpha_f, alpha_r = self.model.compute_slip_angles(state, delta)

        # Front slip angle should be negative of steering angle
        # Rear slip angle should be zero
        np.testing.assert_almost_equal(alpha_f, -delta, decimal=6)
        np.testing.assert_almost_equal(alpha_r, 0.0, decimal=6)

    def test_normal_forces_static(self):
        """Test normal forces in static condition"""
        Fxf = 0.0
        Fxr = 0.0
        Fzf, Fzr = self.model.compute_normal_forces(Fxf, Fxr)

        # Calculate expected static loads
        total_weight = self.params.m * 9.81
        expected_Fzf = total_weight * (self.params.b / self.params.L)
        expected_Fzr = total_weight * (self.params.a / self.params.L)

        np.testing.assert_almost_equal(Fzf, expected_Fzf, decimal=2)
        np.testing.assert_almost_equal(Fzr, expected_Fzr, decimal=2)
        np.testing.assert_almost_equal(Fzf + Fzr, total_weight, decimal=2)

    def test_normal_forces_acceleration(self):
        """Test normal forces during acceleration"""
        Fxf = 1000.0
        Fxr = 1000.0
        Fzf, Fzr = self.model.compute_normal_forces(Fxf, Fxr)

        # During acceleration, rear normal force should increase
        static_Fzf, static_Fzr = self.model.compute_normal_forces(0.0, 0.0)
        self.assertLess(Fzf, static_Fzf)
        self.assertGreater(Fzr, static_Fzr)

    def test_lateral_force_linear_region(self):
        """Test lateral force calculation in linear region"""
        alpha = jnp.deg2rad(1.0)  # Small slip angle
        Fz = 5000.0
        Fx = 0.0
        Ca = 100000.0
        mu = 1.0

        Fy = self.model.compute_lateral_force(alpha, Fz, Fx, Ca, mu)

        # Compute Fy_max for brush tire model
        Fy_max = jnp.sqrt((mu * Fz) ** 2 - Fx**2)

        # Full brush tire model equation for linear region
        expected_Fy = (
            -Ca * jnp.tan(alpha)
            + (Ca**2 / (3 * Fy_max)) * jnp.abs(jnp.tan(alpha)) * jnp.tan(alpha)
            - (Ca**3 / (27 * (Fy_max) ** 2)) * jnp.tan(alpha) ** 3
        )

        np.testing.assert_almost_equal(Fy, expected_Fy, decimal=4)

    def test_lateral_force_saturation(self):
        """Test lateral force saturation"""
        alpha = jnp.deg2rad(20.0)  # Large slip angle
        Fz = 5000.0
        Fx = 0.0
        Ca = 100000.0
        mu = 1.0

        Fy = self.model.compute_lateral_force(alpha, Fz, Fx, Ca, mu)

        # Force should be saturated at mu*Fz
        self.assertLessEqual(abs(Fy), mu * Fz)

    def test_state_derivatives_straight(self):
        """Test state derivatives for straight driving"""
        state = jnp.array(
            [0.0, 0.0, 0.0, 20.0, 0.0, 0.0]
        )  # [x, y, psi, ux, uy, r]
        controls = (0.0, 0.0, 0.0)  # No steering or forces

        derivatives = self.model.state_derivatives(state, controls)

        # Should only have forward motion
        np.testing.assert_almost_equal(derivatives[0], 20.0)  # x_dot
        np.testing.assert_almost_equal(derivatives[1], 0.0)  # y_dot
        np.testing.assert_almost_equal(derivatives[2], 0.0)  # psi_dot
        self.assertLess(derivatives[3], 0.0)  # ux_dot (drag deceleration)
        np.testing.assert_almost_equal(derivatives[4], 0.0)  # uy_dot
        np.testing.assert_almost_equal(derivatives[5], 0.0)  # r_dot

    def test_forces_continuity(self):
        """Test continuity of force calculations"""
        state = jnp.array([10.0, 0.0, 0.0])
        controls = (0.1, 100.0, 100.0)

        # Get gradient of forces with respect to state
        forces_grad = jit(
            grad(lambda s: sum(self.model.compute_forces(s, controls)))
        )

        # Gradient should exist and be finite
        grad_values = forces_grad(state)
        self.assertTrue(jnp.all(jnp.isfinite(grad_values)))


if __name__ == "__main__":
    unittest.main()
