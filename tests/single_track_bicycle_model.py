from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit


@dataclass
class VehicleParams:
    """Vehicle parameters for single track model"""

    m: float  # vehicle mass [kg]
    Izz: float  # moment of inertia around z-axis [kg*m^2]
    a: float  # distance from CG to front axle [m]
    b: float  # distance from CG to rear axle [m]
    h_cm: float  # height of center of mass [m]
    Ca_f: float = 100000.0  # front cornering stiffness [N/rad]
    Ca_r: float = 100000.0  # rear cornering stiffness [N/rad]
    mu: float = 1.0  # friction coefficient [-]

    @property
    def L(self) -> float:
        """Total wheelbase [m]"""
        return self.a + self.b


class SingleTrackModel:
    """
    Single track vehicle model with dynamic weight transfer

    State vector: [x, y, psi, ux, uy, r]
        x: global x position [m]
        y: global y position [m]
        psi: heading angle [rad]
        ux: longitudinal velocity [m/s]
        uy: lateral velocity [m/s]
        r: yaw rate [rad/s]

    Control inputs: (delta, Fxf, Fxr)
        delta: steering angle [rad]
        Fxf: front longitudinal force [N]
        Fxr: rear longitudinal force [N]
    """

    def __init__(self, params: VehicleParams):
        self.p = params

    @partial(jit, static_argnums=(0,))
    def compute_slip_angles(
        self, state: jnp.ndarray, delta: float
    ) -> Tuple[float, float]:
        ux, uy, r = state
        v_threshold = 0.5  # m/s

        # Full calculation
        alpha_f_full = jnp.arctan2(uy + self.p.a * r, ux) - delta
        alpha_r_full = jnp.arctan2(uy - self.p.b * r, ux)

        # Small angle approximation
        alpha_f_small = (uy + self.p.a * r) / jnp.maximum(
            jnp.abs(ux), 0.1
        ) - delta
        alpha_r_small = (uy - self.p.b * r) / jnp.maximum(jnp.abs(ux), 0.1)

        # Blend based on velocity
        v = jnp.sqrt(ux**2 + uy**2)
        blend = jnp.clip(v / v_threshold, 0.0, 1.0)

        alpha_f = blend * alpha_f_full + (1 - blend) * alpha_f_small
        alpha_r = blend * alpha_r_full + (1 - blend) * alpha_r_small

        return alpha_f, alpha_r

    @partial(jit, static_argnums=(0,))
    def compute_normal_forces(
        self, Fxf: float, Fxr: float
    ) -> Tuple[float, float]:
        """
        Compute normal forces including longitudinal weight transfer

        Args:
            Fxf: front longitudinal force [N]
            Fxr: rear longitudinal force [N]

        Returns:
            Fzf, Fzr: front and rear normal forces [N]
        """
        g = 9.81  # gravitational acceleration [m/s^2]

        Fzf = (1 / self.p.L) * (
            self.p.b * self.p.m * g - self.p.h_cm * (Fxf + Fxr)
        )

        Fzr = (1 / self.p.L) * (
            self.p.a * self.p.m * g + self.p.h_cm * (Fxf + Fxr)
        )

        return Fzf, Fzr

    @partial(jit, static_argnums=(0,))
    def compute_lateral_force(
        self, alpha: float, Fz: float, Fx: float, Ca: float, mu: float
    ) -> float:
        """
        Compute lateral force using simplified brush tire model

        Args:
            alpha: slip angle [rad]
            Fz: normal load [N]
            Fx: longitudinal force [N]
            Ca: cornering stiffness [N/rad]
            mu: friction coefficient [-]

        Returns:
            Fy: lateral force [N]
        """
        # Compute maximum lateral force
        Fy_max = jnp.sqrt((mu * Fz) ** 2 - Fx**2)

        # Compute critical slip angle
        alpha_crit = jnp.arctan(3 * Fy_max / Ca)

        # Compute forces for both cases
        linear_case = (
            -Ca * jnp.tan(alpha)
            + (Ca**2 / (3 * Fy_max)) * jnp.abs(jnp.tan(alpha)) * jnp.tan(alpha)
            - (Ca**3 / (27 * (Fy_max) ** 2)) * jnp.tan(alpha) ** 3
        )

        saturated_case = -Fy_max * jnp.sign(alpha)

        # Use where for smooth conditional
        return jnp.where(
            jnp.abs(alpha) < alpha_crit, linear_case, saturated_case
        )

    @partial(jit, static_argnums=(0,))
    def compute_forces(
        self, state: jnp.ndarray, controls: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Compute the tire forces given the current state and controls

        Args:
            state: vehicle state [ux, uy, r]
            controls: (delta, Fxf, Fxr) steering angle and longitudinal forces

        Returns:
            Fyf, Fyr, Fxdrag: lateral forces and drag [N]
        """
        delta, Fxf, Fxr = controls

        # Compute slip angles
        alpha_f, alpha_r = self.compute_slip_angles(state, delta)

        # Compute normal loads with weight transfer
        Fz_f, Fz_r = self.compute_normal_forces(Fxf, Fxr)

        # Compute lateral forces using brush tire model
        Fyf = self.compute_lateral_force(
            alpha_f, Fz_f, Fxf, self.p.Ca_f, self.p.mu
        )
        Fyr = self.compute_lateral_force(
            alpha_r, Fz_r, Fxr, self.p.Ca_r, self.p.mu
        )

        # Simple quadratic drag model
        Fxdrag = 0.5 * state[0] ** 2  # proportional to velocity squared

        return Fyf, Fyr, Fxdrag

    @partial(jit, static_argnums=(0,))
    def state_derivatives(
        self, state: jnp.ndarray, controls: Tuple[float, float, float]
    ) -> jnp.ndarray:
        """
        Compute the state derivatives given current state and inputs

        Args:
            state: [x, y, psi, ux, uy, r]
            controls: (delta, Fxf, Fxr)

        Returns:
            state derivatives [x_dot, y_dot, psi_dot, ux_dot, uy_dot, r_dot]
        """
        # Extract states
        x, y, psi, ux, uy, r = state
        delta, Fxf, Fxr = controls

        # Compute forces
        dynamic_state = state[3:]  # [ux, uy, r]
        Fyf, Fyr, Fxdrag = self.compute_forces(dynamic_state, controls)

        # Kinematic derivatives
        x_dot = ux * jnp.cos(psi) - uy * jnp.sin(psi)
        y_dot = ux * jnp.sin(psi) + uy * jnp.cos(psi)
        psi_dot = r

        # Dynamic derivatives
        r_dot = (1 / self.p.Izz) * (
            self.p.a * Fyf * jnp.cos(delta)
            + self.p.a * Fxf * jnp.sin(delta)
            - self.p.b * Fyr
        )

        uy_dot = (1 / self.p.m) * (
            Fyf * jnp.cos(delta) + Fxf * jnp.sin(delta) + Fyr
        ) - ux * r

        ux_dot = (1 / self.p.m) * (
            -Fyf * jnp.sin(delta) + Fxf * jnp.cos(delta) + Fxr - Fxdrag
        ) + uy * r

        return jnp.array([x_dot, y_dot, psi_dot, ux_dot, uy_dot, r_dot])


def main():
    """Example usage of the single track model"""
    # Create vehicle parameters
    params = VehicleParams(
        m=1500.0,  # kg
        Izz=2500.0,  # kg*m^2
        a=1.0,  # m
        b=1.5,  # m
        h_cm=0.5,  # m
        Ca_f=100000.0,  # N/rad
        Ca_r=100000.0,  # N/rad
        mu=1.0,  # friction coefficient
    )

    # Create model
    model = SingleTrackModel(params)

    # Initial state [x, y, psi, ux, uy, r]
    state = jnp.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0])

    # Example control inputs
    delta = jnp.deg2rad(5.0)  # 5 degrees steering
    Fxf = 1000.0  # 1000 N front longitudinal force
    Fxr = 1000.0  # 1000 N rear longitudinal force
    controls = (delta, Fxf, Fxr)

    # Compute derivatives
    derivatives = model.state_derivatives(state, controls)
    print(
        "\nState derivatives:",
        {
            "x_dot": f"{derivatives[0]:.2f} m/s",
            "y_dot": f"{derivatives[1]:.2f} m/s",
            "psi_dot": f"{jnp.rad2deg(derivatives[2]):.2f} deg/s",
            "ux_dot": f"{derivatives[3]:.2f} m/s^2",
            "uy_dot": f"{derivatives[4]:.2f} m/s^2",
            "r_dot": f"{jnp.rad2deg(derivatives[5]):.2f} deg/s^2",
        },
    )


if __name__ == "__main__":
    main()
