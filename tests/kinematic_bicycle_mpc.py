"""
Vehicle trajectory optimization using Model Predictive Control (MPC) with a kinematic bicycle model.
Uses JAX for acceleration and automatic differentiation.
"""

from dataclasses import dataclass
from functools import partial
from timeit import default_timer as timer
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from plot_utils import plot_kinematic_bicycle_results
from trajax import optimizers

from environment.vehicle_env import create_track_reference
from math_lib.signed_distance import point_to_polyline_signed_distance

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)


class MPCConfig:
    """Configuration parameters for MPC."""

    horizon: int = 50
    dt: float = 0.1
    max_iterations: int = 500
    simulation_steps: int = 1700

    # Cost weights
    w_xy: float = 0.5  # Position error weight
    w_yaw: float = 0.9  # Yaw error weight
    w_vx: float = 1.0  # Velocity error weight
    w_kappa: float = 0.1  # Curvature error weight
    w_control: float = 10.0  # Control input weight
    w_constraint: float = 1000.0  # Constraint violation weight

    # Cost weights
    w_xy: float = 0.5  # Position error weight
    w_yaw: float = 0.9  # Yaw error weight
    w_vx: float = 1.0  # Velocity error weight
    w_kappa: float = 0.1  # Curvature error weight
    w_control: float = 10.0  # Control input weight
    w_constraint: float = 1000.0  # Constraint violation weight


class VehicleState(NamedTuple):
    """Vehicle state representation."""

    x: float  # X position
    y: float  # Y position
    psi: float  # Yaw angle
    vx: float  # Longitudinal velocity
    kappa: float  # Curvature
    ax: float  # Longitudinal acceleration


class VehicleControl(NamedTuple):
    """Vehicle control inputs."""

    swirl: float  # Rate of curvature change
    jerk: float  # Rate of acceleration change


@jax.jit
def kinematic_bicycle_dynamics(
    state: jnp.ndarray, control: jnp.ndarray, timestep: float
) -> jnp.ndarray:
    """
    Kinematic bicycle model dynamics.

    Args:
        state: Current vehicle state [x, y, psi, vx, kappa, ax]
        control: Control inputs [swirl, jerk]
        timestep: Integration timestep (unused in this model)

    Returns:
        State derivatives [dx/dt, dy/dt, dpsi/dt, dvx/dt, dkappa/dt, dax/dt]
    """
    _, _, psi, vx, kappa, ax = state
    swirl, jerk = control

    return jnp.array(
        [
            vx * jnp.cos(psi),  # x_dot
            vx * jnp.sin(psi),  # y_dot
            kappa * vx,  # psi_dot
            ax,  # vx_dot
            swirl,  # kappa_dot
            jerk,  # ax_dot
        ]
    )


@jax.jit
def get_target_state(state: jnp.ndarray, track_reference) -> jnp.ndarray:
    """
    Get target state based on spatial projection to reference line.

    Args:
        state: Current state vector [x, y, psi, vx, kappa, ax]
        track_reference: Track reference object containing waypoints and target values

    Returns:
        Target state vector [cross_track_error, psi, vx, kappa]
    """
    position = state[:2]

    # Calculate signed distance and closest point index
    e_y, min_idx = point_to_polyline_signed_distance(
        position, track_reference.get_reference_line()
    )

    return jnp.array(
        [
            e_y,
            track_reference.psi[min_idx],
            track_reference.vx[min_idx],
            track_reference.kappa[min_idx],
        ]
    )


def create_stage_cost_fn(config: MPCConfig, track_reference):
    """Creates the stage cost function for the MPC optimization."""

    @jax.jit
    def cost(x: jnp.ndarray, u: jnp.ndarray, t: int) -> float:
        """
        Compute stage cost for MPC optimization.

        Args:
            x: Current state
            u: Control inputs
            t: Current timestep

        Returns:
            Total cost including tracking errors and control penalties
        """
        target = get_target_state(x, track_reference)

        # Compute tracking errors
        err_xy = target[0]
        err_yaw = x[2] - target[1]
        err_vx = x[3] - target[2]
        err_kappa = x[4] - target[3]

        # Compute constraint violations
        cv = (
            jnp.maximum(jnp.abs(u[1]) - 2.0, 0.0)  # Jerk limits
            + jnp.maximum(x[5] - 4.0, 0.0)  # Max acceleration
            + jnp.maximum(-10.0 - x[5], 0.0)  # Min acceleration
            + jnp.maximum(-x[3], 0.0)  # Non-negative velocity
        )

        # Weighted sum of tracking errors
        tracking_cost = (
            config.w_xy * jnp.dot(err_xy, err_xy)
            + config.w_vx * jnp.dot(err_vx, err_vx)
            + config.w_yaw * jnp.dot(err_yaw, err_yaw)
            + config.w_kappa * jnp.dot(err_kappa, err_kappa)
        )

        # Add control and constraint costs (zero for terminal stage)
        control_cost = config.w_control * jnp.dot(
            u, u
        ) + config.w_constraint * jnp.dot(cv, cv)

        return jnp.where(
            t == config.horizon,
            tracking_cost,  # Terminal cost
            tracking_cost + control_cost,  # Stage cost
        )

    return cost


def create_dynamics_fn(dt: float):
    """Creates the discrete dynamics function using forward Euler integration."""

    def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: float) -> jnp.ndarray:
        return x + dt * kinematic_bicycle_dynamics(x, u, t)

    return jax.jit(dynamics)


@partial(jax.jit, static_argnums=(1, 2, 3))
def run_mpc(
    initial_state: jnp.ndarray,
    simulation_steps: int,
    mpc_horizon: int,
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run Model Predictive Control simulation.

    Args:
        initial_state: Initial vehicle state
        simulation_steps: Number of simulation timesteps
        mpc_horizon: MPC prediction horizon length
        dt: Timestep size

    Returns:
        Tuple of (states, controls) arrays over the simulation horizon
    """
    # Initial control sequence for warm starting
    u0 = jnp.zeros([mpc_horizon, 2])

    def mpc_step(carry, t):
        """Single step of MPC using iLQR."""
        current_state, u0 = carry

        # Run iLQR optimization
        X, U, _, _, _, _, _ = optimizers.ilqr(
            cost_fn,
            dynamics_fn,
            current_state,
            u0,
            maxiter=MPCConfig.max_iterations,
        )

        # Extract first control action and simulate forward
        control = U[0]
        next_state = dynamics_fn(current_state, control, t)

        # Prepare warm start for next iteration
        next_u0 = jnp.vstack([U[1:], U[-1]])

        return (next_state, next_u0), (current_state, control)

    # Run the MPC loop using scan
    (final_state, _), (states, controls) = jax.lax.scan(
        mpc_step, (initial_state, u0), jnp.arange(simulation_steps)
    )

    # Add final state to trajectory
    states = jnp.vstack([states, final_state[None, :]])

    return states, controls


def main():
    """Main execution function."""
    # Create configuration
    config = MPCConfig()

    # Load track reference
    track_reference = create_track_reference(
        track_name="Nuerburgring", target_speed=30, ds=1.0
    )

    # Create cost and dynamics functions
    global cost_fn  # Need global for jitted mpc_step
    global dynamics_fn
    cost_fn = create_stage_cost_fn(config, track_reference)
    dynamics_fn = create_dynamics_fn(config.dt)

    # Set initial state
    x0 = jnp.array(
        [
            track_reference.x[0],
            track_reference.y[0],
            track_reference.psi[0],
            track_reference.vx[0],
            track_reference.kappa[0],
            0.0,
        ]
    )

    # Run MPC simulation
    print("Starting MPC simulation...")
    start = timer()
    states, controls = run_mpc(
        x0, config.simulation_steps, config.horizon, config.dt
    )
    states.block_until_ready()  # Ensure computation is complete
    end = timer()
    print(f"MPC simulation completed in {end - start:.4f} seconds")

    # Plot results
    print("Plotting results...")
    plot_kinematic_bicycle_results(
        states, controls, config.dt, "mpc_results", track_reference
    )


if __name__ == "__main__":
    main()
