from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from plot_utils import plot_kinematic_bicycle_results
from trajax import optimizers

from environment.vehicle_env import create_track_reference
from math_lib.signed_distance import point_to_polyline_signed_distance
from primal_dual_ilqr.optimizers import primal_dual_ilqr

jax.config.update("jax_enable_x64", True)


@jax.jit
def kinematic_single_track(state, control, timestep):
    del timestep
    _, _, psi, vx, kappa, ax = state
    swirl, jerk = control

    # equations of motion
    x_dot = vx * jnp.cos(psi)
    y_dot = vx * jnp.sin(psi)
    psi_dot = kappa * vx
    vx_dot = ax
    kappa_dot = swirl
    ax_dot = jerk

    return jnp.array([x_dot, y_dot, psi_dot, vx_dot, kappa_dot, ax_dot])


track_reference = create_track_reference(
    track_name="Nuerburgring", target_speed=30, ds=1.0
)

horizon: int = 50
dt: float = 0.1
start_idx: int = 0


def dynamics(x, u, t):
    return x + dt * kinematic_single_track(x, u, t)


@jax.jit
def get_target_state(state: jnp.ndarray, track_reference):
    """
    Get target state based on spatial projection to reference line.

    Args:
        state: Current state vector [x, y, psi, vx, kappa, ax]
        t: Current time (not used in spatial projection but kept for API consistency)
        track_reference: Track reference object with attributes:
            - reference_line: Array of points defining the reference line
            - x, y: Arrays of x and y coordinates
            - psi: Array of heading angles
            - vx: Array of target velocities
            - kappa: Array of curvatures

    Returns:
        jnp.ndarray: Target state vector [x, y, psi, vx, kappa, ax]
    """
    # Extract position from state
    position = state[:2]

    # Calculate signed distance and closest point index
    e_y, min_idx = point_to_polyline_signed_distance(
        position, track_reference.get_reference_line()
    )

    # Get target state components
    psi = track_reference.psi[min_idx]
    vx = track_reference.vx[min_idx]
    kappa = track_reference.kappa[min_idx]

    return jnp.array([e_y, psi, vx, kappa])


def cost(x, u, t):
    target = get_target_state(x, track_reference)
    err_xy = target[0]
    err_yaw = x[2] - target[1]
    err_vx = x[3] - target[2]
    err_kappa = x[4] - target[3]

    w_xy = 0.5
    w_yaw = 0.9
    w_vx = 1.0
    w_kappa = 0.1

    cv = (
        jnp.maximum(jnp.abs(u[1]) - 2.0, 0.0)
        + jnp.maximum(x[5] - 4.0, 0.0)
        + jnp.maximum(-10.0 - x[5], 0.0)
        + jnp.maximum(-x[3], 0.0)
    )

    stage_cost = (
        w_xy * jnp.dot(err_xy, err_xy)
        + w_vx * jnp.dot(err_vx, err_vx)
        + w_yaw * jnp.dot(err_yaw, err_yaw)
        + w_kappa * jnp.dot(err_kappa, err_kappa)
        + 10.0 * jnp.dot(u, u)
        + 1000.0 * jnp.dot(cv, cv)
    )
    final_cost = (
        w_xy * jnp.dot(err_xy, err_xy)
        + w_vx * jnp.dot(err_vx, err_vx)
        + w_yaw * jnp.dot(err_yaw, err_yaw)
        + w_kappa * jnp.dot(err_kappa, err_kappa)
    )

    return jnp.where(t == horizon, final_cost, stage_cost)


x0 = jnp.array(
    [
        track_reference.x[start_idx],
        track_reference.y[start_idx],
        track_reference.psi[start_idx],
        track_reference.vx[start_idx],
        track_reference.kappa[start_idx],
        0.0,
    ]
)
u0 = jnp.zeros([horizon, 2])
X_warm_start = optimizers.rollout(dynamics, u0, x0)
V0 = jnp.zeros([horizon + 1, 6])

# from timeit import default_timer as timer

MAX_ITER: int = 500


# @jax.jit
# def work_ilqr():
#     return optimizers.ilqr(cost, dynamics, x0, u0, maxiter=MAX_ITER)


# X, U, obj, grad, adj, lqr, iter = work_ilqr()
# X.block_until_ready()
# start = timer()
# n = 1
# for i in range(n):
#     X, _, _, _, _, _, _ = work_ilqr()
#     X.block_until_ready()
# end = timer()
# t = (end - start) / n
# print(f"Trajax iLQR result: {obj=} {iter=}, time: {t:.4f} seconds")
# plot_kinematic_bicycle_results(X, U, dt, "ilqr_results", track_reference)


# @jax.jit
# def work_primal_dual():
#     return primal_dual_ilqr(
#         cost,
#         dynamics,
#         x0,
#         X_warm_start,
#         u0,
#         V0,
#         max_iterations=MAX_ITER,
#     )


# X, U, V, iter, obj, c, no_errors = work_primal_dual()
# X.block_until_ready()
# start = timer()
# n = 1
# for i in range(n):
#     X, _, _, _, _, _, _ = work_primal_dual()
#     X.block_until_ready()
# end = timer()
# t = (end - start) / n
# print(f"Primal dual result: {obj=} {iter=}, time: {t:.4f} seconds")
# plot_kinematic_bicycle_results(X, U, dt, "primal_dual_results", track_reference)


def mpc_step(carry, t, horizon):
    """Single step of MPC using iLQR.

    Args:
        carry: tuple of (current_state, u0)
        t: time step
        horizon: MPC horizon (static)

    Returns:
        tuple: (next_carry, (state, control))
    """
    current_state, u0 = carry

    # Run iLQR optimization
    X, U, _, _, _, _, _ = optimizers.ilqr(
        cost, dynamics, current_state, u0, maxiter=MAX_ITER
    )

    # Extract first control action
    control = U[0]

    # Simulate one step forward
    next_state = dynamics(current_state, control, t)

    # Prepare warm start for next iteration (shift controls forward)
    next_u0 = jnp.vstack([U[1:], U[-1]])

    return (next_state, next_u0), (current_state, control)


@partial(jax.jit, static_argnums=(1, 2, 3))
def run_mpc(initial_state, simulation_steps, mpc_horizon, dt):
    """
    Run Model Predictive Control using iLQR and lax.scan.

    Args:
        initial_state: Initial state vector [x, y, psi, vx, kappa, ax]
        simulation_steps: Number of simulation steps (static)
        mpc_horizon: MPC prediction horizon (static)
        dt: Time step size (static)

    Returns:
        tuple: Arrays of states and controls over time
    """
    # Initial control sequence for warm starting
    u0 = jnp.zeros([mpc_horizon, 2])

    # Run the MPC loop using lax.scan
    (final_state, final_u0), (states, controls) = jax.lax.scan(
        lambda c, t: mpc_step(c, t, mpc_horizon),
        (initial_state, u0),
        jnp.arange(simulation_steps),
    )

    # Add final state to trajectory
    states = jnp.vstack([states, final_state[None, :]])

    return states, controls


simulation_steps = 1700  # Total simulation steps
mpc_horizon = 50  # MPC prediction horizon

from timeit import default_timer as timer

start = timer()
X_mpc, U_mpc = run_mpc(x0, simulation_steps, mpc_horizon, dt)
X_mpc.block_until_ready()
end = timer()
t = end - start
print(f"MPC using trajax: {t:.4f} seconds")

plot_kinematic_bicycle_results(X_mpc, U_mpc, dt, "mpc_results", track_reference)
