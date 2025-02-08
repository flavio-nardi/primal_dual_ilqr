from enum import Enum, auto
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from plot_utils import plot_kinematic_bicycle_results
from trajax import optimizers

from primal_dual_ilqr.constrained_optimizers import constrained_primal_dual_ilqr
from primal_dual_ilqr.optimizers import primal_dual_ilqr


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


horizon = 300
dt = 0.1


def dynamics(x, u, t):
    return x + dt * kinematic_single_track(x, u, t)


target_state = jnp.array([100.0, 30.0, 0.0, 0.0, 0.0, 0.0])
n = 10

horizon = 200
dt = 0.1
x0 = jnp.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
u0 = jnp.zeros([horizon, 2])
X_warm_start = optimizers.rollout(dynamics, u0, x0)
V0 = jnp.zeros([horizon + 1, 6])

target_state = jnp.array([0.0, 11.0, 3.14, 0.0, 0.0, 0.0])


def al_cost(x, u, t):
    err_xy = x[:2] - target_state[:2]
    err_yaw = x[2] - target_state[2]
    err_vx = x[3] - target_state[3]
    w_xy = 10.0
    w_yaw = 100.0
    w_vx = 10.0

    stage_cost = 10.0 * jnp.dot(u, u)
    final_cost = (
        w_xy * jnp.dot(err_xy, err_xy)
        + w_vx * jnp.dot(err_vx, err_vx)
        + w_yaw * jnp.dot(err_yaw, err_yaw)
    )
    return jnp.where(t == horizon, final_cost, stage_cost)


inequality_constraint = jnp.array

start = timer()
for i in range(n):
    X, U, V, iteration_ilqr, iteration_al, no_errors = (
        constrained_primal_dual_ilqr(
            al_cost,
            dynamics,
            x0,
            X_warm_start,
            u0,
            V0,
            equality_constraint=lambda x, u, t: jnp.empty(1),
            inequality_constraint=lambda x, u, t: jnp.array(
                [x[2] - jnp.pi, -x[3], x[4] - 0.18]
            ),
        )
    )
    X.block_until_ready()
end = timer()
t = (end - start) / n

print(
    f"Primal dual aug lag result: {iteration_ilqr=} {iteration_al=}, time: {t:.4f} seconds"
)
plot_kinematic_bicycle_results(X, U, dt, "primal_dual_aug_lag_results")
