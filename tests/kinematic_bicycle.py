from enum import Enum, auto

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from plot_utils import plot_kinematic_bicycle_results
from trajax import optimizers

from primal_dual_ilqr.constrained_optimizers import constrained_primal_dual_ilqr
from primal_dual_ilqr.optimizers import primal_dual_ilqr


class DiscretizationMethod(Enum):
    EULER = auto()
    EXPM = auto()


discretization_method: DiscretizationMethod = DiscretizationMethod.EULER


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


def cost(x, u, t):
    err_xy = x[:2] - target_state[:2]
    err_yaw = x[2] - target_state[2]
    err_vx = x[3] - target_state[3]
    w_xy = 1000.0
    w_yaw = 1000.0
    w_vx = 1000.0

    cv = (
        jnp.maximum(jnp.abs(u[1]) - 2.0, 0.0)
        + jnp.maximum(x[5] - 4.0, 0.0)
        + jnp.maximum(-10.0 - x[5], 0.0)
        + jnp.maximum(-x[3], 0.0)
    )
    stage_cost = 100.0 * jnp.dot(u, u) + 1000.0 * jnp.dot(cv, cv)
    final_cost = (
        w_xy * jnp.dot(err_xy, err_xy)
        + w_vx * jnp.dot(err_vx, err_vx)
        + w_yaw * jnp.dot(err_yaw, err_yaw)
    )
    return jnp.where(t == horizon, final_cost, stage_cost)


x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
u0 = jnp.zeros([horizon, 2])
X_warm_start = optimizers.rollout(dynamics, u0, x0)
V0 = jnp.zeros([horizon + 1, 6])

from timeit import default_timer as timer

MAX_ITER: int = 500


@jax.jit
def work_ilqr():
    return optimizers.ilqr(cost, dynamics, x0, u0, maxiter=MAX_ITER)


X, U, obj, grad, adj, lqr, iter = work_ilqr()
X.block_until_ready()

start = timer()

n = 10
for i in range(n):
    X, _, _, _, _, _, _ = work_ilqr()
    X.block_until_ready()

end = timer()

t = (end - start) / n

print(f"Trajax iLQR result: {obj=} {iter=}, time: {t:.4f} seconds")
plot_kinematic_bicycle_results(X, U, dt, "ilqr_results")


@jax.jit
def work_primal_dual():
    return primal_dual_ilqr(
        cost,
        dynamics,
        x0,
        X_warm_start,
        u0,
        V0,
        max_iterations=MAX_ITER,
    )


X, U, V, iter, obj, c, no_errors = work_primal_dual()
X.block_until_ready()


start = timer()

for i in range(n):
    X, _, _, _, _, _, _ = work_primal_dual()
    X.block_until_ready()

end = timer()

t = (end - start) / n
print(f"Primal dual result: {obj=} {iter=}, time: {t:.4f} seconds")
plot_kinematic_bicycle_results(X, U, dt, "primal_dual_results")
