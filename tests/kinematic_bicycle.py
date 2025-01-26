import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from trajax import optimizers

from primal_dual_ilqr.optimizers import primal_dual_ilqr
from plot_utils import plot_kinematic_bicycle_results

from enum import Enum, auto


class DiscretizationMethod(Enum):
    EULER = auto()
    EXPM = auto()


discretization_method: DiscretizationMethod = DiscretizationMethod.EULER


@jax.jit
def kinematic_model_expm_discretization(state, control, timestep):
    A = jnp.array(
        [
            [0, 0, -state[3] * jnp.sin(state[2]), jnp.cos(state[2]), 0, 0],
            [0, 0, state[3] * jnp.cos(state[2]), jnp.sin(state[2]), 0, 0],
            [0, 0, 0, state[4], state[3], 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    B = jnp.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]])

    if discretization_method == DiscretizationMethod.EULER:
        next_state = A * timestep @ state + B * timestep @ control
    elif discretization_method == DiscretizationMethod.EXPM:
        BigA = jnp.block(
            [
                [A, B],
                [
                    jnp.zeros((control.shape[0], state.shape[0])),
                    jnp.zeros((control.shape[0], control.shape[0])),
                ],
            ]
        )

        BigAd = expm(BigA * timestep)

        Ad = BigAd[: state.shape[0], : state.shape[0]]
        Bd = BigAd[: state.shape[0], state.shape[0] :]

        next_state = Ad @ state + Bd @ control
    else:
        raise ValueError(
            f"Unknown discretization method: {discretization_method}"
        )

    return next_state


horizon = 200
dt = 0.1


def dynamics(x, u, t):
    return kinematic_model_expm_discretization(x, u, t)


target_state = jnp.array([0.0, 3.0, 0.0, 10.0, 0.0, 0.0])


def cost(x, u, t):
    y_error = x[1:] - target_state[1:]
    yaw_error = x[2:] - target_state[2:]
    speed_error = x[3:] - target_state[3:]
    w_y = 0.1
    w_yaw = 0.0
    w_speed = 0.05

    cv = jnp.maximum(jnp.abs(u[1]) - 2.0, 0.0)
    stage_cost = (
        w_y * jnp.dot(y_error, y_error)
        + w_speed * jnp.dot(speed_error, speed_error)
        + w_yaw * jnp.dot(yaw_error, yaw_error)
        + 10.0 * jnp.dot(u, u)
        + 10000.0 * jnp.dot(cv, cv)
    )
    final_cost = (
        w_y * jnp.dot(y_error, y_error)
        + w_speed * jnp.dot(speed_error, speed_error)
        + w_yaw * jnp.dot(yaw_error, yaw_error)
    )
    return jnp.where(t == horizon, final_cost, stage_cost)


x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
u0 = jnp.zeros([horizon, 2])
X_warm_start = optimizers.rollout(dynamics, u0, x0)
V0 = jnp.zeros([horizon + 1, 6])

from timeit import default_timer as timer


@jax.jit
def work_ilqr():
    return optimizers.ilqr(cost, dynamics, x0, u0, maxiter=1000)


X, U, obj, grad, adj, lqr, iter = work_ilqr()
X.block_until_ready()

start = timer()

n = 100
for i in range(n):
    X, _, _, _, _, _, _ = work_ilqr()
    X.block_until_ready()

end = timer()

t = (end - start) / n

print(f"Trajax iLQR result: {obj=} {iter=}, time: {t:.4f} seconds")
plot_kinematic_bicycle_results(X, U, dt)


@jax.jit
def work_primal_dual():
    return primal_dual_ilqr(
        cost,
        dynamics,
        x0,
        X_warm_start,
        u0,
        V0,
        max_iterations=1000,
    )


X, U, V, iter, obj, c, no_errors = work_primal_dual()
X.block_until_ready()


start = timer()

n = 100
for i in range(n):
    X, _, _, _, _, _, _ = work_primal_dual()
    X.block_until_ready()

end = timer()

t = (end - start) / n
print(f"Primal dual result: {obj=} {iter=}, time: {t:.4f} seconds")
