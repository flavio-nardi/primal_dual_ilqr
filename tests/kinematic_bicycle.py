import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from trajax import optimizers

from primal_dual_ilqr.optimizers import primal_dual_ilqr


@jax.jit
def kinematic_model_expm_discretization(state, control, timestep):
    A = jnp.array(
        [
            [0, 0, -state[3] * jnp.sin(state[2]), jnp.cos(state[2]), 0, 0],
            [0, 0, state[3] * jnp.cos(state[2]), jnp.sin(state[2]), 0, 0],
            [0, 0, 0, 0, state[3], 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    B = jnp.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]])

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
    return next_state


horizon = 50
dt = 0.1


def dynamics(x, u, t):
    return kinematic_model_expm_discretization(x, u, t)


target_state = jnp.array([10, 0.0, 0.0, 0.0, 0.0, 0.0])


def cost(x, u, t):
    err = x - target_state
    # cv = jnp.maximum(jnp.abs(u[1]) - 4.0, 0.0)
    stage_cost = (
        1.0 * jnp.dot(err, err)
        + 1.0 * jnp.dot(u, u)
        # + 10000.0 * jnp.dot(cv, cv)
    )
    final_cost = 1.0 * jnp.dot(err, err)
    return jnp.where(t == horizon, final_cost, stage_cost)


x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
u0 = jnp.zeros([horizon, 2])
X_warm_start = optimizers.rollout(dynamics, u0, x0)
V0 = jnp.zeros([horizon + 1, 6])

from timeit import default_timer as timer


@jax.jit
def work():
    return optimizers.ilqr(cost, dynamics, x0, u0, maxiter=1000)


X, U, obj, grad, adj, lqr, iter = work()
X.block_until_ready()

start = timer()

n = 100
for i in range(n):
    X, _, _, _, _, _, _ = work()
    X.block_until_ready()

end = timer()

t = (end - start) / n

print(f"Trajax iLQR result: {obj=} {iter=}, time: {t:.4f} seconds")


@jax.jit
def work():
    return primal_dual_ilqr(
        cost,
        dynamics,
        x0,
        X_warm_start,
        u0,
        V0,
        max_iterations=1000,
    )


X, U, V, iter, obj, c, no_errors = work()
X.block_until_ready()


start = timer()

n = 100
for i in range(n):
    X, _, _, _, _, _, _ = work()
    X.block_until_ready()

end = timer()

t = (end - start) / n
print(f"Primal dual result: {obj=} {iter=}, time: {t:.4f} seconds")
