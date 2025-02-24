import json
import os
from dataclasses import dataclass
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
from plot_utils import plot_speed_trajectory
from trajax import optimizers

from primal_dual_ilqr.constrained_optimizers import constrained_primal_dual_ilqr


def interpolate_curvature(s_actual, reference):
    s_ref = reference[:, 0]
    kappa_ref = reference[:, 1]

    # Find indices of reference points that bound each actual point
    idx = jnp.searchsorted(s_ref, s_actual)
    print(idx)
    idx_low = jnp.maximum(idx - 1, 0)
    idx_high = jnp.minimum(idx, len(s_ref) - 1)

    # Get bounding reference points
    s_low = s_ref[idx_low]
    s_high = s_ref[idx_high]
    kappa_low = kappa_ref[idx_low]
    kappa_high = kappa_ref[idx_high]

    # Handle edge cases where s_actual is outside reference range
    equal_points = s_low == s_high
    weights = jnp.where(
        equal_points, 1.0, (s_actual - s_low) / (s_high - s_low)
    )

    # Linear interpolation
    kappa_actual = kappa_low + weights * (kappa_high - kappa_low)

    return kappa_actual


def main():
    ds = 0.5  # Spatial discretization

    @jax.jit
    def vehicle_kinematics(state, control, t):
        del t

        s_dot = state[1]
        s_ddot = state[2]
        jerk = control[0]

        return jnp.array([s_dot, s_ddot, jerk, 1.0])

    max_speed = 40.0  # Maximum speed in m/s
    max_lat_accel = 10.0  # Maximum lateral acceleration in m/s^2

    base_dir = os.path.dirname(os.path.dirname(__file__))

    # track_name: str = "Austin"
    track_name: str = "Nuerburgring"
    file_path = os.path.join(base_dir, "tests", f"{track_name}.json")

    with open(file_path, "r") as f:
        data = json.load(f)

    M = len(data["s"])
    print(len(data["x"][:M]))

    s_ref = jnp.array(data["s"])
    kappa_ref = jnp.array(data["kappa"])
    vx_ref = jnp.minimum(
        jnp.sqrt(max_lat_accel / jnp.maximum(jnp.abs(kappa_ref), 1e-6)),
        max_speed,
    )

    dt = 0.1
    # T = 245
    T = 385
    N = jnp.floor(T / dt).astype(jnp.int32)
    horizon = N - 1
    u0 = jnp.zeros((horizon, 1))

    # Initial state now includes time: [x, y, psi, ds_dt, kappa, ax]
    x0 = jnp.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    # Warm start needs to be updated for new state dimension
    X_warm_start = jnp.zeros((horizon + 1, 4))

    V0 = jnp.zeros([horizon + 1, 4])

    def dynamics(x, u, t):
        return x + dt * vehicle_kinematics(x, u, t)

    def al_cost(x, u, t):
        time = x[3]
        jerk = u[0]

        stage_cost = jnp.dot(jerk, jerk)

        final_cost = time
        return jnp.where(jnp.equal(t, horizon), final_cost, stage_cost)

    def equality_constraint(x, u, t):
        s = x[0]
        ds_dt = x[1]
        ds2_dt2 = x[2]
        return jnp.array(
            [
                jnp.where(jnp.equal(t, horizon), s - s_ref[-1], 0.0),
                jnp.where(jnp.equal(t, horizon), ds_dt, 0.0),
                jnp.where(jnp.equal(t, horizon), ds2_dt2, 0.0),
            ]
        )

    def inequality_constraint(x, u, t):
        s = x[0]  # Current position
        ds_dt = x[1]  # Current speed
        ds2_dt2 = x[2]  # Current speed

        # Find closest reference point
        distances = jnp.abs(s_ref - s)
        closest_idx = jnp.argmin(distances)
        vx_max = vx_ref[closest_idx]

        # distance constraint
        distance_constraint = jnp.array([s - s_ref[-1]])

        # Speed constraints (2 constraints)
        speed_constraints = jnp.array([ds_dt - vx_max, -ds_dt])
        long_accel_constraints = jnp.array([ds2_dt2 - 4.0, -ds2_dt2 - 10.0])

        return jnp.concatenate(
            [
                distance_constraint,
                speed_constraints,
                long_accel_constraints,
            ]
        )

    X, U, V, iteration_ilqr, iteration_al, no_errors = (
        constrained_primal_dual_ilqr(
            al_cost,
            dynamics,
            x0,
            X_warm_start,
            u0,
            V0,
            equality_constraint=equality_constraint,  # lambda x, u, t: jnp.empty(1),
            inequality_constraint=inequality_constraint,
        )
    )

    reference = jnp.column_stack((s_ref, kappa_ref))
    kappa_actual = interpolate_curvature(
        X[:, 0], reference
    )  # where s is your actual position
    lateral_acceleration = kappa_actual * X[:, 1] * X[:, 1]

    print(f"Primal dual aug lag result: {iteration_ilqr=} {iteration_al=}")
    plot_speed_trajectory(
        X, U, "speed_optimization_results", dt, lateral_acceleration
    )
    print(s_ref[-1])


if __name__ == "__main__":
    main()
