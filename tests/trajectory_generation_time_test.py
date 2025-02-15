import json
import os
from dataclasses import dataclass
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
from plot_utils import plot_optimal_time_trajectory
from trajax import optimizers

from primal_dual_ilqr.constrained_optimizers import constrained_primal_dual_ilqr


def main():
    ds = 0.1  # Spatial discretization

    @jax.jit
    def vehicle_kinematics(state, control, t):
        del t

        _, _, psi, ds_dt, kappa, ax = state
        swirl, jerk = control

        # Position equations
        x_dot = ds_dt * jnp.cos(psi)
        y_dot = ds_dt * jnp.sin(psi)
        psi_dot = ds_dt * kappa

        # Speed and acceleration equations (temporal derivatives converted to spatial)
        ds_dt = ax
        kappa_dot = swirl
        ax_dot = jerk

        return jnp.array(
            [
                x_dot,
                y_dot,
                psi_dot,
                ds_dt,
                kappa_dot,
                ax_dot,
            ]
        )

    max_speed = 20.0  # Maximum speed in m/s
    max_lat_accel = 10.0  # Maximum lateral acceleration in m/s^2

    base_dir = os.path.dirname(os.path.dirname(__file__))

    track_name: str = "Austin"
    # track_name: str = "Nuerburgring"
    file_path = os.path.join(base_dir, "tests", f"{track_name}.json")

    with open(file_path, "r") as f:
        data = json.load(f)

    x_ref = jnp.array(data["x"])
    y_ref = jnp.array(data["y"])
    psi_ref = jnp.array(data["psi"])
    kappa_ref = jnp.array(data["kappa"])
    vx_ref = max_speed * jnp.ones_like(x_ref)
    ax_ref = 0.0 * jnp.ones_like(x_ref)

    horizon = x_ref.shape[0] - 1
    u0 = jnp.zeros((horizon, 2))
    # Initial state now includes time: [x, y, psi, ds_dt, kappa, ax]
    x0 = jnp.array(
        [
            x_ref[0],
            y_ref[0],
            psi_ref[0],
            vx_ref[0],
            kappa_ref[0],
            ax_ref[0],
        ]
    )

    # Warm start needs to be updated for new state dimension
    X_warm_start = jnp.zeros((horizon + 1, 6))
    X_warm_start = X_warm_start.at[:, 0].set(x_ref)
    X_warm_start = X_warm_start.at[:, 1].set(y_ref)
    X_warm_start = X_warm_start.at[:, 2].set(psi_ref)
    X_warm_start = X_warm_start.at[:, 3].set(jnp.ones_like(x_ref))
    X_warm_start = X_warm_start.at[:, 4].set(kappa_ref)
    X_warm_start = X_warm_start.at[:, 5].set(ax_ref)

    V0 = jnp.zeros([horizon + 1, 6])

    def dynamics(x, u, s):
        return x + (ds / x[3]) * vehicle_kinematics(x, u, s)

    def al_cost(x, u, t):
        err_xy = x[:2] - jnp.array([x_ref[t], y_ref[t]])
        rot_L_P = jnp.array(
            [
                [jnp.cos(psi_ref[t]), jnp.sin(psi_ref[t])],
                [-jnp.sin(psi_ref[t]), jnp.cos(psi_ref[t])],
            ]
        )
        _, err_ey = rot_L_P @ err_xy
        err_yaw = x[2] - psi_ref[t]
        err_vx = x[3] - vx_ref[t]
        err_kappa = x[4] - kappa_ref[t]

        w_ey = 0.1
        w_yaw = 0.2
        w_swirl = 0.2
        w_jerk = 2.0

        stage_cost = (
            w_ey * jnp.dot(err_ey, err_ey)
            + w_yaw * jnp.dot(err_yaw, err_yaw)
            + w_yaw * jnp.dot(err_kappa, err_kappa)
            + 1e-5 * jnp.dot(err_vx, err_vx)
            + w_swirl * jnp.dot(u[0], u[0])
            + w_jerk * jnp.dot(u[1], u[1])
        )

        final_cost = w_ey * jnp.dot(err_ey, err_ey) + w_yaw * jnp.dot(
            err_yaw, err_yaw
        )
        return jnp.where(jnp.equal(t, horizon), final_cost, stage_cost)

    # Update inequality constraints to include time-based constraints if needed
    def inequality_constraint(x, u, t):
        ds_dt = x[3]
        kappa = x[4]

        speed_constraints = jnp.array([ds_dt - max_speed, -ds_dt + 0.5])
        accel_constraints = jnp.array([x[5] - 4.0, -x[5] - 10.0])
        swirl_constraints = jnp.array([u[0] - 0.2, -u[0] - 0.2])
        jerk_constraints = jnp.array([u[1] - 2.0, -u[1] - 2.0])
        lat_accel = kappa * ds_dt * ds_dt
        lat_accel_constraints = jnp.array(
            [lat_accel - max_lat_accel, -lat_accel - max_lat_accel]
        )

        return jnp.concatenate(
            [
                speed_constraints,
                accel_constraints,
                lat_accel_constraints,
                swirl_constraints,
                jerk_constraints,
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
            equality_constraint=lambda x, u, t: jnp.empty(1),
            inequality_constraint=inequality_constraint,
        )
    )

    reference = jnp.column_stack((x_ref, y_ref, psi_ref))
    print(f"Primal dual aug lag result: {iteration_ilqr=} {iteration_al=}")
    plot_optimal_time_trajectory(
        X, U, "trajectory_optimization_results", reference, ds
    )
    print("Track time: ", X[:10, 3])


if __name__ == "__main__":
    main()
