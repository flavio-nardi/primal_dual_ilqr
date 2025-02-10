import json
import os
from dataclasses import dataclass
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
from plot_utils import plot_optimal_trajectory
from trajax import optimizers

from data.read_track_data import (
    calculate_distance_along,
    read_raceline_data,
    read_track_data,
    resample,
)
from primal_dual_ilqr.constrained_optimizers import constrained_primal_dual_ilqr


def main():
    ds = 0.5  # Spatial discretization

    @jax.jit
    def vehicle_kinematics(state, control, s):
        del s
        # State now includes time as a state variable
        _, _, psi, ds_dt, kappa, ax, _ = state
        swirl, jerk = control

        # Calculate dt for this spatial step
        dt = ds / (ds_dt + 1e-6)  # Small epsilon to avoid division by zero

        # Position equations (scaled by ds since we're integrating in space)
        x_prime = jnp.cos(psi)  # dx/ds
        y_prime = jnp.sin(psi)  # dy/ds
        psi_prime = kappa  # dpsi/ds

        # Speed and acceleration equations (temporal derivatives converted to spatial)
        ds_dt_prime = ax / ds_dt
        kappa_prime = swirl / ds_dt
        ax_prime = jerk / ds_dt
        t_prime = jnp.abs(dt)

        return jnp.array(
            [
                x_prime,
                y_prime,
                psi_prime,
                ds_dt_prime,
                kappa_prime,
                ax_prime,
                t_prime,
            ]
        )

    max_speed = 50.0  # Maximum speed in m/s
    max_lat_accel = 10.0  # Maximum lateral acceleration in m/s^2

    base_dir = os.path.dirname(os.path.dirname(__file__))

    track_name: str = "Nuerburgring"
    file_path = os.path.join(base_dir, "tests", f"{track_name}.json")

    with open(file_path, "r") as f:
        data = json.load(f)

    x_ref = jnp.array(data["x"])
    y_ref = jnp.array(data["y"])
    psi_ref = jnp.array(data["psi"])
    kappa_ref = jnp.array(data["kappa"])
    vx_ref = 0.0 * jnp.ones_like(x_ref)

    horizon = x_ref.shape[0] - 1
    u0 = jnp.zeros((horizon, 2))
    # Initial state now includes time: [x, y, psi, ds_dt, kappa, ax, t]
    x0 = jnp.array(
        [x_ref[0], y_ref[0], psi_ref[0], 1.0, kappa_ref[0], 0.0, 0.0]
    )
    # Note: Initialize ds_dt to small positive value to avoid division by zero

    # Warm start needs to be updated for new state dimension
    X_warm_start = jnp.zeros((horizon + 1, 7))  # Now 7 states instead of 6
    X_warm_start = X_warm_start.at[:, 0].set(x_ref)
    X_warm_start = X_warm_start.at[:, 1].set(y_ref)
    X_warm_start = X_warm_start.at[:, 2].set(psi_ref)
    X_warm_start = X_warm_start.at[:, 3].set(1.0)  # Initial guess for ds_dt
    X_warm_start = X_warm_start.at[:, 4].set(kappa_ref)
    X_warm_start = X_warm_start.at[:, 5].set(0.0)  # ax
    X_warm_start = X_warm_start.at[:, 6].set(0.2)  # time estimate
    V0 = jnp.zeros([horizon + 1, 7])

    def dynamics(x, u, s):
        # Now we're integrating in space domain directly
        return x + ds * vehicle_kinematics(x, u, s)

    def al_cost(x, u, t):
        err_x = x[0] - x_ref[t]
        err_y = x[1] - y_ref[t]
        err_yaw = x[2] - psi_ref[t]
        err_kappa = x[4] - kappa_ref[t]
        err_vx = x[3] - vx_ref[t]

        w_x = 1.0
        w_y = 1.0
        w_yaw = 10.0
        w_speed = 0.0
        w_swirl = 2.0
        w_jerk = 0.2

        stage_cost = (
            w_x * jnp.dot(err_x, err_x)
            + w_y * jnp.dot(err_y, err_y)
            + w_yaw * jnp.dot(err_yaw, err_yaw)
            + w_yaw * jnp.dot(err_kappa, err_kappa)
            + w_speed * jnp.dot(err_vx, err_vx)
            + w_swirl * jnp.dot(u[0], u[0])
            + w_jerk * jnp.dot(u[1], u[1])
        )

        final_cost = (
            w_x * jnp.dot(err_x, err_x)
            + w_y * jnp.dot(err_y, err_y)
            + w_yaw * jnp.dot(err_yaw, err_yaw)
            + 100.0 * x[6]
            + 10.0 * jnp.dot(x[3], x[3])
            # + 10.0 * jnp.dot(x[5], x[5])
        )
        return jnp.where(jnp.equal(t, horizon), final_cost, stage_cost)

    # Update inequality constraints to include time-based constraints if needed
    def inequality_constraint(x, u, t):
        ds_dt = x[3]
        kappa = x[4]
        dt = 1.0 / (ds_dt + 1e-6)

        speed_constraints = jnp.array([ds_dt - max_speed, -ds_dt])
        accel_constraints = jnp.array([x[5] - 4.0, -x[5] - 10.0])
        swirl_constraints = jnp.array([u[0] - 0.5, -u[0] - 0.5])
        jerk_constraints = jnp.array([u[1] - 5.0, -u[1] - 5.0])
        lat_accel = kappa * ds_dt * ds_dt
        lat_accel_constraints = jnp.array(
            [lat_accel - max_lat_accel, -lat_accel - max_lat_accel]
        )
        time_constraints = jnp.array(
            [
                -dt,
                dt - 0.5,
            ]
        )

        return jnp.concatenate(
            [
                speed_constraints,
                accel_constraints,
                lat_accel_constraints,
                time_constraints,
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
    plot_optimal_trajectory(X, U, 0.1, "stupid_test", reference)
    print(X[:, 6])


if __name__ == "__main__":
    main()
