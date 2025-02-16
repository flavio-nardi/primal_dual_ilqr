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

        _, _, psi, vx, kappa, ax, _ = state
        swirl, jerk = control

        return jnp.array(
            [
                vx * jnp.cos(psi),  # x_dot
                vx * jnp.sin(psi),  # y_dot
                kappa * vx,  # psi_dot
                ax,  # vx_dot
                swirl,  # kappa_dot
                jerk,  # ax_dot
                vx,
            ]
        )

    max_speed = 20.0  # Maximum speed in m/s
    max_lat_accel = 10.0  # Maximum lateral acceleration in m/s^2

    base_dir = os.path.dirname(os.path.dirname(__file__))

    # track_name: str = "Austin"
    track_name: str = "Nuerburgring"
    file_path = os.path.join(base_dir, "tests", f"{track_name}.json")

    with open(file_path, "r") as f:
        data = json.load(f)

    N = len(data["x"]) - 20000
    print(len(data["x"][:N]))
    x_ref = jnp.array(data["x"][:N])
    y_ref = jnp.array(data["y"][:N])
    psi_ref = jnp.array(data["psi"][:N])
    kappa_ref = jnp.array(data["kappa"][:N])
    ax_ref = 0.0 * jnp.ones_like(x_ref)

    horizon = x_ref.shape[0] - 1
    u0 = jnp.zeros((horizon, 2))
    vx_0 = 0.5
    # Initial state now includes time: [x, y, psi, ds_dt, kappa, ax]
    x0 = jnp.array(
        [
            x_ref[0],
            y_ref[0],
            psi_ref[0],
            vx_0,
            kappa_ref[0],
            ax_ref[0],
            0.0,
        ]
    )

    # Warm start needs to be updated for new state dimension
    X_warm_start = jnp.zeros((horizon + 1, 7))
    X_warm_start = X_warm_start.at[:, 0].set(x_ref)
    X_warm_start = X_warm_start.at[:, 1].set(y_ref)
    X_warm_start = X_warm_start.at[:, 2].set(psi_ref)
    X_warm_start = X_warm_start.at[:, 3].set(vx_0 * jnp.ones_like(x_ref))
    X_warm_start = X_warm_start.at[:, 4].set(kappa_ref)
    X_warm_start = X_warm_start.at[:, 5].set(jnp.zeros_like(x_ref))
    X_warm_start = X_warm_start.at[:, 6].set(jnp.ones_like(x_ref))

    V0 = jnp.zeros([horizon + 1, 7])

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
        err_kappa = x[4] - kappa_ref[t]
        err_vx = x[3] - max_speed

        w_ey = 0.1
        w_yaw = 0.2
        w_swirl = 0.2
        w_jerk = 0.2

        stage_cost = (
            w_ey * jnp.dot(err_ey, err_ey)
            + w_yaw * jnp.dot(err_kappa, err_kappa)
            + w_swirl * jnp.dot(u[0], u[0])
            + w_jerk * jnp.dot(u[1], u[1])
            + 0.005 * jnp.dot(err_vx, err_vx)
        )

        final_cost = (
            w_ey * jnp.dot(err_ey, err_ey)
            + w_yaw * jnp.dot(err_yaw, err_yaw)
            + 50 * jnp.dot(x[3], x[3])
            # + 100 * jnp.dot(x[5], x[5])
        )
        return jnp.where(jnp.equal(t, horizon), final_cost, stage_cost)

    # Update inequality constraints to include time-based constraints if needed
    def inequality_constraint(x, u, t):
        ds_dt = x[3]
        kappa = x[4]

        speed_constraints = jnp.array([ds_dt - max_speed, -ds_dt])
        accel_constraints = jnp.array([x[5] - 4.0, -x[5] - 10.0])
        swirl_constraints = jnp.array([u[0] - 0.2, -u[0] - 0.2])
        jerk_constraints = jnp.array([u[1] - 5.0, -u[1] - 5.0])
        lat_accel = kappa * ds_dt * ds_dt
        lat_accel_constraints = jnp.array(
            [lat_accel - max_lat_accel, -lat_accel - max_lat_accel]
        )

        return jnp.concatenate(
            [
                speed_constraints,
                accel_constraints,
                lat_accel_constraints,
                # swirl_constraints,
                # jerk_constraints,
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
    print("Track time: ", X[-1, 6])


if __name__ == "__main__":
    main()
