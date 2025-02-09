import json
import os
from dataclasses import dataclass
from enum import Enum, auto
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from plot_utils import plot_track_results
from trajax import optimizers

from data.read_track_data import (
    calculate_distance_along,
    read_raceline_data,
    read_track_data,
    resample,
)
from primal_dual_ilqr.constrained_optimizers import constrained_primal_dual_ilqr


@dataclass(frozen=True)
class Path:
    s: jnp.ndarray
    x: jnp.ndarray
    y: jnp.ndarray
    psi: jnp.ndarray
    kappa: jnp.ndarray
    dkappa_ds: jnp.ndarray


@jax.jit
def spatial_kinematics(state, control, s):
    del s
    _, _, psi, kappa = state
    u = control[0]  # Extract scalar control value

    # equations of motion
    dx_ds = jnp.cos(psi)
    dy_ds = jnp.sin(psi)
    dpsi_ds = kappa
    dkappa_ds = u

    # Ensure all values have the same shape before concatenation
    dx_ds = jnp.reshape(dx_ds, (1,))
    dy_ds = jnp.reshape(dy_ds, (1,))
    dpsi_ds = jnp.reshape(dpsi_ds, (1,))
    dkappa_ds = jnp.reshape(dkappa_ds, (1,))

    return jnp.concatenate([dx_ds, dy_ds, dpsi_ds, dkappa_ds])


def dump_path_to_json(path: Path, track_name: str, curr_dir: str):
    """
    Save reconstructed track path to JSON file.

    Args:
        path: Path object containing centerline data
        track_name: Name of track for filename
        curr_dir: Directory to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    path_dict = {
        "s": path.s.tolist(),
        "x": path.x.tolist(),
        "y": path.y.tolist(),
        "psi": path.psi.tolist(),
        "kappa": path.kappa.tolist(),
        "dkappa_ds": path.dkappa_ds.tolist(),
    }

    file_path = os.path.join(curr_dir, track_name + ".json")

    # Write to JSON file
    with open(file_path, "w") as f:
        json.dump(path_dict, f, indent=2)


def main():
    ds = 1.0

    base_dir = os.path.dirname(os.path.dirname(__file__))

    track_name: str = "Nuerburgring"
    file_path = os.path.join(base_dir, "data")
    x, y, w_right, w_left = read_track_data(track_name, file_path)
    x_raceline, y_raceline = read_raceline_data(
        track_name + "_raceline", file_path
    )
    raceline = jnp.vstack((x_raceline, y_raceline))
    track_boundaries: jnp.ndarray = jnp.vstack((w_left, w_right))

    resampled_track_waypoints, resampled_raceline = resample(
        jnp.vstack((x, y)), track_boundaries, raceline, ds
    )
    x_ref = resampled_track_waypoints[0, :]
    y_ref = resampled_track_waypoints[1, :]
    route_segments = jnp.array(
        resampled_track_waypoints[:, 1:] - resampled_track_waypoints[:, :-1]
    )
    psi_ref = jnp.unwrap(
        jnp.arctan2(route_segments[1, :], route_segments[0, :])
    )
    psi_ref = jnp.append(psi_ref, psi_ref[-1])
    horizon = x_ref.shape[0] - 1

    x0 = jnp.array([x_ref[0], y_ref[0], psi_ref[0], 0.0])
    u0 = jnp.zeros((horizon, 1))

    x_ref = jnp.array(x_ref)
    y_ref = jnp.array(y_ref)
    psi_ref = jnp.array(psi_ref)
    kappa_ref = jnp.zeros((horizon + 1, 1))
    X_warm_start = jnp.column_stack((x_ref, y_ref, psi_ref, kappa_ref))
    V0 = jnp.zeros([horizon + 1, 4])

    def dynamics(x, u, s):
        return x + ds * spatial_kinematics(x, u, s)

    def al_cost(x, u, t):
        err_x = x[0] - x_ref[t]
        err_y = x[1] - y_ref[t]
        err_yaw = x[2] - psi_ref[t]
        w_x = 1.0
        w_y = 1.0
        w_yaw = 10.0

        stage_cost = (
            w_x * jnp.dot(err_x, err_x)
            + w_y * jnp.dot(err_y, err_y)
            + w_yaw * jnp.dot(err_yaw, err_yaw)
            + 100000.0 * jnp.dot(u, u)
        )
        final_cost = (
            w_x * jnp.dot(err_x, err_x)
            + w_y * jnp.dot(err_y, err_y)
            + w_yaw * jnp.dot(err_yaw, err_yaw)
        )
        return jnp.where(t == horizon, final_cost, stage_cost)

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
                [x[3] - 0.2, -x[3] - 0.2]
            ),
        )
    )

    reference = jnp.column_stack((x_ref, y_ref, psi_ref))
    print(f"Primal dual aug lag result: {iteration_ilqr=} {iteration_al=}")
    plot_track_results(X, U, ds, "track_reconstruction", reference)

    arc_length = calculate_distance_along(
        jnp.vstack(
            (
                X[:, 0],
                X[:, 1],
            )
        )
    )

    path = Path(
        s=arc_length,
        x=X[:, 0],
        y=X[:, 1],
        psi=X[:, 2],
        kappa=X[:, 3],
        dkappa_ds=U[:, 0],
    )

    dump_path_to_json(path, track_name, os.path.join(base_dir, "tests"))


if __name__ == "__main__":
    main()
