import json
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

# Type aliases for arrays that can be either JAX or NumPy
ArrayType = Union[Array, NDArray[np.float64]]
FloatList = List[float]  # Type alias for list of floats

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from data.read_track_data import (
    calculate_distance_along,
    read_raceline_data,
    read_track_data,
    resample,
)
from primal_dual_ilqr.constrained_optimizers import primal_dual_ilqr


@dataclass(frozen=True)
class Path:
    s: ArrayType
    x: ArrayType
    y: ArrayType
    psi: ArrayType
    kappa: ArrayType
    dkappa_ds: ArrayType


def plot_track_results(
    X: ArrayType,
    U: ArrayType,
    ds: float,
    plot_name: str,
    reference: Optional[ArrayType] = None,
) -> None:
    """Plot track reconstruction results."""
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    dist = jnp.arange(X.shape[0]) * ds
    fig, axs = plt.subplots(2, 2, figsize=(24, 18))

    axs[0, 0].plot(
        X[:, 0],
        X[:, 1],
        "b-",
        label="Position (m)",
    )
    if reference is not None:
        axs[0, 0].plot(
            reference[:, 0], reference[:, 1], "r-", label="reference"
        )
    axs[0, 0].set_xlabel("x (m)")
    axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[0, 1].plot(
        dist,
        X[:, 2],
        "b-",
        label="Yaw (rad)",
    )
    if reference is not None:
        axs[0, 1].plot(dist, reference[:, 2], "r-", label="reference")
    axs[0, 1].set_xlabel("Distance along (s)")
    axs[0, 1].set_ylabel("Yaw")
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[1, 0].plot(
        dist,
        X[:, 3],
        "b-",
        label="Curvature (1/m)",
    )
    axs[1, 0].set_xlabel("Distance along (s)")
    axs[1, 0].set_ylabel("Curvature")
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[1, 1].plot(
        dist[:-1],
        U[:, 0],
        "b-",
        label="dkappa/ds (1/m²)",
    )
    axs[1, 1].set_xlabel("Distance along (s)")
    axs[1, 1].set_ylabel("dkappa_ds")
    axs[1, 1].legend()
    axs[1, 1].grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(curr_dir, f"{plot_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )


@jax.jit
def spatial_kinematics(
    state: ArrayType,
    control: ArrayType,
    s: float,
) -> ArrayType:
    """Calculate spatial kinematics."""
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


def dump_path_to_json(path: Path, track_name: str, curr_dir: str) -> None:
    """
    Save reconstructed track path to JSON file.

    Args:
        path: Path object containing centerline data
        track_name: Name of track for filename
        curr_dir: Directory to save JSON file
    """
    # Convert to lists for JSON serialization
    path_dict: Dict[str, FloatList] = {
        "s": cast(FloatList, np.asarray(path.s).tolist()),
        "x": cast(FloatList, np.asarray(path.x).tolist()),
        "y": cast(FloatList, np.asarray(path.y).tolist()),
        "psi": cast(FloatList, np.asarray(path.psi).tolist()),
        "kappa": cast(FloatList, np.asarray(path.kappa).tolist()),
        "dkappa_ds": cast(FloatList, np.asarray(path.dkappa_ds).tolist()),
    }

    file_path = os.path.join(curr_dir, f"{track_name}.json")

    # Write to JSON file
    with open(file_path, "w") as f:
        json.dump(path_dict, f, indent=2)


def main() -> None:
    """Main function to generate and optimize track trajectory."""
    ds = 0.5

    base_dir = os.path.dirname(os.path.dirname(__file__))

    track_name: str = "Nuerburgring"
    file_path = os.path.join(base_dir, "data")
    x, y, w_right, w_left = read_track_data(track_name, file_path)
    x_raceline, y_raceline = read_raceline_data(
        f"{track_name}_raceline", file_path
    )
    raceline = jnp.vstack((x_raceline, y_raceline))
    track_boundaries: jnp.ndarray = jnp.vstack((w_left, w_right))

    # Convert numpy arrays to jax arrays before resampling
    track_waypoints = jnp.asarray(jnp.vstack((x, y)))
    track_bounds = jnp.asarray(track_boundaries)
    race_line = jnp.asarray(raceline)

    resampled_track_waypoints = cast(
        ArrayType,
        resample(
            cast(NDArray[np.float64], track_waypoints),
            cast(NDArray[np.float64], track_bounds),
            cast(NDArray[np.float64], race_line),
            ds,
        )[0],
    )  # Add [0] to get first element of tuple
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

    def dynamics(x: ArrayType, u: ArrayType, s: float) -> ArrayType:
        return x + ds * spatial_kinematics(x, u, s)

    def al_cost(x: ArrayType, u: ArrayType, t: int) -> ArrayType:
        err_x = x[0] - x_ref[t]
        err_y = x[1] - y_ref[t]
        err_yaw = x[2] - psi_ref[t]
        w_x = 0.001
        w_y = 0.001
        w_yaw = 0.01

        stage_cost = (
            w_x * jnp.dot(err_x, err_x)
            + w_y * jnp.dot(err_y, err_y)
            + w_yaw * jnp.dot(err_yaw, err_yaw)
            + jnp.dot(u, u)
        )
        final_cost = (
            w_x * jnp.dot(err_x, err_x)
            + w_y * jnp.dot(err_y, err_y)
            + w_yaw * jnp.dot(err_yaw, err_yaw)
        )
        return jnp.where(t == horizon, final_cost, stage_cost)

    X, U, V, iteration_ilqr, g, c, no_errors = primal_dual_ilqr(
        al_cost, dynamics, x0, X_warm_start, u0, V0
    )

    reference = jnp.column_stack((x_ref, y_ref, psi_ref))
    print(f"Primal dual aug lag result: {iteration_ilqr=}")
    plot_track_results(X, U, ds, "track_reconstruction", reference)

    # Calculate path lengths
    positions = jnp.vstack((X[:, 0], X[:, 1]))
    arc_length = cast(
        ArrayType,
        calculate_distance_along(
            cast(NDArray[np.float64], np.asarray(positions))
        ),
    )

    path = Path(
        s=arc_length,
        x=X[:, 0],
        y=X[:, 1],
        psi=X[:, 2],
        kappa=X[:, 3],
        dkappa_ds=U[:, 0],
    )

    dump_path_to_json(
        path, track_name, os.path.join(base_dir, "trajectory_optimization")
    )


if __name__ == "__main__":
    main()
