import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

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

# Type alias for arrays that can be either JAX or NumPy
ArrayType = Any  # Using Any to avoid mypy conflicts between JAX and NumPy types


@dataclass(frozen=True)
class Path:
    """Track path representation containing spatial coordinates and curvature data."""

    s: ArrayType  # Distance along path
    x: ArrayType  # X coordinates
    y: ArrayType  # Y coordinates
    psi: ArrayType  # Yaw angle
    kappa: ArrayType  # Curvature
    dkappa_ds: ArrayType  # Curvature derivative


def plot_track_results(
    X: ArrayType,
    U: ArrayType,
    ds: float,
    plot_name: str,
    reference: Optional[ArrayType] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Plot track reconstruction results.

    Args:
        X: State trajectory
        U: Control inputs
        ds: Spatial step size
        plot_name: Name for the output file
        reference: Optional reference trajectory for comparison
        output_dir: Directory to save the plot (defaults to current directory)
    """
    curr_dir = output_dir or os.path.dirname(os.path.realpath(__file__))
    dist = jnp.arange(X.shape[0]) * ds  # Calculate distance along track

    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(24, 18))

    # Plot position (x, y)
    axs[0, 0].plot(X[:, 0], X[:, 1], "b-", label="Position (m)")
    if reference is not None:
        axs[0, 0].plot(
            reference[:, 0], reference[:, 1], "r-", label="Reference"
        )
    axs[0, 0].set_xlabel("x (m)")
    axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].set_aspect("equal")

    # Plot yaw angle
    axs[0, 1].plot(dist, X[:, 2], "b-", label="Yaw (rad)")
    if reference is not None:
        axs[0, 1].plot(dist, reference[:, 2], "r-", label="Reference")
    axs[0, 1].set_xlabel("Distance along (s)")
    axs[0, 1].set_ylabel("Yaw")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot curvature
    axs[1, 0].plot(dist, X[:, 3], "b-", label="Curvature (1/m)")
    axs[1, 0].set_xlabel("Distance along (s)")
    axs[1, 0].set_ylabel("Curvature")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot curvature derivative
    axs[1, 1].plot(dist[:-1], U[:, 0], "b-", label="dkappa/ds (1/m²)")
    axs[1, 1].set_xlabel("Distance along (s)")
    axs[1, 1].set_ylabel("dkappa_ds")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(curr_dir, f"{plot_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)  # Close the figure to free memory


@jax.jit
def spatial_kinematics(
    state: ArrayType,
    control: ArrayType,
    s: float,
) -> jax.Array:
    """
    Calculate spatial kinematics for track optimization.

    Args:
        state: Current state [x, y, psi, kappa]
        control: Control input [u] (dkappa_ds)
        s: Distance along track (unused but required for API compatibility)

    Returns:
        State derivatives [dx_ds, dy_ds, dpsi_ds, dkappa_ds]
    """
    del s  # Unused parameter
    _, _, psi, kappa = state
    u = control[0]  # Extract scalar control value

    # Equations of motion
    return jnp.array(
        [
            jnp.cos(psi),  # dx_ds
            jnp.sin(psi),  # dy_ds
            kappa,  # dpsi_ds
            u,  # dkappa_ds
        ]
    )


def dump_path_to_json(path: Path, track_name: str, output_dir: str) -> None:
    """
    Save reconstructed track path to JSON file.

    Args:
        path: Path object containing centerline data
        track_name: Name of track for filename
        output_dir: Directory to save JSON file
    """
    try:
        # Convert JAX arrays to lists for JSON serialization
        path_dict = {
            attr: [float(x) for x in np.asarray(getattr(path, attr))]
            for attr in ["s", "x", "y", "psi", "kappa", "dkappa_ds"]
        }
    except TypeError as e:
        print(f"Error converting path to JSON: {e}")
        # Fallback for potential JAX tracers
        path_dict = {
            attr: [float(x) for x in np.asarray(getattr(path, attr))]
            for attr in ["s", "x", "y", "psi", "kappa", "dkappa_ds"]
        }

    file_path = os.path.join(output_dir, f"{track_name}.json")

    # Write to JSON file with proper indentation
    with open(file_path, "w") as f:
        json.dump(path_dict, f, indent=2)

    print(f"Saved path to {file_path}")


def create_track_reconstruction_cost_function(
    x_ref: ArrayType,
    y_ref: ArrayType,
    psi_ref: ArrayType,
    track_reconstruction_horizon: int,
    w_x: float = 0.001,
    w_y: float = 0.001,
    w_yaw: float = 0.01,
) -> Callable:
    """
    Create a cost function for track optimization.

    Args:
        x_ref: Reference x coordinates
        y_ref: Reference y coordinates
        psi_ref: Reference yaw angles
        track_reconstruction_horizon: Optimization horizon
        w_x: Weight for x coordinate error
        w_y: Weight for y coordinate error
        w_yaw: Weight for yaw angle error

    Returns:
        Cost function for use in optimization
    """
    # Convert references to JAX arrays to ensure proper indexing
    x_ref_jax = jnp.asarray(x_ref)
    y_ref_jax = jnp.asarray(y_ref)
    psi_ref_jax = jnp.asarray(psi_ref)

    def cost(x: ArrayType, u: ArrayType, t: int) -> ArrayType:
        """Cost function for optimization, compatible with JAX tracing."""
        # Use JAX pure indexing to avoid tracer issues
        t_idx = jnp.asarray(t, dtype=jnp.int32)

        # Calculate position and orientation errors
        err_x = x[0] - jnp.take(x_ref_jax, t_idx)
        err_y = x[1] - jnp.take(y_ref_jax, t_idx)
        err_yaw = x[2] - jnp.take(psi_ref_jax, t_idx)

        # Calculate weighted squared errors
        weighted_errors = (
            w_x * jnp.square(err_x)
            + w_y * jnp.square(err_y)
            + w_yaw * jnp.square(err_yaw)
        )

        # For non-terminal states, add control penalty
        stage_cost = weighted_errors + jnp.sum(jnp.square(u))

        # Return final cost for terminal state, stage cost otherwise
        return jnp.where(
            t_idx == track_reconstruction_horizon, weighted_errors, stage_cost
        )

    return cost


def create_track_reconstruction_dynamics_function(ds: float) -> Callable:
    """
    Create a dynamics function for track optimization.

    Args:
        ds: Spatial step size

    Returns:
        Dynamics function for use in optimization
    """

    def dynamics(x: ArrayType, u: ArrayType, s: float) -> Any:
        return x + ds * spatial_kinematics(x, u, s)

    return dynamics


def main() -> None:
    """Main function to generate and optimize track trajectory."""
    # Set up directories
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "trajectory_optimization")
    os.makedirs(output_dir, exist_ok=True)

    # Configuration parameters
    ds = 0.5  # Spatial step size
    track_name = "Nuerburgring"

    # Optimization weights
    w_x = 0.001  # Weight for x position error
    w_y = 0.001  # Weight for y position error
    w_yaw = 0.01  # Weight for yaw angle error

    # Read track data
    print(f"Reading track data for {track_name}...")
    x, y, w_right, w_left = read_track_data(track_name, data_dir)
    x_raceline, y_raceline = read_raceline_data(
        f"{track_name}_raceline", data_dir
    )

    # Prepare data arrays - convert to numpy for compatibility with resample function
    raceline_np = np.asarray(jnp.vstack((x_raceline, y_raceline)))
    track_boundaries_np = np.asarray(jnp.vstack((w_left, w_right)))
    track_waypoints_np = np.asarray(jnp.vstack((x, y)))

    # Resample track data
    print(f"Resampling track with step size {ds}...")
    resampled_track_waypoints = resample(
        track_waypoints_np,
        track_boundaries_np,
        raceline_np,
        ds,
    )[0]

    # Extract reference trajectory
    x_ref = resampled_track_waypoints[0, :]
    y_ref = resampled_track_waypoints[1, :]

    # Calculate reference yaw angles
    route_segments = jnp.array(
        resampled_track_waypoints[:, 1:] - resampled_track_waypoints[:, :-1]
    )
    psi_ref = jnp.unwrap(
        jnp.arctan2(route_segments[1, :], route_segments[0, :])
    )
    psi_ref = jnp.append(psi_ref, psi_ref[-1])

    # Set up optimization
    track_reconstruction_horizon = x_ref.shape[0] - 1

    # Initial and reference states
    track_reconstruction_x0 = jnp.array([x_ref[0], y_ref[0], psi_ref[0], 0.0])
    track_reconstruction_u0 = jnp.zeros((track_reconstruction_horizon, 1))

    # Create warm start for optimization
    kappa_ref = jnp.zeros((track_reconstruction_horizon + 1, 1))
    track_reconstruction_warm_start = jnp.column_stack(
        (x_ref, y_ref, psi_ref, kappa_ref)
    )
    track_reconstruction_V0 = jnp.zeros([track_reconstruction_horizon + 1, 4])

    # Create cost and dynamics functions
    track_reconstruction_cost = create_track_reconstruction_cost_function(
        x_ref, y_ref, psi_ref, track_reconstruction_horizon, w_x, w_y, w_yaw
    )
    track_reconstruction_dynamics = (
        create_track_reconstruction_dynamics_function(ds)
    )

    # Ensure arrays are JAX arrays for compatibility
    track_reconstruction_x0 = jnp.asarray(track_reconstruction_x0)
    track_reconstruction_warm_start = jnp.asarray(
        track_reconstruction_warm_start
    )
    track_reconstruction_u0 = jnp.asarray(track_reconstruction_u0)
    track_reconstruction_V0 = jnp.asarray(track_reconstruction_V0)

    # Run optimization
    print("Running optimization...")
    (
        track_reconstruction_X,
        track_reconstruction_U,
        _,
        iteration_ilqr,
        _,
        _,
        no_errors,
    ) = primal_dual_ilqr(
        track_reconstruction_cost,
        track_reconstruction_dynamics,
        track_reconstruction_x0,
        track_reconstruction_warm_start,
        track_reconstruction_u0,
        track_reconstruction_V0,
    )

    # Print results
    print(f"Track reconstruction completed in {iteration_ilqr} iterations")
    print(
        f"Track reconstruction status: {'Success' if no_errors else 'Failed'}"
    )

    # Create reference for plotting
    reference = jnp.column_stack((x_ref, y_ref, psi_ref))

    # Plot results
    print("Plotting results...")
    plot_track_results(
        track_reconstruction_X,
        track_reconstruction_U,
        ds,
        track_name + "_track_reconstruction",
        reference,
        output_dir,
    )

    # Calculate path lengths
    positions = np.vstack(
        (track_reconstruction_X[:, 0], track_reconstruction_X[:, 1])
    )
    arc_length = calculate_distance_along(positions)

    # Create Path object
    path = Path(
        s=jnp.asarray(arc_length),
        x=track_reconstruction_X[:, 0],
        y=track_reconstruction_X[:, 1],
        psi=track_reconstruction_X[:, 2],
        kappa=track_reconstruction_X[:, 3],
        dkappa_ds=jnp.append(
            track_reconstruction_U[:, 0], track_reconstruction_U[-1, 0]
        ),  # Append last control to match length
    )

    # Save results
    print("Saving results...")
    dump_path_to_json(path, track_name, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
