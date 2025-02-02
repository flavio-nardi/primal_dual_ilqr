import json
import os
from dataclasses import dataclass
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from casadi import Opti, cos, mtimes, sin, transpose, vertcat
from scipy.signal import savgol_filter

from data.read_track_data import (
    calculate_distance_along,
    read_raceline_data,
    read_track_data,
    resample,
)

MAX_CURVATURE = 0.22
W_CONTROL = 100.0
W_POSITION = 1.0
MAX_ITERATIONS = 20


@dataclass(frozen=True)
class Path:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    kappa: np.ndarray
    dkappa_ds: np.ndarray


@dataclass(frozen=True)
class LaneBoundaries:
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray


def track_reconstruction(xy_ref: np.ndarray, ds: float) -> Path:
    """
    Reconstruct a smooth track path from reference waypoints using optimization.

    Solves an optimization problem to find a smooth path that:
    - Minimizes deviation from reference points
    - Minimizes curvature rate of change
    - Satisfies maximum curvature constraints
    - Maintains continuity in position, heading and curvature

    Args:
        xy_ref: Reference waypoints of shape (2, N) containing [x, y] coordinates
        ds: Step size for path discretization in meters

    Returns:
        Path: Optimized path containing:
            - s: Distance along path
            - x, y: Optimized waypoint coordinates
            - psi: Heading angle at each point
            - kappa: Path curvature at each point
            - dkappa_ds: Rate of change of curvature

    Notes:
        - Uses CasADI optimizer to solve the nonlinear program
        - Constraints curvature between -0.22 and 0.22 (about 4.5m radius)
        - Weights in objective: position error (1.0), curvature rate (100.0)
        - Discretizes dynamics using 4th order Runge-Kutta
    """
    x_ref = xy_ref[0, :]
    y_ref = xy_ref[1, :]
    route_segments = np.array(xy_ref[:, 1:] - xy_ref[:, :-1])
    psi_ref = np.unwrap(np.arctan2(route_segments[1, :], route_segments[0, :]))
    psi_ref = np.append(psi_ref, psi_ref[-1])

    N = x_ref.shape[0] - 1
    opti = Opti()

    # Decision variables.
    X = opti.variable(4, N + 1)

    pos_x = X[0, :]
    pos_y = X[1, :]
    psi = X[2, :]
    kappa = X[3, :]

    U = opti.variable(1, N)
    dkappa_ds = U[0, :]
    pos_x_err = transpose(pos_x) - x_ref
    pos_y_err = transpose(pos_y) - y_ref

    opti.minimize(
        W_CONTROL * mtimes(dkappa_ds, dkappa_ds.T)
        + W_POSITION * mtimes(pos_x_err.T, pos_x_err)
        + W_POSITION * mtimes(pos_y_err.T, pos_y_err)
    )

    # Discretization via RungeKutta4
    for k in range(N):
        x_next = step_runge_kutta_4(spatial_kinematics, ds, X[:, k], U[:, k])
        opti.subject_to(X[:, k + 1] == x_next)

    # Path constraints.
    opti.subject_to(opti.bounded(-MAX_CURVATURE, kappa, MAX_CURVATURE))

    # Boundary conditions.
    opti.subject_to(pos_x[0] == x_ref[0])
    opti.subject_to(pos_x[N] == x_ref[N])
    opti.subject_to(pos_y[0] == y_ref[0])
    opti.subject_to(pos_y[N] == y_ref[N])
    opti.subject_to(psi[0] == psi_ref[0])
    opti.subject_to(psi[N] == psi_ref[N])

    # Initial values for solver.
    opti.set_initial(psi, psi_ref)
    opti.set_initial(pos_x, x_ref)
    opti.set_initial(pos_y, y_ref)
    opti.set_initial(kappa, np.zeros((1, N + 1)))
    opti.set_initial(dkappa_ds, np.zeros((1, N)))

    # Solve NLP.
    p_opts = {"expand": True}
    s_opts = {"max_iter": MAX_ITERATIONS}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()

    # Calculate arc-length.
    arc_length = calculate_distance_along(
        np.vstack(
            (
                sol.value(pos_x),
                sol.value(pos_y),
            )
        )
    )

    return Path(
        arc_length,
        sol.value(pos_x),
        sol.value(pos_y),
        sol.value(psi),
        sol.value(kappa),
        sol.value(dkappa_ds),
    )


def step_runge_kutta_4(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    step_size: float,
    X: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    """
    Perform one step of RK4 integration.

    Args:
        f: Function computing state derivatives
        step_size: Integration step size
        X: Current state vector
        U: Current control input

    Returns:
        np.ndarray: Next state vector
    """
    k1 = f(X, U)
    k2 = f(X + step_size / 2 * k1, U)
    k3 = f(X + step_size / 2 * k2, U)
    k4 = f(X + step_size * k3, U)

    return X + step_size / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def spatial_kinematics(states: np.ndarray, controls: np.ndarray) -> List[float]:
    """
    Calculate spatial derivatives of vehicle states with respect to path distance.

    Implements the spatial kinematic model:
        dx/ds = cos(psi)
        dy/ds = sin(psi)
        dpsi/ds = kappa
        dkappa/ds = u

    Args:
        states: np.ndarray [4] containing:
            - x: position x
            - y: position y
            - psi: heading angle
            - kappa: path curvature
        controls: np.ndarray [1] containing:
            - u: rate of change of curvature

    Returns:
        List[float]: Spatial derivatives [dx/ds, dy/ds, dpsi/ds, dkappa/ds]
            - dx/ds: spatial change in x position
            - dy/ds: spatial change in y position
            - dpsi/ds: spatial change in heading (curvature)
            - dkappa/ds: spatial change in curvature (control input)
    """
    psi = states[2]
    kappa = states[3]
    u = controls[0]

    dx_ds = cos(psi)
    dy_ds = sin(psi)
    dpsi_ds = kappa
    dkappa_ds = u

    return vertcat(dx_ds, dy_ds, dpsi_ds, dkappa_ds)


def dump_path_to_json(
    path: Path, track_boundaries: LaneBoundaries, track_name: str, curr_dir: str
):
    """
    Save track path and boundaries to JSON file.

    Args:
        path: Path object containing centerline data
        track_boundaries: Track boundary offsets
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
        "left_x": track_boundaries.left_x.tolist(),
        "left_y": track_boundaries.left_y.tolist(),
        "right_x": track_boundaries.right_x.tolist(),
        "right_y": track_boundaries.right_y.tolist(),
    }

    file_path = os.path.join(curr_dir, track_name + ".json")

    # Write to JSON file
    with open(file_path, "w") as f:
        json.dump(path_dict, f, indent=2)


def read_track_from_json(track_name: str):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(curr_dir, track_name + ".json")

    with open(file_path, "r") as f:
        data = json.load(f)

    path = Path(
        s=np.array(data["s"]),
        x=np.array(data["x"]),
        y=np.array(data["y"]),
        psi=np.array(data["psi"]),
        kappa=np.array(data["kappa"]),
        dkappa_ds=np.array(data["dkappa_ds"]),
    )

    # Create LaneBoundaries object
    lane_boundaries = LaneBoundaries(
        left_x=np.array(data["left_x"]),
        left_y=np.array(data["left_y"]),
        right_x=np.array(data["right_x"]),
        right_y=np.array(data["right_y"]),
    )

    return path, lane_boundaries


def get_reconstructed_track(
    track_name: str, ds: float
) -> Tuple[Path, LaneBoundaries, Path]:
    """
    Get reconstructed track data by loading, resampling and optimizing track waypoints.

    This function:
    1. Loads centerline and raceline data for given track
    2. Resamples track waypoints at specified intervals
    3. Reconstructs smooth paths for both centerline and raceline
    4. Computes track boundaries in local coordinates

    Args:
        track_name: Name of the track to load (e.g., "Austin")
        ds: Step size for track discretization in meters

    Returns:
        Tuple containing:
            - Path: Reconstructed centerline path containing optimized waypoints
                and curvature with fields (s, x, y, psi, kappa, dkappa_ds)
            - np.ndarray: Track boundaries of shape (2, N) containing [left, right]
                boundaries in local coordinates relative to centerline
            - Path: Reconstructed raceline path with same fields as centerline Path
                representing the optimal racing line
    """
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    x, y, w_right, w_left = read_track_data(track_name, curr_dir)
    x_raceline, y_raceline = read_raceline_data(
        track_name + "_raceline", curr_dir
    )
    raceline = np.vstack((x_raceline, y_raceline))
    track_boundaries: np.ndarray = np.vstack((w_left, w_right))

    resampled_track_waypoints, resampled_raceline = resample(
        np.vstack((x, y)), track_boundaries, raceline, ds
    )
    centerline_path = track_reconstruction(resampled_track_waypoints[:2, :], ds)
    left_boundary, right_boundary = from_path_to_local_vectorized(
        centerline_path.x,
        centerline_path.y,
        centerline_path.psi,
        resampled_track_waypoints[3:, :],
    )
    return (
        centerline_path,
        LaneBoundaries(
            left_x=left_boundary[0, :],
            left_y=left_boundary[1, :],
            right_x=right_boundary[0, :],
            right_y=right_boundary[1, :],
        ),
        track_reconstruction(resampled_raceline, ds),
    )


def from_path_to_local_vectorized(
    x: np.ndarray, y: np.ndarray, psi: np.ndarray, track_boundaries: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert track boundaries from centerline offsets to global Euclidean coordinates.

    Takes track boundaries specified as perpendicular offsets from the centerline
    and converts them to (x,y) coordinates in the global frame, accounting for the
    path heading angle at each point.

    Args:
        x: np.ndarray of shape (N,) containing x-coordinates of centerline
        y: np.ndarray of shape (N,) containing y-coordinates of centerline
        psi: np.ndarray of shape (N,) containing heading angles in radians
        track_boundaries: np.ndarray of shape (2, N) containing [left_width, right_width]
                        as perpendicular offsets from centerline

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays of shape (2, N) containing:
            - Left boundary points (x,y) in global coordinates
            - Right boundary points (x,y) in global coordinates

    Raises:
        ValueError: If input array shapes are inconsistent
    """
    # Get the number of points
    N = track_boundaries.shape[1]

    # Ensure psi has the correct shape
    if psi.shape[0] != N:
        raise ValueError(
            f"psi shape {psi.shape[0]} doesn't match track_boundaries shape {N}"
        )

    # Convert left and right width from track centerline to Euclidean coordinates
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    left_x = -sin_psi * track_boundaries[0, :] + x
    left_y = cos_psi * track_boundaries[0, :] + y

    right_x = sin_psi * track_boundaries[1, :] + x
    right_y = -cos_psi * track_boundaries[1, :] + y

    return (np.vstack((left_x, left_y)), np.vstack((right_x, right_y)))


def analyze_curvature_derivative(
    dkappa_ds: np.ndarray,
    window: int = 51,
    poly_order: int = 3,
    threshold: float = 1.0,
) -> np.ndarray:
    """
    Analyze noisy curvature derivative after smoothing.

    Args:
        dkappa_ds: Raw curvature derivative array
        window: Window size for smoothing (must be odd)
        poly_order: Polynomial order for Savitzky-Golay filter
        threshold: Minimum magnitude to consider significant

    Returns:
        np.ndarray: Indices of significant points
    """

    smoothed = savgol_filter(dkappa_ds, window, poly_order)

    # Find zero crossings in smoothed signal
    zero_crossings = np.where(np.diff(np.signbit(smoothed)))[0]

    return zero_crossings


def plot_track(
    centerline_path: Path,
    raceline_path: Path,
    track_boundaries: np.ndarray,
    apex_idx: np.ndarray,
) -> None:
    # Calculate direction of first two waypoints
    dx_start = centerline_path.x[50] - centerline_path.x[0]
    dy_start = centerline_path.y[50] - centerline_path.y[0]

    """Plot track centerline, raceline and boundaries."""
    plt.arrow(
        centerline_path.x[0],
        centerline_path.y[0],
        dx_start,
        dy_start,
        head_width=6,
        head_length=6,
        width=3,
        fc="k",
        ec="k",
    )
    plt.plot(
        centerline_path.x[-1], centerline_path.y[-1], "r+", label="Centerline"
    )
    plt.plot(raceline_path.x, raceline_path.y, color="blue", label="Raceline")
    plt.plot(
        track_boundaries.left_x,
        track_boundaries.left_y,
        "r--",
        label="Left boundary",
    )
    plt.plot(
        track_boundaries.right_x,
        track_boundaries.right_y,
        "g--",
        label="Right boundary",
    )
    for idx in apex_idx:
        plt.plot(
            centerline_path.x[idx], centerline_path.y[idx], "ro", markersize=6
        )
        plt.annotate(
            f"{idx}",
            (centerline_path.x[idx], centerline_path.y[idx]),
            xytext=(-20, 20),  # Offset more from point
            textcoords="offset points",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.7, pad=2),
            arrowprops=dict(arrowstyle="->"),
        )
    plt.axis("equal")
    plt.legend()
    plt.show()


def main() -> None:
    """
    Main function to reconstruct a race track and save the path data.

    Loads the Austin track data, reconstructs a smooth path with curvature
    optimization using a 1-meter discretization step, and saves the results
    to a JSON file.
    """
    track_name: str = "Nuerburgring"
    (
        centerline_path,
        resampled_track_boundaries,
        raceline_path,
    ) = get_reconstructed_track(track_name, ds=1.0)
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    dump_path_to_json(
        centerline_path, resampled_track_boundaries, track_name, curr_dir
    )

    # find apexes. this is a hack
    high_curvature_idx = np.where(np.abs(centerline_path.kappa) > 0.005)[0]
    zero_crossings = analyze_curvature_derivative(centerline_path.dkappa_ds)
    apex_idxs = np.intersect1d(high_curvature_idx, zero_crossings)
    plot_track(
        centerline_path, raceline_path, resampled_track_boundaries, apex_idxs
    )


if __name__ == "__main__":
    main()
