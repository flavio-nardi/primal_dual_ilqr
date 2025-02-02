import math
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def read_track_data(
    track_name: str, curr_dir: str
) -> Tuple[NDArray[np.float64], ...]:
    """
    Reads trajectory data from a CSV file into numpy arrays.

    Args:
        track_name: Name of the track file (without .csv extension)
        curr_dir: Directory containing the track file

    Returns:
        Tuple containing arrays of x coordinates, y coordinates,
        right widths, and left widths
    """
    file_path = os.path.join(curr_dir, f"{track_name}.csv")

    # Initialize lists for each column
    x_coords: list[float] = []
    y_coords: list[float] = []
    w_right: list[float] = []
    w_left: list[float] = []

    with open(file_path, "r") as file:
        # Skip the header line
        next(file)

        for line in file:
            if not line.strip():
                continue

            # Split the line and convert to float
            x, y, wr, wl = map(float, line.strip().split(","))

            x_coords.append(x)
            y_coords.append(y)
            w_right.append(wr)
            w_left.append(wl)

    return (
        np.array(x_coords),
        np.array(y_coords),
        np.array(w_right),
        np.array(w_left),
    )


def read_raceline_data(
    track_name: str, curr_dir: str
) -> Tuple[NDArray[np.float64], ...]:
    """
    Read racing line coordinates from a CSV file.

    Reads x,y coordinates of the optimal racing line from a CSV file with format:
    x,y
    1.23,4.56
    7.89,0.12
    ...

    Args:
        track_name: Name of the track file (without .csv extension)
        curr_dir: Directory containing the track data files

    Returns:
        Tuple containing:
            - NDArray[np.float64]: x-coordinates of racing line points
            - NDArray[np.float64]: y-coordinates of racing line points

    Raises:
        FileNotFoundError: If track file does not exist
        ValueError: If CSV file is malformed or empty
    """
    file_path = os.path.join(curr_dir, f"{track_name}.csv")

    # Initialize lists for each column
    x_coords: list[float] = []
    y_coords: list[float] = []

    with open(file_path, "r") as file:
        # Skip the header line
        next(file)

        for line in file:
            if not line.strip():
                continue

            # Split the line and convert to float
            x, y = map(float, line.strip().split(","))

            x_coords.append(x)
            y_coords.append(y)

    return (np.array(x_coords), np.array(y_coords))


def plot_track_data(
    x_coords: NDArray[np.float64],
    y_coords: NDArray[np.float64],
    w_right: NDArray[np.float64],
    w_left: NDArray[np.float64],
    curr_dir: str,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Creates a comprehensive plot of trajectory data with width measurements.

    Args:
        x_coords: X coordinates
        y_coords: Y coordinates
        w_right: Right width measurements
        w_left: Left width measurements
        curr_dir: Current working directory
        save_path: Optional path to save the plot. If None, display instead

    Returns:
        Tuple containing the figure and a tuple of the two axes objects
    """
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Track Analysis", fontsize=14)

    # Plot trajectory
    ax1.plot(x_coords, y_coords, "b-", label="Track")

    # Add width indicators
    step = 5
    for i in range(0, len(x_coords), step):
        # Calculate perpendicular direction
        if i < len(x_coords) - 1:
            dx = x_coords[i + 1] - x_coords[i]
            dy = y_coords[i + 1] - y_coords[i]
        else:
            dx = x_coords[i] - x_coords[i - 1]
            dy = y_coords[i] - y_coords[i - 1]

        # Normalize and rotate 90 degrees
        length = np.sqrt(dx**2 + dy**2)
        perpx, perpy = -dy / length, dx / length

        # Draw width lines
        ax1.plot(
            [x_coords[i] - perpx * w_left[i], x_coords[i] + perpx * w_right[i]],
            [y_coords[i] - perpy * w_left[i], y_coords[i] + perpy * w_right[i]],
            "r-",
            alpha=0.3,
        )

    # Configure first subplot
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.set_title("Track with Width Measurements")
    ax1.grid(True)
    ax1.axis("equal")
    ax1.legend()

    # Calculate trajectory distance
    trajectory_distance = np.zeros_like(x_coords)
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i - 1]
        dy = y_coords[i] - y_coords[i - 1]
        trajectory_distance[i] = trajectory_distance[i - 1] + np.sqrt(
            dx**2 + dy**2
        )

    # Plot width measurements
    ax2.plot(trajectory_distance, w_right, "r-", label="Right Width")
    ax2.plot(trajectory_distance, w_left, "b-", label="Left Width")
    ax2.set_xlabel("Distance Along Trajectory (m)")
    ax2.set_ylabel("Width (m)")
    ax2.set_title("Width Measurements vs. Distance")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        full_save_path = os.path.join(curr_dir, save_path)
        plt.savefig(full_save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return fig, (ax1, ax2)


def calculate_distance_along(track_waypoints: np.ndarray) -> np.ndarray:
    """
    Calculate cumulative distance along a sequence of waypoints.

    Args:
        waypoints: np.ndarray of shape (2, N) containing x,y coordinates of waypoints

    Returns:
        np.ndarray of shape (N,) containing cumulative distances, starting with 0
    """
    # Calculate segments between consecutive points
    track_segments = track_waypoints[:, 1:] - track_waypoints[:, :-1]

    # Calculate length of each segment using Euclidean distance
    segment_lengths = np.hypot(track_segments[0, :], track_segments[1, :])

    # Calculate cumulative sum and prepend 0 for starting point
    distance_along = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    return distance_along


def resample(
    track_waypoints: NDArray[np.float64],
    track_boundaries: NDArray[np.float64],
    track_raceline: NDArray[np.float64],
    ds: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Resamples track waypoints at fixed distance intervals.

    Args:
        track_waypoints: Array of shape (2, N) containing x,y coordinates
        ds: Distance interval for resampling

    Returns:
        Resampled track waypoints array of shape (2, M)
    """

    distance_along = calculate_distance_along(track_waypoints)

    # Create new distance array with fixed intervals
    num_points = math.floor(distance_along[-1] / ds) + 1
    s = np.linspace(0.0, distance_along[-1], num=num_points)

    # Interpolate x and y coordinates
    resampled_x = np.interp(s, distance_along, track_waypoints[0, :])
    resampled_y = np.interp(s, distance_along, track_waypoints[1, :])
    resampled_left_boundary = np.interp(
        s, distance_along, track_boundaries[0, :]
    )
    resampled_right_boundary = np.interp(
        s, distance_along, track_boundaries[1, :]
    )

    raceline_distance_along = calculate_distance_along(track_raceline)
    num_points = math.floor(raceline_distance_along[-1] / ds) + 1
    s = np.linspace(0.0, raceline_distance_along[-1], num=num_points)
    resampled_raceline_x = np.interp(
        s, raceline_distance_along, track_raceline[0, :]
    )
    resampled_raceline_y = np.interp(
        s, raceline_distance_along, track_raceline[1, :]
    )

    resampled_track_waypoints = np.vstack((resampled_x, resampled_y))

    resampled_track_segments = (
        resampled_track_waypoints[:, 1:] - resampled_track_waypoints[:, :-1]
    )
    resampled_psi = np.unwrap(
        np.arctan2(
            resampled_track_segments[1, :], resampled_track_segments[0, :]
        )
    )
    resampled_psi = np.concatenate((resampled_psi, [resampled_psi[-1]]))

    return np.vstack(
        (
            resampled_x,
            resampled_y,
            resampled_psi,
            resampled_left_boundary,
            resampled_right_boundary,
        )
    ), np.vstack((resampled_raceline_x, resampled_raceline_y))


def plot_resampled_track(
    original_track: NDArray[np.float64],
    resampled_track: NDArray[np.float64],
    resampled_raceline: NDArray[np.float64],
    title: str = "Track Comparison",
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot original and resampled track data with heading angle.

    Args:
        original_track: Original track waypoints array of shape (2, N)
        resampled_track: Resampled track waypoints array with shape (3, M)
                        containing x, y, and psi
        title: Plot title

    Returns:
        Figure and tuple of Axes objects
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    # Plot 1: Track layout
    ax1.plot(
        original_track[0, :],
        original_track[1, :],
        "b.",
        label="Original",
        alpha=0.5,
        markersize=10,
    )
    ax1.plot(
        resampled_track[0, :],
        resampled_track[1, :],
        "r.",
        label="Resampled",
        markersize=6,
    )
    ax1.plot(
        resampled_raceline[0, :],
        resampled_raceline[1, :],
        "k.",
        label="Raceline",
        markersize=6,
    )

    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.grid(True)
    ax1.axis("equal")
    ax1.legend()

    # Plot 2: Heading angle vs distance
    # Calculate distance along for resampled track
    track_segments = resampled_track[:2, 1:] - resampled_track[:2, :-1]
    segment_lengths = np.hypot(track_segments[0, :], track_segments[1, :])
    distance_along = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    ax2.plot(
        distance_along, np.rad2deg(resampled_track[2, :]), "r-", label="Heading"
    )
    ax2.set_xlabel("Distance Along Track (m)")
    ax2.set_ylabel("Heading Angle (degrees)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    return fig, (ax1, ax2)


def get_track_reference(ds: float) -> NDArray[np.float64]:
    """
    Retrieves and resamples the Austin track reference data.

    This function reads the track data from the Austin.csv file and resamples
    it at fixed 10-meter intervals. The resampling process includes calculation
    of track heading angles (psi).

    Returns:
        NDArray[np.float64]: Resampled track data with shape (3, N) where:
            - Row 0: X coordinates (meters)
            - Row 1: Y coordinates (meters)
            - Row 2: Heading angles (radians)
        N is the number of points after resampling at 10m intervals.

    Note:
        The function assumes the existence of an 'Austin.csv' file in the same
        directory as the script, and that read_track_data() and resample()
        functions are available.
    """
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    x, y, w_right, w_left = read_track_data("Austin", curr_dir)
    x_raceline, y_raceline = read_raceline_data("Austin_raceline", curr_dir)

    return resample(
        np.vstack((x, y)),
        np.vstack((w_left, w_right)),
        np.vstack((x_raceline, y_raceline)),
        ds,
    )


def main() -> None:
    """Main function to read and plot track data."""
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    # read track data (x, y, left width, right width)
    x, y, w_right, w_left = read_track_data("Austin", curr_dir)
    # read the raceline data (x, y)
    x_raceline, y_raceline = read_raceline_data("Austin_raceline", curr_dir)
    track_waypoints = np.vstack((x, y))
    track_boundaries = np.vstack((w_left, w_right))
    track_raceline = np.vstack((x_raceline, y_raceline))
    resampled_track_waypoints, resampled_raceline = resample(
        track_waypoints, track_boundaries, track_raceline, 10
    )
    print(resampled_track_waypoints.shape)

    plot_track_data(x, y, w_right, w_left, curr_dir)

    plot_resampled_track(
        track_waypoints,
        resampled_track_waypoints,
        resampled_raceline,
        "Austin Track Resampling",
    )
    plt.show()


if __name__ == "__main__":
    main()
