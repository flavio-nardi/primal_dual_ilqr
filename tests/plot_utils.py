import os
from typing import Any, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_kinematic_bicycle_results(
    X,
    U,
    dt,
    plot_name: str,
    target_states: Optional[jnp.ndarray] = None,
    cross_track_error: Optional[jnp.array] = None,
    heading_error: Optional[jnp.array] = None,
    speed_error: Optional[jnp.array] = None,
):
    """Plot optimization results."""
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    time = jnp.arange(X.shape[0]) * dt
    fig, axs = plt.subplots(3, 3, figsize=(24, 18))

    axs[0, 0].plot(
        X[:, 0],
        X[:, 1],
        "x-",
        label="Position (m)",
    )
    if target_states is not None:
        axs[0, 0].plot(target_states.x, target_states.y, "r--", label="Target")
    axs[0, 0].set_xlabel("x (m)")
    axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[0, 1].plot(
        time,
        X[:, 3],
        "x-",
        label="Speed (mps)",
    )
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Speed")
    axs[0, 1].set_ylim([0, 50])
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[0, 2].plot(
        time,
        X[:, 5],
        "x-",
        label="Acceleration (mps2)",
    )
    axs[0, 2].set_xlabel("Time (s)")
    axs[0, 2].set_ylabel("Acceleration")
    axs[0, 2].legend()
    axs[0, 2].grid()

    axs[1, 0].plot(
        time[:-1],
        U[:, 1],
        markersize=10,
        label="Jerk (mps3)",
    )
    axs[1, 0].set_xlabel("time (s)")
    axs[1, 0].set_ylabel("Jerk (mps3)")
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[1, 1].plot(
        time,
        X[:, 2],
        label="Yaw (rad)",
    )
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Yaw")
    axs[1, 1].legend()
    axs[1, 1].grid()

    axs[1, 2].plot(
        time,
        X[:, 4],
        label="Curvature (1pm)",
    )
    axs[1, 2].set_xlabel("Time (s)")
    axs[1, 2].set_ylabel("Curvature")
    axs[1, 2].legend()
    axs[1, 2].grid()

    if target_states is not None:
        axs[2, 0].plot(
            time[:-1],
            cross_track_error,
            label="Cross track error (m)",
        )
        axs[2, 0].set_xlabel("Time (s)")
        axs[2, 0].set_ylabel("Cross track error")
        axs[2, 0].legend()
        axs[2, 0].grid()

        axs[2, 1].plot(
            time[:-1],
            heading_error,
            label="Heading error (rad)",
        )
        axs[2, 1].set_xlabel("Time (s)")
        axs[2, 1].set_ylabel("Heading error")
        axs[2, 1].legend()
        axs[2, 1].grid()

        axs[2, 2].plot(
            time[:-1],
            speed_error,
            label="Speed error (rad)",
        )
        axs[2, 2].set_xlabel("Time (s)")
        axs[2, 2].set_ylabel("Speed error")
        axs[2, 2].legend()
        axs[2, 2].grid()

    plt.tight_layout()

    plt.savefig(
        curr_dir + "/" + plot_name + ".png",
        dpi=300,
        bbox_inches="tight",
    )


def plot_track_results(
    X,
    U,
    ds,
    plot_name: str,
    reference: Optional[jnp.ndarray] = None,
):
    """Plot track reconstruction results."""
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    dist = jnp.arange(X.shape[0]) * ds
    fig, axs = plt.subplots(2, 2, figsize=(24, 18))

    axs[0, 0].plot(
        X[:, 0],
        X[:, 1],
        "x-",
        label="Position (m)",
    )
    if reference is not None:
        axs[0, 0].plot(reference[:, 0], reference[:, 1], "r-", label="refence")
    axs[0, 0].set_xlabel("x (m)")
    axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[0, 1].plot(
        dist,
        X[:, 2],
        "x-",
        label="Yaw (rad)",
    )
    if reference is not None:
        axs[0, 1].plot(dist, reference[:, 2], "r-", label="refence")
    axs[0, 1].set_xlabel("Distance along (s)")
    axs[0, 1].set_ylabel("Yaw")
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[1, 0].plot(
        dist,
        X[:, 3],
        "x-",
        label="Curvature (1pm)",
    )
    axs[1, 0].set_xlabel("Distance along (s)")
    axs[1, 0].set_ylabel("Curvature")
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[1, 1].plot(
        dist[:-1],
        U[:, 0],
        "x-",
        label="dkappa ds (1pm2)",
    )
    axs[1, 1].set_xlabel("Dstance along (s)")
    axs[1, 1].set_ylabel("dkappa_ds")
    axs[1, 1].legend()
    axs[1, 1].grid()

    plt.tight_layout()

    plt.savefig(
        curr_dir + "/" + plot_name + ".png",
        dpi=300,
        bbox_inches="tight",
    )


def plot_optimal_trajectory(X, U, plot_name: str, reference):
    """Plot optimization results."""
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    time = X[:, 6]
    fig, axs = plt.subplots(3, 3, figsize=(24, 18))

    axs[0, 0].plot(
        X[:, 0],
        X[:, 1],
        "x-",
        label="Position (m)",
    )
    axs[0, 0].plot(
        reference[:, 0],
        reference[:, 1],
        "r-",
        label="Reference (m)",
    )
    axs[0, 0].set_xlabel("x (m)")
    axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[0, 1].plot(
        time,
        X[:, 3],
        "x-",
        label="Speed (mps)",
    )
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Speed")
    axs[0, 1].set_ylim([0, 60])
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[0, 2].plot(
        time,
        X[:, 5],
        "x-",
        label="Acceleration (mps2)",
    )
    axs[0, 2].set_xlabel("Time (s)")
    axs[0, 2].set_ylabel("Acceleration")
    axs[0, 2].legend()
    axs[0, 2].grid()

    axs[1, 0].plot(
        time[:-1],
        U[:, 1],
        markersize=10,
        label="Jerk (mps3)",
    )
    axs[1, 0].set_xlabel("time (s)")
    axs[1, 0].set_ylabel("Jerk (mps3)")
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[1, 1].plot(
        time,
        X[:, 2],
        "x-",
        label="Yaw",
    )
    axs[1, 1].plot(
        time,
        reference[:, 2],
        "r-",
        label="Reference",
    )
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Yaw (rad)")
    axs[1, 1].legend()
    axs[1, 1].grid()

    axs[1, 2].plot(
        time,
        X[:, 4],
        label="Curvature (1pm)",
    )
    axs[1, 2].set_xlabel("Time (s)")
    axs[1, 2].set_ylabel("Curvature")
    axs[1, 2].legend()
    axs[1, 2].grid()

    axs[2, 0].plot(
        time,
        X[:, 4] * X[:, 3] * X[:, 3],
        label="Lateral acceleration rear axle (mps2)",
    )
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Lateral acceleration rear axle (mps2)")
    axs[2, 0].legend()
    axs[2, 0].grid()

    axs[2, 1].plot(
        time[:-1],
        U[:, 0],
        label="Swirl (1pmps)",
    )
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Swirl")
    axs[2, 1].legend()
    axs[2, 1].grid()

    # axs[2, 2].plot(
    #     time,
    #     speed_error,
    #     label="Speed error (rad)",
    # )
    # axs[2, 2].set_xlabel("Time (s)")
    # axs[2, 2].set_ylabel("Speed error")
    # axs[2, 2].legend()
    # axs[2, 2].grid()

    plt.tight_layout()

    plt.savefig(
        curr_dir + "/" + plot_name + ".png",
        dpi=300,
        bbox_inches="tight",
    )


def plot_optimal_time_trajectory(X, U, plot_name: str, reference, ds):
    """Plot optimization results."""
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    # dt = ds / jnp.abs(X[:, 3])
    # print(dt[:30])
    # time = jnp.cumsum(dt)
    # print(time[-1])

    fig, axs = plt.subplots(3, 3, figsize=(24, 18))

    axs[0, 0].plot(
        X[:, 0],
        X[:, 1],
        "x-",
        label="Position (m)",
    )
    axs[0, 0].plot(
        reference[:, 0],
        reference[:, 1],
        "r-",
        label="Reference (m)",
    )
    axs[0, 0].set_xlabel("x (m)")
    axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[0, 1].plot(
        # time,
        X[:, 3],
        "x-",
        label="Speed (mps)",
    )
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Speed")
    axs[0, 1].set_ylim([0, 60])
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[0, 2].plot(
        # time,
        X[:, 5],
        "x-",
        label="Acceleration (mps2)",
    )
    axs[0, 2].set_xlabel("Time (s)")
    axs[0, 2].set_ylabel("Acceleration")
    axs[0, 2].legend()
    axs[0, 2].grid()

    axs[1, 0].plot(
        # time[:-1],
        U[:, 1],
        markersize=10,
        label="Jerk (mps3)",
    )
    axs[1, 0].set_xlabel("time (s)")
    axs[1, 0].set_ylabel("Jerk (mps3)")
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[1, 1].plot(
        # time,
        X[:, 2],
        "x-",
        label="Yaw",
    )
    axs[1, 1].plot(
        # time,
        reference[:, 2],
        "r-",
        label="Reference",
    )
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Yaw (rad)")
    axs[1, 1].legend()
    axs[1, 1].grid()

    axs[1, 2].plot(
        # time,
        X[:, 4],
        label="Curvature (1pm)",
    )
    axs[1, 2].set_xlabel("Time (s)")
    axs[1, 2].set_ylabel("Curvature")
    axs[1, 2].legend()
    axs[1, 2].grid()

    axs[2, 0].plot(
        # time,
        X[:, 4] * X[:, 3] * X[:, 3],
        label="Lateral acceleration rear axle (mps2)",
    )
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Lateral acceleration rear axle (mps2)")
    axs[2, 0].legend()
    axs[2, 0].grid()

    axs[2, 1].plot(
        # time[:-1],
        U[:, 0],
        label="Swirl (1pmps)",
    )
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Swirl")
    axs[2, 1].legend()
    axs[2, 1].grid()

    # axs[2, 2].plot(
    #     time,
    #     speed_error,
    #     label="Speed error (rad)",
    # )
    # axs[2, 2].set_xlabel("Time (s)")
    # axs[2, 2].set_ylabel("Speed error")
    # axs[2, 2].legend()
    # axs[2, 2].grid()

    plt.tight_layout()

    plt.savefig(
        curr_dir + "/" + plot_name + ".png",
        dpi=300,
        bbox_inches="tight",
    )
