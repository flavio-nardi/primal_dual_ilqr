import os
from typing import Any, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_kinematic_bicycle_results(
    X, U, dt, plot_name: str, target_states: Optional[jnp.ndarray] = None
):
    """Plot optimization results."""
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    time = jnp.arange(X.shape[0]) * dt
    fig, axs = plt.subplots(2, 3, figsize=(24, 18))

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
    axs[0, 1].set_ylim([0, 30])
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

    plt.tight_layout()

    plt.savefig(
        curr_dir + "/" + plot_name + ".png",
        dpi=300,
        bbox_inches="tight",
    )
