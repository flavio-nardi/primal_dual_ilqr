import os
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


def plot_kinematic_bicycle_results(X, U, dt):
    """Plot optimization results."""
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    time = jnp.arange(X.shape[0]) * dt
    fig, axs = plt.subplots(2, 3, figsize=(24, 18))

    axs[0, 0].plot(
        X[:, 0],
        X[:, 1],
        "x",
        label="Position (m)",
    )
    axs[0, 0].set_xlabel("x (m)")
    axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].legend()

    axs[0, 1].plot(
        time,
        X[:, 3],
        "x",
        label="Speed (mps)",
    )
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Speed")
    axs[0, 1].legend()

    axs[0, 2].plot(
        time,
        X[:, 5],
        "x",
        label="Acceleration (mps2)",
    )
    axs[0, 2].set_xlabel("Time (s)")
    axs[0, 2].set_ylabel("Acceleration")
    axs[0, 2].legend()

    axs[1, 0].plot(
        time[:-1],
        U[:, 1],
        "x",
        markersize=10,
        label="Jerk (mps3)",
    )
    axs[1, 0].set_xlabel("time (s)")
    axs[1, 0].set_ylabel("Jerk (mps3)")
    axs[1, 0].legend()

    axs[1, 1].plot(
        time,
        X[:, 2],
        label="Yaw (rad)",
    )
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Yaw")
    axs[1, 1].legend()

    # axs[1, 2].plot(
    #     result.time[: config.T - config.mpc_N],
    #     result.u_history[:, KinBikeControlIdx.SWIRL],
    #     label="Swirl (1pms)",
    # )
    # axs[1, 2].set_xlabel("Time (s)")
    # axs[1, 2].set_ylabel("Swirl")
    # axs[1, 2].legend()

    # ay_ra = (
    #     result.x_history[:, 0, KinBikeStateIdx.KAPPA]
    #     * result.x_history[:, 0, KinBikeStateIdx.VX] ** 2
    # )
    # axs[2, 0].plot(
    #     result.time,
    #     ay_ra,
    #     label="Lateral acceleration at the rear axle (mps2)",
    # )
    # axs[2, 0].set_xlabel("Time (s)")
    # axs[2, 0].set_ylabel("Lateral acceleration at the rear axle")
    # axs[2, 0].legend()

    # axs[2, 1].plot(
    #     result.time,
    #     result.metrics,
    #     label="Crosstrack error (m)",
    # )
    # axs[2, 1].set_xlabel("Time (s)")
    # axs[2, 1].set_ylabel("Crosstrack error rear axle (m)")
    # axs[2, 1].legend()

    plt.tight_layout()
    # plt.show()

    plt.savefig(
        curr_dir + "/kinematic_bicycle_results.png",
        dpi=300,
        bbox_inches="tight",
    )
