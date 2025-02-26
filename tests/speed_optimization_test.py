import argparse
import json
import os

import jax
import jax.numpy as jnp
from plot_utils import plot_speed_trajectory

from primal_dual_ilqr.constrained_optimizers import constrained_primal_dual_ilqr

# Constants
MAX_SPEED = 60.0  # Maximum speed in m/s
MAX_LAT_ACCEL = 10.0  # Maximum lateral acceleration in m/s^2
MAX_LONGITUDINAL_ACCEL = 4.0  # m/s²
MAX_LONGITUDINAL_DECEL = 10.0  # m/s²
SPATIAL_DISCRETIZATION = 0.5  # Spatial discretization in m
TIME_STEP = 0.1  # Time step in seconds
EPS = 1e-8  # Small epsilon for numerical stability


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Trajectory optimization")
    parser.add_argument(
        "--track",
        type=str,
        default="Nuerburgring",
        choices=["Nuerburgring", "Austin"],
        help="Track to optimize for",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=TIME_STEP,
        help="Time step for discretization",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=MAX_SPEED,
        help="Maximum vehicle speed in m/s",
    )
    parser.add_argument(
        "--max-lat-accel",
        type=float,
        default=MAX_LAT_ACCEL,
        help="Maximum lateral acceleration in m/s^2",
    )
    return parser.parse_args()


def load_track_data(track_name):
    """Load track data from a JSON file.

    Args:
        track_name: Name of the track file (without extension)

    Returns:
        data: Dictionary containing track data
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, "tests", f"{track_name}.json")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Track data file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in track data file: {file_path}")


@jax.jit
def interpolate_curvature(s_actual, reference):
    """Interpolate curvature value at a given position.

    Args:
        s_actual: Position along the track
        reference: Array of (position, curvature) pairs

    Returns:
        kappa_actual: Interpolated curvature at s_actual
    """
    s_ref = reference[:, 0]
    kappa_ref = reference[:, 1]

    # Find indices of reference points that bound each actual point
    idx = jnp.searchsorted(s_ref, s_actual)
    idx_low = jnp.maximum(idx - 1, 0)
    idx_high = jnp.minimum(idx, len(s_ref) - 1)

    # Get bounding reference points
    s_low = s_ref[idx_low]
    s_high = s_ref[idx_high]
    kappa_low = kappa_ref[idx_low]
    kappa_high = kappa_ref[idx_high]

    # Handle edge cases where s_actual is outside reference range
    equal_points = s_low == s_high
    weights = jnp.where(
        equal_points, 0.0, (s_actual - s_low) / jnp.maximum(s_high - s_low, EPS)
    )

    # Linear interpolation
    kappa_actual = kappa_low + weights * (kappa_high - kappa_low)

    return kappa_actual


@jax.jit
def vehicle_kinematics(state, control, t):
    """Define vehicle kinematics.

    Args:
        state: Current state [s, s_dot, s_ddot, time]
        control: Control input [jerk]
        t: Time step (unused)

    Returns:
        State derivatives [s_dot, s_ddot, jerk, 1.0]
    """
    del t  # Unused argument

    s_dot = state[1]
    s_ddot = state[2]
    jerk = control[0]

    return jnp.array([s_dot, s_ddot, jerk, 1.0])


def setup_optimization_problem(data, args):
    """Setup the optimization problem parameters based on track data.

    Args:
        data: Track data dictionary
        args: Command-line arguments

    Returns:
        Tuple of (s_ref, kappa_ref, vx_ref, dt, N, horizon, x0, u0)
    """
    # Extract reference data
    s_ref = jnp.array(data["s"])
    kappa_ref = jnp.array(data["kappa"])

    # Calculate reference velocity based on curvature
    vx_ref = jnp.minimum(
        jnp.sqrt(args.max_lat_accel / jnp.maximum(jnp.abs(kappa_ref), EPS)),
        args.max_speed,
    )

    # Time discretization
    dt = args.dt

    # Determine simulation horizon based on track length
    if "Nuerburgring" in args.track:
        T = 190  # Time horizon for Nuerburgring
    else:  # Austin or default
        T = 200  # Time horizon for Austin

    N = jnp.floor(T / dt).astype(jnp.int32)
    horizon = N - 1

    # Initial control and state
    u0 = jnp.zeros((horizon, 1))
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])  # [s, s_dot, s_ddot, time]

    return s_ref, kappa_ref, vx_ref, dt, N, horizon, x0, u0


@jax.jit
def dynamics(x, u, t, dt):
    """System dynamics function.

    Args:
        x: Current state
        u: Control input
        t: Time step
        dt: Time discretization step

    Returns:
        Next state
    """
    return x + dt * vehicle_kinematics(x, u, t)


@jax.jit
def al_cost(x, u, t, horizon):
    """Augmented Lagrangian cost function.

    Args:
        x: Current state
        u: Control input
        t: Time step
        horizon: Optimization horizon

    Returns:
        Cost value
    """
    time = x[3]
    jerk = u[0]

    stage_cost = 0.01 * jnp.dot(jerk, jerk)  # Minimize jerk
    final_cost = 500.0 * time  # Minimize final time

    return jnp.where(jnp.equal(t, horizon), final_cost, stage_cost)


@jax.jit
def equality_constraint(x, u, t, horizon, s_ref_end):
    """Equality constraints for the optimization problem.

    Args:
        x: Current state
        u: Control input
        t: Time step
        horizon: Optimization horizon
        s_ref_end: End position of the reference trajectory

    Returns:
        Array of equality constraint values
    """
    s = x[0]
    ds_dt = x[1]
    ds2_dt2 = x[2]

    return jnp.array(
        [
            jnp.where(
                jnp.equal(t, horizon), s - s_ref_end, 0.0
            ),  # End position constraint
            jnp.where(jnp.equal(t, horizon), ds_dt, 0.0),  # Zero final velocity
            jnp.where(
                jnp.equal(t, horizon), ds2_dt2, 0.0
            ),  # Zero final acceleration
        ]
    )


@jax.jit
def inequality_constraint(x, u, t, s_ref, vx_ref, s_ref_end):
    """Inequality constraints for the optimization problem.

    Args:
        x: Current state
        u: Control input
        t: Time step
        s_ref: Reference positions
        vx_ref: Reference velocities
        s_ref_end: End position of the reference trajectory

    Returns:
        Array of inequality constraint values
    """
    s = x[0]  # Current position
    ds_dt = x[1]  # Current speed
    ds2_dt2 = x[2]  # Current acceleration

    # Use interpolation for reference velocity
    reference_pos_vel = jnp.column_stack((s_ref, vx_ref))
    vx_max = interpolate_curvature(s, reference_pos_vel)

    # Distance constraint
    distance_constraint = jnp.array([s - s_ref_end])

    # Speed constraints
    speed_constraints = jnp.array([ds_dt - vx_max, -ds_dt])

    # Acceleration constraints
    long_accel_constraints = jnp.array(
        [ds2_dt2 - MAX_LONGITUDINAL_ACCEL, -ds2_dt2 - MAX_LONGITUDINAL_DECEL]
    )

    return jnp.concatenate(
        [
            distance_constraint,
            speed_constraints,
            long_accel_constraints,
        ]
    )


def analyze_results(X, U, s_ref_end, kappa_actual, lateral_acceleration, dt):
    """Analyze optimization results and print key metrics.

    Args:
        X: Optimized state trajectory
        U: Optimized control inputs
        s_ref_end: End position of the reference trajectory
        kappa_actual: Actual curvature along the trajectory
        lateral_acceleration: Lateral acceleration along the trajectory
        dt: Time step
    """
    total_time = X[-1, 3]
    max_speed = jnp.max(X[:, 1])
    max_lat_accel = jnp.max(lateral_acceleration)
    final_position = X[-1, 0]
    position_error = jnp.abs(final_position - s_ref_end)

    print("\nOptimization Results:")
    print(f"  Total lap time: {total_time:.2f} seconds")
    print(f"  Maximum speed: {max_speed:.2f} m/s ({max_speed * 3.6:.2f} km/h)")
    print(f"  Maximum lateral acceleration: {max_lat_accel:.2f} m/s²")
    print(f"  Final position: {final_position:.2f} m")
    print(f"  Position error: {position_error:.6f} m")
    print(f"  Average speed: {s_ref_end / total_time:.2f} m/s")

    # Check constraint violations
    max_jerk = jnp.max(jnp.abs(U[:, 0]))
    max_long_accel = jnp.max(X[:, 2])
    max_long_decel = jnp.min(X[:, 2])

    print("\nConstraint Verification:")
    print(f"  Maximum jerk: {max_jerk:.2f} m/s³")
    print(f"  Maximum longitudinal acceleration: {max_long_accel:.2f} m/s²")
    print(f"  Maximum longitudinal deceleration: {-max_long_decel:.2f} m/s²")

    # Check if constraints are violated
    if max_long_accel > MAX_LONGITUDINAL_ACCEL + EPS:
        print(
            f"  WARNING: Longitudinal acceleration constraint violated by {max_long_accel - MAX_LONGITUDINAL_ACCEL:.2f} m/s²"
        )
    if -max_long_decel > MAX_LONGITUDINAL_DECEL + EPS:
        print(
            f"  WARNING: Longitudinal deceleration constraint violated by {-max_long_decel - MAX_LONGITUDINAL_DECEL:.2f} m/s²"
        )


def main():
    """Main function to run the trajectory optimization."""
    # Parse command-line arguments
    args = parse_args()

    # Load track data
    data = load_track_data(args.track)
    print(f"Loaded {args.track} track data with {len(data['s'])} points.")

    # Setup optimization problem
    s_ref, kappa_ref, vx_ref, dt, N, horizon, x0, u0 = (
        setup_optimization_problem(data, args)
    )

    # Create warm start trajectories
    X_warm_start = jnp.zeros((horizon + 1, 4))
    V0 = jnp.zeros([horizon + 1, 4])

    # Create dynamics function with fixed dt
    dynamics_with_dt = lambda x, u, t: dynamics(x, u, t, dt)

    # Create cost and constraint functions with fixed parameters
    s_ref_end = s_ref[-1]
    cost_fn = lambda x, u, t: al_cost(x, u, t, horizon)
    eq_constraint_fn = lambda x, u, t: equality_constraint(
        x, u, t, horizon, s_ref_end
    )
    ineq_constraint_fn = lambda x, u, t: inequality_constraint(
        x, u, t, s_ref, vx_ref, s_ref_end
    )

    print(f"Starting optimization with horizon: {horizon}, dt: {dt}")
    print(f"Track length: {s_ref_end:.2f} m")

    # Run optimization
    X, U, V, iteration_ilqr, iteration_al, no_errors = (
        constrained_primal_dual_ilqr(
            cost_fn,
            dynamics_with_dt,
            x0,
            X_warm_start,
            u0,
            V0,
            equality_constraint=eq_constraint_fn,
            inequality_constraint=ineq_constraint_fn,
        )
    )

    print(
        f"Optimization completed: {iteration_ilqr=}, {iteration_al=}, {no_errors=}"
    )

    # Calculate actual curvature and lateral acceleration
    reference = jnp.column_stack((s_ref, kappa_ref))
    kappa_actual = interpolate_curvature(X[:, 0], reference)
    lateral_acceleration = kappa_actual * X[:, 1] * X[:, 1]

    # Analyze results
    analyze_results(X, U, s_ref_end, kappa_actual, lateral_acceleration, dt)

    # Plot results
    plot_speed_trajectory(
        X,
        U,
        f"{args.track}_speed_optimization_results",
        dt,
        lateral_acceleration,
    )

    return X, U


if __name__ == "__main__":
    X, U = main()
