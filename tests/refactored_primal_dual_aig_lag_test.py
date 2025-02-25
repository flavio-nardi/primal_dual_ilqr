from enum import Enum, auto
from timeit import default_timer as timer
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from plot_utils import plot_kinematic_bicycle_results
from trajax import optimizers
from primal_dual_ilqr.constrained_optimizers import constrained_primal_dual_ilqr
from primal_dual_ilqr.optimizers import primal_dual_ilqr

# Kinematic single track model (unchanged from original)
@jax.jit
def kinematic_single_track(state, control, timestep):
    del timestep
    x, y, psi, vx, kappa, ax = state
    swirl, jerk = control
    # equations of motion
    x_dot = vx * jnp.cos(psi)
    y_dot = vx * jnp.sin(psi)
    psi_dot = kappa * vx
    vx_dot = ax
    kappa_dot = swirl
    ax_dot = jerk
    return jnp.array([x_dot, y_dot, psi_dot, vx_dot, kappa_dot, ax_dot])

# Configuration
horizon = 200
dt = 0.1
n = 10

# Initial and target states
x0 = jnp.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
target_state = jnp.array([0.0, 11.0, 3.14, 0.0, 0.0, 0.0])

# Keep the Euler integration but improve it later if needed
def dynamics(x, u, t):
    return x + dt * kinematic_single_track(x, u, t)

# Improved warm start - simple linear interpolation
def get_better_warm_start(x0, target_state, horizon):
    # Create linearly interpolated states
    traj = jnp.zeros((horizon + 1, x0.shape[0]))
    
    for i in range(6):
        traj = traj.at[:, i].set(
            jnp.linspace(x0[i], target_state[i], horizon + 1)
        )
    
    # Calculate a simple control sequence using finite differences
    u0 = jnp.zeros((horizon, 2))
    for i in range(horizon):
        # Simple finite difference for curvature change
        if i < horizon - 1:
            u0 = u0.at[i, 0].set((traj[i+1, 4] - traj[i, 4]) / dt)
        
        # Simple finite difference for acceleration change
        if i < horizon - 1:
            u0 = u0.at[i, 1].set((traj[i+1, 5] - traj[i, 5]) / dt)
    
    return traj, u0

# Generate improved warm start
X_warm_start, u0 = get_better_warm_start(x0, target_state, horizon)
V0 = jnp.zeros([horizon + 1, 6])

# Improved cost function with better weighting
def al_cost(x, u, t):
    # Extract state components
    x_pos, y_pos, yaw, vel, curvature, accel = x
    target_x, target_y, target_yaw, target_vel, target_curvature, target_accel = target_state
    
    # Calculate errors
    err_xy = jnp.array([x_pos - target_x, y_pos - target_y])
    err_yaw = yaw - target_yaw
    # Normalize yaw error to [-pi, pi]
    err_yaw = (err_yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi
    err_vel = vel - target_vel
    
    # Weights
    w_xy = 10.0
    w_yaw = 100.0
    w_vel = 10.0
    w_control = 10.0
    
    # Stage cost - penalize control effort
    stage_cost = w_control * jnp.dot(u, u)
    
    # Terminal cost - reach the target state
    final_cost = (
        w_xy * jnp.dot(err_xy, err_xy) +
        w_yaw * (err_yaw ** 2) +
        w_vel * (err_vel ** 2)
    )
    
    return jnp.where(t == horizon, final_cost, stage_cost)

# Improved inequality constraint scaling
def inequality_constraint(x, u, t):
    # Scale constraints for better numerical stability
    return jnp.array([
        (x[2] - jnp.pi) / jnp.pi,  # Normalized heading constraint
        -x[3] / 5.0,               # Scaled velocity constraint (negative = violation)
        (x[4] - 0.18) / 0.2        # Scaled curvature constraint
    ])

# Run optimization with improved parameters
start = timer()
for i in range(n):
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
            # Optimization parameters
            max_iterations=50,       # Increased from default
            max_al_iterations=10,         # Augmented Lagrangian iterations
            penalty_init=10.0,               # Initial penalty parameter
        )
    )
    X.block_until_ready()
end = timer()
t = (end - start) / n

print(
    f"Primal dual aug lag result: {iteration_ilqr=} {iteration_al=}, time: {t:.4f} seconds"
)

# Analyze constraint violations
def analyze_constraints(X, U):
    max_heading_violation = jnp.max(jnp.abs(X[:, 2] - jnp.pi))
    min_velocity = jnp.min(X[:, 3])
    max_curvature_violation = jnp.max(jnp.abs(X[:, 4] - 0.18))
    
    print(f"Maximum heading constraint violation: {max_heading_violation}")
    print(f"Minimum velocity: {min_velocity}")
    print(f"Maximum curvature constraint violation: {max_curvature_violation}")

# Plot results
plot_kinematic_bicycle_results(X, U, dt, "improved_primal_dual_aug_lag_results")
analyze_constraints(X, U)

# Quick function to verify the warm start quality
def verify_warm_start(X, u):
    # Calculate endpoint error
    end_xy_error = jnp.linalg.norm(X[-1, :2] - target_state[:2])
    end_yaw_error = jnp.abs((X[-1, 2] - target_state[2] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
    
    print(f"Warm start endpoint position error: {end_xy_error:.4f}")
    print(f"Warm start endpoint yaw error: {end_yaw_error:.4f}")

# Verify warm start quality
verify_warm_start(X_warm_start, u0)
