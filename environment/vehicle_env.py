from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from data.track_reconstruction import (
    get_reconstructed_track,
    read_track_from_json,
)
from math_lib.signed_distance import point_to_polyline_signed_distance


@dataclass(frozen=True)
class SimulationResult:
    """Container for simulation results."""

    x_history: jnp.ndarray
    u_history: jnp.ndarray
    xy_ref: jnp.ndarray
    metrics: jnp.ndarray
    time: jnp.ndarray


class TrackReference(NamedTuple):
    """Container for track reference data."""

    s: jnp.ndarray
    x: jnp.ndarray
    y: jnp.ndarray
    psi: jnp.ndarray
    vx: jnp.ndarray
    kappa: jnp.ndarray
    left_x: jnp.ndarray
    left_y: jnp.ndarray
    right_x: jnp.ndarray
    right_y: jnp.ndarray

    def get_track_obs(
        self, point: jnp.ndarray, speed: float, horizon: float, ds: float
    ) -> "TrackReference":
        reference_line = self.get_reference_line()
        _, min_idx = point_to_polyline_signed_distance(point, reference_line)

        N = int(speed * horizon / ds)

        return TrackReference(
            s=jax.lax.dynamic_slice(self.s, (min_idx,), (N,)),
            x=jax.lax.dynamic_slice(reference_line[:, 0], (min_idx,), (N,)),
            y=jax.lax.dynamic_slice(reference_line[:, 1], (min_idx,), (N,)),
            psi=jax.lax.dynamic_slice(self.psi, (min_idx,), (N,)),
            vx=jax.lax.dynamic_slice(self.vx, (min_idx,), (N,)),
            kappa=jax.lax.dynamic_slice(self.kappa, (min_idx,), (N,)),
            left_x=jax.lax.dynamic_slice(self.left_x, (min_idx,), (N,)),
            left_y=jax.lax.dynamic_slice(self.left_y, (min_idx,), (N,)),
            right_x=jax.lax.dynamic_slice(self.right_x, (min_idx,), (N,)),
            right_y=jax.lax.dynamic_slice(self.right_y, (min_idx,), (N,)),
        )

    def get_reference_line(self) -> jnp.ndarray:
        return jnp.column_stack((self.x, self.y))


def create_track_reference(
    track_name: str, target_speed: float, ds: float
) -> TrackReference:
    """Create a reference trajectory for the track."""
    # track_data, boundaries, _ = get_reconstructed_track(track_name, ds)
    track_data, boundaries = read_track_from_json(track_name)
    reference_psi = jnp.array(track_data.psi)
    reference_vx = jnp.full_like(reference_psi, target_speed)

    return TrackReference(
        s=jnp.array(track_data.s),
        x=jnp.array(track_data.x),
        y=jnp.array(track_data.y),
        psi=reference_psi,
        vx=reference_vx,
        kappa=jnp.array(track_data.kappa),
        left_x=jnp.array(boundaries.left_x),
        left_y=jnp.array(boundaries.left_y),
        right_x=jnp.array(boundaries.right_x),
        right_y=jnp.array(boundaries.right_y),
    )


@jax.jit
def update_kinematics(x, y, psi, v_x, delta, L):
    """Static kinematics function."""
    x_dot = v_x * jnp.cos(psi)
    y_dot = v_x * jnp.sin(psi)
    psi_dot = v_x * jnp.tan(delta) / L
    return x_dot, y_dot, psi_dot


class BicycleEnv(gym.Env):
    """A bicycle model environment for autonomous vehicle control simulation."""

    L_F: float = 1.5
    MAX_STEER: float = 35.0 * jnp.pi / 180
    MAX_LONG_ACCEL: float = 10.0
    MAX_SPEED: float = 50.0
    TRACK_OBSERVATION_HORIZON: float = 6.0

    def __init__(
        self,
        dt: float,
        l_r: float,
        target_speed: float,
        ds: float,
        track_name: str,
    ) -> None:
        super().__init__()
        self.L = self.L_F + l_r
        self.l_r = l_r
        self.dt = dt
        self.target_speed = target_speed
        self.ds = ds
        self.track_name = track_name
        self.track_reference = create_track_reference(
            self.track_name, self.target_speed, self.ds
        )
        self.state = jnp.zeros(4)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, 0], dtype=np.float32),
            high=np.array(
                [np.inf, np.inf, np.pi, self.MAX_SPEED], dtype=np.float32
            ),
            dtype=np.float32,
        )

    @property
    def get_wheel_base(self) -> float:
        return self.l_r + self.L_F

    def get_l_r(self) -> float:
        return self.l_r

    def get_max_road_wheel_angle(self) -> float:
        return self.MAX_STEER

    def get_max_acceleration(self) -> float:
        return self.MAX_LONG_ACCEL

    def get_target_speed(self) -> float:
        return self.target_speed

    def _update_kinematics(self, x, y, psi, v_x, delta):
        return update_kinematics(x, y, psi, v_x, delta, self.L)

    def get_track_reference(
        self,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        reference_line = jnp.column_stack(
            (self.track_reference.x, self.track_reference.y)
        )
        return (
            reference_line,
            self.track_reference.psi,
            self.track_reference.vx,
            self.track_reference.kappa,
        )

    def reset(
        self,
        x_0: float,
        y_0: float,
        psi_0: float,
        vx_0: float,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[jnp.ndarray, TrackReference, Dict[str, Any]]:
        super().reset(seed=seed)

        # self.track_data, _, _ = get_reconstructed_track(self.track_name, self.ds)
        self.track_data, _ = read_track_from_json(self.track_name)
        self.state = jnp.array([x_0, y_0, psi_0, vx_0])

        track_obs = self.track_reference.get_track_obs(
            jnp.array([x_0, y_0]), vx_0, self.TRACK_OBSERVATION_HORIZON, self.ds
        )
        return self.state, track_obs, {}

    def _unnormalize_actions(self, action: jnp.ndarray) -> Tuple[float, float]:
        """Convert normalized actions to physical values."""
        delta = action[0] * self.MAX_STEER
        acceleration = action[1] * self.MAX_LONG_ACCEL
        return delta, acceleration

    def step(
        self, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, TrackReference, float, bool, Dict[str, Any]]:
        if not isinstance(self.state, jnp.ndarray):
            raise RuntimeError("Environment must be reset before calling step")

        delta, acceleration = self._unnormalize_actions(action)
        x, y, psi, v_x = self.state

        track_obs = self.track_reference.get_track_obs(
            jnp.array([x, y]), v_x, self.TRACK_OBSERVATION_HORIZON, self.ds
        )

        # Update speed using JAX's clip function
        v_x = jnp.clip(v_x + acceleration * self.dt, 0, self.MAX_SPEED)
        x_dot, y_dot, psi_dot = self._update_kinematics(x, y, psi, v_x, delta)

        # Euler integration with JAX operations
        x += x_dot * self.dt
        y += y_dot * self.dt
        psi_new = (psi + psi_dot * self.dt) % (2 * jnp.pi)
        psi = jnp.arctan2(jnp.sin(psi_new), jnp.cos(psi_new))

        self.state = jnp.array([x, y, psi, v_x])

        reward = 0.0
        terminated = False
        info = {"speed": v_x}

        return self.state, track_obs, reward, terminated, info
