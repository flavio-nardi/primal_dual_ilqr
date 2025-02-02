from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


def point_to_segment_signed_distance(
    q: jnp.ndarray, segment_vertices: jnp.ndarray
) -> jnp.ndarray:
    """Calculate signed distance from point q to a line segment defined by two vertices.
    Maintains gradient computation through JAX's autodiff.

    Args:
        q: Point coordinates array [e, n]
        segment_vertices: Segment vertices array [[e1, n1], [e2, n2]]

    Returns:
        Signed distance from point to line segment (with gradient information through JAX)
    """
    e_1 = segment_vertices[0, 0]
    e_2 = segment_vertices[1, 0]
    n_1 = segment_vertices[0, 1]
    n_2 = segment_vertices[1, 1]
    e = q[0]
    n = q[1]

    numerator = (n_1 - n_2) * e + (e_2 - e_1) * n + (e_1 * n_2 - e_2 * n_1)
    denominator = jnp.sqrt((e_2 - e_1) ** 2 + (n_2 - n_1) ** 2)

    return numerator / denominator


@jax.jit
def point_to_polyline_signed_distance(
    q: jnp.ndarray, vertices: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate signed distance and segment index from point q to a polyline.
    Uses dot product to find the closest segment. JIT-compiled for performance.

    Args:
        q: Point coordinates as array [2] containing [e, n]
        vertices: Polyline vertices as array [N, 2] containing N points with [e, n] coordinates

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: (signed distance, segment index)
    """
    # Convert numpy arrays to JAX arrays if needed
    if isinstance(q, np.ndarray):
        q = jnp.array(q)
    if isinstance(vertices, np.ndarray):
        vertices = jnp.array(vertices)

    # Ensure float64 precision
    q = q.astype(jnp.float64)
    vertices = vertices.astype(jnp.float64)

    # Get start and end points of each segment
    starts = vertices[:-1]  # Shape: [N-1, 2]
    ends = vertices[1:]  # Shape: [N-1, 2]

    # Calculate segment vectors
    segments = ends - starts  # Shape: [N-1, 2]
    segment_lengths = jnp.sqrt(jnp.sum(segments**2, axis=1))  # Shape: [N-1]

    # Calculate vectors from start points to q
    to_point = jnp.broadcast_to(q, starts.shape) - starts  # Shape: [N-1, 2]

    # Calculate normalized dot products
    dot_products = jnp.sum(segments * to_point, axis=1) / (
        segment_lengths**2
    )  # Shape: [N-1]

    # Clamp dot products to [0,1] to find if projection falls on segment
    clamped_dot_products = jnp.clip(dot_products, 0.0, 1.0)

    # Calculate closest points on each segment
    closest_points = starts + (clamped_dot_products[:, jnp.newaxis] * segments)

    # Calculate distances to closest points
    distances_to_closest = jnp.linalg.norm(closest_points - q, axis=1)

    # Find index of minimum distance
    min_idx = jnp.argmin(distances_to_closest)

    # Calculate signed distance for chosen segment
    e_1 = starts[min_idx, 0]
    n_1 = starts[min_idx, 1]
    e_2 = ends[min_idx, 0]
    n_2 = ends[min_idx, 1]
    e = q[0]
    n = q[1]

    # Calculate signed distance
    numerator = (n_1 - n_2) * e + (e_2 - e_1) * n + (e_1 * n_2 - e_2 * n_1)
    denominator = segment_lengths[min_idx]
    distance = numerator / denominator

    return distance, min_idx
