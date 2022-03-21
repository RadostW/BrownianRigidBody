# Following code is implementation of simulation published in:
# "Brownian Dynamics Simulations of Rotational Diffusion Using
# the Cartesian Components of the Rotation Vector as Generalized Coordinates"
# T. R. Evensen, S. N. Naess & A. Elgsaeter
# Macromol. Theory Simul. (2008)
# doi:10.1002/mats.200800031

import pychastic
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np

mobility = jnp.eye(3)
mobility_d = jnp.linalg.cholesky(mobility)  # Compare with equation: Evensen2008.6


def spin_matrix(q):
    # Antisymmetric matrix dual to q
    return jnp.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])


def rotation_matrix(q):
    # Compare with equation: Evensen2008.11
    unsafephi = jnp.sqrt(jnp.sum(q ** 2))
    phi = jnp.maximum(unsafephi, jnp.array(0.01))

    rot = (
        (jnp.sin(phi) / phi) * spin_matrix(q)
        + jnp.cos(phi) * jnp.eye(3)
        + ((1.0 - jnp.cos(phi)) / phi ** 2) * q.reshape(1, 3) * q.reshape(3, 1)
    )
    return jnp.where(phi == unsafephi, rot, 1.0 * jnp.eye(3) + spin_matrix(q))


def transformation_matrix(q):
    # Compare with equation: Evensen2008.12
    unsafephi = jnp.sqrt(jnp.sum(q ** 2))
    phi = jnp.maximum(unsafephi, jnp.array(0.01))

    scale1 = jnp.where(
        phi == unsafephi,
        (1.0 / phi ** 2 - (jnp.sin(phi) / (2.0 * phi * (1.0 - jnp.cos(phi))))),
        1.0 / 12,
    )

    scale2 = jnp.where(
        phi == unsafephi, (phi * jnp.sin(phi) / (1.0 - jnp.cos(phi))), 2.0
    )

    trans = 0.5 * (
        scale1 * q.reshape(1, 3) * q.reshape(3, 1)
        + spin_matrix(q)
        + scale2 * jnp.eye(3)
    )

    return trans


def metric_force(q):
    # Compare with equation: Evensen2008.10
    unsafephi = jnp.sqrt(jnp.sum(q ** 2))
    phi = jnp.maximum(unsafephi, jnp.array(0.01))

    scale = jnp.where(
        phi == unsafephi,
        jnp.sin(phi) / (1.0 - jnp.cos(phi)) - 2.0 / phi,
        -unsafephi / 6.0,
    )

    return jnp.where(phi == unsafephi, (q / phi) * scale, jnp.array([0.0, 0.0, 0.0]))


def t_mobility(q):
    # Mobility matrix transformed to coordinates.
    # Compare with equation: Evensen2008.2
    return transformation_matrix(q) @ mobility @ (transformation_matrix(q).T)


def drift(q):
    # Drift term.
    # Compare with equation: Evensen2008.5
    # jax.jacobian has differentiation index last (like mu_ij d_k) so divergence is contraction of first and last axis.
    return t_mobility(q) @ metric_force(q) + jnp.einsum(
        "iji->j", jax.jacobian(t_mobility)(q)
    )


def noise(q):
    # Noise term.
    # Compare with equation: Evensen2008.5
    return jnp.sqrt(2) * transformation_matrix(q) @ (rotation_matrix(q).T) @ mobility_d


def canonicalize_coordinates(q):
    unsafephi = jnp.sqrt(jnp.sum(q ** 2))
    phi = jnp.maximum(unsafephi, jnp.array(0.01))

    max_phi = jnp.pi
    canonical_phi = jnp.fmod(phi + max_phi, 2.0 * max_phi) - max_phi

    return jax.lax.select(
        phi > max_phi,  # and phi == unsafephi
        (canonical_phi / phi) * q,
        q,
    )


problem = pychastic.sde_problem.SDEProblem(
    drift, noise, tmax=2.0, x0=jnp.array([0.0, 0.0, 0.00001])
)


solver = pychastic.sde_solver.SDESolver(dt=0.01, scheme="euler")

trajectories = solver.solve_many(
    problem,
    step_post_processing=canonicalize_coordinates,
    n_trajectories=10000,
    chunk_size=8,
    chunks_per_randomization=2,
)

rotation_matrices = jax.vmap(jax.vmap(rotation_matrix))(trajectories["solution_values"])

epsilon_tensor = jnp.array(
    [
        [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
        [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
    ]
)

delta_u = -0.5 * jnp.einsum("kij,abij->abk", epsilon_tensor, rotation_matrices)

cor = jnp.mean(delta_u ** 2, axis=0)


t_a = trajectories["time_values"][0]
plt.plot(t_a, cor[:, 0])
D = 1.0
plt.plot(
    t_a,
    1.0 / 6.0
    - (5.0 / 12.0) * jnp.exp(-6.0 * D * t_a)
    + (1.0 / 4.0) * jnp.exp(-2.0 * D * t_a),
    label="theoretical",
)

plt.show()
