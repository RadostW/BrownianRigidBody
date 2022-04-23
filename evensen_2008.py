# Following code is implementation of simulation published in:
# "Brownian Dynamics Simulations of Rotational Diffusion Using
# the Cartesian Components of the Rotation Vector as Generalized Coordinates"
# T. R. Evensen, S. N. Naess & A. Elgsaeter
# Macromol. Theory Simul. (2008)
# doi:10.1002/mats.200800031

# Theoretical correlations published in:
# "Brownian motion of a particle with arbitrary shape"
# B. Cichocki, M. L. Ekiel-Jezewska & E. Wajnryb
# J. Chem-Phys (2015)
# doi:10.1063/1.4921729

import pychastic
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np

mobility = jnp.eye(6)
mobility_d = jnp.linalg.cholesky(mobility)  # Compare with equation: Evensen2008.6


def spin_matrix(q):
    # Antisymmetric matrix dual to q
    return jnp.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])


def rotation_matrix(xq):
    # Compare with equation: Evensen2008.11
    q = xq[3:]
    unsafe_phi_squared = jnp.sum(q ** 2)
    phi_squared = jnp.maximum(unsafe_phi_squared, jnp.array(0.01) ** 2)
    phi = jnp.sqrt(phi_squared)

    rot = jnp.where(
        phi_squared == unsafe_phi_squared,
        (jnp.sin(phi) / phi) * spin_matrix(q)
        + jnp.cos(phi) * jnp.eye(3)
        + ((1.0 - jnp.cos(phi)) / phi ** 2) * q.reshape(1, 3) * q.reshape(3, 1),
        (1.0 - 0.5 * unsafe_phi_squared) * jnp.eye(3)
        + spin_matrix(q)
        + 0.5 * q.reshape(1, 3) * q.reshape(3, 1),
    )
    
    return jnp.concatenate(
    	(jnp.concatenate((rot, jnp.zeros((3,3))), axis=1),
	jnp.concatenate((jnp.zeros((3,3)), rot), axis=1)), axis=0)


def rotation_matrix_r(q):
    # Compare with equation: Evensen2008.11
    unsafe_phi_squared = jnp.sum(q ** 2)
    phi_squared = jnp.maximum(unsafe_phi_squared, jnp.array(0.01) ** 2)
    phi = jnp.sqrt(phi_squared)

    rot = jnp.where(
        phi_squared == unsafe_phi_squared,
        (jnp.sin(phi) / phi) * spin_matrix(q)
        + jnp.cos(phi) * jnp.eye(3)
        + ((1.0 - jnp.cos(phi)) / phi ** 2) * q.reshape(1, 3) * q.reshape(3, 1),
        (1.0 - 0.5 * unsafe_phi_squared) * jnp.eye(3)
        + spin_matrix(q)
        + 0.5 * q.reshape(1, 3) * q.reshape(3, 1),
    )
    
    return rot
    

def transformation_matrix(xq):
    # Compare with equation: Evensen2008.12 - there are typos!
    # Compare with equation: Ilie2014.A9-A10 - no typos (except from Taylor-expanded terms)
    q = xq[3:]
    unsafe_phi_squared = jnp.sum(q ** 2)
    phi_squared = jnp.maximum(unsafe_phi_squared, jnp.array(0.01) ** 2)
    phi = jnp.sqrt(phi_squared)

    c = phi * jnp.sin(phi) / (1.0 - jnp.cos(phi))

    trans = jnp.where(
        phi_squared == unsafe_phi_squared,
        ((1.0 - 0.5 * c) / (phi ** 2)) * q.reshape(1, 3) * q.reshape(3, 1)
        + 0.5 * spin_matrix(q)
        + 0.5 * c * jnp.eye(3),
        (1.0 / 12.0) * q.reshape(1, 3) * q.reshape(3, 1)
        + 0.5 * spin_matrix(q)
        + 0.5 * 2.0 * jnp.eye(3),
    )

    return jnp.concatenate(
    	(jnp.concatenate((jnp.eye(3), jnp.zeros((3,3))), axis=1),
	jnp.concatenate((jnp.zeros((3,3)), trans), axis=1)), axis=0)
	
def transformation_matrix_E(xq):
    # Compare with equation: Evensen2008.12 - there are typos!
    q = xq[3:]
    unsafe_phi_squared = jnp.sum(q ** 2)
    phi_squared = jnp.maximum(unsafe_phi_squared, jnp.array(0.01) ** 2)
    phi = jnp.sqrt(phi_squared)

    c = phi * jnp.sin(phi) / (1.0 - jnp.cos(phi))
    
    trans = jnp.where(
        phi_squared == unsafe_phi_squared,
        0.5 * ((1.0 - 0.5 * c) / (phi ** 2)) * q.reshape(1, 3) * q.reshape(3, 1)
        + spin_matrix(q)
        + c * jnp.eye(3),
        0.5 * (1.0 / 12.0) * q.reshape(1, 3) * q.reshape(3, 1)
        + spin_matrix(q)
        + 2.0 * jnp.eye(3),
    )

    return jnp.concatenate(
    	(jnp.concatenate((jnp.eye(3), jnp.zeros((3,3))), axis=1),
	jnp.concatenate((jnp.zeros((3,3)), trans), axis=1)), axis=0)


def metric_force(xq):
    # Compare with equation: Evensen2008.10
    q = xq[3:]
    unsafephi = jnp.sqrt(jnp.sum(q ** 2))
    phi = jnp.maximum(unsafephi, jnp.array(0.01))

    scale = jnp.where(
        phi == unsafephi,
        jnp.sin(phi) / (1.0 - jnp.cos(phi)) - 2.0 / phi,
        -unsafephi / 6.0,
    )

    force = jnp.where(phi == unsafephi, (q / phi) * scale, jnp.array([0.0, 0.0, 0.0]))
    
    return jnp.concatenate((jnp.zeros(3), force), axis = None)


def t_mobility(xq):
    # Mobility matrix transformed to coordinates.
    # Compare with equation: Evensen2008.2
    return (
        transformation_matrix(xq)
        @ (rotation_matrix(xq).T)
        @ mobility
        @ rotation_matrix(xq)
        @ (transformation_matrix(xq).T)
    )

def t_mobility_E(xq):
    # Mobility matrix transformed to coordinates, with the unproper Evensen equation.
    return (
        transformation_matrix_E(xq)
        @ (rotation_matrix(xq).T)
        @ mobility
        @ rotation_matrix(xq)
        @ (transformation_matrix_E(xq).T)
    )

def drift(xq):
    # Drift term.
    # Compare with equation: Evensen2008.5
    # jax.jacobian has differentiation index last (like mu_ij d_k) so divergence is contraction of first and last axis.
    return t_mobility(xq) @ metric_force(xq) + jnp.einsum(
        "iji->j", jax.jacobian(t_mobility)(xq)
    )
    
def drift_E(xq):
    # Drift term with the unproper Evensen equation.
    return t_mobility_E(xq) @ metric_force(xq) + jnp.einsum(
        "iji->j", jax.jacobian(t_mobility_E)(xq)
    )


def noise(xq):
    # Noise term.
    # Compare with equation: Evensen2008.5
    return jnp.sqrt(2) * transformation_matrix(xq) @ (rotation_matrix(xq).T) @ mobility_d

def noise_E(xq):
    # Noise term with the unproper Evensen equation.
    return jnp.sqrt(2) * transformation_matrix_E(xq) @ (rotation_matrix(xq).T) @ mobility_d


def canonicalize_coordinates(xq):
    q = xq[3:]
    unsafephi = jnp.sqrt(jnp.sum(q ** 2))
    phi = jnp.maximum(unsafephi, jnp.array(0.01))

    max_phi = jnp.pi
    canonical_phi = jnp.fmod(phi + max_phi, 2.0 * max_phi) - max_phi

    q = jax.lax.select(
        phi > max_phi, (canonical_phi / phi) * q, q  # and phi == unsafephi
    )
    
    return jnp.concatenate((xq[:3],q), axis=None)

# Stating the problem and solving its SDEs
print("\nSolving proper equations\n")


problem = pychastic.sde_problem.SDEProblem(
    drift, noise, tmax=2.0, x0=jnp.array([1.0, 0.0, 1.0, 0.001, 0.0, 0.0])
)


solver = pychastic.sde_solver.SDESolver(dt=0.001, scheme="euler")

trajectories = solver.solve_many(
    problem,
    step_post_processing=canonicalize_coordinates,
    n_trajectories=1000,
    chunk_size=8,
    chunks_per_randomization=2,
)


# The same solving, now for the unproper Evensen equation.
print("\nSolving Evensen equations\n")


problem_E = pychastic.sde_problem.SDEProblem(
    drift_E, noise_E, tmax=2.0, x0=jnp.array([1.0, 0.0, 1.0, 0.001, 0.0, 0.0])
)

trajectories_E = solver.solve_many(
    problem_E,
    step_post_processing=canonicalize_coordinates,
    n_trajectories=1000,
    chunk_size=8,
    chunks_per_randomization=2,
)

t_n = trajectories["time_values"][0]
t_t = jnp.arange(0.0, trajectories["time_values"][0][-1], 0.005)

# Rotational-rotational correlations.
# Compare with equation: Cichocki2015.71
rotation_matrices = jax.vmap(jax.vmap(rotation_matrix_r))(trajectories["solution_values"][:,:,3:])
rotation_matrices = jnp.einsum(
    "ij,abjk", (rotation_matrix_r(problem.x0[3:]).T), rotation_matrices
)

rotation_matrices_E = jax.vmap(jax.vmap(rotation_matrix_r))(trajectories_E["solution_values"][:,:,3:])
rotation_matrices_E = jnp.einsum(
    "ij,abjk", (rotation_matrix_r(problem.x0[3:]).T), rotation_matrices_E
)

epsilon_tensor = jnp.array(
    [
        [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
        [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
    ]
)

delta_u = -0.5 * jnp.einsum("kij,abij->abk", epsilon_tensor, rotation_matrices)
r_cor = jnp.mean(delta_u ** 2, axis=0)

delta_u_E = -0.5 * jnp.einsum("kij,abij->abk", epsilon_tensor, rotation_matrices_E)
r_cor_E = jnp.mean(delta_u_E ** 2, axis=0)

plt.figure(0)

plt.plot(t_n, r_cor[:, 0], 'b', label = "00-coeff")
plt.plot(t_n, r_cor[:, 1], 'm', label = "11-coeff")
plt.plot(t_n, r_cor[:, 2], 'g', label = "22-coeff")
	
plt.plot(t_n, r_cor_E[:, 0], 'bo', label = "00-coeff (Evensen)", markersize = 1)
plt.plot(t_n, r_cor_E[:, 1], 'mo', label = "11-coeff (Evensen)", markersize = 1)
plt.plot(t_n, r_cor_E[:, 2], 'go', label = "22-coeff (Evensen)", markersize = 1)

D = 1.0
plt.plot(
    t_t,
    1.0 / 6.0
    - (5.0 / 12.0) * jnp.exp(-6.0 * D * t_t)
    + (1.0 / 4.0) * jnp.exp(-2.0 * D * t_t),
    'r',
    label="theoretical",
)

plt.title("$<\Delta u(t) \Delta u(t)>_0$")
plt.legend()

plt.savefig("dudu.png")

f = open("rot_corr.dat", "w")
g = open("rot_corr_E.dat", "w")
t = open("t.dat", "w")

for i in range(len(r_cor[:, 0])):
    g.write(str(r_cor_E[i, 0]) + '\t' + str(r_cor_E[i, 1]) + '\t' + str(r_cor_E[i, 2]) + '\n')
    f.write(str(r_cor[i, 0]) + '\t' + str(r_cor[i, 1]) + '\t' + str(r_cor[i, 2]) + '\n')
    t.write(str(t_n[i]) + '\n')

f.close()
g.close()
t.close()

'''
# Translational-translational correlations.
# Compare with equation: Cichocki2015.57

x_0 = jnp.swapaxes(
    jnp.full(
        (len(t_n), trajectories["solution_values"].shape[0], 3),
        problem.x0[:3]
    ),
    0, 1
    )

delta_R = trajectories["solution_values"][:,:,:3] - x_0
t_cor = jnp.mean(delta_R ** 2, axis = 0)

plt.figure(1)
plt.plot(t_n, t_cor[:, 0], label = "00-coeff")
plt.plot(t_n, t_cor[:, 1], label = "11-coeff")
plt.plot(t_n, t_cor[:, 2], label = "22-coeff")

plt.plot(
    t_t,
    2 * D * t_t,
    label = "theoretical"
)

plt.title("$<\Delta R(t) \Delta R(t)>_0$")
plt.legend()

plt.savefig("dRdR.png")

# Translational-rotational correlations.
# Compare with equation: Cichocki2015.81

delta_u_delta_R = delta_u * delta_R
tr_cor = jnp.mean(delta_u_delta_R, axis = 0)

plt.figure(2)
plt.plot(t_n, tr_cor[:, 0], label = "00-coeff")
plt.plot(t_n, tr_cor[:, 1], label = "11-coeff")
plt.plot(t_n, tr_cor[:, 2], label = "22-coeff")

plt.plot(
    t_t,
    0 * t_t,
    label = "theoretical"
)

plt.title("$<\Delta u(t) \Delta R(t)>_0$")
plt.legend()

plt.savefig("dudR.png")
'''
