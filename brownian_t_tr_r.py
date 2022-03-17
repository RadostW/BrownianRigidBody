import pychastic
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np

mobility = jnp.diag(jnp.array([.258342,.229993,.229993,.554606,.267374,.267374]))
#mobility = 2.0*jnp.eye(6)
mobility_t = mobility[0:3,0:3]
mobility_r = mobility[3:6,3:6]
mobility_dt = jnp.linalg.cholesky(mobility_t)
mobility_dr = jnp.linalg.cholesky(mobility_r) # Compare with equation: Evensen2008.6

def spin_matrix(q):
    # Antisymmetric matrix dual to q
    return jnp.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])


def rotation_matrix(xq):
    # Compare with equation: Evensen2008.11
    q = xq[3:]
    phi = jnp.sqrt(jnp.sum(q ** 2))
    rot = jax.lax.cond( phi > .01,
    
        lambda: (jnp.sin(phi) / phi) * spin_matrix(q)
        + jnp.cos(phi) * jnp.eye(3)
        + ((1.0 - jnp.cos(phi)) / phi ** 2) * q.reshape(1, 3) * q.reshape(3, 1),
        
        lambda: jnp.eye(3)
        #lambda: (1 - phi**2 / 6 + phi**4 / 120) * spin_matrix(q)
        #+ jnp.cos(phi) * jnp.eye(3)
        #+ (.5 - phi**2 / 24 + phi**4 / 720) * q.reshape(1, 3) * q.reshape(3, 1)
        )        
    return jnp.concatenate(
    	(jnp.concatenate((rot, jnp.zeros((3,3))), axis=1),
	jnp.concatenate((jnp.zeros((3,3)), rot), axis=1)), axis=0)
	
def rotation_matrix_r(xq):
    # Compare with equation: Evensen2008.11
    q = xq[3:]
    phi = jnp.sqrt(jnp.sum(q ** 2))
    return jax.lax.cond( phi > .01,
        lambda: (jnp.sin(phi) / phi) * spin_matrix(q)
        + jnp.cos(phi) * jnp.eye(3)
        + ((1.0 - jnp.cos(phi)) / phi ** 2) * q.reshape(1, 3) * q.reshape(3, 1),
        lambda: (1 - phi**2 / 6 + phi**4 / 120) * spin_matrix(q)
        + jnp.cos(phi) * jnp.eye(3)
        + (.5 - phi**2 / 24 + phi**4 / 720) * q.reshape(1, 3) * q.reshape(3, 1))

def transformation_matrix(xq):
    # Compare with equation: Evensen2008.12
    q = xq[3:]
    phi = jnp.sqrt(jnp.sum(q ** 2))
    trans = jax.lax.cond( phi > .01,
        lambda: 0.5
        * (1.0 / phi ** 2 - (jnp.sin(phi) / (2.0 * phi * (1.0 - jnp.cos(phi)))))
        * q.reshape(1, 3)
        * q.reshape(3, 1)
        + spin_matrix(q)
        + (phi * jnp.sin(phi) / (1.0 - jnp.cos(phi))) * jnp.eye(3),
        lambda: 0.5
        * (1./12 + phi**2 / 720 + phi**4 / 30240)
        * q.reshape(1, 3)
        * q.reshape(3, 1)
        + spin_matrix(q)
        + (2. - phi**2 / 6 - phi**4 / 360) * jnp.eye(3))
    return jnp.concatenate(
    	(jnp.concatenate((jnp.eye(3), jnp.zeros((3,3))), axis=1),
	jnp.concatenate((jnp.zeros((3,3)), trans), axis=1)), axis=0)


def metric_force(xq):
    # Compare with equation: Evensen2008.10
    q = xq[3:]
    phi = jnp.sqrt(jnp.sum(q ** 2))
    scale = jax.lax.cond(
        phi < 0.01,
        lambda: -phi / 6 - phi**3 / 360,
        lambda: jnp.sin(phi) / (1 - jnp.cos(phi)) - 2 / phi,)
    return jnp.concatenate(
        (jnp.zeros(3),
        jax.lax.cond(
        	phi > 0.,
        	lambda: (q / phi) * scale,
        	lambda: jnp.zeros(3))),
        axis = None)


def mobility_f(xq):
    return transformation_matrix(xq) @ rotation_matrix(xq).T @ mobility @ rotation_matrix(xq) @ transformation_matrix(xq).T


def drift(xq):
    # Drift term.
    # Compare with equation: Evensen2008.5 
    # jax.jacobian has differentiation index last (like mu_ij d_k) so divergence is contraction of first and last axis.
    return mobility_f(xq) @ metric_force(xq) + jnp.einsum("iji->j", jax.jacobian(mobility_f)(xq))


def noise(xq):
    # Noise term.
    # Compare with equation: Evensen2008.5
   return jnp.sqrt(2) * transformation_matrix(xq) @ rotation_matrix(xq).T @ jnp.linalg.cholesky(mobility)


def canonicalize_coordinates(xq):
    q = xq[3:]
    phi = jnp.sqrt(jnp.sum(q ** 2))
    max_phi = jnp.pi
    canonical_phi = jnp.fmod(phi + max_phi, 2.0 * max_phi) - max_phi
    q = jax.lax.cond(
        phi > max_phi,
        lambda canonical_phi, phi, q: (canonical_phi / phi) * q,
        lambda canonical_phi, phi, q: q,
        canonical_phi,
        phi,
        q,)
    return jnp.concatenate((xq[:3],q), axis=None)

tmax_value = 0.1    
problem = pychastic.sde_problem.SDEProblem(
    drift, noise, tmax=tmax_value, x0 = jnp.reshape(jnp.array([0.,0.,0.,0., 0., 0.]),(6,))
)

solver = pychastic.sde_solver.SDESolver(dt=0.01)

trajectories = solver.solve_many(
    problem,
    step_post_processing=canonicalize_coordinates,
    n_trajectories=2,
    chunk_size=1,
    chunks_per_randomization=1,
)


######## post-processing ########

exit()

def trans_corr(xq_0, xq_t):
    # Translational-translational correlations.
    # Compare with equations: Cichocki2015.23,57
    return (xq_t[:3] - xq_0[:3]) ** 2
    
    
def rot_corr(xq_0, xq_t):
    # Rotational-rotational correlations.
    # Compare with equations: Cichocki2015.29,72
    R_0 = rotation_matrix_r(xq_0);	R_t = rotation_matrix_r(xq_t)
    return .25 * jnp.sum(jnp.cross(R_0.T, R_t.T),axis=0) ** 2

rotation_vectors = trajectories["solution_values"][:,:,3:]
v_rotation_matrix = jax.numpy.vectorize(rotation_matrix_r, signature='(k)->(k,k)')

mathcal_r = rotation_matrix_r(rotation_vectors)


######################		PLOTS		##############################

n_trajectories = trajectories["solution_values"][:,-1,0].shape[0]
t_len = trajectories["solution_values"][1,:,0].shape[0]
t_a = np.linspace(0, tmax_value, t_len)


final_angles = np.array(
    jnp.sqrt(jnp.sum(trajectories["solution_values"][:, -1, 3:] ** 2, axis=1))
)

#'''		PROBABILITY DENSITY OF PHI
plt.figure(0)
xvals = np.arange(0, np.pi, 0.01)
yvals = (np.pi / t_len) * len(final_angles) * ((1.0 - np.cos(xvals)) / np.pi)
plt.plot(xvals, yvals)
plt.hist(final_angles, 20)
plt.savefig('hist_r.png')
#'''
#'''		SQUARE DISPLACEMENT
displacement = np.sqrt(np.sum(trajectories["solution_values"][:, -1, :3] ** 2, axis=1))
plt.figure(1)
plt.hist(displacement, 20)
plt.savefig('hist_displacement.png')
#'''
'''		EULER ANGLES
euler_angles = np.zeros((n_trajectories,3))
for i in range(n_trajectories):
	euler_angles[i] = transformation_matrix(trajectories["solution_values"][i,-1,3:]) #BUG!
	# BUG! There is no defined function to find Euler angles.

plt.figure(2)
plt.hist(euler_angles[:,0],20)
x=np.linspace(0,np.pi,100)
plt.plot(x,(np.pi / 20) * len(final_angles) * .5 * np.sin(x))
plt.savefig('thetas.png')
'''
#'''		TRANSLATIONAL CORRELATIONS

translational_correlations_array = np.zeros((t_len,3))
for t in range(1, t_len):
	print(t, end='\r')
	translational_correlations = 0
	for i in range(n_trajectories):
		xq_0 = trajectories["solution_values"][i,0,:]
		xq_t = trajectories["solution_values"][i,t,:]
		translational_correlations += trans_corr(xq_0, xq_t)
	translational_correlations_array[t] = translational_correlations/n_trajectories

D1 = mobility[5,5]
D3 = mobility[3,3]

t_a2 = np.linspace(0, tmax_value, t_len)
t_a = np.linspace(0, tmax_value, 100*t_len)

plt.figure(3)
plt.plot(t_a2, translational_correlations_array[:,0], 'ro', label='numerical', markersize = 1)
plt.plot(t_a, 2*(D1*t_a + (D1 - D3) * (1 - np.exp(-6*D1*t_a)) / (9*D1)), label = 'theoretical')
plt.legend(loc = 8)
#plt.ylim(0,.3)
plt.savefig('dR_11.png')

plt.figure(4)
plt.plot(t_a2, translational_correlations_array[:,1], 'ro', label='numerical', markersize = 1)
plt.plot(t_a, 2*(D1*t_a + (D1 - D3) * (1 - np.exp(-6*D1*t_a)) / (18*D1)), label = 'theoretical')
plt.legend(loc = 8)
#plt.ylim(0,.3)
plt.savefig('dR_22.png')

plt.figure(5)
plt.plot(t_a2, translational_correlations_array[:,2], 'ro', label='numerical', markersize = 1)
plt.plot(t_a, 2*(D1*t_a + (D1 - D3) * (1 - np.exp(-6*D1*t_a)) / (18*D1)), label = 'theoretical')
plt.legend(loc = 8)
#plt.ylim(0,.3)
plt.savefig('dR_33.png')

#'''
#'''		ROTATIONAL CORRELATIONS

rotational_correlations_array = np.zeros((t_len,3))
for t in range(1, t_len):
	print(t, end='\r')
	rotational_correlations = 0
	for i in range(n_trajectories):
		xq_0 = trajectories["solution_values"][i,0,:]
		xq_t = trajectories["solution_values"][i,t,:]
		rotational_correlations += rot_corr(xq_0, xq_t)
	rotational_correlations_array[t] = rotational_correlations/n_trajectories

D1 = mobility[5,5]
D3 = mobility[3,3]

t_a2 = np.linspace(0, tmax_value, t_len)
t_a = np.linspace(0, tmax_value, 100*t_len)

with open('t_a.txt', 'w') as f:
	f.write(str(t_a))
	f.close()
	
with open('t_a2.txt', 'w') as f:
	f.write(str(t_a2))
	f.close()

plt.figure(6)
plt.plot(t_a2, rotational_correlations_array[:,0], 'ro', label='numerical', markersize = 1)
plt.plot(t_a, 1/6 + 1/12 * np.exp(-6*D1*t_a) - 1/2 * np.exp(-(2*D1+4*D3)*t_a) + 1/4 * np.exp(-2*D1*t_a), label = 'theoretical')
with open('du_11.txt', 'w') as f:
	f.write(str(rotational_correlations_array[:,0]))
	f.close()
plt.legend(loc = 8)
plt.ylim(0,.4)
plt.savefig('du_11.png')

plt.figure(7)
plt.plot(t_a2, rotational_correlations_array[:,1], 'ro', label='numerical', markersize = 1)
plt.plot(t_a, 1/6 - 1/6 * np.exp(-6*D1*t_a) - 1/4 * np.exp(-(5*D1+D3)*t_a) + 1/4 * np.exp(-(D1+D3)*t_a), label = 'theoretical')
with open('du_22.txt', 'w') as f:
	f.write(str(rotational_correlations_array[:,1]))
	f.close()
plt.legend(loc = 8)
plt.ylim(0,.4)
plt.savefig('du_22.png')

plt.figure(8)
plt.plot(t_a2, rotational_correlations_array[:,2], 'ro', label='numerical', markersize = 1)
plt.plot(t_a, 1/6 - 1/6 * np.exp(-6*D1*t_a) - 1/4 * np.exp(-(5*D1+D3)*t_a) + 1/4 * np.exp(-(D1+D3)*t_a), label = 'theoretical')
with open('du_33.txt', 'w') as f:
	f.write(str(rotational_correlations_array[:,2]))
	f.close()
plt.legend(loc = 8)
plt.ylim(0,.4)
plt.savefig('du_33.png')
#'''



