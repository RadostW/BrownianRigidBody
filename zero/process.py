# The following code is post-processing of the data obtained in evensen_2008.py

import numpy as np
import matplotlib.pyplot as plt


# Font parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams['mathtext.fontset'] = 'cm'

# Size of the files
with open('t.dat') as f:
    length = sum(1 for _ in f)
f.close()


# Reading data from the files
rot_corr = np.empty((length, 3))
rot_corr_E = np.empty((length, 3))
t = np.empty((length))

r = open("rot_corr.dat", "r")
r_E = open("rot_corr_E.dat", "r")
t_file = open("t.dat", "r")

for i, line in enumerate(r):
    l = line.split()
    rot_corr[i, :] = l
    
for i, line in enumerate(r_E):
    l = line.split()
    rot_corr_E[i, :] = l
    
for i, line in enumerate(t_file):
    t[i] = line
    
r.close()
r_E.close()
t_file.close()


# Plotting the correlations
plt.figure(0)

plt.plot(t, rot_corr[:, 0], 'c', label = "Waszkiewicz et al. [2022]")
plt.plot(t, rot_corr[:, 1], 'c')
plt.plot(t, rot_corr[:, 2], 'c')
	
plt.plot(t, rot_corr_E[:, 0], 'g', label = "Evensen et al. [2008]", markersize = 1)
plt.plot(t, rot_corr_E[:, 1], 'g', markersize = 1)
plt.plot(t, rot_corr_E[:, 2], 'g', markersize = 1)

D = 1.0
plt.plot(
    t,
    1.0 / 6.0
    - (5.0 / 12.0) * np.exp(-6.0 * D * t)
    + (1.0 / 4.0) * np.exp(-2.0 * D * t),
    'r--',
    label="Cichocki et al. [2015]",
    linewidth = 2
)

plt.xlabel("Time $t$ [s]")
plt.ylabel("$<\Delta u(t) \Delta u(t)>_0$")
plt.title("Rotational correlations")
plt.legend()

plt.xlim(min(t), max(t))
plt.ylim(bottom = 0.0)

plt.savefig("rotational_correlations.png")
