# The following code is post-processing of the data obtained in evensen_2008.py

import numpy as np
import matplotlib.pyplot as plt


# Font parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams['mathtext.fontset'] = 'cm'


# Size of the files
with open('phi.dat') as f:
    length = sum(1 for _ in f)
f.close()


# Reading data from the files
phis = np.empty((length))
phis_E = np.empty((length))

p = open("phi.dat", "r")
p_E = open("phi_E.dat", "r")

for i, line in enumerate(p):
    phis[i] = line
    
for i, line in enumerate(p_E):
    phis_E[i] = line
    
p.close()
p_E.close()


# Plotting the stationary probability density
plt.figure(0)

plt.hist(phis, bins = 20, range = (0, np.pi), edgecolor = "c", density = True,
    color = "c", label = "Waszkiewicz et al. [2022]", linewidth = 3.0)
plt.hist(phis_E, bins = 20, range = (0, np.pi), edgecolor = "g", density = True,
    fill = False, label = "Evensen et al. [2008]", linewidth = 3.0)

angles = np.linspace(0, np.pi, 200)
plt.plot(angles, (1 - np.cos(angles)) / np.pi, "r", 
    label = "Eq. (13) Evensen et al. [2008]", linewidth = 2.0)

plt.xlabel("Angle of rotation $\Phi$")
plt.ylabel("Probability distribution")
plt.title("Stationary distribution of the angle of rotation $\Phi$")
plt.legend()

plt.xlim(0, np.pi)

plt.savefig("phi_histogram.png")
