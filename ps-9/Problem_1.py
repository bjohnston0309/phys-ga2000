# Importing necessary libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


# Setting up initial variables
L = 1
x = np.linspace(0, L, 1001)
a = L/1000
h = 10**-4
time = np.arange(0, 0.1, h)


# Setting up initial conditions
x_0 = L/2.
sigma = L/10.
kappa = 50./L
psi = np.exp(-1* ((x - x_0)**2)/(2*sigma**2)) * np.exp(1j * kappa * x)
psi[0] = psi[-1] = 0


# Setting up a1, a2, b1, b2
a1 = 1 + 1j*(h/(2*a**2))
a2 = -1j*(h/(4*a**2))
b1 = 1 - 1j*(h/(2*a**2))
b2 = 1j*(h/(4*a**2))


# Creating matrix A
N = 1001
A_diag = np.ones(N, dtype=complex)*a1
A_u = np.ones(N, dtype=complex) * a2
A_u[0] = 0
A_l = np.ones(N, dtype=complex) * a2
A_l[-1] = 0
# build matrix
A = np.array([A_u, A_diag, A_l])


# Getting psi values 
psi_values = []

for t in time:

    psi_values.append(psi)
    psiold = psi
    
    # calculate v
    psiold = np.concatenate(([0],psi,[0])) 
    v = b1*psiold[1:-1] + b2*(psiold[2:]+psiold[:-2])
    
    # Solve matrix
    psi = linalg.solve_banded((1,1), A, v)
    psi[0] = psi[-1] = 0


# Convert psi values list to array
psi_values = np.array(psi_values, dtype=complex)
real_parts = np.real(psi_values)


# Plot for time t=0
plt.plot(x, real_parts[0], label='Psi(t=0.0000)')
plt.xlabel('Position ($1x10^{-8}$m)', fontsize = 16)
plt.ylabel('Amplitude', fontsize = 16)
plt.ylim(-1.1,1.1)
plt.legend()
plt.savefig('t0000.png')
plt.cla()


# Plot for t=0.0012
plt.plot(x, real_parts[120], label='Psi(t=0.0012)', color = 'orange')
plt.xlabel('Position ($1x10^{-8}$m)', fontsize = 16)
plt.ylabel('Amplitude', fontsize = 16)
plt.ylim(-1.1,1.1)
plt.legend()
plt.savefig('t0012.png')
plt.cla()


# Plot for t=0.0020
plt.plot(x, real_parts[200], label='Psi(t=0.0020)', color = 'green')
plt.xlabel('Position ($1x10^{-8}$m)', fontsize = 16)
plt.ylabel('Amplitude', fontsize = 16)
plt.ylim(-1.1,1.1)
plt.legend()
plt.savefig('t0020.png')
plt.cla()


# Plot for t=0.0099
plt.plot(x, real_parts[999], label='Psi(t=0.0099)', color = 'red')
plt.xlabel('Position ($1x10^{-8}$m)', fontsize = 16)
plt.ylabel('Amplitude', fontsize = 16)
plt.ylim(-1.1,1.1)
plt.legend()
plt.savefig('t0099.png')
plt.cla()