# Importing the necessary libraries 
import numpy as np
import math
from scipy import integrate
import matplotlib.pyplot as plt
import warnings

# Ignoring warnings printed to screen
warnings.filterwarnings("ignore")

# Setting constants
rho = 6.022e28
V = 0.001
theta = 428
k = 1.380649e-23

# Function setting up integral
def integral(x):

    return (x**4*np.exp(x))/((np.exp(x)-1)**2)

# Function to calculate integral using Gaussian quadrature
def heat_capacity(T, N):

    const = (9*V*rho*k)*((T/theta)**3)

    (gauss_integral, none) = integrate.fixed_quad(integral, 0, theta/T, n=N)

    return const*gauss_integral

# Calculating heat capacity as a function of T
temp = (np.linspace(5,500,100)).astype(int)
N = 50
heat_cap = np.zeros(len(temp))

for i in range(len(temp)):

    heat_cap[i] = heat_capacity(temp[i],N)

# Testing convergence 
N1 = [10, 20, 30, 40, 50, 60, 70]
heat_cap_convergence = np.zeros((len(N1), len(temp)))

heat_cap_convergence[0,1] = 1

for i in range(len(N1)):
    for j in range(len(temp)):
        
        heat_cap_convergence[i,j] = heat_capacity(temp[j],N1[i])

# Calculating difference between consecutive values to get convergence
diffs = np.zeros((len(N1), len(temp)))
total_diff = np.zeros(len(N1))

for i in range(len(N1)-1):
    for j in range(len(temp)):

        diffs[i,j] = heat_cap_convergence[i+1,j] - heat_cap_convergence[i,j]

for i in range(len(N1)):

    total_diff[i] = np.sum(diffs[i,:])

log_diff = np.log(np.abs(total_diff))

plt.plot(N1, log_diff)
plt.xlabel('Number of sample points', fontsize=16)
plt.ylabel('log($summed difference$)', fontsize=16)
plt.savefig("Problem_1b.png")

print('At a value of N=70 Python encountered a RunTimeWarning of divide by zero encountered, meaning that the difference between N=60 and N=70 was zero, hence the calculation had converged at this point')
