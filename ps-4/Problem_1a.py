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

# Plotting heat capacity as a function of T
plt.plot(temp, heat_cap)
plt.xlabel('T ($K$)', fontsize=16)
plt.ylabel('$C_V$ ($JK^{-1}m^{-3}$)', fontsize=16)
plt.savefig("Problem_1a.png")