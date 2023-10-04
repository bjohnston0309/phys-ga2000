# Importing necessary libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate
import warnings

# Ignoring warnings printed to screen
warnings.filterwarnings("ignore")

# Settingup arrays and constants
a = np.linspace(0,2,100)
T = np.zeros(len(a))
N = 20

# Funciton setting up integral
def integral(x):

    return (1/np.sqrt(a[i]**4-x**4))

# Writing function to determine time period as a function of amplitude
def time_period(y, N):
    const = np.sqrt(8)

    (gauss_integral, none) = integrate.fixed_quad(integral, 0, y, n=N)

    return const*gauss_integral

# Determining period for increasing values of amplitude
for i in range(len(a)):

    T[i] = time_period(a[i], N)

# Plotting graph of period as a function of amplitude
plt.plot(a, T)
plt.xlabel('Amplitude ($m$)', fontsize=16)
plt.ylabel('Time Period ($s$)', fontsize=16)
plt.savefig("Problem_2.png")