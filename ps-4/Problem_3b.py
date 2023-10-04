# Importing necessary libraries 
import numpy as np 
import math
import scipy as sc
from scipy import integrate
import matplotlib.pyplot as plt

# Function for Hermite Polynomial calculation
def H(n,x):

    if n ==0:
        
        return 1
    
    elif n==1:

        return 2*x
    
    else:

        return 2*x*H(n-1,x)-2*(n-1)*H(n-2,x) 

# Function for wavefunction
def wavefunction(n,x):

    const = 1/np.sqrt(2**n*math.factorial(n)*np.sqrt(math.pi))
    
    wave_func = const*np.exp(-x**2/2)*H(n,x)

    return wave_func

# Part b with n = 30
x1 = np.linspace(-10,10,1000)
n1 = 30
W30 = np.zeros(len(x1))

plt.plot(x1, wavefunction(n1, x1))
plt.xlabel('x (length)', fontsize=16)
plt.ylabel('$\psi_{30}$ (1/$\sqrt{length}$)', fontsize=16)
plt.savefig("Problem_3b.png")