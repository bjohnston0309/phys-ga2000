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

# Setting up arrays
x = np.linspace(-4,4,1000)
N = np.arange(0,4)

# Plotting wavefunctions
for i in N:

    plt.plot(x, wavefunction(i, x))

plt.xlim(-4,4)
plt.xlabel('$x$ (length)', fontsize=16)
plt.ylabel('$\psi$ (1/$\sqrt{length}$)', fontsize=16)
plt.legend(['n = 0', 'n = 1', 'n = 2', 'n = 3'])
plt.savefig("Problem_3a.png")