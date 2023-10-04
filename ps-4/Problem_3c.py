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

# Function to set up integral
def transform(x):

    return x/(1-x**2)

# Function to determine intrgral
def x_squared(x,n):

    return (1+x**2)/(1-x**2)**2*(transform(x))**2*wavefunction(n, transform(x))**2

# Function to calculate integral and square root
def integral_calc(points, n):

    roots, weights = sc.special.roots_legendre(points)

    sum = 0

    for i in np.arange(points):
        sum += weights[i]*x_squared(roots[i], n)

    return np.sqrt(sum)

# Calculating integral
print(integral_calc(100,5))