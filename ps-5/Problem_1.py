# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy as sc

# Function to compute integrand
def integrand(x,a):

    return x**(a-1)*np.exp(-x)

# PLotting integrand for different values of a
x = np.linspace(0,5,1000)
a = [2, 3, 4]

for i in range(len(a)):

    plt.plot(x, integrand(x,a[i]))
    plt.legend(['a = 2', 'a = 3', 'a = 4'])
    plt.xlabel('x', fontsize=16)
    plt.ylabel('Integrand', fontsize=16)
    plt.savefig('Problem_1.png')

# Defining transform for x
def transform(x,a):

    return (a-1)*x/(1-x)

# Function for altered gamma function
def integrand_altered(x,a):

    return np.exp((a-1)*np.log(x)-x)

# Function to compute gamma function using Gaussian quadrature
def gamma(a):

    gam = lambda x: integrand_altered(transform(x,a), a)*(a-1)/((1-x)**2)
    (s, none) = integrate.fixed_quad(gam, 0, 1, n=100)

    return s

print('For a=3/2 the gamma function is', gamma(3/2))
print('For a=3 the gamma function is', gamma(3))
print('For a=6 the gamma function is', gamma(6))
print('For a=10 the gamma function is', gamma(10))