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
def wavefunction_GH(n,x):

    const = 1/np.sqrt(2**n*math.factorial(n)*np.sqrt(math.pi))
    
    wave_func_GH = const*H(n,x)

    return wave_func_GH

# Function to determine intrgral
def x_squared_GH(x,n):

    return x**2*wavefunction_GH(n,x)**2

# Gauss-Hermite quadrature calculation
def integral_calc_GH(points_GH,n):

    roots_GH, weights_GH = sc.special.roots_hermite(points_GH)

    sum_GH = 0

    for i in np.arange(points_GH):
        sum_GH += weights_GH[i]*x_squared_GH(roots_GH[i], n)

    return np.sqrt(sum_GH)

# Calculating integral using Gauss-Hermite quadrature 
print(integral_calc_GH(100,5))

# Checking for correct integral
num_points = (np.linspace(1,1000,999)).astype(int)
int_check = np.zeros(len(num_points))

for i in range(len(num_points)):

    int_check[i] = integral_calc_GH(num_points[i], 5)

plt.plot( num_points, int_check)
plt.xlabel('Number of integration points', fontsize=16)
plt.ylabel('Uncertainty in position', fontsize=16)
plt.savefig('Problem_3d.png')