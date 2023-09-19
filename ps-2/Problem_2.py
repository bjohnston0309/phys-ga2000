# Run in ipython to make this work

# Importing necessary libraries
from math import sqrt
import numpy as np
import warnings
import timeit

# Ignoring warnings printed to screen
warnings.filterwarnings("ignore")

# Function using a for loop

def madelung_forloop(L):

    M = 0.0

    for i in range(-L,L+1):
        for j in range(-L,L+1):
            for k in range(-L,L+1):
                if i==j==k==0:
                    continue
                M += (-1)**(i+j+k)/sqrt((i**2+j**2+k**2))
    
    return(M)
    
# Function for calculating Madelung constant without for loops

def madelung_noforloop(L):

    x = np.arange(-L,L+1, dtype=float)

    i,j,k = np.meshgrid(x,x,x)

    M2 = np.where((i!=0) | (j!=0) | (k!=0), (-1)**(i+j+k)/np.sqrt(i**2+j**2+k**2),0).sum()
    
    return(M2)

get_ipython().run_line_magic('timeit','madelung_forloop(100)')
get_ipython().run_line_magic('timeit', 'madelung_noforloop(100)')

M1 = madelung_forloop(100)
M2 = madelung_noforloop(100)

#t1 = timeit.timeit(madelung_forloop(100))

print(M1)
print(M2)