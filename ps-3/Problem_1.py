# Importing necessary libraries
import numpy as np
import math

# Function upon which the derivative will be performed
def deriv_func(x):

    f = x*(x-1)

    return f

# Performing derivative
x = 1
dx = [10e-2, 10e-4, 10e-6, 10e-8, 10e-10, 10e-12, 10e-14]
df = np.zeros(len(dx))

for i in range(len(dx)):
    df[i] = (deriv_func(x+dx[i]) - deriv_func(x))/dx[i]

print("For dx=10e-2, df =", df[0])
print("For dx=10e-4, df =", df[1])
print("For dx=10e-6, df =", df[2])
print("For dx=10e-8, df =", df[3])
print("For dx=10e-10, df =", df[4])
print("For dx=10e-12, df =", df[5])
print("For dx=10e-14, df =", df[6])
print("Analytical derivative at x=1 is 1")
