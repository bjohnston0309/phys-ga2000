import math
import numpy as np
import quadratic

# Function that takes 3 numbers and prints out solutions to a quadratic equation
def quadratic_solver(a,b,c):

    x1 = (-b + np.sqrt(b**2-(4*a*c)))/(2*a)
    x2 = (-b - np.sqrt(b**2-(4*a*c)))/(2*a)

    x11 = (2*c)/(-b-np.sqrt(b**2-4*a*c))
    x22 = (2*c)/(-b+np.sqrt(b**2-4*a*c))

    return x1, x2, x11, x22

# Solving quadratric for parts a and b 
sol_b = quadratic_solver(0.001,1000,0.001)

print("Solutions for part a are", sol_b[0], "and", sol_b[1])
print("Solutions for part b are", sol_b[2], "and", sol_b[3])
