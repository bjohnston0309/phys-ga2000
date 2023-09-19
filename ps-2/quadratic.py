# Function to determine roots accurately in all cases 
import numpy as np

def quadratic(a, b, c):

    x1 = (-b + np.sqrt(b**2-(4*a*c)))/(2*a)
    x2 = (-b - np.sqrt(b**2-(4*a*c)))/(2*a)

    return x1, x2