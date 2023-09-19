import numpy as np
import matplotlib.pyplot as plt
import math

#Function for calculating the Mandelbrot set 
def mandelbrot(x, y, num_it):
    c = complex(x, y)
    z = 0
    for i in range(1, num_it):
        if abs(z) > 2:
            return False
        z = z * z + c
    return True


# Determining the Mandelbrot set
size=2000
size_squared=size**2
num_it = 500
d=False
x = np.linspace(-2,2, size+1)
y = np.linspace(-2,2, size+1)
xvals = np.zeros(size_squared)
yvals = np.zeros(size_squared)   

i=0
for a in x:
    for b in y:
       d=mandelbrot(a,b,num_it)

       if d:
           xvals[i] = a
           yvals[i] = b
           i=i+1
        
# Plotting the Mandelbrot set
plt.scatter(xvals, yvals, color='black', s=.1)
plt.xlabel("Real Part of $c$")
plt.ylabel("Imaginary part of $c$")
plt.title("Mandelbrot Set")
plt.savefig("mandelbrot.png")