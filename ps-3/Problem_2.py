# Importing necessary libraries 
import numpy as np
import matplotlib.pyplot as plt
import time

# Function to measure time
def measure_time():
    start_time = time.time()
    end_time = time.time()
    return end_time - start_time

# Defining a function for matrix multiplication using for loops as in example
def matrix_mult_forloop(N):

    A1 = np.ones([N,N], float)
    B1 = np.ones([N,N], float)
    C1 = np.zeros([N,N], float)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                C1[i,j] += A1[i,k]*B1[k,j]

    return C1

# Defining a function for matrix multiplication kusing dot() method 
def matrix_mult_dot(N):

    A2 = np.ones([N,N], float)
    B2 = np.ones([N,N], float)

    C2 = np.dot(A2, B2)

    return(C2)

# Calculating computation time for for loops
N1 = (np.linspace(1,100,20)).astype(int)
t1 = [[] for i in range(len(N1))]

for i in range(len(N1)):

    start_time = time.time()
    matrix_mult_forloop(N1[i])
    end_time = time.time()
    total_time = end_time - start_time
    t1[i] = end_time - start_time

# Plotting computation time using for loops 
plt.plot(N1,t1, 'x')
plt.plot(N1, 1/100**3/4*N1**3)
plt.title('Computational time using for loops')
plt.xlabel('Matrix size, N')
plt.ylabel('t ($s$)')
plt.legend(['Computational time', '$\sim N^3$'])
plt.savefig("for_loop.png")

# Calculating computation time for dot method
N2 = (np.linspace(1,100,20)).astype(int)
t2 = [[] for i in range(len(N2))]

for i in range(len(N2)):

    start_time = time.time()
    matrix_mult_dot(N2[i])
    end_time = time.time()
    total_time = end_time - start_time
    t2[i] = end_time - start_time

# Plotting computation time using dot()
plt.figure()
plt.plot(N2,t2, 'x')
plt.title('Computational time using dot() method')
plt.xlabel('Matrix size, N')
plt.ylabel('t ($s$)')
plt.legend(['Computational time'])
plt.savefig("dot.png")