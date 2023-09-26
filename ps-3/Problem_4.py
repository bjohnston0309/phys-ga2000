# Importing necessary libraries 
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Setting up necessary arrays and constants
N = np.arange(1000)
dt = 1
tau = 3.053*60
mu = np.log(2)/tau
random_nums = np.random.uniform(0,1,1000)
x = -(1/mu)*np.log(1-random_nums)
x_sorted = -np.sort(-x)

# Plotting decay 
plt.plot(x_sorted, N)
plt.xlabel('Time (s)')
plt.ylabel('Number of atoms')
plt.savefig("prob_4.png")