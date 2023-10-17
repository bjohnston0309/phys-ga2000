# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Importing signla data
data = np.genfromtxt('signal.dat',
                     skip_header=1, 
                     skip_footer=1,
                     dtype=None,
                     delimiter='|')

data1 = data.transpose()

# Plotting signal
time = data1[1]
signal = data1[2]
plt.plot(time, signal, '.')
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Signal', fontsize=16)
plt.savefig('Problem_2a.png')