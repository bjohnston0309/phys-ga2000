# Importing necessary libraries 
import numpy as np
from random import random
import matplotlib.pyplot as plt

# Constants and arrays
dt = 1
max_time = int(20000)
NBi_213 = np.zeros(max_time+dt)
NBi_213[0] = int(10000)
NTl_209 = np.zeros(max_time+dt)
NPb_209 = np.zeros(max_time+dt)
NBi_209 = np.zeros(max_time+dt)
HL_PB209 = 3.3*60
HL_Tl209 = 2.2*60
HL_Bi_213 = 46*60
time = np.arange(0,max_time+dt, dt)

# Probabilities of decay and weights
p_Pb209 = 1-2**(-dt/HL_PB209)
p_Tl209 = 1-2**(-dt/HL_Tl209)
p_Bi213 = 1-2**(-dt/HL_Bi_213)
weight_Tl209 = 0.0209
weight_Pb209 = 0.9791

# Calculating decays
for i in time[0:20000]:

    # Updating for each loop 
    NPb_209[i+1] = NPb_209[i]
    NTl_209[i+1] = NTl_209[i]
    NBi_209[i+1] = NBi_209[i]
    NBi_213[i+1] = NBi_213[i]
    
    # Decay to Bi_209 from Pb_209
    for j in range(int(NPb_209[i])):
        if random() < p_Pb209:
            NBi_209[i+1] += 1
            NPb_209[i+1] -= 1
    
    # Decay to Pb_209 from Tl_209
    for j in range(int(NTl_209[i])):
        if random() < p_Tl209:
            NPb_209[i+1] += 1
            NTl_209[i+1] -= 1

    # Decay to Pb_209 from Bi_213
    for j in range(int(NBi_213[i])):
        if random() < p_Bi213*weight_Pb209:
            NPb_209[i+1] += 1
            NBi_213[i+1] -= 1

    # Decay to Tl_209 from Bi_213
    for j in range(int(NBi_213[i])):
        if random() < p_Bi213*weight_Tl209:
            NTl_209[i+1] += 1
            NBi_213[i+1] -= 1

# Plotting graph
plt.plot(time, NBi_213)
plt.plot(time, NTl_209)
plt.plot(time, NPb_209)
plt.plot(time, NBi_209)
plt.legend(['Bi 213', 'Tl 209', 'Pb 209', 'Bi 209'])
plt.xlabel('Time (seconds)')
plt.ylabel('Number of atoms')
plt.savefig("prob_3.png")
    
