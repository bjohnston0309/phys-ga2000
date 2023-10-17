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

time = data1[1]
signal = data1[2]

# SVD technique for 3rd order polynomial
time_p = time/1e9

M = np.zeros((len(time_p), 4))
M[:, 0] = 1.
M[:, 1] = time_p
M[:, 2] = time_p**2
M[:, 3] = time_p**3

(U, w, Vt) = np.linalg.svd(M, full_matrices=False)
V = Vt.T
Ainv = Vt.transpose().dot(np.diag(1. / w)).dot(U.transpose())
c = Ainv.dot(signal)
ym = M.dot(c)

# Plotting the data and model with thirs order polynomial
plt.plot(time_p, signal, '.', label='data')
plt.plot(time_p, ym, '.', label='model')
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Signal', fontsize=16)
plt.savefig('Problem_2b.png')
plt.legend()

# Calculating and plotting residuals between model and predicted data
r = signal - ym

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_size_inches(14,5)
ax1.plot(time_p, r, '.')
ax2.plot(ym, r, '.')
ax1.set(xlabel='Time (s)', ylabel='Residuals')
ax2.set(xlabel='Predicted data', ylabel='Residuals')
ax1.xaxis.label.set(fontsize=16)
ax1.yaxis.label.set(fontsize=16)
ax2.xaxis.label.set(fontsize=16)
ax2.yaxis.label.set(fontsize=16)
fig.savefig('Problem_2c.png')
