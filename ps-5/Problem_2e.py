# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# Importing signla data
data = np.genfromtxt('signal.dat',
                     skip_header=1, 
                     skip_footer=1,
                     dtype=None,
                     delimiter='|')

data1 = data.transpose()

time = data1[1]
signal = data1[2]


# Estimating linear trend
time_p = time/10e8 #(time - time.mean()) / time.std()

A = np.zeros((len(time_p), 2))
A[:, 0] = 1.
A[:, 1] = time_p
(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(signal)
ym = A.dot(c)

cond_num = np.linalg.cond(ainv)

print('Condition number of design matrix here is', cond_num)


# Subtracting off linear trend
flat_signal = signal-ym


# Setting up SVD calculation
N = len(time_p)
T = time_p[N-1]-time_p[0]
yf = fft(flat_signal)
xf = fftfreq(N, T)[:N//2]
A = np.zeros((len(time_p), 4))
omega = 2*np.pi*xf[np.argsort(2.0/N * np.abs(yf[0:N//2]))[::-1]][0]

A[:, 0] = 1.
A[:, 1] = time_p
A[:, 2] = np.cos(omega*2*np.pi*time_p)
A[:, 3] = np.sin(2*np.pi*omega*time_p)


# SVD calculation
(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(signal)
ym = A.dot(c) 


# Plotting data
plt.plot(time_p, signal, '.', label='data')
plt.plot(time_p, ym, '.', label='model')
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Signal', fontsize=16)
plt.legend()
plt.savefig('Problem_2e.png')

# Determing period
period = 1/omega

print('The period of the signal is', period)