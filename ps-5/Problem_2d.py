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
time_p = time/1e9

# SVD with a much higher order polynomial
N = 30
M_1 = np.zeros((len(time_p), N))

for i in range(len(M_1.transpose())):

    M_1[:,i] = time_p**i

(U_1, w_1, Vt_1) = np.linalg.svd(M_1, full_matrices=False)
V_1 = Vt_1.T
Ainv_1 = Vt_1.transpose().dot(np.diag(1. / w_1)).dot(U_1.transpose())
c_1 = Ainv_1.dot(signal)
ym_1 = M_1.dot(c_1)

# Plotting the data and model with thirs order polynomial
plt.plot(time_p, signal, '.', label='data')
plt.plot(time_p, ym_1, '.', label='model')
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Signal', fontsize=16)
plt.savefig('Problem_2di')
plt.legend()

cond1 = np.linalg.cond(Ainv_1)

print('Condition number for matrix here is ', cond1)

# Calculating condition numbers for increasing value of polynomial number
N_2 = np.linspace(1,100,99).astype(int)
cond_nums = np.zeros(len(N_2))

for i in range(len(N_2)):

    M_2 = np.zeros((len(time_p), N_2[i]))

    for j in range(len(M_2.transpose())):

        M_2[:,j] = time_p**j

    (U_2, w_2, Vt_2) = np.linalg.svd(M_2, full_matrices=False)
    V_2 = Vt_2.T
    Ainv_2 = Vt_2.transpose().dot(np.diag(1. / w_2)).dot(U_2.transpose())

    cond_nums[i] = np.linalg.cond(Ainv_2)

fig1, (ax3, ax4) = plt.subplots(1,2)
fig1.set_size_inches(14,5)
ax3.plot(N_2, cond_nums)
ax4.plot(N_2[0:10], cond_nums[0:10])
ax3.set(xlabel='Polynomial order', ylabel='Condition Number')
ax4.set(xlabel='Polynomial order', ylabel='Condition Number')
ax3.xaxis.label.set(fontsize=16)
ax3.yaxis.label.set(fontsize=16)
ax4.xaxis.label.set(fontsize=16)
ax4.yaxis.label.set(fontsize=16)
fig1.savefig('Problem_2dii.png')