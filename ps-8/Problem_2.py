# Importing necessary libraries
import numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate


# Function for Lorentz equations (could have included variables for sigma, r and b but for the purposes of this problem there is no need as we have specified variables to use)
def lorentz(t, w):

    x, y, z, = w

    xdot = 10.*(y-x)
    ydot = 28.*x - y - x*z
    zdot = x*y - (8./3.)*z

    return np.array([xdot, ydot, zdot])


# Function for Jacobian for solving set of equations
def lotentz_jac(t, w):

    x, y, z = w 

    jac = np.zeros((3, 3), dtype=np.float32)
    jac[0, :] = [-10., 10., 0.]
    jac[1, :] = [28. - 1.*z, -1., -1.*x]
    jac[2, :] = [1.*y, 1.*x, -(8./3.)]
    return(jac)


# Solving Lorentz equations
results = scipy.integrate.solve_ivp(lorentz, [0., 50.], [0., 1., 0.], jac=lotentz_jac, method='Radau')
print(results)


# Plotting y as a function of t
plt.plot(results.t, results.y[1,:])
plt.xlabel('Time ($s$)', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.savefig('y_vs_t.png')
plt.cla()


# Plotting z as a function of x
plt.plot(results.y[0,:], results.y[2,:])
plt.xlabel('x', fontsize=16)
plt.ylabel('z', fontsize=16)
plt.savefig('z_vs_x.png')
plt.cla()