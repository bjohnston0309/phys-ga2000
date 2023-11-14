# Importing necessary libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import brent


# Definining function to be minimised
def f(x):

    y = (x-0.3)**2*math.exp(x)

    return y


# Golden search function
def golden(func=None, astart=None, bstart=None, cstart=None, tol=1.e-7):
    xgrid = -12. + 25. * np.arange(10000) / 10000. 
    #plt.plot(xgrid, func(xgrid))
    gsection = (3. - np.sqrt(5)) / 2
    a = astart
    b = bstart
    c = cstart
    while(np.abs(c - a) > tol):
        # Split the larger interval
        if((b - a) > (c - b)):
            x = b
            b = b - gsection * (b - a)
        else:
            x = b + gsection * (c - b)
        step = np.array([b, x])
        #plt.plot(step, func(step), color='black')
        #plt.plot(step, func(step), '.', color='black')
        fb = func(b)
        fx = func(x)
        if(fb < fx):
            c = x
        else:
            a = b
            b = x 
    return(b)


# Parabolic minimisation
def s_quad_interp(a, b, c):
    """
    inverse quadratic interpolation
    """
    epsilon = 1e-7 #for numerical stability
    s0 = a*f(b)*f(c) / (epsilon + (f(a)-f(b))*(f(a)-f(c)))
    s1 = b*f(a)*f(c) / (epsilon + (f(b)-f(a))*(f(b)-f(c)))
    s2 = c*f(a)*f(b) / (epsilon + (f(c)-f(a))*(f(c)-f(b)))
    return s0+s1+s2


# Optimisation function
def optimize():
    #define interval
    a = -1
    b = 1
    tol = 1e-7
    if abs(f(a)) < abs(f(b)):
        a, b = b, a #swap bounds
    c = a
    flag = True
    err = abs(b-a)
    err_list, b_list = [err], [b]
    while err > tol:
        s = s_quad_interp(a,b,c)
        if ((s >= b))\
            or ((flag == True) and (abs(s-b) >= abs(b-c)))\
            or ((flag == False) and (abs(s-b) >= abs(c-d))):
            s = golden(func=f, astart=s, bstart=b, cstart=c)
            flag = True
        else:
            flag = False
        c, d = b, c # d is c from previous step
        #if f(a)*f(s) < 0:
        #    b = s
        #else:
        a = s
        if abs(f(a)) < abs(f(b)):
            a, b = b, a #swap if needed
        err = abs(b-a) #update error to check for convergence
        err_list.append(err)
        b_list.append(b)
    print(f'minimum = {b}')

    b_opt = b

    return b_list, err_list, b_opt


# Plotting function
def plot(b_list, err_list):
    log_err = [np.log10(err) for err in err_list]
    fig, axs = plt.subplots(2,1, sharex=False)
    ax0, ax1 = axs[0], axs[1]
    ax0.scatter(range(len(b_list)), b_list, marker = 'o', facecolor = 'red', edgecolor = 'k')
    ax0.plot(range(len(b_list)), b_list, 'r-', alpha = .5)
    ax1.plot(range(len(err_list)), log_err,'.-')
    ax1.set_xlabel('Number of Iterations', fontsize=16)
    ax0.set_ylabel(r'$x_{min}$', fontsize=16)
    ax1.set_ylabel(r'$\log{\delta}$', fontsize=16)
    plt.savefig('convergence.png')


# Performing minimisation without scipy
if __name__ == "__main__":
    b_list, err_list, b_opt = optimize()
    plot(b_list, err_list)


# Performing 1D Brent minimisation using scipy.optimize.brent
brent_calc_scipy = brent(f)


# Plotting function and minimised points from both methods (zoomed in on minimised points)
x1 = np.linspace(-10,10,10000)
y1 = np.zeros(len(x1))
x2 = np.linspace(0.1,0.4,1000)
y2 = np.zeros(len(x2))
x3 = np.linspace(0.29999999, 0.30000001,100)
y3 = np.zeros(len(x3))


for i in range(len(x1)):

    y1[i] = f(x1[i])

for i in range(len(x2)):

    y2[i] = f(x2[i])


fig, (ax1, ax2, ax3) = plt.subplots(1,3)
fig.set_size_inches(14,5)
ax1.plot(x1, y1)
ax1.plot(brent_calc_scipy, 0, 'x')
ax1.plot(b_opt, 0, 'x')
ax2.plot(x2, y2)
ax2.plot(brent_calc_scipy, 0, 'x')
ax2.plot(b_opt, 0, 'x')
ax3.plot(x1, y1)
ax3.plot(brent_calc_scipy, 0, 'x')
ax3.plot(b_opt, 0, 'x')
ax1.set(xlabel='$x$', ylabel='$(x-0.3)^2e^x$')
ax2.set(xlabel='$x$', ylabel='$(x-0.3)^2e^x$')
ax3.set(xlabel='$x$', ylabel='$(x-0.3)^2e^x$')
ax2.set(xlim=(0.1,0.4))
ax3.set(xlim=(0.29999999, 0.30000001))
ax1.xaxis.label.set(fontsize=12)
ax1.yaxis.label.set(fontsize=12)
ax2.xaxis.label.set(fontsize=12)
ax2.yaxis.label.set(fontsize=12)
ax3.xaxis.label.set(fontsize=12)
ax3.yaxis.label.set(fontsize=12)
fig.legend(['Function', 'Minimised point (scipy)', 'Minimised point (no scipy)'])
fig.savefig('minimised.png')
