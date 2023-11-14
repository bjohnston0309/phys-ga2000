# Importing necessary libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import optimize


# Importing data and setting to numpy
data = pd.read_csv('survey.csv')
x = data['age'].to_numpy()
y = data['recognized_it'].to_numpy()
x_sort = np.argsort(x)
x = x[x_sort]
y = y[x_sort]


# Function defining logistic function
def p(x, beta):
    
    return 1/(1+np.exp(-1*(beta[0]+beta[1]*x)))


# Covariance matrix of parameters
def Covariance(hess_inv, resVariance):
    return hess_inv * resVariance


#Error of parameters
def error(hess_inv, resVariance):
    covariance = Covariance(hess_inv, resVariance)
    return np.sqrt( np.diag( covariance ))


# Function defining log likelihood
def log_likelihood(beta, xs, ys):
    beta_0 = beta[0]
    beta_1 = beta[1]
    epsilon = 1e-16
    l_list = [ys[i]*np.log(p(xs[i], beta)/(1-p(xs[i], beta)+epsilon)) 
              + np.log(1-p(xs[i], beta)+epsilon) for i in range(len(xs))]
    ll = np.sum(np.array(l_list), axis = -1)
    return -ll # return log likelihood


# Setting up beta for the problem
beta_0 = np.linspace(-5.,5., 100)
beta_1 = np.linspace(-5.,5.,100)
beta = np.meshgrid(beta_0, beta_1)


# Plotting initial log likelihood
ll = log_likelihood(beta, x, y)
plt.pcolormesh(*beta, ll)
plt.colorbar()
plt.xlabel(r'$\beta_0$', fontsize = 16)
plt.ylabel(r'$\beta_1$', fontsize = 16)
plt.title(r'$\mathcal{L}(\beta_0, \beta_1)$', fontsize = 16)
plt.savefig('log_likelihood.png')
plt.clf()


# Starting values for beta_0 and beta_1
beta_0_start = 1
beta_1_start = 2
beta_start = [beta_0_start, beta_1_start]
errFunc = lambda beta, x, y: log_likelihood(beta, x, y) - y


# Performing calculation
result = optimize.minimize(lambda beta,x,y: np.sum(errFunc(beta,x,y)**2), beta_start,  args=(x,y))
hess_inv = result.hess_inv # inverse of hessian matrix
var = result.fun/(len(y)-len(beta_start)) 
dFit = error( hess_inv,  var)
print('Optimal parameters and error:\n\tbeta: ' , result.x, '\n\tdbeta: ', dFit)
print('Covariance matrix of optimal parameters:\n\tC: ' , Covariance( hess_inv,  var))


# Plotting logistic function with optimised parameters
plt.plot(x, p(x, result.x), '.')
plt.xlabel('Age', fontsize=16)
plt.ylabel('Logistic function', fontsize=16)
plt.savefig('logistic_function.png')


# Printing condition number of covariance matrix
print('Condition number of covariance matrix is', np.linalg.cond(Covariance( hess_inv,  var)))