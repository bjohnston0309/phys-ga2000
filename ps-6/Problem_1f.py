# Importing necessary libraries
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

# Reading in data 
hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

# Attempting to normalise fluxes
normalisation = np.sum(flux, axis=1)
normalised_flux = flux/np.tile(normalisation, (np.shape(flux)[1], 1)).T

# Determining average from normalised spectra
av_norm = np.mean(normalised_flux, axis = 1)
residuals = normalised_flux-np.tile(av_norm, (np.shape(flux)[1], 1)).T

# Function to compute eigenvalues
def sorted_eigs(r, return_eigvalues = False):
    """
    Calculate the eigenvectors and eigenvalues of the correlation matrix of r
    -----------------------------------------------------
    """
    corr=r.T@r
    eigs=np.linalg.eig(corr) 
    arg=np.argsort(eigs[0])[::-1] 
    eigvec=eigs[1][:,arg] 
    eig = eigs[0][arg]
    if return_eigvalues == True:
        return eig, eigvec
    else:
        return eigvec
    

# Definition for covariance matrix
def cov_matrix(residuals):

    Rij = np.matrix(residuals)

    return Rij.T*Rij


# Calculating eigenvalues for PCA
eigvals_pca, eigvecs_pca = sorted_eigs(residuals, return_eigvalues = True)

C = residuals.T@residuals


# SVD calculation
(u, w, vt) = linalg.svd(residuals, full_matrices=True)

# SVD manipulation
eigvecs_svd = vt.T
eigvals_svd = w**2
svd_sort = np.argsort(eigvals_svd)[::-1]
eigvecs_svd = eigvecs_svd[:,svd_sort]
eigvals_svd = eigvals_svd[svd_sort]


# Condition numbers
print('Condition number of covariance matrix is', np.abs(max(eigvals_pca)/min(eigvals_pca)))
print('Condition number of R is', max(w)/min(w))


# Plotting eigenvectors for both PCA and SVD
[plt.plot(eigvecs_svd[:,i], eigvecs_pca[:,i], 'o')for i in range(500)]
plt.plot(np.linspace(-0.2, 0.2), np.linspace(-0.2, 0.2))
plt.xlabel('SVD eigenvectors', fontsize = 16)
plt.ylabel('PCA eigenvectors', fontsize = 16)
plt.savefig('EigenvectorsPCA_vs_eigenvectorsSVD.png')

# PLotting eigenvalues for both PCA and SVD
fig, (ax1) = plt.subplots(1,1)
ax1.plot(eigvals_svd[:500], eigvals_pca[:500], 'o')
ax1.set(xlabel='SVD eigenvalues', ylabel='PCA eigenvalues')
ax1.xaxis.label.set(fontsize=16)
ax1.yaxis.label.set(fontsize=16)
fig.savefig('EigenvaluesPCA_vs_eigenvaluesSVD.png', bbox_inches='tight')