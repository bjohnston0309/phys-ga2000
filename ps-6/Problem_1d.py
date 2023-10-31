# Importing necessary libraries
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import time

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

t1 = time.time()

# Calculating eigenvalues for PCA
eigvals_pca, eigvecs_pca = sorted_eigs(residuals, return_eigvalues = True)

C = residuals.T@residuals
t2 = time.time()

print('Dimensions of correlation matrix are', C.shape)
print('Time taken to compute eigenvectors is', t2-t1)

# Plotting first 5 PCA eigenvectors
for i in np.arange(5):
    
    plt.plot(10**logwave/10, eigvecs_pca.T[:,i], linewidth=0.5)
    plt.xlabel('Wavelength (nm)', fontsize=16)
    plt.ylabel('Eigenvectors PCA', fontsize=16)
    plt.legend(['Eigenvector 1', 'Eigenvector 2', 'Eigenvector 3', 'Eigenvector 4', 'Eigenvector 5'])

plt.savefig('PCA_eigenvectors.png')