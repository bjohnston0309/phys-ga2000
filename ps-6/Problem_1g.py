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


# Dimensionality reduction PCA
def PCA(l, r, eigenvecs, project = True):
    """
    Perform PCA dimensionality reduction
    --------------------------------------------------------------------------------------
    """
    reduced_wavelength_data= np.dot(eigenvecs[:,:l].T,r.T) 
    if project == False:
        return reduced_wavelength_data.T 
    else: 
        return np.dot(eigenvecs[:,:l], reduced_wavelength_data).T 


# Calculating eigenvalues for PCA
eigvals_pca, eigvecs_pca = sorted_eigs(residuals, return_eigvalues = True)

C = residuals.T@residuals


# Using first 5 eigenvectors to model data
plt.plot(10**logwave/10, residuals[1,:], label = 'original data')
plt.plot(10**logwave/10, PCA(5,residuals, eigvecs_pca)[1,:], label = 'l = 5')
plt.ylabel('Flux ($10^{-17}serg s^{-1} cm^{-2} \AA^{-1}$)', fontsize=16)
plt.xlabel('Wavelength (nm)', fontsize = 16)
plt.legend()
plt.savefig('reduction_5_eigvecs.png')


# Checking all 4001 eigenvectors reproduces data exactly
fig, (ax1) = plt.subplots(1,1)
ax1.plot(10**logwave/10, residuals[1,:], label = 'original data')
ax1.plot(10**logwave/10, PCA(4001,residuals[1,:], eigvecs_pca), label = 'l = 4001')
ax1.set(xlabel='Wavelength (nm)', ylabel='Flux ($10^{-17}serg s^{-1} cm^{-2} \AA^{-1}$)')
ax1.xaxis.label.set(fontsize=16)
ax1.yaxis.label.set(fontsize=16)
ax1.legend()
fig.savefig('reduction_4001_eigvecs.png')
