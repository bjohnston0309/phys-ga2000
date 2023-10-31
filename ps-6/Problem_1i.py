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


# Calculating least squares residuals 
N = np.linspace(1,20,19).astype(int)
squared_residuals = np.zeros(len(N))

for i in np.arange(19):

    least_vecs = np.asarray(PCA(i, residuals, eigvecs_pca))
    squared_residuals[i] = np.sum((least_vecs - residuals)**2)


# Plotting squared residuals as a function of N
plt.plot(N, squared_residuals)
plt.xlabel('N', fontsize=16)
plt.ylabel('Squared residuals', fontsize=16)
plt.savefig('least_squared_residuals.png')