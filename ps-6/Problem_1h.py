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
    

# Definition to determine the coefficients
def weights_pca(num_spectra, data, eigenvecs):

    vecs = eigenvecs[:,: num_spectra]

    return np.dot(vecs.T, data.T)


# Calculating eigenvalues for PCA
eigvals_pca, eigvecs_pca = sorted_eigs(residuals, return_eigvalues = True)

C = residuals.T@residuals


# Calculating weights for PCA
weight_num = 5

coef_PCA = weights_pca(weight_num, residuals, eigvecs_pca)


# Plotting c_0 vs c_1 and c_0 vs c_1
fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_size_inches(14,5)
ax1.plot(coef_PCA[0,:], coef_PCA[1,:], 'o', linewidth=0.5)
ax2.plot(coef_PCA[0,:], coef_PCA[2,:], 'o')
ax1.set(xlabel='$c_0$', ylabel='$c_1$')
ax2.set(xlabel='$c_0$', ylabel='$c_2$')
ax2.set(ylim=(-0.0004,0.0005))
ax1.xaxis.label.set(fontsize=16)
ax1.yaxis.label.set(fontsize=16)
ax2.xaxis.label.set(fontsize=16)
ax2.yaxis.label.set(fontsize=16)
fig.savefig('c0_c1_c2_graphs.png')
