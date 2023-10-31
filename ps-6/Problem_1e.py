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

t1 = time.time()

# SVD calculation
(u, w, vt) = linalg.svd(residuals, full_matrices=True)

t2 = time.time()

print('Time taken to compute eigenvectors is', t2-t1)

# SVD manipulation
eigvecs_svd = vt.T
eigvals_svd = w**2
svd_sort = np.argsort(eigvals_svd)[::-1]
eigvecs_svd = eigvecs_svd[:,svd_sort]
eigvals_svd = eigvals_svd[svd_sort]

# Plotting first 5 SVD eigenvectors
for i in range(5):

    plt.plot(10**logwave/10, eigvecs_svd.T[:,i], linewidth=0.5)
    plt.xlabel('Wavelength (nm)', fontsize=16)
    plt.ylabel('Eigenvectors SVD', fontsize=16)
    plt.legend(['Eigenvector 1', 'Eigenvector 2', 'Eigenvector 3', 'Eigenvector 4', 'Eigenvector 5'])

plt.savefig('SVD_eigenvectors.png')