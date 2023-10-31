# Importing necessary libraries
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

# Reading in data 
hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

# Plotting 1st galaxy - not normalised
plt.plot(logwave, np.abs(flux[0]), '-', linewidth=0.5)
plt.xlabel('$log_{10}(\lambda)$ ($log_{10}(\AA)$)', fontsize=16)
plt.ylabel('Flux ($10^{-17}serg s^{-1} cm^{-2} \AA^{-1}$)', fontsize=16)
plt.savefig('not_normalised_flux.png')

# Attempting to normalise fluxes
normalisation = np.sum(flux, axis=1)
normalised_flux = flux/np.tile(normalisation, (np.shape(flux)[1], 1)).T

# Checking data is properly normalised
plt.plot(np.sum(normalised_flux, axis = 1))
plt.ylim(0,2)
plt.xlabel('Data points', fontsize=16)
plt.ylabel('Sum of flux', fontsize=16)
plt.savefig('normalised_check.png')

# Determining average from normalised spectra
av_norm = np.mean(normalised_flux, axis = 1)
residuals = normalised_flux-np.tile(av_norm, (np.shape(flux)[1], 1)).T

# Plotting final spectra
fig, (ax1) = plt.subplots(1,1)
ax1.plot(10**logwave/10, residuals[0], '-', linewidth=0.5)
ax1.set(xlabel='Wavelength (nm)', ylabel='Flux ($10^{-17}serg s^{-1} cm^{-2} \AA^{-1}$)')
ax1.xaxis.label.set(fontsize=16)
ax1.yaxis.label.set(fontsize=16)
fig.savefig('normalised_flux.png', bbox_inches='tight')

