'''
Plotting the NIRSpec spectrum, exploring it.

Goals:  Figure out what's wrong with the equivalent width calculation?
        One you do, set up a series of runs where you make a subfolder within
        plots-data for line spectra?  Could input a linelist and then have the
        code run through the spectrum and calculate the info for each line
        --> line flux, centroid, fwhm, equivalent width, etc.

'''


__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fitting_ifu_spectra import *


name = 'S1723'
z = 1.3293

all_lines = {'nii.a':0.654981, 'niii.b':0.658523,
           'ha':0.656461,
           'sii.a':0.6718, 'sii.b':0.6732,
           'hei':0.587569}

# reading in random spaxels I saved
x,y = 32,20 # spaxel defined in "exploring-1D-spectra.py"

spectrum = np.loadtxt(f'plots-data/{name}-spaxel-{x}-{y}-1D.txt',delimiter='\t')
spec = pd.DataFrame({'wave':spectrum[:,0],'flux':spectrum[:,1],'ferr':spectrum[:,2]})

# converting spectrum flux to cgs units first
spec['flux'] *= 1e-23 # Jy --> erg/s/cm/Hz (fnu)
spec['flux'] *= 2.998e14 / (spec.wave.values)**2 # fnu --> flam

# converting spectrum error to cgs units first
spec['ferr'] *= 1e-23 # Jy --> erg/s/cm/Hz (fnu)
spec['ferr'] *= 2.998e14 / (spec.wave.values)**2 # fnu --> flam



# -------- #
# PLOTTING #
# -------- #

yscale = 1e-9 # erg/s/cm2/A

# looking for Ha & [NII]
plt.figure(figsize=(10,5))

plt.step(spec.wave,spec.flux/yscale,where='mid')

for l in [0.654981, 0.656461, 0.658523]:
    zline = l*(1+z)
    plt.axvline(zline,ls=':',color='k',zorder=0)
    plt.axvspan(zline-0.0014,zline+0.0014,alpha=0.2)


plt.ylim(-0.25,2)  
plt.xlim(1.51,1.55)

plt.ylabel('flux [10 $^{%s}$ erg/s/cm$^2$/$\AA$]'%int(np.log10(yscale)))
plt.xlabel('wavelength [microns]')

plt.tight_layout()
plt.show()
plt.close('all')


# looking for [SII]
plt.figure(figsize=(10,5))

plt.step(spec.wave,spec.flux/yscale,where='mid')

for l in [0.6718,0.6732]:
    zline = l*(1+z)
    plt.axvline(zline,ls=':',color='k',zorder=0)
    plt.axvspan(zline-0.0014,zline+0.0014,alpha=0.2)


plt.ylim(-0.25,1)  
plt.xlim(1.55,1.58)

plt.ylabel('flux [10 $^{%s}$ erg/s/cm$^2$/$\AA$]'%int(np.log10(yscale)))
plt.xlabel('wavelength [microns]')

plt.tight_layout()
plt.show()
plt.close('all')


# looking for HeI5875
plt.figure(figsize=(10,5))

plt.step(spec.wave,spec.flux/yscale,where='mid')

for l in [0.587569]:
    zline = l*(1+z)
    plt.axvline(zline,ls=':',color='k',zorder=0)
    plt.axvspan(zline-0.0014,zline+0.0014,alpha=0.2)

plt.ylim(-0.25,1)  
plt.xlim(1.36,1.382)

plt.ylabel('flux [10 $^{%s}$ erg/s/cm$^2$/$\AA$]'%int(np.log10(yscale)))
plt.xlabel('wavelength [microns]')

plt.tight_layout()
plt.show()
plt.close('all')



# CALCULATING
# -----------

# measuring the line fluxes & line centroids
for line in ['sii.a']:#list(all_lines.keys()):
    l = all_lines[line]
    zline = l*(1+z)
    wave_range = [zline-0.0014,zline+0.0014]
    
    print(f'For line {line}:')
    # emission = measure_emission_line(spec,wave_range)
    ew = measure_ew(spec,wave_range,cont=0.01)
    print()