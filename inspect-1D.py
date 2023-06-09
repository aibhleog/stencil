'''
Basic script for just inspecting the spectrum of a galaxy.  

The commented out code below puls one spaxel and allows me to looking around.

The code below that makes a 4-panel plot that pulls the spectra for 4 spaxels
and then plots the 2D image slice on the left, with the locatin of the 
chosen spaxels scatter plotted on top.
        
    
TODO:

    current playing with fitting the [SII] doublet in the SPT0418 data.
    trying to figure out how to make a double gaussian peak, this may help:
    -- https://specutils.readthedocs.io/en/stable/fitting.html#double-peak-fit-two-separate-peaks-with-two-windows
    maybe this too
    -- https://github.com/astropy/past-astropy-workshops/blob/master/aas_233_workshop/09b-Specutils/Specutils_analysis.ipynb


'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'

import time
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import pandas as pd
from fitting_ifu_spectra import * # written by TAH
import smoothing as sm # written by TAH
import add_lines # written by TAH
import json, sys


# specify which galaxy
# --------------------
target = 'SGAS1723'

saveit = True # save the figure?



# returns dictionary of info for chosen galaxy
# also path to reduced FITS cubes
galaxy, path, grating = get_galaxy_info(target)




# defining some values
# --------------------
name,filename = galaxy['name'],galaxy['grating'][grating]['filename']

scale,z = galaxy['grating'][grating]['scale'], galaxy['z']
x,y = galaxy['grating'][grating]['x,y']

# reduced cube
data, header = fits.getdata(path+f'{name}/{filename}', header=True) 
error = fits.getdata(path+f'{name}/{filename}',ext=2) 

# sigma clipped cube BY LAYERS
# data, header = fits.getdata(f'plots-data/test-{name}-sigmaclipped-s3d-{grating}.fits', header=True) 
# error = fits.getdata(f'plots-data/test-{name}-sigmaclipped-s3d-{grating}.fits',ext=2) 



# setting wavelenth array & making spec
# -------------------------------------
wave = np.arange(header['CRVAL3'], 
                 header['CRVAL3']+(header['CDELT3']*len(data)), 
                 header['CDELT3'])



# # pulling spectrum
# # doing it this way to circumvent the Big Endian pandas error
# dat = [float(f) for f in data[:,int(y),int(x)].copy()]
# err = [float(f) for f in error[:,int(y),int(x)].copy()]

# spec = pd.DataFrame({'wave':wave,'flux':dat,'ferr':err})
# spec = convert_MJy_cgs(spec.copy())



# # smoothing spectrum
# spec['smoothed'] = sm.smooth(spec.flam,window_len=5)


# # fitting line
# line = galaxy['grating'][grating]['lines']['sii']

# lspec = spec_wave_range(spec.copy(),line['wave_range'])
# lspec['flux'] = lspec.smoothed

# spectrum = make_spectrum(lspec,line['wave_range'])

# cgs = u.erg * u.cm**-2 * u.s**-1 * u.AA**-1
# x0_1, x0_2 = line['x0']

# # setting up a Gaussian model with my initial guesses
# g1_init = models.Gaussian1D(amplitude = x0_1[0] * cgs, 
#                            mean = x0_1[1] * u.um, 
#                            stddev = x0_1[2] * u.um)

# g2_init = models.Gaussian1D(amplitude = x0_2[0] * cgs, 
#                            mean = x0_2[1] * u.um, 
#                            stddev = x0_2[2] * u.um)

# # fitting the line
# g_fit = fit_lines(spectrum, g1_init)
# y_fit = g_fit(spectrum.spectral_axis)






# # plotting
# # --------
# onepix = header['CDELT3'] # the wavelength step of one pixel


# plt.figure(figsize=(7,5))

# plt.step(spec.wave,spec.flam/scale,where='mid')
# plt.step(spec.wave,spec.smoothed/scale,where='mid',color='k')

# for l in [.654981,.65628,.658523,.6717,.6731,.726276,.90686,.95311]:
#     plt.axvline(l*(1+z)+onepix,color='k',ls=':')
    
# plt.step(lspec.wave,lspec.flam/scale,where='mid')
# plt.step(lspec.wave,y_fit/scale,where='mid',color='g')


# plt.xlim(3.35,3.7)
# plt.ylim(0,0.6)
# # plt.ylim(-0.05,0.2)


# plt.tight_layout()
# plt.show()
# plt.close('all')


# sys.exit(0) # to stop here and not continue




# EXPANDING TO LOOK AT A NUMBER OF SPECTRA
# ----------------------------------------

count = 0

plt.figure(figsize=(15,8))
gs0 = gridspec.GridSpec(1,2,width_ratios=[1.95,2],wspace=0)


gsL = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=gs0[0],
								height_ratios=[1,8,1],hspace=0)

gsR = gridspec.GridSpecFromSubplotSpec(4,1,subplot_spec=gs0[1],
								height_ratios=[1,1,1,1],hspace=0)



# plotting an image of the galaxy, centered around Halpha
image = data[galaxy['grating'][grating]['slice']-1].copy()
image[image == 0] = np.nan # to remove non-IFU areas

ax_image = plt.subplot(gsL[1])

ax_image.imshow(image,origin='lower',clim=(-1,3.5),cmap='viridis')
ax_image.axis('off')


cmap = plt.get_cmap('viridis')
colors = [cmap(j) for j in np.linspace(0,0.8,4)]

if target == 'SPT0418':
    pixels = [[15,31],[6,13],[28,12],[36,28]]
elif target == 'SGAS1723':
    pixels = [[20,44],[22,36],[29,24],[35,17]]
elif target == 'SPT2147':
    pixels = [[14,25],[18,27],[29,10],[39,29]]
    

    
if target[:3] == 'SPT': scale /= 1e3 


for x,y in pixels:
    
    # adding pixel point to image plot
    ax_image.scatter(x,y,marker='o',s=200,color=colors[count],edgecolor='k',lw=3)
    
    
    # spectra
    ax = plt.subplot(gsR[count])

    # pulling spectrum
    # doing it this way to circumvent the Big Endian pandas error
    dat = [float(f) for f in data[:,int(y),int(x)].copy()]
    err = [float(f) for f in error[:,int(y),int(x)].copy()]

    spec = pd.DataFrame({'wave':wave,'flam':dat,'flamerr':err})
    spec = convert_MJy_cgs(spec.copy())

    
    # smoothing spectrum
    smoothed = sm.smooth(spec.flam,window_len=5)
    
    
    # plotting
    # --------
    # txt = ax.text(0.977,0.83,f"spaxel ({x},{y})",transform=ax.transAxes,fontsize=15,ha='right',color='k')
    # txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground=colors[count])])
    
    
    ax.step(spec.wave,spec.flam/scale,where='mid',color=colors[count],lw=4)
    ax.step(spec.wave,smoothed/scale,where='mid',color='k',lw=2.5)
    
    
    if target == 'SGAS1723':
        for l in [.4863,.4959,.5007]: #,.5876,.654981,.65628,.658523]:
            ax.axvline(l*(1+z),color='k',ls=':')
        
        # ax.set_xlim(1.1,1.2)
        # ax.set_xlim(1.1,1.2)
        ax.set_xlim(1.5,1.58)
        ax.set_ylim(-0.2,2)
        ax.set_yticklabels([])
    
    elif target == 'SPT0418':
        for l in [.654981,.65628,.658523,.6717,.6731,.90686,.95311]:
            ax.axvline(l*(1+z),color='k',ls=':')
        
        ax.set_xlim(3.36,3.55)
        ax.set_ylim(-0.2,2.4)
        ax.set_yticklabels([])
        
    elif target == 'SPT2147':
        for l in [.654981,.65628,.658523,.6717,.6731,.90686,.95311]:
            ax.axvline(l*(1+z),color='k',ls=':')
        
        ax.set_xlim(3.05,3.23)
        ax.set_ylim(-0.2,3.6)
        ax.set_yticklabels([])
        
        
    # adding lines  
    if count == 1:
        if target[:3] == 'SPT':
            add_lines.draw_lines(ax,2.5,z,xlim=['nii1','sii'],xo=12,
                             fontsize=13,alpha=0.8)
        
    if count != 3:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('observed wavelength [microns]')
        
    if count == 2:
        # ax.set_ylabel('\t\t\tflux density [10$^{%s}$ erg/s/cm$^2$/$\AA$]' 
                      # %round(np.log10(scale)), labelpad=5)
        ax.set_ylabel('\t\t\tflux density [erg/s/cm$^2$/$\AA$]', labelpad=5)
    
    count += 1
    
    
    
plt.tight_layout()
if saveit == True: plt.savefig(f'plots-data/{name}-various1D.pdf')
plt.show()
plt.close('all')

