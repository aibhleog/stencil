'''
Basic script for just inspecting the spectrum of a galaxy.  

The commented out code below puls one spaxel and allows me to looking around.
The code below that makes a 4-panel plot that pulls the spectra for 4 spaxels
and then plots the 2D image slice on the left, with the locatin of the 
chosen spaxels scatter plotted on top.
        
    

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
import json, sys


# specify which galaxy
# --------------------
target = 'SGAS1723'



# ---------------------- #
# reading in galaxy info #
# ---------------------- #

# reading in file that has all of the galaxy values
with open('plots-data/galaxies.txt') as f:
    data = f.read()

# reconstructing dictionary
galaxies = json.loads(data)
path = galaxies['path']

# double-checking names are right for dictionary
try: galaxy = galaxies[target]
except KeyError: 
    print(f'The available targets are: {list(galaxies.keys())[1:]}')
    sys.exit(0) # exiting script

# ---------------------- #
# ---------------------- #




# defining some values
# --------------------
name,filename = galaxy['name'],galaxy['filename']

scale,z = galaxy['scale'], galaxy['z']
x,y = 16,33#galaxy['x,y']

# continuum-subtracted cube
data, header = fits.getdata(path+f'{name}/{filename}', header=True) 
error = fits.getdata(path+f'{name}/{filename}',ext=2) 




# setting wavelenth array & making spec
# -------------------------------------
wave = np.arange(header['CRVAL3'], 
                 header['CRVAL3']+(header['CDELT3']*len(data)), 
                 header['CDELT3'])



# pulling spectrum
# doing it this way to circumvent the Big Endian pandas error
dat = [float(f) for f in data[:,int(y),int(x)].copy()]
err = [float(f) for f in error[:,int(y),int(x)].copy()]

spec = pd.DataFrame({'wave':wave,'flux':dat,'ferr':err})
spec = convert_MJy_cgs(spec.copy())


# # window range
# window = [(wavemin*u.um,(spec.wave.mean()-0.0007)*u.um),
#           ((spec.wave.mean()+0.0007)*u.um,wavemax*u.um)]

# exclude = []
# if len(line['exclude']) > 0:
#     for win_range in line['exclude']:
#         win_range = [i*u.um for i in win_range]
#         exclude.append(win_range)
# else: exclude = None

# spec = fitting_continuum_1D(spec.copy(),window=window,exclude=exclude)


# plotting
# --------

# smoothing spectrum
smoothed = sm.smooth(spec.flux,window_len=5)


# plt.figure(figsize=(7,5))

# plt.step(spec.wave,spec.flux/scale,where='mid')
# plt.step(spec.wave,smoothed/scale,where='mid',color='k')
# # plt.step(spec.wave,spec.cont_sub/scale,where='mid',color='k')

# # plt.plot(spec.wave,spec.cont/scale)
# plt.plot(spec.wave,np.zeros(len(spec)),color='k')

# for l in [.654981,.65628,.658523,.6717,.6731,.726276,.90686,.95311]:
#     plt.axvline(l*(1+z),color='k',ls=':')

# # plt.axvspan(spec.wave.values[3],spec.wave.values[-3],alpha=0.2)
# # plt.axvspan(1.1635,1.1647,alpha=0.4)

# plt.xlim(4.7,5.02)
# plt.ylim(-0.05,0.1)


# plt.tight_layout()
# plt.show()
# plt.close('all')


# sys.exit(0)


# EXPANDING TO LOOK AT A NUMBER OF SPECTRA
# ----------------------------------------

count = 0

plt.figure(figsize=(12,8))
gs0 = gridspec.GridSpec(1,2,width_ratios=[1,2])


gsL = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=gs0[0],
								height_ratios=[1,3,1],hspace=0)

gsR = gridspec.GridSpecFromSubplotSpec(4,1,subplot_spec=gs0[1],
								height_ratios=[1,1,1,1],hspace=0)



# plotting an image of the galaxy, centered around Halpha
image = data[galaxy['slice']].copy()
image[image == 0] = np.nan # to remove non-IFU areas

ax_image = plt.subplot(gsL[1])

ax_image.imshow(image,origin='lower',clim=(-0.01,0.035))
ax_image.axis('off')


cmap = plt.get_cmap('viridis')
colors = [cmap(j) for j in np.linspace(0,0.8,4)]

if target == 'SPT0418':
    pixels = [[14,31],[6,14],[28,12],[36,28]]
elif target == 'SGAS1723':
    pixels = [[20,44],[22,37],[30,25],[35,18]]
elif target == 'SPT2147':
    pixels = [[14,25],[18,26],[29,10],[39,29]]
    


for x,y in pixels:
    
    # adding pixel point to image plot
    ax_image.scatter(x,y,marker='o',s=60,color=colors[count],edgecolor='k')
    
    
    # spectra
    ax = plt.subplot(gsR[count])

    # pulling spectrum
    # doing it this way to circumvent the Big Endian pandas error
    dat = [float(f) for f in data[:,int(y),int(x)].copy()]
    err = [float(f) for f in error[:,int(y),int(x)].copy()]

    spec = pd.DataFrame({'wave':wave,'flux':dat,'ferr':err})
    spec = convert_MJy_cgs(spec.copy())

    
    # smoothing spectrum
    smoothed = sm.smooth(spec.flux,window_len=5)
    
    
    # plotting
    # --------
    txt = ax.text(0.977,0.83,f"spaxel ({x},{y})",transform=ax.transAxes,fontsize=15,ha='right',color='k')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground=colors[count])])
    
    
    ax.step(spec.wave,spec.flux/scale,where='mid',color=colors[count])
    ax.step(spec.wave,smoothed/scale,where='mid',color='k',lw=2)
    
    
    if target == 'SGAS1723':
        for l in [.4863,.4959,.5007]: #,.5876,.654981,.65628,.658523]:
            ax.axvline(l*(1+z),color='k',ls=':')
        
        ax.set_xlim(1.1,1.2)
        # ax.set_xlim(1.1,1.2)
        # ax.set_xlim(1.45,1.6)
        ax.set_ylim(-0.2,2.3)
    
    elif target == 'SPT0418':
        for l in [.654981,.65628,.658523,.6717,.6731,.90686,.95311]:
            ax.axvline(l*(1+z),color='k',ls=':')
        
        ax.set_xlim(3.3,3.6)
        ax.set_ylim(-0.02,0.24)
        
    elif target == 'SPT2147':
        for l in [.654981,.65628,.658523,.6717,.6731,.90686,.95311]:
            ax.axvline(l*(1+z),color='k',ls=':')
        
        ax.set_xlim(3,3.3)
        ax.set_ylim(-0.02,0.36)
        
        
        
    if count != 3:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('observed wavelength [microns]')
        
    if count == 2:
        ax.set_ylabel('\t\t\tflux [10$^{%s}$ erg/s/cm$^2$/$\AA$]' 
                      %round(np.log10(scale),2), labelpad=5)
    
    count += 1
    
    
    
plt.tight_layout()
plt.savefig(f'plots-data/{name}-various1D.pdf')
plt.show()
plt.close('all')

