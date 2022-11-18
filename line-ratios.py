'''
Zooming in around the line to fit continuum & subtract, then picking out 
just the emission line to measure the lineflux (no fit).

NOTES:  

    I proved to myself that measuring the lineflux of the emission line vs 
    the fit doesn't seem to matter.  So because of this, I'm just measuring
    the emission line fluxes, not fitting them.


TODO:
    
    Get code working for SPT sources.  But also need to decide what to do
    with upper limits?  Cause it looks like neither see [SII]...
    
    Would love to be able to -- 1) plot a 3" line on the 2D image
                                2) maybe rotate the cube?
                                3) plot wcs grid but no axis?        
    

'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'

import time
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import pandas as pd
from fitting_ifu_spectra import *
import json, sys
# import warnings
# warnings.filterwarnings("error")
# warnings.filterwarnings("ignore")


# specify which galaxy
# --------------------
target = 'SPT2147'

saveit = True # saving FITS image of line fluxes


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
x,y = galaxy['x,y']

# continuum-subtracted cube
data, header = fits.getdata(path+f'{name}/{filename}', header=True) 
error = fits.getdata(path+f'{name}/{filename}',ext=2) 


# emission lines
# --------------
line1 = galaxy['lines']['ha']
line2 = galaxy['lines']['siib']

line = line1


# setting wavelenth array & making spec
# -------------------------------------
bigwave = np.arange(header['CRVAL3'], 
                 header['CRVAL3']+(header['CDELT3']*len(data)), 
                 header['CDELT3'])


# slicing out line cube
wavemin,wavemax = line['wave_range'][0]-0.0055, line['wave_range'][1]+0.0055
mask = np.where((bigwave<wavemax)&(bigwave>wavemin),bigwave,-1)
mask = np.arange(len(mask))[mask > 0]

wave = bigwave[mask]
ldata = data[mask].copy()
lerror = error[mask].copy()


# pulling spectrum
# doing it this way to circumvent the Big Endian pandas error
dat = [float(f) for f in ldata[:,int(y),int(x)].copy()]
err = [float(f) for f in lerror[:,int(y),int(x)].copy()]

spec = pd.DataFrame({'wave':wave,'flux':dat,'ferr':err})
spec = spec_wave_range(spec,[wavemin,wavemax])
spec = convert_MJy_cgs(spec.copy())


# window range
window = [(wavemin*u.um,(spec.wave.mean()-0.0007)*u.um),
          ((spec.wave.mean()+0.0007)*u.um,wavemax*u.um)]

exclude = []
if len(line['exclude']) > 0:
    for win_range in line['exclude']:
        win_range = [i*u.um for i in win_range]
        exclude.append(win_range)
else: exclude = None

spec = fitting_continuum_1D(spec.copy(),window=window,exclude=exclude)


# fit = fit_emission_line(spec,line['wave_range'],line['x0'],contsub=True)
# lspec = spec_wave_range(spec,line['wave_range'])
# lspec['fit'] = fit['fit'].value


# lineflux = measure_emission_line(lspec,line['wave_range'],contsub=True)['lineflux']#[0].value




plt.figure(figsize=(5,5))

plt.step(spec.wave,spec.flux/scale,where='mid')
plt.step(spec.wave,spec.cont_sub/scale,where='mid',color='k')
# plt.step(spec.wave,spec.fit/scale,where='mid',color='k',label='fit')

# plt.plot(lspec.wave,lspec.fit/scale,color='magenta')

plt.plot(spec.wave,spec.cont/scale)
plt.plot(spec.wave,np.zeros(len(spec)),color='k')

for l in [.6717,.6731]:
    plt.axvline(l*(1+z),color='k',ls=':')

plt.axvspan(spec.wave.values[3],spec.wave.values[-3],alpha=0.2)
# plt.axvspan(1.1635,1.1647,alpha=0.4)

plt.tight_layout()
plt.show()
plt.close('all')


sys.exit(0)



# making list of spaxel coordinates based on data shape
x0,y0 = np.arange(0,data.shape[2]),np.arange(0,data.shape[1])
g = np.meshgrid(x0,y0)
coords = list(zip(*(c.flat for c in g))) # coords we'll walk through
# ------------------




# --------------- #
# RUNNING FITTING #
# --------------- #
# making a linemap FITS image
# each 2 layers is a line's flux and uncertainty
linemap = np.zeros((4,data.shape[1],data.shape[2]))
linemap[:] = np.nan

count = 0 # counts the coordinate we're on
n = 0 # counts the line we're on
error_count = [0,0] # keeping track of the except's thrown


# timing how long it takes
t0 = time.time()

# for each line
for line in [line1,line2]:
    
    # slicing out line cube
    wavemin,wavemax = line['wave_range'][0]-0.0055, line['wave_range'][1]+0.0055
    mask = np.where((bigwave<wavemax)&(bigwave>wavemin),bigwave,-1)
    mask = np.arange(len(mask))[mask > 0]

    wave = bigwave[mask].copy()
    ldata = data[mask].copy()
    lerror = error[mask].copy()
    
    
    # running through coordinates
    for x,y in coords:

        if count % 100 == 0: print(f'At spaxel {count}/{len(coords)}')

        # pulling spectrum
        # doing it this way to circumvent the Big Endian pandas error
        dat = [float(f) for f in ldata[:,int(y),int(x)].copy()]
        err = [float(f) for f in lerror[:,int(y),int(x)].copy()]

        spec = pd.DataFrame({'wave':wave,'flux':dat,'ferr':err})
        spec = spec_wave_range(spec.copy(),[wavemin,wavemax])

        # if actual spaxel:
        if set(spec.flux.values) != {0}:
            
            spec = convert_MJy_cgs(spec.copy())

            # window range
            window = [(wavemin*u.um,(spec.wave.mean()-0.0007)*u.um),
                      ((spec.wave.mean()+0.0007)*u.um,wavemax*u.um)]

            exclude = []
            if len(line['exclude']) > 0:
                for win_range in line['exclude']:
                    filler = [i*u.um for i in win_range]
                    exclude.append(filler)
            else: exclude = None


            try:
                spec = fitting_continuum_1D(spec.copy(),window=window,exclude=exclude)
                
                # measuring emission line flux
                lspec = spec_wave_range(spec.copy(),line['wave_range'])
                lineflux = measure_emission_line(lspec,line['wave_range'],contsub=True)

                # saving value to map
                linemap[0+n,int(y),int(x)] = lineflux['lineflux'][0].value
                linemap[1+n,int(y),int(x)] = lineflux['lineflux'][1].value

            except ValueError: # spaxel with NaNs
                error_count[0] += 1
                continue
                
            except TypeError: # spaxel with issues?
                error_count[1] += 1
                continue


        count += 1

    if n == 0: print('\nRepeating for second line',end='\n\n')
    count = 0
    n = 2



# marking end time
# ----------------
t1 = time.time()
total = t1-t0
print(f'\nThis took {total/60} minutes.',end='\n\n')


print(f"Number of ValueError's thrown: {error_count[0]}\n" +
      f"Number of TypeError's thrown: {error_count[1]}")


# sys.exit(0)



# looking at ratio
# ----------------

plt.figure(figsize=(8,6))
plt.axis('off')

rat = plt.imshow(linemap[0]/linemap[2],origin='lower',clim=(3,8),cmap='viridis')

cbar = plt.colorbar(rat,pad=0.03)
cbar.ax.set_yticks([3,4,5,6,7,8,])
cbar.ax.set_yticklabels(['<3','4','5','6','7','8+',])

ratio_name = r'[OIII] $\lambda$5007 / H$\beta$' # SGAS1723
ratio_name = r'[SIII] $\lambda$ / [SII] $\lambda$6717,6731' # SPT0418 & SPT2147
cbar.set_label(ratio_name, rotation=270, labelpad=15, fontsize=17)

plt.gca().set_yticklabels([])
plt.gca().set_xticklabels([])

plt.tight_layout()
plt.savefig(f'plots-data/{name}-linefluxes.pdf')
plt.show()
plt.close('all')




# writing out continuum subtracted line flux FITS cube to file
# using the header from the reduced s3d cube to preserve WCS & wavelength
# -----------------------------
if saveit == True:
    hdu = fits.PrimaryHDU(linemap, header=header)
    hdu.writeto(f'plots-data/{name}-linefluxes-s3d.fits',overwrite=True)
    print('Continuum-subtracted line flux FITS cube saved.  Exiting script...')





