'''
This is going to be a very basic script (hopefully) where I fit 
a polynomial curve to the arc in S1723 so that I can get some 
pixel coordinates to plot.

'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'


import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from highlight_colorbar import * # written by TAH


# setting path
path = '/Users/tahutch1/data/raw/jwst/ers/templates/reduced/'


# ----------------- #
# defining galaxies #
# ----------------- #
# note about the lines dictionary:
#    name = name of the emission line doublet
#    waves = the peak wavelengths of the lines
#    clims = the color upper & lower limits for plt.imshow
#    ratio = the observational/theoretical flux ratio 
#    dratio = 1) the uncertainty/range you want, 2) helps with colorbar


# SDSS1723
# --------
name = 'S1723'
data, header = fits.getdata(path+'SGAS1723/Level3_g140h-f100lp_s3d_BW.fits', header=True) # S1723
z = 1.3293 # from Rigby+2021
lines = {'name':'OIII','waves':[4959,5007],
         'indexes':[790,837],
         'clims':(-0.1,0.2),'ratio':2.98, 'dratio':[0.5,62]}



# -------- #
# PLOTTING #
# -------- #

slice_1 = data[lines['indexes'][0]]
slice_2 = data[lines['indexes'][1]]


# plotting the two slices together
# --------------------------------

plt.figure()

plt.imshow(slice_1,clim=lines['clims'],origin='lower',cmap='Greys')
plt.imshow(slice_2,clim=lines['clims'],origin='lower',cmap='Greys',alpha=0.65)

plt.text(0.97,0.905,f'{name}',ha='right',transform=plt.gca().transAxes,fontsize=20)


# fitting the shape of the arc by hand
def polyfit(x,a,b,c):
    return a*x**2 + b*x + c

x = np.arange(17,37)
# these are the values for a 2nd order polynominal
x0 = [0.06,-4.7,109]
fit = polyfit(x,*x0)

plt.plot(x,fit,color='g',lw=4,label='by-hand approx.')

# making y values integers
y = [round(i) for i in fit]

# saving fit
np.savetxt(f'plots-data/{name}-polyfit-arc.txt',list(zip(x,y)),delimiter=',')


plt.legend(loc=3)
plt.gca().set_yticklabels([])
plt.gca().set_xticklabels([])

plt.tight_layout()
plt.savefig(f'plots-data/{name}-polyfit-arc.pdf')
plt.show()
plt.close('all')