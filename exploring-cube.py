'''
This is code I'm using to explore IFU cubes and gain experience with the relevant coding software I need to use for these data.

Current goal:   combine the data in multiple slices to properly encompass
                the emission line(s).  Also continuum subtract.

--------------------------------------------------
Theoretical & observational values for [OIII] and [SIII] line ratios:
based upon papers I've googled (albeit quickly), I've found

>> [OIII] line flux ratios with a median of 2.98
    - https://inspirehep.net/files/fc28972f2f825524151452902a991af8
    - https://academic.oup.com/mnras/article/312/4/813/1017382
    - https://arxiv.org/abs/astro-ph/0610848
    
>> [SIII] line flux ratio, theoretical is 2.5
    - https://arxiv.org/pdf/1910.13594.pdf
---------------------------------------------

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
# data, header = fits.getdata(path+'L3/Level3_g140h-f100lp_s3d.fits', header=True) # S1723
data, header = fits.getdata(path+'SGAS1723/Level3_g140h-f100lp_s3d_BW.fits', header=True) # S1723
z = 1.3293 # from Rigby+2021
lines = {'name':'OIII','waves':[4959,5007],
         'indexes':[790,837], 
         'clims':(-0.1,0.2),'ratio':2.98, 'dratio':[0.5,62]}

# # SPT2147
# # -------
# name = 'SPT2147'
# data, header = fits.getdata(path+'jw01355-o019_t005_nirspec_g395m-f290lp_s3d.fits', header=True) #SPT2147
# z = 3.76 # from Gururajan+2022
# lines = {'name':'SIII','waves':[9069,9531],
#          'indexes':[810,932],
#          'clims':(-2e-2,3e-2),'ratio':2.5,'dratio':[0.25,32]}



# setting wavelenth array & slicing
wave = np.arange(header['CRVAL3'], 
                 header['CRVAL3']+(header['CDELT3']*len(data)), 
                 header['CDELT3'])

slice_1 = data[lines['indexes'][0]]
slice_2 = data[lines['indexes'][1]]




# -------- #
# PLOTTING #
# -------- #


# plotting the two slices together
# --------------------------------

plt.figure(figsize=(8,6))

plt.imshow(slice_1,clim=lines['clims'],origin='lower',cmap='Blues')
plt.imshow(slice_2,clim=lines['clims'],origin='lower',cmap='Reds',alpha=0.65)

plt.text(0.035,0.11,f'[{lines["name"]}] $\lambda${lines["waves"][0]}',
         transform=plt.gca().transAxes,fontsize=14,color='blue')
plt.text(0.035,0.047,f'[{lines["name"]}] $\lambda${lines["waves"][1]}',
         transform=plt.gca().transAxes,fontsize=14,color='red')

plt.text(0.975,0.92,f'{name}',transform=plt.gca().transAxes,ha='right',fontsize=20)


plt.gca().set_yticklabels([])
plt.gca().set_xticklabels([])

plt.tight_layout()
plt.savefig(f'plots-data/{name}-{lines["name"]}-overlaid.pdf')
plt.show()
plt.close('all')


    
    
# looking at flux ratio of lines in 2D
# ------------------------------------

plt.figure(figsize=(8,6))

s1 = slice_1.copy()
s1[s1==0] = np.nan # to make the code stop yelling when dividing

s2 = slice_2.copy()
s2[s2==0] = np.nan # to make the code stop yelling when dividing


# I just wanted to remove the values below 0
r = s2/s1
r[r<0] = np.nan


# highlighting a specific value
dmax = 10
dmin = np.nanmin(r)
cen = lines['ratio']
x = cen/(dmax-dmin)

tmap = make_colormap_range(cen,lines['dratio'][0],dmin,
                           dmax,len(r),span=lines['dratio'][1])
norm = FixPointNormalize(fixme=cen,fixhere=x,vmin=dmin,vmax=dmax)

# plotting it
co = plt.imshow(r,origin='lower',clim=(0,10),cmap=tmap)#, norm=norm)
cb = plt.colorbar(co, ax=plt.gca())


# defining a white background for the text
white_background = dict(facecolor='white', alpha=0.65, edgecolor='none')

plt.text(0.97,0.88,
        f'Ratio of [{lines["name"]}]:\n{lines["waves"][1]}/{lines["waves"][0]}  ',
         transform=plt.gca().transAxes,fontsize=16,ha='right',
         bbox=white_background)

plt.text(0.035,0.04,f'median value from lit: {lines["ratio"]}',
         transform=plt.gca().transAxes,fontsize=14,color='#FF4D27')

plt.text(0.97,0.84,'(not continuum subtracted)',
         transform=plt.gca().transAxes,fontsize=7,ha='right',
        bbox=white_background)


plt.gca().set_yticklabels([])
plt.gca().set_xticklabels([])

plt.tight_layout()
plt.savefig(f'plots-data/{name}-{lines["name"]}-ratio.pdf')
plt.show()
plt.close('all')