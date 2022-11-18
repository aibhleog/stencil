'''
This is code I'm using to explore 1D spectroscopy part of the IFU cubes and gain experience with the relevant coding software I need to use for these data.

Current goal:   plot a spectrum of S1723 from ONE of the spaxels


NOTES:  The x1d.fits files are set up like recarrays -- also I didn't end up 
        using them beccause they confused me...
        
'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'


import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
error = fits.getdata(path+'SGAS1723/Level3_g140h-f100lp_s3d_BW.fits',ext=2) # S1723
z = 1.3293 # from Rigby+2021
lines = {'name':'OIII','waves':[4959,5007],
         'indexes':[790,837],
         'clims':(-0.1,0.2),'ratio':2.98, 'dratio':[0.5,62]}



# setting wavelenth array & slicing
wave = np.arange(header['CRVAL3'], 
                 header['CRVAL3']+(header['CDELT3']*len(data)), 
                 header['CDELT3'])

slice_1 = data[lines['indexes'][0]]
slice_2 = data[lines['indexes'][1]]


# reading in polyfit to the arc
arc = np.loadtxt(f'plots-data/{name}-polyfit-arc.txt',delimiter=',')



# -------- #
# PLOTTING #
# -------- #

cmap = plt.get_cmap('viridis_r')
colors = [cmap(j) for j in np.linspace(0,1,len(arc))]


# plotting both the path of the spaxels
# and the series of spectra
# -------------------------

f = plt.figure(figsize=(13,4.5))
gs0 = gridspec.GridSpec(1,2,width_ratios=[0.45,1])#,wspace=0.1) 

ax1 = plt.subplot(gs0[0])
ax2 = plt.subplot(gs0[1])


# ax1 subplot

ax1.imshow(slice_1,clim=lines['clims'],origin='lower',cmap='Greys')
ax1.imshow(slice_2,clim=lines['clims'],origin='lower',cmap='Greys',alpha=0.65)

i = 0
for x,y in arc:
    ax1.scatter(x,y, marker='s',edgecolor='k',color=colors[i])
    i += 1


ax1.text(0.03,0.05,f'{name}',transform=ax1.transAxes,fontsize=20)    
    
ax1.set_yticklabels([])
ax1.set_xticklabels([])

    
    
# ax2 subplot

i = 0
for x,y in arc:
    spec = data[:,int(y),int(x)].copy()
    ax2.step(wave,spec,where='mid',color=colors[i],lw=2,alpha=0.7)
    i += 1
    
    
# marking out lines
ax2.text(0.68,0.605,'[OIII]',transform=ax2.transAxes,fontsize=20)
ax2.text(0.17,0.4,r'H$\beta$',transform=ax2.transAxes,fontsize=20)
    
ax2.plot([1.156,1.1595],[2.05,3.34],ls=':',color='k',lw=2) # OIIIa
ax2.plot([1.165,1.1608],[2.6,3.35],ls=':',color='k',lw=2) # OIIIb
ax2.plot([1.1335,1.1356],[1.2,2.05],ls=':',color='k',lw=2) # hbeta
    
# ax2.axvline(wave[lines['indexes'][1]])
    
ax2.set_xlabel('wavelength [microns]')
ax2.set_ylim(-0.5,6.2)
ax2.set_xlim(1.129,1.172)
ax2.set_yticklabels([])



plt.tight_layout()
# plt.savefig(f'plots-data/{name}-polyfit-2D-1D.pdf')
plt.show()
plt.close('all')



# saving a few lines to compare with Rigby+2021
print('Saving the following spaxels:')
tosave = [3,15,18] # random
for s in tosave:
    x,y = arc[s]
    spec = data[:,int(y),int(x)].copy()
    err = error[:,int(y),int(x)].copy()
    np.savetxt(f'plots-data/{name}-spaxel-{int(x)}-{int(y)}-1D.txt',
               list(zip(wave,spec,err)),delimiter='\t')
    print(int(x),int(y))



# plotting the path of spaxels
# ----------------------------

# plt.figure()

# plt.imshow(slice_1,clim=lines['clims'],origin='lower',cmap='Greys')
# plt.imshow(slice_2,clim=lines['clims'],origin='lower',cmap='Greys',alpha=0.65)

# i = 0
# for x,y in arc:
#     plt.scatter(x,y, marker='s',edgecolor='k',color=colors[i])
#     i += 1

    
# plt.text(0.03,0.05,f'{name}',transform=plt.gca().transAxes,fontsize=20)    

    
# plt.tight_layout()
# plt.savefig(f'plots-data/{name}-polyfit-overlaid.pdf')
# plt.show()
# plt.close('all')



# plotting a series of spectra
# ----------------------------

# plt.figure(figsize=(10,4.5))

# i = 0
# for x,y in arc:
#     spec = data[:,int(y),int(x)].copy()
#     plt.step(wave,spec,where='mid',color=colors[i],lw=2,alpha=0.7)
#     i += 1
    
    
# # marking out lines
# plt.text(0.58,0.6,'[OIII]',transform=plt.gca().transAxes,fontsize=20)
# plt.text(0.17,0.4,r'H$\beta$',transform=plt.gca().transAxes,fontsize=20)
    
# plt.plot([1.156,1.1595],[2.05,3.34],ls=':',color='k',lw=2) # OIIIa
# plt.plot([1.165,1.1608],[2.6,3.35],ls=':',color='k',lw=2) # OIIIb
# plt.plot([1.1335,1.1356],[1.2,2.05],ls=':',color='k',lw=2) # hbeta
    
    
# plt.xlabel('wavelength [microns]')
# plt.ylim(-0.5,6.2)
# plt.xlim(1.125,1.182)

# plt.tight_layout()
# plt.show()
# plt.close('all')