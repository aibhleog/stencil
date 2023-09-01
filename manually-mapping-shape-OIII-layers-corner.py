'''
Taking the galaxy mask I made and then making layers, so I can sigma clip
JUST those layers and then piece it all together again.


ORDER OF OPERATIONS:
>> manually-mapping-shape-[line].py; measuring the flux centered around the brighest line
   (or blended lines) and saving output to a 2D fits file.
>> manually-masking-[galaxy].py; starts with making the S/N map of the galaxy and then
   applies a S/N cut.  From there, it gets a little more hands on with masking out spaxels
   that made it past the cut but are obviously not associated with the galaxy.
   Saves full galaxy mask (0=not galaxy; 1=galaxy) to fits file.
>> THIS SCRIPT; takes the full galaxy mask made in the previous
   step and creates contour layers based on S/N.  Saves total galaxy mask and then each
   layer mask to a single fits file.  This file will be used in the sigma-clipping stage.

'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'

import time
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import pandas as pd
from fitting_ifu_spectra import * # written by TAH
import json, sys

from matplotlib.patches import Ellipse,Circle,Wedge,Rectangle
from astropy.convolution import Gaussian2DKernel


# specify which galaxy
# --------------------
target = 'SGAS1723'

saveit = False
# True or False
minsnr = 3 # used to be 5


# returns dictionary of info for chosen galaxy
# also path to reduced FITS cubes
galaxy, path, grating = get_galaxy_info(target)


# reading in data
# ---------------
# defining some values
name,filename = galaxy['name'],galaxy['grating'][grating]['filename']

# fits image
# slices are flux, ferr
data,header = fits.getdata(f'plots-data/{name}-brightOIII-mapping-alt.fits',header=True)


# using mask to focus in on galaxy spaxels only
mask = get_mask(name,array_2d=True)
snr = data.copy()
snr[mask<1] = np.nan 




print("\nPicking out the layers.",end='\n\n')


# looking at S/N map
plt.figure(figsize=(14,6))
gs = gridspec.GridSpec(1,2,width_ratios=[1,1.25])

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])


# using contours to choose layers
# -------------------------------
# making list of spaxel coordinates based on data shape
x0,y0 = np.arange(0,snr.shape[1]),np.arange(0,snr.shape[0])
g = np.meshgrid(x0,y0)


if target == 'SGAS1723': 
    levels = np.array([25,70,180]) # S/N levels
    # levels = np.array([15,55,150]) # S/N levels
    # levels = np.array([30,110])
    maxsnr = 300

if target == 'SGAS1226': 
    levels = np.array([15,30,65]) # S/N levels
    maxsnr = 45

    
# making array colors
cmap = plt.get_cmap('Blues_r')
colors = [cmap(j) for j in np.linspace(0,0.7,len(levels))]

# plotting contours
contours = ax1.contour(g[0],g[1],snr, levels, origin='upper',alpha=0.6,linewidths=3,colors=colors)
plt.clabel(contours, inline=True, fontsize=14,colors='k')

ax1.text(0.05,0.07,'picking out layers using contours',
         transform=ax1.transAxes,fontsize=17)



# looking at data
im = ax2.imshow(snr,origin='lower',cmap='viridis',clim=(minsnr,maxsnr))
cbar = plt.colorbar(im)

ratio_name = r'S/N of [OIII]5007'
cbar.set_label(ratio_name, rotation=270, labelpad=25, fontsize=17)

ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticklabels([])


plt.tight_layout()
# plt.savefig(f'plots-data/data-reduction/layers-{name}.pdf')
# plt.show()
plt.close('all')


# sys.exit(0)



print("\nLet's look at the layers individually.",end='\n\n')




# looking at S/N map
plt.figure(figsize=(12,7.21))
gs0 = gridspec.GridSpec(1,2,width_ratios=[2,1])

ax = plt.subplot(gs0[1])

ax.axis('off')
ax.imshow(snr,origin='lower',cmap='viridis',clim=(minsnr,maxsnr))
cbar = plt.colorbar(im,ax=ax,fraction=0.07)

ratio_name = r'S/N of [OIII]5007'
cbar.set_label(ratio_name, rotation=270, labelpad=25, fontsize=17)

ax.text(0.5,1.1,name,transform=ax.transAxes,fontsize=17)

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_ylim(9,50)
ax.set_xlim(3,48)


# LOOKING AT LEVELS
gs = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs0[0],
                    width_ratios=[1,1],height_ratios=[1,1],
                    wspace=0,hspace=0)

levels = np.concatenate(([minsnr],levels,[round(np.nanmax(snr)+10)]))

# setting these if statements cause SGAS1226 fainter than SGAS1723
if target == 'SGAS1723':
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    axes = [ax1,ax2,ax3,ax4]
if target == 'SGAS1226':
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    axes = [ax1,ax2,ax3,ax4]


for count,ax in enumerate(axes):
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    condition = (snr>=levels[count]) & (snr<levels[count+1])
    snr_cut = np.where(condition,snr,-1)
    snr_cut[snr_cut < 0] = np.nan
    
    ax.imshow(snr_cut,origin='lower',cmap='viridis',clim=(minsnr,maxsnr))
    ax.text(0.05,0.07,f'{levels[count]} $\leq$ S/N < {levels[count+1]}',
         transform=ax.transAxes,fontsize=17)

    if count == 0:
        spaxels_sli1 = np.array([[15,45],[15,44],[14,44],[14,43],
                                 [15,43],[22,47],[22,46]])
        ax.scatter(spaxels_sli1[:,0],spaxels_sli1[:,1],color='r',s=10)
        
    elif count == 1:
        spaxels_sli0 = np.array([[28,39],[27,38],[28,38],
                                 [29,35],[30,35],[30,36],
                                 [36,30],[36,29],[35,29]])
        ax.scatter(spaxels_sli0[:,0],spaxels_sli0[:,1],color='r',s=10)
        
        spaxels_sli2 = np.array([[18,48],[17,48],[19,48],[16,47],
                                 [20,48],[20,47],[20,46],
                                 [16,45],[16,44],[18,49]])
        ax.scatter(spaxels_sli2[:,0],spaxels_sli2[:,1],color='r',s=10)
        
    elif count == 2:
        spaxels_sli23 = np.array([[17,47],[18,47],[17,46]])
        ax.scatter(spaxels_sli23[:,0],spaxels_sli23[:,1],color='r',s=10)
        
    elif count == 3:
        spaxels_star = np.array([[17,41],[17,40],[16,40],
                                 [18,41],[18,40],[16,41]])
        ax.scatter(spaxels_star[:,0],spaxels_star[:,1],color='r',s=10)
    # if count == 2:
    #     spaxels_sli3 = np.array([[16,46],[17,46],[17,47],[18,47],[19,47]])
    #     ax.scatter(spaxels_sli3[:,0],spaxels_sli3[:,1],color='r',s=10)


    

plt.tight_layout()
# plt.savefig(f'plots-data/data-reduction/layer-slices-{name}.pdf')
plt.show()
plt.close('all')



# sys.exit(0)


print("\nLet's look at the mask layers.",end='\n\n')



# sigma clipping slices
map_layers = []
map_layers_ranges = []

for i in range(len(levels)-1): # -1 cause of the upper limit
    condition = (snr>=levels[i]) & (snr<levels[i+1])
    snr_cut = np.where(condition,snr,-1)
    # snr_cut[snr_cut < 0] = np.nan
    
    new_map = np.zeros(snr_cut.shape)
    new_map[snr_cut>0] = 1
    
    # manually adding in the galaxy spaxels I had to add
    # in the other script
    if i == 0:
        new_map[34,47] = 0        
        new_map[spaxels_sli0[:,1],spaxels_sli0[:,0]] = 1
        new_map[spaxels_sli1[:,1],spaxels_sli1[:,0]] = 0
    elif i == 1:
        new_map[49,17] = 1 # adding these edge pixels in
        new_map[47,15] = 1 # adding these edge pixels in
        new_map[46,15] = 1 # adding these edge pixels in
        new_map[spaxels_sli0[:,1],spaxels_sli0[:,0]] = 0
        new_map[spaxels_sli2[:,1],spaxels_sli2[:,0]] = 0
        new_map[spaxels_star[:,1],spaxels_star[:,0]] = 0
        new_map[spaxels_sli1[:,1],spaxels_sli1[:,0]] = 1
    elif i == 2:
        new_map[spaxels_sli2[:,1],spaxels_sli2[:,0]] = 1
        new_map[spaxels_star[:,1],spaxels_star[:,0]] = 0
        new_map[spaxels_sli23[:,1],spaxels_sli23[:,0]] = 0
    elif i == 3:
        new_map[spaxels_star[:,1],spaxels_star[:,0]] = 0
        new_map[spaxels_sli23[:,1],spaxels_sli23[:,0]] = 1
        

    map_layers.append(new_map)
    map_layers_ranges.append([levels[i],levels[i+1]])
    

# adding star as final layer
new_map = np.zeros(snr_cut.shape)
new_map[spaxels_star[:,1],spaxels_star[:,0]] = 1

map_layers.append(new_map)
map_layers_ranges.append([-99,-99]) # because it's the star
    


# LOOKING AT FINAL MAP WITH LAYERS
# --------------------------------
plt.figure(figsize=(11,7.05))
gs0 = gridspec.GridSpec(1,2,width_ratios=[2,1])

ax = plt.subplot(gs0[1])

ax.axis('off')

full_mask = np.asarray(map_layers).sum(axis=0) # adding layers together
ax.imshow(full_mask,origin='lower',cmap='viridis')
ax.text(0.1,0.1,name,transform=ax.transAxes,fontsize=17,color='w')
ax.set_title('full galaxy mask',fontsize=16)

ax.set_yticklabels([])
ax.set_xticklabels([])
# ax.set_ylim(9,50)
# ax.set_xlim(7,45)


# LOOKING AT LEVELS
gs = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs0[0],
                    width_ratios=[1,1],height_ratios=[1,1],
                    wspace=0,hspace=0)

if target == 'SGAS1723':
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    axes = [ax1,ax2,ax3,ax4]
    

count = 0
for ax in axes:
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')

    layer = map_layers[count].copy()
    # layer[layer<1] = np.nan
    
    ax.imshow(layer,origin='lower',cmap='viridis')
    ax.text(0.05,0.07,f'{levels[count]} $\leq$ S/N < {levels[count+1]}',
         transform=ax.transAxes,fontsize=17,color='w')

    count += 1


plt.tight_layout()
# plt.savefig(f'plots-data/data-reduction/mask-layer-slices-{name}.pdf')
plt.show()
plt.close('all')


# sys.exit(0)

# WRITING OUT NEW MAP LAYERS
# to be used for all line ratio codes for this galaxy
layers_list = [[minsnr,map_layers_ranges[-1][-1]]]
t = [layers_list.append(m) for m in map_layers_ranges]

full = np.asarray(map_layers).sum(axis=0) # adding layers together


if saveit == True:
    # making FITS slices
    hdu_list = []
    hdu = fits.PrimaryHDU(full)
    
    for i in range(len(map_layers)):
        hdu_list.append(fits.ImageHDU(map_layers[i]))
    
    hdul = fits.HDUList([hdu,*hdu_list])

    # saving layers info
    np.savetxt(f'plots-data/{name}-mask-layers.txt',layers_list,
               delimiter='\t',fmt='%.0f')
    
    # saving FITS image
    print('\nWriting out mask.')
    hdul.writeto(f'plots-data/{name}-mask-layers.fits',overwrite=True)
    print(f'Mask layers for {name} saved.  Exiting script...',end='\n\n')