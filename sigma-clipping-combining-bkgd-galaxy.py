'''

Part 3 in sigma clipping routine
---------------------------------

This script is modest & straightforward.  The two separate clipped pieces
(the "bkgd"/not-galaxy cube and the layer-clipped galaxy cube) are pieced
together to make the final layered sigma clipped cube to be used for science.


'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'

import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fitting_ifu_spectra import * # written by TAH


# specify which galaxy
# --------------------
target = 'SGAS1723'

saveit = True # True or False


# returns dictionary of info for chosen galaxy
# also path to reduced FITS cubes
galaxy, path, grating = get_galaxy_info(target,grat='g395h')


# since updated pmap:
pmap_scale = 1e2 # was 1e4 for pmap1027 with old pipeline version


# defining some values
# --------------------
name,filename = galaxy['name'],galaxy['grating'][grating]['filename']
scale,z = galaxy['grating'][grating]['scale']/pmap_scale, galaxy['z']

# sli = 674 # SGAS1723, not an emission line
# sli = 300 # SPT2147, not an emission line
sli = galaxy['grating'][grating]['slice'] 


# getting mask
mask = get_mask(name,array_2d=True)



# reading in both bkgd clipped & galaxy clipped
# --(yes this seems a tedious way to read them in, but I'm lazy and this was faster)
pieces_path = 'plots-data/data-reduction/sigma-clipping-pieces/'

bkgd = fits.getdata(f'{pieces_path}/{name}-sigmaclipping-bkgd-{grating}-s3d.fits')
bkgd_err = fits.getdata(f'{pieces_path}/{name}-sigmaclipping-bkgd-{grating}-s3d.fits',ext=2)
bkgd_clipped = fits.getdata(f'{pieces_path}/{name}-sigmaclipping-bkgd-{grating}-s3d.fits',ext=3)
gal = fits.getdata(f'{pieces_path}/{name}-sigmaclipping-galaxy-{grating}-s3d.fits')
gal_err = fits.getdata(f'{pieces_path}/{name}-sigmaclipping-galaxy-{grating}-s3d.fits',ext=2)
gal_clipped = fits.getdata(f'{pieces_path}/{name}-sigmaclipping-galaxy-{grating}-s3d.fits',ext=3)


# setting up final cubes
final_clipped = np.zeros(bkgd.shape)
final_clipped[:] = np.nan

final_clipped_error = np.zeros(bkgd.shape)
final_clipped_error[:] = np.nan

# making cube of zeros to log when a pixel has been clipped
clipped_pixels = np.zeros(bkgd.shape) # will be 1 in slice if pixel is clipped


# combining data slice by slice
for i in range(len(bkgd)):
    # data
    bkgd_slice = bkgd[i].copy()
    gal_slice = gal[i].copy()
    
    # error
    bkgd_slice_error = bkgd_err[i].copy()
    gal_slice_error = gal_err[i].copy()
    
    # just to be safe, masking out bkgd, galaxy
    bkgd_slice[mask>0] = np.nan
    gal_slice[mask<1] = np.nan
    
    bkgd_slice_error[mask>0] = np.nan
    gal_slice_error[mask<1] = np.nan
    
    # pieceing together
    filler_slice = bkgd_slice.copy()
    filler_slice_error = bkgd_slice_error.copy()
    
    filler_slice[mask>0] = gal_slice[mask>0].copy()
    filler_slice_error[mask>0] = gal_slice_error[mask>0].copy()
    
    # adding to new cube
    final_clipped[i] = filler_slice.copy()
    final_clipped_error[i] = filler_slice_error.copy()

    # adding together the clipped pixel tracker cubes
    clipped_pixels[i][mask<1] = bkgd_clipped[i][mask<1].copy()
    clipped_pixels[i][mask>0] = gal_clipped[i][mask>0].copy()
    

    
    
# plotting pieced together clipped slices
# -----------------------------------------
plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(1,3,width_ratios=[1,1,1],wspace=0)


# original data cube straight from the pipeline
data,header = fits.getdata(path+f'{name}/{filename}',header=True)

ax = plt.subplot(gs[0])
ax.set_title('original slice')
ax.imshow(data[sli],clim=(-5e-3*pmap_scale,5e-2*pmap_scale),origin='lower',
           cmap='viridis')

ax.set_yticklabels([])
ax.set_xticklabels([])


# slice from final sigma clipped cube
ax = plt.subplot(gs[1])
ax.set_title('layered sigma clipped slice')
ax.imshow(final_clipped[sli],clim=(-5e-3*pmap_scale,5e-2*pmap_scale),origin='lower',
           cmap='viridis')

ax.set_yticklabels([])
ax.set_xticklabels([])


# slice from clipping pixel tracker
ax = plt.subplot(gs[2])
ax.set_title('pixels clipped in slice')
ax.imshow(mask,origin='lower',cmap='Greys',zorder=0,alpha=0.3)
ax.imshow(clipped_pixels[sli],origin='lower',cmap='Blues',alpha=0.5)
ax.text(0.047,0.927,'galaxy mask',color='grey',transform=ax.transAxes,fontsize=13)
ax.text(0.047,0.87,'clipped pixel',color='C0',transform=ax.transAxes,fontsize=13,alpha=0.8)

ax.set_yticklabels([])
ax.set_xticklabels([])


plt.tight_layout()
plt.show()
plt.close('all')
print(end='\n\n')

    
    
# BEFORE & AFTER STATS
print('Stats for the galaxy spaxels, before & after clipping:',end='\n\n')
for d in [data[sli].copy(),final_clipped[sli].copy()]:
    d[mask<1] = np.nan
    print('Median:',np.nanmedian(d))
    print('Mean:',np.nanmean(d))
    print('Standard Deviation:',np.nanstd(d))
    print('Max,Min:',np.nanmax(d),np.nanmin(d),end='\n\n')

    
print('Stats for the off-galaxy spaxels, before & after clipping:',end='\n\n')
for d in [data[sli].copy(),final_clipped[sli].copy()]:
    d[mask>0] = np.nan
    print('Median:',np.nanmedian(d))
    print('Mean:',np.nanmean(d))
    print('Standard Deviation:',np.nanstd(d))
    print('Max,Min:',np.nanmax(d),np.nanmin(d),end='\n\n')

print()
    
    
    
    
# sys.exit(0)


# writing out sigma clipped line flux FITS cube to file
# using the header from the reduced s3d cube to preserve WCS & wavelength
# -----------------------------
if saveit == True:
    hdu = fits.PrimaryHDU(header=header)
    hdu1 = fits.ImageHDU(final_clipped,header=header) # the data cube
    hdu2 = fits.ImageHDU(final_clipped_error,header=header) # the error cube
    hdu3 = fits.ImageHDU(clipped_pixels,header=header) # the clipped pixel tracker
    hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3])
    hdul.writeto(f'plots-data/{name}-sigmaclipped-{grating}-s3d.fits',overwrite=True)
    print('\nsigma clipped FITS cube saved.  Exiting script...',end='\n\n')
    
    
    

    