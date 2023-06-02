'''

Part 1 in sigma clipping routine
---------------------------------

Sigma clipping on the not-galaxy spaxels.  This script effectively runs like regular sigma clipping.
However, the galaxy spaxels are first masked out.


'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'

import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fitting_ifu_spectra import * # written by TAH


# specify which galaxy
# --------------------
target = 'SPT2147'

saveit = True # True or False
sigma = 5


# returns dictionary of info for chosen galaxy
# also path to reduced FITS cubes
galaxy, path, grating = get_galaxy_info(target)#,grat='g395h')


# since updated pmap:
pmap_scale = 1e2 # was 1e4 for pmap1027 with old pipeline version


# defining some values
# --------------------
name,filename = galaxy['name'],galaxy['grating'][grating]['filename']
scale,z = galaxy['grating'][grating]['scale']/pmap_scale, galaxy['z']

x,y = galaxy['grating'][grating]['x,y']



# before & after NSClean
# filename = 'testing-nsclean-not/pmap1084-88/Level3_SGAS1723_g140h-f100lp_s3d_nsclean.fits'
# filename = 'testing-nsclean-not/pmap1084-88/Level3_SGAS1723_g140h-f100lp_s3d_orig.fits'
# filename = 'testing-nsclean-not/pmap1084-88/Level3_SGAS1723_g395h-f290lp_s3d_nsclean.fits'
# filename = 'testing-nsclean-not/pmap1084-88/Level3_SGAS1723_g395h-f290lp_s3d_orig.fits'



# sli = 674 # for SGAS1723, not an emission line
# sli = 300 # for SPT2147, not an emission line
sli = galaxy['grating'][grating]['slice']


if name == 'SGAS1723' and grating == 'g140h':
    x_c,y_c = 7,30 # +bad spaxel
    x_n,y_n = 45,29 # bkgd spaxel
    x_b,y_b = 13,40 # -bad spaxel
elif name == 'SGAS1723' and grating == 'g395h':
    x_c,y_c = 7,30 # +bad spaxel
    x_n,y_n = 45,20 # bkgd spaxel
    x_b,y_b = 8,14 # -bad spaxel
elif name == 'SPT0418':
    x_c,y_c = 40,6 # +bad spaxel
    x_n,y_n = 43,29 # bkgd spaxel
    x_b,y_b = 13,40 # -bad spaxel
elif name == 'SPT2147':
    x_c,y_c = 28,46 # +bad spaxel
    x_n,y_n = 33,8 # bkgd spaxel
    x_b,y_b = 8,30 # -bad spaxel
elif name == 'SGAS1226':
    x_c,y_c = 26,36 # +bad spaxel
    x_n,y_n = 39,7 # bkgd spaxel
    x_b,y_b = 7,30 # -bad spaxel


# cube
data, header = fits.getdata(path+f'{name}/{filename}', header=True) 
error = fits.getdata(path+f'{name}/{filename}', ext=2) 



slice_1 = data[sli].copy()



# plotting slice
# ---------------

plt.figure(figsize=(8,6))

# plt.imshow(slice_1,clim=(1e-10*pmap_scale,5e-2*pmap_scale),origin='lower',
plt.imshow(slice_1,clim=(-5e-3*pmap_scale,5e-2*pmap_scale),origin='lower',
            cmap='viridis')#,norm=LogNorm())

plt.scatter(x,y,s=40,edgecolor='k',color='C0',lw=1.5) # good pixel
plt.scatter(x_c,y_c,s=40,edgecolor='k',color='g',lw=1.5) # +bad pixel
plt.scatter(x_n,y_n,s=40,edgecolor='k',color='#CB9E0E',lw=1.5) # bkgd pixel
plt.scatter(x_b,y_b,s=40,edgecolor='k',color='r',lw=1.5) # -bad pixell

plt.text(0.975,0.92,f'{name}',transform=plt.gca().transAxes,ha='right',fontsize=20,color='w')

plt.gca().set_yticklabels([])
plt.gca().set_xticklabels([])

plt.tight_layout()
plt.show()
plt.close('all')




# sys.exit(0)




# LOOKING AT 1D AROUND HOT PIXEL
# setting wavelenth array & making spec
# -------------------------------------
bigwave = np.arange(header['CRVAL3'], 
                 header['CRVAL3']+(header['CDELT3']*len(data)), 
                 header['CDELT3'])

sli_wave = bigwave[sli]

# slicing out line cube
if grating[-1] == 'h': w_win = 0.0085
elif grating[-1] == 'm': w_win = 0.05

wavemin,wavemax = sli_wave-w_win, sli_wave+w_win
wave_mask = np.where((bigwave<wavemax)&(bigwave>wavemin),bigwave,-1)
wave_mask = np.arange(len(wave_mask))[wave_mask > 0]

wave = bigwave[wave_mask]
ldata = data[wave_mask].copy()
lerror = error[wave_mask].copy()



def get_spec(x,y,d=ldata,e=lerror):
    # doing it this way to circumvent the Big Endian pandas error
    dat = [float(f) for f in d[:,int(y),int(x)].copy()]
    err = [float(f) for f in e[:,int(y),int(x)].copy()]

    spec = pd.DataFrame({'wave':wave,'flam':dat,'flamerr':err})
    spec = spec_wave_range(spec,[wavemin,wavemax])
    spec = convert_MJy_cgs(spec.copy())
    return spec.copy()





# pulling spectrum
spec = get_spec(x,y)
spec_good = get_spec(x_c,y_c)
spec_bkgd = get_spec(x_n,y_n)
spec_nbad = get_spec(x_b,y_b)


plt.figure(figsize=(8,5))

plt.step(spec.wave,spec.flam/scale,where='mid')
plt.axvline(sli_wave,color='k')

plt.step(spec_good.wave,spec_good.flam/scale,where='mid',color='g')
plt.step(spec_bkgd.wave,spec_bkgd.flam/scale,where='mid',color='#CB9E0E')
plt.step(spec_nbad.wave,spec_nbad.flam/scale,where='mid',color='r')

y_scale = np.nanmean([spec_good.median().flam,spec_bkgd.median().flam,
                        spec_nbad.median().flam]) / scale
if target == 'SGAS1723' and grating == 'g140h': y_scale *= 1e1
plt.ylim(-50*y_scale,80*y_scale)

plt.tight_layout()
plt.show()
plt.close('all')


# sys.exit(0)


# sigma clipping NOT the galaxy

# mask = get_mask(name,array_2d=True) # want the 2D array, not coord list
mask, mask_info = get_mask(name,array_2d=True,layers=True)#,grating='g395h')
full_mask = mask[0].copy()

masked_slice = slice_1.copy()
masked_slice[full_mask>0] = np.nan



# sigma clipping masked slice
with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    masked_slice_clipped = sigma_clip(masked_slice, sigma=sigma, maxiters=5)




# plotting slice with mask
# ------------------------
plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(1,2,width_ratios=[1,1],wspace=0)


ax = plt.subplot(gs[0])

ax.imshow(masked_slice,clim=(-5e-3*pmap_scale,5e-2*pmap_scale),origin='lower',
           cmap='viridis')#,norm=LogNorm())

ax.scatter(x,y,s=40,edgecolor='k',color='C0',lw=1.5) # good pixel
ax.scatter(x_c,y_c,s=40,edgecolor='k',color='g',lw=1.5) # +bad pixel
ax.scatter(x_n,y_n,s=40,edgecolor='k',color='#CB9E0E',lw=1.5) # bkgd pixel
ax.scatter(x_b,y_b,s=40,edgecolor='k',color='r',lw=1.5) # -bad pixell

ax.text(0.975,0.92,f'{name}',transform=ax.transAxes,ha='right',fontsize=20,color='w')
ax.set_yticklabels([])
ax.set_xticklabels([])


ax = plt.subplot(gs[1])

ax.imshow(masked_slice_clipped,clim=(-5e-3*pmap_scale,5e-2*pmap_scale),origin='lower',
           cmap='viridis')#,norm=LogNorm())

ax.text(0.975,0.92,f'{name}',transform=ax.transAxes,ha='right',fontsize=20,color='w')
ax.set_yticklabels([])
ax.set_xticklabels([])



plt.tight_layout()
plt.show()
plt.close('all')


# sys.exit(0)




# clipping slices
# ---------------

# making cube of zeros to log when a pixel has been clipped
clipped_pixels = np.zeros(data.shape) # will be 1 in slice if pixel is clipped

# copy of original cube & error
data_clipped = data.copy()
error_clipped = error.copy()

for i in range(len(data_clipped)):
    data_slice = data[i].copy()
    error_slice = error[i].copy()

    # first, masking out galaxy spaxels
    data_slice[full_mask>0] = np.nan
    error_slice[full_mask>0] = np.nan
    
    mask_data_slice = data_slice.copy()
    mask_error_slice = error_slice.copy()
    
    # marking the pixels that were nans before
    nans_before = np.zeros(mask_data_slice.shape)
    nans_before[np.isnan(mask_data_slice) == True] = 1
    
    # first checking for the all-NaN slices in IFU
    # if yes, nothing happens for that slice and it just
    # gets added to the final cube
    if np.isnan(mask_data_slice).all() == False:
        # sigma clipping
        with warnings.catch_warnings():  # Ignore warnings
            warnings.simplefilter('ignore')

            clip_mask = sigma_clip(mask_data_slice, sigma=sigma, maxiters=5).mask

        clip_mask[full_mask>0] = False # re-masking out the galaxy

        # replaced flagged things with nanmedian (WAS nanmean)
        mask_data_slice[clip_mask] = np.nanmedian(mask_data_slice)
        mask_error_slice[clip_mask] = np.nanmedian(mask_error_slice)
    
        
        # logging pixels that were clipped in separate cube
        clipped_pixels[i][clip_mask] = 1
        # removing pixels that were NaNs before
        clipped_pixels[i][nans_before == 1] = 0
        
    
        # adding back in nans that were there before
        mask_data_slice[nans_before == 1] = np.nan
        mask_error_slice[nans_before == 1] = np.nan
    
    
    # adding back into cube
    data_clipped[i] = mask_data_slice.copy()
    error_clipped[i] = mask_error_slice.copy()
    
    
    
# LOOKING AT CHANGES
wave = bigwave[wave_mask]
ldata_clipped = data_clipped[wave_mask].copy()
lerror_clipped = error_clipped[wave_mask].copy()


# pulling spectrum
spec = get_spec(x,y,d=ldata_clipped,e=lerror_clipped)
spec_good = get_spec(x_c,y_c,d=ldata_clipped,e=lerror_clipped)
spec_bkgd = get_spec(x_n,y_n,d=ldata_clipped,e=lerror_clipped)
spec_nbad = get_spec(x_b,y_b,d=ldata_clipped,e=lerror_clipped)


plt.figure(figsize=(8,5))

plt.step(spec.wave,spec.flam/scale,where='mid')
plt.axvline(sli_wave,color='k')

plt.step(spec_good.wave,spec_good.flam/scale,where='mid',color='g')
plt.step(spec_bkgd.wave,spec_bkgd.flam/scale,where='mid',color='#CB9E0E')
plt.step(spec_nbad.wave,spec_nbad.flam/scale,where='mid',color='r')

# plt.ylim(-5*pmap_scale,8*pmap_scale)
plt.ylim(-50*y_scale,80*y_scale)

plt.tight_layout()
plt.show()
plt.close('all')

    
    
    
    
    


# sys.exit(0)


# writing out sigma clipped line flux FITS cube to file
# using the header from the reduced s3d cube to preserve WCS & wavelength
# -----------------------------
if saveit == True:
    pieces_path = 'plots-data/data-reduction/sigma-clipping-pieces/'
    
    hdu = fits.PrimaryHDU(header=header)
    hdu1 = fits.ImageHDU(data_clipped,header=header) # the data cube
    hdu2 = fits.ImageHDU(error_clipped,header=header) # the error cube
    hdu3 = fits.ImageHDU(clipped_pixels,header=header) # the clipped pixels logging
    hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3])
    hdul.writeto(f'{pieces_path}/{name}-sigmaclipping-bkgd-{grating}-s3d.fits',overwrite=True)
    print('\nsigma clipped FITS cube saved.  Exiting script...',end='\n\n')
    
    
    
    
    
    
    
    
    
    
# CODE TO USE IN TROUBLESHOOTING
# -----------------------------

# meant to be used for a subset of the full cube
# ldata instead of data


    
    
# # pulling spectrum
# cspec = get_spec(x,y,d=ldata_clipped,e=lerror_clipped)
# cspec_good = get_spec(x_c,y_c,d=ldata_clipped,e=lerror_clipped)
# cspec_bkgd = get_spec(x_n,y_n,d=ldata_clipped,e=lerror_clipped)
# cspec_nbad = get_spec(x_b,y_b,d=ldata_clipped,e=lerror_clipped)


# plt.figure(figsize=(8,5))

# plt.step(cspec.wave,cspec.flam/scale,where='mid')
# plt.axvline(sli_wave,color='k')

# plt.step(cspec_good.wave,cspec_good.flam/scale,where='mid',color='g')
# plt.step(cspec_bkgd.wave,cspec_bkgd.flam/scale,where='mid',color='#CB9E0E')
# plt.step(cspec_nbad.wave,cspec_nbad.flam/scale,where='mid',color='r')

# plt.ylim(-5,8)

# plt.tight_layout()
# plt.show()
# plt.close('all')




# # plotting slice with mask
# # ------------------------
# plt.figure(figsize=(12,6))
# gs = gridspec.GridSpec(1,2,width_ratios=[1,1],wspace=0)

# new_sli = 36

# ax = plt.subplot(gs[0])

# ax.imshow(ldata[new_sli],clim=(-5e-3,5e-2),origin='lower',
#            cmap='viridis')#,norm=LogNorm())

# ax.set_yticklabels([])
# ax.set_xticklabels([])


# ax = plt.subplot(gs[1])

# ax.imshow(ldata_clipped[new_sli],clim=(-5e-3,5e-2),origin='lower',
#            cmap='viridis')#,norm=LogNorm())

# ax.set_yticklabels([])
# ax.set_xticklabels([])


# plt.tight_layout()
# plt.show()
# plt.close('all')