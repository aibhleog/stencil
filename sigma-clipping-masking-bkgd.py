'''

Part 2 in sigma clipping routine
---------------------------------

Sigma clipping on the galaxy spaxels.  For a given slice, this code runs through the
S/N layer maps defined in a previous script and sigma clips each S/N layer separately.
Then, it adds them all together and that final added-together slice is the layered
clipping completed for that slice of the cube.

The process then repeats itself for each slice.




NOTES:  Some fine-tuning still needed; likely in the S/N contour layer phase.
        Overall, though, it does work!


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
sigma = 5


# returns dictionary of info for chosen galaxy
# also path to reduced FITS cubes
galaxy, path, grating = get_galaxy_info(target)#,grat='g395h')


endname = '-alt-outlier' #'-nsclean'


# since updated pmap:
pmap_scale = 1e2 # was 1e4 for pmap1027 with old pipeline version


# defining some values
# --------------------
name,filename = galaxy['name'],galaxy['grating'][grating]['filename']
scale,z = galaxy['grating'][grating]['scale']/pmap_scale, galaxy['z']

x,y = galaxy['grating'][grating]['x,y']
# x,y = 13,26



# before & after NSClean
# filename = 'testing-nsclean-not/pmap1084-88/Level3_SGAS1723_g140h-f100lp_s3d_nsclean.fits'
# filename = 'testing-nsclean-not/pmap1084-88/Level3_SGAS1723_g140h-f100lp_s3d_orig.fits'
# filename = 'testing-nsclean-not/pmap1084-88/Level3_SGAS1723_g395h-f290lp_s3d_nsclean.fits'
# filename = 'testing-nsclean-not/pmap1084-88/Level3_SGAS1723_g395h-f290lp_s3d_orig.fits'


# TESTING OUTLIER DETECTION STEP
# NOW THAT IT'S "NEW AND IMPROVED"
filename = 'testing-outlier-detection/pmap1105/Level3_SGAS1723_BGSUB_OUTLIER_g140h-f100lp_s3d.fits'



# sli = 674 # not an emission line
# sli = galaxy['grating'][grating]['slice']
# sli = 2618
# sli = 730 # for sgas1723 gratingg g140
sli = 707
# sli = 547
# sli = 29
# sli = 152


if name == 'SGAS1723' and grating == 'g140h':
    x_c,y_c = 28,37 # random spaxel
    x_n,y_n = 25,29 # random spaxel
    x_b,y_b = 18,48 # random spaxel
elif name == 'SGAS1723' and grating == 'g395h':
    x_c,y_c = 23,32 # 20,42 # random spaxel
    x_n,y_n = 25,29 # random spaxel
    x_b,y_b = 37,14 # random spaxel
    sli -= 1
elif name == 'SPT0418':
    x_c,y_c = 15,10 # +bad spaxel
    x_n,y_n = 27,32 # bkgd spaxel
    x_b,y_b = 15,31 # -bad spaxel
elif name == 'SPT2147':
    x_c,y_c = 27,13 # +bad spaxel
    x_n,y_n = 35,34 # bkgd spaxel
    x_b,y_b = 15,30 #14,18 # -bad spaxel
elif name == 'SGAS1226':
    x_c,y_c = 27,23 # +bad spaxel
    x_n,y_n = 32,25 # bkgd spaxel
    x_b,y_b = 14,18 # -bad spaxel
    
    


# cube
data, header = fits.getdata(path+f'{name}/{filename}', header=True) 
error = fits.getdata(path+f'{name}/{filename}', ext=2) 

# slice of cube
slice_1 = data[sli].copy()
slice_1err = error[sli].copy()


# getting masking layers & info
mask, mask_info = get_mask(name,array_2d=True,layers=True)#,grating='g395h')
full_mask = mask[0].copy()


# plotting slice
# ---------------

plt.figure(figsize=(8,6))

slice_1_masked = slice_1.copy()
slice_1_masked[full_mask<1] = np.nan

plt.imshow(slice_1_masked,clim=(-5e-3*pmap_scale,5e-2*pmap_scale),origin='lower',
           cmap='viridis')

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

# plt.ylim(-5*pmap_scale,10*pmap_scale)
y_scale = np.nanmean([spec_good.median().flam,spec_bkgd.median().flam,
                        spec_nbad.median().flam]) / scale
if target == 'SGAS1723' and grating == 'g140h': y_scale *= 1e1
plt.ylim(-5*y_scale,8*y_scale)

plt.tight_layout()
plt.show()
plt.close('all')


# looking at the S/N spec
# plt.figure(figsize=(8,5))

# plt.step(spec.wave,spec.flam/spec.flamerr,where='mid')
# plt.axvline(sli_wave,color='k')

# plt.step(spec_good.wave,spec_good.flam/spec_good.flamerr,where='mid',color='g')
# plt.step(spec_bkgd.wave,spec_bkgd.flam/spec_bkgd.flamerr,where='mid',color='#CB9E0E')
# plt.step(spec_nbad.wave,spec_nbad.flam/spec_nbad.flamerr,where='mid',color='r')

# # plt.ylim(-5*pmap_scale,10*pmap_scale)

# plt.tight_layout()
# plt.show()
# plt.close('all')



# sys.exit(0)



# sigma clipping the galaxy NOT the background
# --------------------------------------------

# data
masked_slice = slice_1.copy()
masked_slice[full_mask<1] = np.nan

# error
masked_slice_err = slice_1err.copy()
masked_slice_err[full_mask<1] = np.nan


# sigma clipping masked slice layers one at a time
with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    
    clipped_layers, clipped_layers_err = [], []
    for i in range(len(mask)-1):
        # data
        masked_slice_layer = masked_slice.copy()
        masked_slice_layer[mask[i+1]<1] = np.nan # one layer at a time
        # error
        masked_slice_err_layer = masked_slice_err.copy()
        masked_slice_err_layer[mask[i+1]<1] = np.nan # one layer at a time
        
        clip_mask = sigma_clip(masked_slice_layer, sigma=sigma, maxiters=5).mask
        
        # data
        # masked_slice_layer[clip_mask] = np.nan
        # masked_slice_layer[clip_mask] = np.nanmean(masked_slice_layer)
        clipped_layers.append(masked_slice_layer)
        # error
        masked_slice_err_layer[clip_mask] = np.nanmean(masked_slice_err_layer)
        clipped_layers_err.append(masked_slice_err_layer)
        
        


# making one fully clipped view
# to see if it works/how it's working
# -- note: nansum isn't a good idea here cause it sets nans to 0
# data
masked_slice_clipped = np.zeros(mask[0].shape)
masked_slice_clipped[:] = np.nan
# error
masked_slice_err_clipped = np.zeros(mask[0].shape)
masked_slice_err_clipped[:] = np.nan

# adding the clipped layers one by one
for i in range(len(clipped_layers)):
    masked_slice_clipped[mask[i+1]>0] = clipped_layers[i][mask[i+1]>0].copy()
    masked_slice_err_clipped[mask[i+1]>0] = clipped_layers_err[i][mask[i+1]>0].copy()

        
# sys.exit(0)

        
        
# plotting pieced together clipped slices
# -----------------------------------------
plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(1,2,width_ratios=[1,1.25],wspace=0)


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

im = ax.imshow(masked_slice_clipped,clim=(-5e-3*pmap_scale,5e-2*pmap_scale),origin='lower',
           cmap='viridis')#,norm=LogNorm())
cbar = plt.colorbar(im)

ax.text(0.975,0.92,f'{name}',transform=ax.transAxes,ha='right',fontsize=20,color='w')
ax.set_yticklabels([])
ax.set_xticklabels([])


if target == 'SGAS1723': 
    levels = np.array([15,55,150]) # S/N levels
    maxsnr = 300
elif target == 'SGAS1226' or target == 'SPT0418' or target == 'SPT2147': 
    levels = np.array([15,25]) # S/N levels
    maxsnr = 45
    
    
# making list of spaxel coordinates based on data shape
snr_map = slice_1.copy() / slice_1err.copy()
x0,y0 = np.arange(0,snr_map.shape[1]),np.arange(0,snr_map.shape[0])
g = np.meshgrid(x0,y0)
    
# making array colors
cmap = plt.get_cmap('Blues_r')
colors = [cmap(j) for j in np.linspace(0,0.7,len(levels))]

# plotting contours
contours = ax.contour(g[0],g[1],snr_map, levels, origin='upper',alpha=0.6,linewidths=3,colors=colors)
# plt.clabel(contours, inline=True, fontsize=14,colors='k')



plt.tight_layout()
# plt.show()
plt.close('all')



# sys.exit(0)





# clipping slices
# ---------------

# making cube of zeros to log when a pixel has been clipped
clipped_pixels = np.zeros(data.shape) # will be 1 in slice if pixel is clipped


# clipping slices based on S/N contours previously defined
data_clipped = data.copy()
error_clipped = error.copy()

for s in range(len(data_clipped)):
    data_slice = data[s].copy()
    error_slice = error[s].copy()

    # first, masking out background spaxels
    data_slice[full_mask<1] = np.nan
    error_slice[full_mask<1] = np.nan
    
    mask_data_slice = data_slice.copy()
    mask_error_slice = error_slice.copy()
    
    # marking the pixels that were nans before
    nans_before = np.zeros(mask_data_slice.shape)
    nans_before[np.isnan(mask_data_slice) == True] = 1
    
    # first checking for the all-NaN slices in IFU
    # if yes, nothing happens for that slice and it just
    # gets added to the final cube
    if np.isnan(mask_data_slice).all() == False:
        # sigma clipping masked slice layers one at a time
        with warnings.catch_warnings():  # Ignore warnings
            warnings.simplefilter('ignore')

            clipped_layers,clipped_layers_error = [],[]
            clipped_pixels_layers = []
            
            for j in range(len(mask)-1):
                
                # setting up slice and layer
                masked_slice_layer = mask_data_slice.copy()
                masked_slice_layer[mask[j+1]<1] = np.nan # one layer at a time

                # for the error, too
                masked_slice_layer_error = mask_error_slice.copy()
                masked_slice_layer_error[mask[j+1]<1] = np.nan # one layer at a time
                
                # for the clipped pixel tracker, too
                masked_clipped_pixels = np.zeros(masked_slice_layer.shape)

                # sigma clipped LAYER in slice
                clip_mask = sigma_clip(masked_slice_layer, sigma=sigma, maxiters=5).mask

                # replaced flagged things in LAYER with nanmedian (WAS nanmean)
                # mask_data_slice[clip_mask] = np.nanmedian(mask_data_slice[~clip_mask])
                # mask_error_slice[clip_mask] = np.nanmedian(mask_error_slice[~clip_mask])

                masked_slice_layer[clip_mask] = np.nanmedian(masked_slice_layer)
                masked_slice_layer_error[clip_mask] = np.nanmedian(masked_slice_layer_error)

                # logging pixels that were clipped in separate cube
                masked_clipped_pixels[clip_mask] = 1
                # removing pixels that were NaNs before
                masked_clipped_pixels[mask[j+1]<1] = 0
                
                # re-masking the rest
                masked_slice_layer[mask[j+1]<1] = np.nan # one layer at a time
                masked_slice_layer_error[mask[j+1]<1] = np.nan # one layer at a time

                clipped_layers.append(masked_slice_layer)
                clipped_layers_error.append(masked_slice_layer_error)
                clipped_pixels_layers.append(masked_clipped_pixels)


            # making one fully clipped view
            # to see if it works/how it's working
            # -- note: nansum isn't a good idea here cause it sets nans to 0
            masked_slice_clipped = np.zeros(full_mask.shape)
            masked_slice_clipped[:] = np.nan

            masked_slice_clipped_error = np.zeros(full_mask.shape)
            masked_slice_clipped_error[:] = np.nan
            
            # adding the clipped layers one by one
            for i in range(len(clipped_layers)):
                masked_slice_clipped[mask[i+1]>0] = clipped_layers[i][mask[i+1]>0].copy()
                masked_slice_clipped_error[mask[i+1]>0] = clipped_layers_error[i][mask[i+1]>0].copy()
                clipped_pixels[s][mask[i+1]>0] = clipped_pixels_layers[i][mask[i+1]>0].copy()

            # adding back in nans that were there before
            masked_slice_clipped[nans_before == 1] = np.nan
            masked_slice_clipped_error[nans_before == 1] = np.nan
            
            # for clipping tracking, removing pixels that were NaNs before
            clipped_pixels[s][nans_before == 1] = 0
                
        # SHOULD be clipped-by-layer slices
        data_clipped[s] = masked_slice_clipped.copy()
        error_clipped[s] = masked_slice_clipped_error.copy()
    




    
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

plt.ylim(-5*y_scale,8*y_scale)

plt.tight_layout()
plt.show()
plt.close('all')
    
    
    
    
        
# plotting pieced together clipped slices
# -----------------------------------------
plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(1,2,width_ratios=[1,1.25],wspace=0)


ax = plt.subplot(gs[0])

ax.imshow(masked_slice,origin='lower',clim=(-5e-3*pmap_scale,5e-2*pmap_scale),
           cmap='viridis')#,norm=LogNorm())

ax.scatter(x,y,s=40,edgecolor='k',color='C0',lw=1.5) # good pixel
ax.scatter(x_c,y_c,s=40,edgecolor='k',color='g',lw=1.5) # +bad pixel
ax.scatter(x_n,y_n,s=40,edgecolor='k',color='#CB9E0E',lw=1.5) # bkgd pixel
ax.scatter(x_b,y_b,s=40,edgecolor='k',color='r',lw=1.5) # -bad pixell

ax.text(0.975,0.92,f'{name}',transform=ax.transAxes,ha='right',fontsize=20,color='w')
ax.set_yticklabels([])
ax.set_xticklabels([])


ax = plt.subplot(gs[1])

im = ax.imshow(data_clipped[sli],origin='lower',clim=(-5e-3*pmap_scale,5e-2*pmap_scale),
           cmap='viridis')#,norm=LogNorm())
cbar = plt.colorbar(im)

ax.text(0.975,0.92,f'{name}',transform=ax.transAxes,ha='right',fontsize=20,color='w')
ax.set_yticklabels([])
ax.set_xticklabels([])


# if target == 'SGAS1723': 
#     levels = np.array([18,55,150]) # S/N levels
#     maxsnr = 300

# if target == 'SGAS1226': 
#     levels = np.array([15,25]) # S/N levels
#     maxsnr = 45

# # making list of spaxel coordinates based on data shape
# snr_map = slice_1.copy() / slice_1err.copy()
# x0,y0 = np.arange(0,snr_map.shape[1]),np.arange(0,snr_map.shape[0])
# g = np.meshgrid(x0,y0)
    
# # making array colors
# cmap = plt.get_cmap('Blues_r')
# colors = [cmap(j) for j in np.linspace(0,0.7,len(levels))]

# # plotting contours
# contours = ax.contour(g[0],g[1],snr_map, levels, origin='upper',alpha=0.6,linewidths=3,colors=colors)
# # plt.clabel(contours, inline=True, fontsize=14,colors='k')



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
    hdul.writeto(f'{pieces_path}/{name}-sigmaclipping-galaxy-{grating}-s3d{endname}.fits',overwrite=True)
    
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




# sys.exit(0)



# # plotting slice with mask
# # ------------------------
# plt.figure(figsize=(12,6))
# gs = gridspec.GridSpec(1,2,width_ratios=[1,1],wspace=0)

# new_sli = 36

# masked_slice = ldata[new_sli].copy()
# masked_slice[full_mask<1] = np.nan

# ax = plt.subplot(gs[0])

# ax.imshow(masked_slice,clim=(-5e-3,5e-2),origin='lower',
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



   
sys.exit(0)

# CODE FOR LOOKING AT SLICES

# cen = -1 # for SPT2147
cen = 0 # for SPT0418

plt.figure(figsize=(15,6.5))
gs = gridspec.GridSpec(1,5,width_ratios=[1,1,1,1,1],wspace=0)

# for i in np.arange(-3,2): # for SPT2147
for i in np.arange(-2,3): # for SPT0418
    ax = plt.subplot(gs[i+2])
    
    im = plt.imshow(data_clipped[sli+i],clim=(-5e-3*pmap_scale,5e-2*pmap_scale),origin='lower',
               cmap='viridis')
    
    plt.text(0.05,0.9,f'slice: {sli+i}',fontsize=15,transform=ax.transAxes)
    if i == cen:  ax.set_title(r'Centered on H$\alpha$ for %s'%target)
    
    ax.set_yticklabels([])
    ax.set_xticklabels([])


plt.tight_layout()
plt.savefig(f'plots-data/{target}-slices-around-Ha-{grating}.pdf')
plt.show()
plt.close('all')





# just the two bad slices from SPT2147
a = -5

plt.figure(figsize=(8,6.5))
gs = gridspec.GridSpec(1,2,width_ratios=[1,1],wspace=0)

for i in np.arange(a,a+2): # for SPT2147
    ax = plt.subplot(gs[i-a])
    
    im = plt.imshow(data_clipped[sli+i],clim=(-5e-3*pmap_scale,5e-2*pmap_scale),origin='lower',
               cmap='viridis')
    
    plt.text(0.05,0.9,f'slice: {sli+i}',fontsize=15,transform=ax.transAxes)
    
    ax.set_yticklabels([])
    ax.set_xticklabels([])


plt.tight_layout()
plt.show()
plt.close('all')










# S/N SPEC OF SPAXELS

plt.figure(figsize=(8,5))

plt.step(spec.wave,spec.flam/spec.flamerr,where='mid')
plt.axvline(sli_wave,color='k')

plt.step(spec_good.wave,spec_good.flam/spec_good.flamerr,where='mid',color='g')
plt.step(spec_bkgd.wave,spec_bkgd.flam/spec_bkgd.flamerr,where='mid',color='#CB9E0E')
plt.step(spec_nbad.wave,spec_nbad.flam/spec_nbad.flamerr,where='mid',color='r')

# plt.ylim(-5*pmap_scale,8*pmap_scale)
plt.ylim(-5,40)

plt.tight_layout()
plt.show()
plt.close('all')
    