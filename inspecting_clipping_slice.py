'''

Looking through the sigma clipped cube, comparing to the original
cube, and also looking at the pixels that were clipped.

Import this script and then you can run it in the ipython terminal
as follows:

    >> check_slice('SGAS1723',30)

to look at slice 30 in the SGAS1723 cube.  You could also run this
in your regular terminal by doing:

    >> python inspecting_clipping_slice.py SGAS1723 30

'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'

import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fitting_ifu_spectra import * # written by TAH



# plots it all together
def check_slice(target,new_sli,saveit=False):

    # returns dictionary of info for chosen galaxy
    # also path to reduced FITS cubes
    galaxy, path, grating = get_galaxy_info(target)#,grat='g395h')

    # defining some values
    # --------------------
    # original data cube straight from the pipeline
    name,filename = galaxy['name'],galaxy['grating'][grating]['filename']
    data = fits.getdata(path+f'{name}/{filename}')

    # getting mask
    mask = get_mask(name,array_2d=True)

    # sigma clipped cube & clipped pixel tracker
    final_clipped = fits.getdata(f'plots-data/{name}-sigmaclipped-{grating}-s3d.fits')
    clipped_pixels = fits.getdata(f'plots-data/{name}-sigmaclipped-{grating}-s3d.fits',ext=3)


    # PLOTTING
    plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(1,3,width_ratios=[1,1,1],wspace=0)

    ax = plt.subplot(gs[0])

    ax.set_title(f'original slice: {new_sli}')
    ax.imshow(data[new_sli],clim=(-5e-1,5),origin='lower',
               cmap='viridis')

    ax.set_yticklabels([])
    ax.set_xticklabels([])


    ax = plt.subplot(gs[1])

    ax.set_title('layered sigma clipped slice')
    ax.imshow(final_clipped[new_sli],clim=(-5e-1,5),origin='lower',
               cmap='viridis')

    ax.set_yticklabels([]) 
    ax.set_xticklabels([])


    ax = plt.subplot(gs[2])

    ax.set_title('pixels clipped in slice')
    ax.imshow(mask,origin='lower',cmap='Greys',zorder=0,alpha=0.3)
    ax.imshow(clipped_pixels[new_sli],origin='lower',cmap='Blues',alpha=0.5)

    ax.text(0.047,0.927,'galaxy mask',color='grey',transform=ax.transAxes,fontsize=13)
    ax.text(0.047,0.87,'clipped pixel',color='C0',transform=ax.transAxes,fontsize=13,alpha=0.8)

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    plt.tight_layout()
    if saveit == True:
        new_path = 'plots-data/data-reduction/bad-spaxels/checking-clipping-pixels/'
        plt.savefig(new_path+f'{name}-slice{new_sli}.pdf')
    plt.show()
    plt.close('all')


# reads in input for scripted version
if __name__ == "__main__":
    import sys
    try: # if saveit flag is True
        check_slice(str(sys.argv[1]),int(sys.argv[2]),bool(sys.argv[3]))
    except: # regular version
        check_slice(str(sys.argv[1]),int(sys.argv[2]))