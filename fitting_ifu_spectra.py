'''
This module take an input pandas dataframe spectrum and does things with it.
My brain works in pandas dataframes, sorry. :P

The functions within this module each have summaries within them of the
inputs required (and optional) and then what the function outputs.


Last updated:  Jan 2023
-- fixed conversion function first convert from MJy/sr --> MJy
   (had incorrectly assumed spectra were already in MJy)


'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'

import warnings
import numpy as np

from astropy import units as u
from specutils import Spectrum1D, SpectralRegion
from astropy.nddata import StdDevUncertainty
from specutils.analysis import line_flux,centroid,equivalent_width
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from specutils.fitting import fit_lines
from astropy.modeling import models
from astropy.stats import sigma_clip
from specutils.fitting.continuum import fit_continuum
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import astropy.io.fits as fits
import sys,json


def get_galaxy_info(target,grat='g140h'):
    # reading in file that has all of the galaxy values
    with open('plots-data/galaxies.txt') as f:
        data = f.read()

    # reconstructing dictionary
    galaxies = json.loads(data)
    path = galaxies['path']

    # double-checking names are right for dictionary
    try: 
        galaxy = galaxies[target] # getting galaxy info
        gratings = list(galaxy['grating'].keys()) # listing gratings used
        
        # chosing grating for SGAS sources
        if len(gratings) > 1: gratings = grat
        else: gratings = gratings[0]
        
        return galaxy,path,gratings
    
    except KeyError: 
        print(f'The available targets are: {list(galaxies.keys())[1:]}')
        sys.exit(0) # exiting script

        

def get_coords(cube_shape):
    '''
    INPUTS:
    >> cube_shape --- a tuple with 3 numbers, representing a cube with 3 dim.
                      can be calculated doing cube.shape
                      
    OUTPUTS:
    >> coordinates -- a list of coordinates from the cube dimensions   
    
    '''
    # making list of spaxel coordinates based on data shape
    x0,y0 = np.arange(0,cube_shape[2]),np.arange(0,cube_shape[1])
    g = np.meshgrid(x0,y0)
    coords = list(zip(*(c.flat for c in g)))
    return coords
        
        
        
def get_mask(galaxy,array_2d=False,layers=False,grating=False,lens=False):
    '''
    INPUTS:
    >> galaxy -------- the name of the galaxy mask I want
    
    OUTPUTS:
    >> coordinates --- a list of coordinates from the mask,
                       specifying where the galaxy light is
    '''
    
    extra = '' # makes it easier to include or not include lens suffix
    # For the rare cases I want the lens, too
    if lens == True:
        if galaxy == 'SPT0418': extra = '-lens-companion'
        elif galaxy == 'SPT2147': extra = '-lens'
    
    
    # just the full galaxy mask
    if layers == False:
        try:
            galaxy_mask = fits.getdata(f'plots-data/{galaxy}-mask{extra}.fits')

            if galaxy == 'SGAS1723' and grating == 'g395h':
                galaxy_mask = fits.getdata(f'plots-data/{galaxy}-mask-{grating}.fits')

            # makings a list of coordinates
            coordinates = list(zip(*np.where(galaxy_mask == 1)))

            if array_2d == False: return coordinates
            else: return galaxy_mask

        except:
            print("\nWrong file and/or file doesn't exist yet.",end='\n\n')
            sys.exit(0) # kills script

            
    # want the mask slices instead
    # note that the first slice is the full galaxy mask
    else:
        try:
            filename = f'plots-data/{galaxy}-mask-layers{extra}.fits'
            
            if galaxy == 'SGAS1723' and grating == 'g395h':
                filename = f'plots-data/{galaxy}-mask-layers-{grating}.fits'
            
            galaxy_mask = fits.getdata(filename)
            mask_layers_info = np.loadtxt(f'{filename[:-5]}.txt',delimiter='\t')
            
            mask_layers = []
                
            with fits.open(filename) as hdul:
                for i in range(len(hdul)):
                    # map layer
                    galaxy_mask = hdul[i].data

                    # makings a list of coordinates
                    coordinates = list(zip(*np.where(galaxy_mask == 1)))

                    if array_2d == False: mask_layers.append(coordinates)
                    else: mask_layers.append(galaxy_mask)
                
            return mask_layers, mask_layers_info
                

        except:
            print("\nWrong layers file and/or file doesn't exist yet.",end='\n\n')
            sys.exit(0) # kills script
        

        
        
def gaussian(xaxis, mean, A1, sig):
    '''
    Model to used in fitting single line.
    
    '''
    g1 = A1*np.exp(-np.power(xaxis - mean, 2.) /( 2 * np.power(sig, 2.)))
    return g1
        
        
        
def gaussian_doublet(xaxis, mean, A1, A2, sig, sep):
    '''
    Model to be used in fitting blended doublet.
    
    '''
    g1 = A1*np.exp(-np.power(xaxis - mean, 2.) /( 2 * np.power(sig, 2.)))
    
    mean2 = mean + sep
    g2 = A2*np.exp(-np.power(xaxis - mean2, 2.) /( 2 * np.power(sig, 2.)))
    
    return g1+g2
        
        
        
        
        
def spec_wave_range(spec,wave_range,index=False):
    '''
    INPUTS:
    >> spec  --------  a pandas dataframe set up with columns
                       "wave", "flam", "ferr"
    >> wave_range  --  a list of 2 numbers, min & max wavelengths
                       describing the wavelength range of the line
    >> index --------  a flag that if True only returns index vals
                       
    OUTPUTS:
    >> results ------  a "zoomed in" dataframe
    '''
    
    # zooms in to specified wavelength range
    wave_query = f'{wave_range[0]}<wave<{wave_range[1]}'
    spec = spec.query(wave_query).copy()
    
    # if I only want the index values, nothing else
    if index == True:
        spec = spec.index.values
    
    return spec


def make_spectrum(spec,wave_range=False,contsub=False):
    '''
    INPUTS:
    >> spec  --------  a pandas dataframe set up with columns
                       "wave", "flam", "ferr"
    >> wave_range  --  (optional)
                       a list of 2 numbers, min & max wavelengths
                       describing the wavelength range of the line
    >> cont_sub -----  a boolean flag to denote if I want to use
                       the continuum-subtracted column ("cont_sub") 
                       for the spectrum instead of the "flam" column
                       
    OUTPUTS:
    >> results ------  a Spectrum1D object
    '''
    # zooms in if specified
    if wave_range != False:
        wavemin, wavemax = wave_range
        lines = spec.query(f'{wavemin} < wave < {wavemax}').copy()
    else:
        lines = spec.copy()
    
    cgs = u.erg * u.cm**-2 * u.s**-1 * u.AA**-1
    
    # turning into a Spectrum object
    waves = lines.wave.values * u.um
    if contsub == True:
        emission = lines.flam_contsub.values * cgs
    else:
        emission = lines.flam.values * cgs
    ferr = lines.flamerr.values * cgs
    ferr = StdDevUncertainty(ferr)

    # making Spectrum1D object
    spectrum = Spectrum1D(spectral_axis=waves, 
                          flux=emission, 
                          uncertainty=ferr)
    
    return spectrum



def convert_MJy_sr_to_MJy(spec):
    '''
    The spectra from the reduced data are in MJy/sr.
    Converting to MJy so they can be converted to cgs units.
    
    INPUTS:
    >> spec  --------  a pandas dataframe set up with columns
                       "wave", "flam", "ferr"
                       
    OUTPUTS:
    >> spec ---------  the same pandas dataframe but in MJy    
    '''
    # taking nominal pixel area from FITS header for data
    pix_area = 2.35040007004737E-13 # in steradians
    
    # converting spectrum flam 
    spec['flam'] *= pix_area # MJy/sr --> MJy

    # converting spectrum error
    spec['flamerr'] *= pix_area # MJy/sr --> MJy
    
    return spec.copy()
    


def convert_MJy_cgs(spec):
    '''
    INPUTS:
    >> spec  --------  a pandas dataframe set up with columns
                       "wave", "flam", "ferr"; assumes wave is 
                       in microns and flam, ferr are in MJy
    OUTPUTS:
    >> spec ---------  the same pandas dataframe but in cgs
    '''
    # converting from MJy/sr to MJy
    spec = convert_MJy_sr_to_MJy(spec.copy()) 
    
    # converting spectrum flam to cgs units
    spec['flam'] *= 1e6 # MJy --> Jy
    spec['flam'] *= 1e-23 # Jy --> erg/s/cm/Hz (fnu)
    spec['flam'] *= 2.998e18 / (spec.wave.values*1e4)**2 # fnu --> flam
    
    # converting spectrum error to cgs units
    spec['flamerr'] *= 1e6 # MJy --> Jy
    spec['flamerr'] *= 1e-23 # Jy --> erg/s/cm/Hz (fnu)
    spec['flamerr'] *= 2.998e18 / (spec.wave.values*1e4)**2 # fnu --> flam
    
    return spec.copy()


def moving_average(a, n=3):
    '''
    INPUTS:
    >> a  ----------  an array of values
    >> n (opt) -----  window size for moving average (for the
                      initial continuum fit)
                       
    OUTPUTS:
    >> avg ---------  a moving average array the same length as "a"
    
    
    The original code for the moving average can be found here:
https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy/54628145
    
    '''
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    
    avg = ret[n-1:]/n
    avg = np.concatenate((np.ones(6)*avg[0], # to make same len as spec
                          avg))
    
    return avg


def fitting_continuum_1D(spec,window,n=7,exclude=None):
    '''
    INPUTS:
    >> spec  --------  a pandas dataframe set up with columns
                       "wave", "flam", "ferr"
    >> window  ------  a list of tuples, wavelengths with units
                       describing the wavelength ranges to be used
                       in fitting the continuum
    >> n (opt) ------  window size for moving average (for the
                       initial continuum fit)
    >> exclude ------  a boolean flag which, if True, makes some
                       SpectralRegion objects with the specified
                       wavelength ranges so that the fitting 
                       routine skips those sections of the spectrum
    
                       
    OUTPUTS:
    >> spec ---------  the same pandas dataframe but with a
                       new column called "cont_sub" which is the
                       continuum-subtracted flam
    '''
    # checking if exclude regions specified, if so, converting to SpectralRegion object
    if exclude != None:
        exclude = [SpectralRegion(e[0],e[1]) for e in exclude]
    
    # setting flam == 0 to NaNs
    spec['flam'][spec.query('flam == 0').index.values] = np.nan

    # making a running average to approx continuum
    avg = moving_average(spec.flam.values,n=7)
    
    # subtracting out continuum to sigma clip
    continuum_sub = spec.flam.values - avg
    
    # sigma clipping
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        continuum_sub_clipped = sigma_clip(continuum_sub,sigma=4)

    # adding continuum back in to sigma clipped spec
    clipped_spec = spec.copy()
    clipped_spec['flam'] = continuum_sub_clipped + avg

    # making Spectrum1D object
    clipped_spectrum = make_spectrum(clipped_spec)

    # fitting continuum again
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        g1_fit = fit_continuum(clipped_spectrum,window=window,exclude_regions=exclude)
    
    # continuum fit
    y_continuum_fitted = g1_fit(clipped_spectrum.spectral_axis)
    spec['cont'] = y_continuum_fitted.value
    
    # making continuum-subtracted spectrum
    continuum_sub = spec.flam.values - y_continuum_fitted.value
    spec['cont_sub'] = continuum_sub
    
    return spec.copy()
    


def fit_emission_line(spec,model,wave_range,p0,exclude,bounds=False):
    '''
    asldkjfwoigenwgn
    '''
    fitspec = spec.copy()
    fitspec = fitspec.query(f'{wave_range[0]} <= wave <= {wave_range[1]+0.01}').copy()

    # setting up continuum subtraction
    window = [(wave_range[0]*u.um,(fitspec.wave.mean()-0.0007)*u.um),
              ((fitspec.wave.mean()+0.0007)*u.um,wave_range[1]*u.um)]

    exclude_formatted = []
    for win_range in exclude:
        win_range = [i*u.um for i in win_range]
        exclude_formatted.append(win_range)

    # continuum subtraction
    fitspec = fitting_continuum_1D(fitspec.copy(),window=window,exclude=exclude_formatted)

    # fitting line!
    # -------------
    
    # if boundaries not specified
    if bounds == False:
        bounds = ((-np.inf,-np.inf,-np.inf),
              (np.inf,np.inf,np.inf))
    
    
    # p0 = [siii_z, 3*scale, 6]
    wavfit,wavcov = curve_fit(model,fitspec.wave*1e4,fitspec.cont_sub,p0=p0,
                              sigma=fitspec.ferr,bounds=bounds)

    return wavfit,wavcov
    
    
    

# def fit_emission_line(spec,wave_range,x0,verbose=False,contsub=False):
#     '''
#     INPUTS:
#     >> spec  --------  a pandas dataframe set up with columns
#                        "wave", "flam", "ferr"
#     >> wave_range  --  a list of 2 numbers, min & max wavelengths
#                        describing the wavelength range of the line
#     >> x0 -----------  a list of initial guesses (try to be fairly accurate)
#     >> verbose ------  if you want it to talk to you about the results
#     >> contsub ------  if you want it to fit the continuum-subtracted
#                        spectrum instead of the "flam" column
#                        --> used in the "make spectrum" function
                       
#     OUTPUTS:
#     >> results ------  a dictionary with the line fit,
#                        as well as the fit values
#     '''
#     assert len(x0) == 3, f"Length of x0 should be 3, it's {len(x0)}"
    
#     # this will feel redundant cause I make the spectrum & also
#     # apply the wave cuts in the next line after these two,
#     # but this is currently the only way I can retain the error
#     # spectrum... will update later
#     err_query = f'{wave_range[0]}<wave<{wave_range[1]}'
#     err = spec.query(err_query).ferr.values
    
#     # making Spectrum1D object
#     spectrum = make_spectrum(spec,wave_range,contsub=contsub)
    
#     cgs = u.erg * u.cm**-2 * u.s**-1 * u.AA**-1

#     # setting up a Gaussian model with my initial guesses
#     g_init = models.Gaussian1D(amplitude = x0[0] * cgs, 
#                                mean = x0[1] * u.um, 
#                                stddev = x0[2] * u.um)

#     # fitting the line
#     g_fit = fit_lines(spectrum, g_init)
#     y_fit = g_fit(spectrum.spectral_axis)
    
#     if verbose == True:
#         print(f'\nAmplitude of fit: {g_fit.amplitude.value}')
#         print(f'Center of fit: {g_fit.mean.value}')
#         print(f'Stddev of fit: {g_fit.stddev.value}',end='\n\n')
    
#     results = {'fit':y_fit, 
#                'wave':spectrum.spectral_axis.value,
#                'amplitude':g_fit.amplitude.value,
#                'mean':g_fit.mean.value,
#                'stddev':g_fit.stddev.value,
#                'ferr':err}
    
#     return results
    

def measure_emission_line(spec,wave_range,verbose=False,contsub=False):
    '''
    INPUTS:
    >> spec  --------  a pandas dataframe set up with columns
                       "wave", "flam", "ferr"
    >> wave_range  --  a list of 2 numbers, min & max wavelengths
                       describing the wavelength range of the line
    >> verbose ------  if you want it to talk to you about the results
    >> contsub ------  if you want it to fit the continuum-subtracted
                       spectrum instead of the "flam" column
                       --> used in the "make spectrum" function
                       
    OUTPUTS:
    >> results ------  a dictionary with the line flam & uncertainty,
                       as well as the centroid of the line
    '''
    
    # making Spectrum1D object
    emission_line = make_spectrum(spec,wave_range,contsub=contsub)
    wavemin, wavemax = wave_range
    
    # measuring both line flam and also the centroid of the line
    lflux = line_flux(emission_line, 
                      SpectralRegion(wavemin*u.um,wavemax*u.um))
    line_cen = centroid(emission_line, 
                        SpectralRegion(wavemin*u.um,wavemax*u.um))

    if verbose == True:
        print(f'\nFlux of line 1: {lflux} +/- {lflux.uncertainty}')
        print(f'S/N of line: {lflux/lflux.uncertainty}',end='\n\n')
        print(f'Centroid of line: {line_cen}',end='\n\n')
    
    results = {'lineflux':[lflux,lflux.uncertainty], 
               'centroid':line_cen}
    
    return results




# FITS FROM jwst_templates CODE

def get_bounds(p0,lower_scale=1,upper_scale=1):
    filler = [[],[]]
    # walking through initial guess parameters
    for p in p0:
        # scaling how wide the bounds will be by the order of mag
        oom = int(np.log10(p)) # order of mag
        if oom > 3: s = 0.001 / oom
        elif oom >= 1: s = 0.2# / oom
        else: s = 0.75

        lower = p - p*s*lower_scale
        upper = p + p*s*upper_scale
        
        if lower < 0: lower = 0. # just in case extending bounds makes it negative
        
        filler[0].append(lower) # lower bound for parameter
        filler[1].append(upper) # upper bound for parameter
        
    bounds = (tuple(filler[0]), tuple(filler[1]))
    return bounds


def check_fit(popt,bounds):
    '''
    Taking the resulting fit parameters and the input boundaries,
    it checks if any of the fits params are at the edges.   
    '''    
    checking = np.log10(abs(np.asarray(bounds)-np.asarray(popt)))
    bad_fits = [False,False]
    
    if any(checking[0] < -2): bad_fits[0] = True
    if any(checking[1] < -2): bad_fits[1] = True
    
    return bad_fits


def fit_jwst(model,spectrum,p0,bounds=False,maxiters=1): # verbose=False
    '''
    
    Assumes spectrum inputted is continuum-subtracted.
    Assumes wavelength is in microns.
    
    bounds : lower and upper bounds on curve fitting
    iterate : determines how many times it tries to refit by extending bounds
    
    '''
    # setting bounds
    if bounds == True: bounds = get_bounds(p0)
    elif bounds == 0: bounds = (0,np.inf)
    else: bounds = (-np.inf,np.inf)
    
    # setting values
    wave = spectrum.wave.values * 1e4
    flux = spectrum.flam_contsub.values 
    ferr = spectrum.flamerr.values 
    
    # now we fit!
    popt, pcov = curve_fit(model, wave, flux, p0=p0, bounds=bounds, sigma=ferr)
    perr = np.sqrt(np.diag(pcov)) # calculate 1-sigma uncertainties on each parameter
    
    return popt,perr
    
    
    
def fit_jwst_testing(model,spectrum,p0,bounds=False,maxiters=0): # verbose=False
    '''
    
    Assumes spectrum inputted is continuum-subtracted.
    Assumes wavelength is in microns.
    
    bounds : lower and upper bounds on curve fitting
    iterate : determines how many times it tries to refit by extending bounds
    
    '''
    # setting bounds
    if bounds == True: bounds = get_bounds(p0)
    else: bounds = (-np.inf,np.inf)
    
    # setting values
    wave = spectrum.wave.values * 1e4
    flux = spectrum.flam_contsub.values
    ferr = spectrum.flamerr.values
    
    # now we fit!
    popt, pcov = curve_fit(model, wave, flux, p0=p0, bounds=bounds, sigma=ferr)
    perr = np.sqrt(np.diag(pcov)) # calculate 1-sigma uncertainties on each parameter
    
    
    # running fitting iterations if maxiters > 0
    iteration = 0
    refit = True
    while refit == True and iteration < maxiters:
        # checking if the fit is on the bounds or not
        #i.e., if it needs to try again with wider bounds
        bad_fits = check_fit(popt,bounds)
        scale = 2 + (i*0.3) # so range increases as iterations increase

        # setting up new bounds OR marking refit as False
        if bad_fits.all() == True: bounds = get_bounds(p0,lower_scale=scale,upper_scale=scale)
        elif bad_fits[0] == True: bounds = get_bounds(p0,lower_scale=scale)
        elif bad_fits[1] == True: bounds = get_bounds(p0,upper_scale=scale)
        else: refit = False

        # if refit is still True, we actually run the refit
        if refit == True: 
            # now we REfit!
            popt, pcov = curve_fit(model, wave, flux, p0=p0, sigma=ferr, bounds=bounds)
            perr = np.sqrt(np.diag(pcov)) # calculate 1-sigma uncertainties on each parameter
        
    return popt,perr
    
    
    
    
    
def flux_jwst(popt,perr,doublet=False):
    
    if doublet == True:     
        x1, a1, s1 = popt[0], popt[1], popt[3]
        x2, a2, s2 = popt[1]+popt[4], popt[2], popt[3]
        
        x1err, a1err, s1err = perr[0], perr[1], perr[3]
        x2err, a2err, s2err = np.sqrt(perr[1]**2+perr[4]**2), perr[2], perr[3]
        
        flux1 = np.sqrt(2*np.pi) * a1 * np.abs(s1)
        flux2 = np.sqrt(2*np.pi) * a2 * np.abs(s2)
        
        ferr1 = np.sqrt((a1err/a1)**2 + (s1err/np.abs(s1))**2) * flux1
        ferr2 = np.sqrt((a2err/a2)**2 + (s2err/np.abs(s2))**2) * flux2
        
        return [flux1,ferr1],[flux2,ferr2]
    
    else:
        x1, a1, s1 = popt[0], popt[1], popt[2]
        x1err, a1err, s1err = perr[0], perr[1], perr[2]
        
        flux1 = np.sqrt(2*np.pi) * a1 * np.abs(s1)
        ferr1 = np.sqrt((a1err/a1)**2 + (s1err/np.abs(s1))**2) * flux1
        
        return [flux1,ferr1]
    
    
#     # MEASURING FLUXES
#     fluxlist = []
#     fluxerrlist = []
#     for n in range(ngauss):
#         i, j, k = 3*n, 3*n+1, 3*n+2

#         fit_amp, fit_mu, fit_sigma = popt[i], popt[j], popt[k]
#         fit_amp /= scale # dividing by scale factor to make outputs make sense. 
#         d_amp, d_mu, d_sigma = perr[i], perr[j], perr[k]
#         d_amp /= scale # thanks code gremlins, look what you make us do!

#         if wlunit == 'um':
#             fit_sigma *= 1e4  #convert sigma from micron to angstrom so output unit makes sense
#         flux = np.sqrt(2*np.pi) * fit_amp * np.abs(fit_sigma) 
#         fluxlist.append(flux)
#         fluxerr = np.sqrt((d_amp/fit_amp)**2 + (d_sigma/np.abs(fit_sigma))**2) * flux
#         fluxerrlist.append(fluxerr)










# def measure_ew(spec,wave_range,cont,verbose=False,contsub=False):
#     '''
#     INPUTS:
#     >> spec  --------  a pandas dataframe set up with columns
#                        "wave", "flam", "ferr"
#     >> wave_range  --  a list of 2 numbers, min & max wavelengths
#                        describing the wavelength range of the line
#     >> cont  --------  a value (in the same units as flam) that
#                        describes the continuum level
#     >> verbose ------  if you want it to talk to you about the results
#     >> contsub ------  if you want it to fit the continuum-subtracted
#                        spectrum instead of the "flam" column
#                        --> used in the "make spectrum" function
                       
#     OUTPUTS:
#     >> results ------  the equvalent width of the line
#     '''
    
#     # making Spectrum1D object
#     emission_line = make_spectrum(spec,wave_range,contsub=contsub)
#     wavemin, wavemax = wave_range
    
#     # measuring the equivalent width of the emission line
#     ew = equivalent_width(spectrum=emission_line, 
#                           regions=SpectralRegion(wavemin*1e4*u.um,
#                                                  wavemax*1e4*u.um),
#                           continuum=cont)

#     if verbose == True: print(f'\nEquivalent width: {ew}')
    
#     results = ew
    
#     return results




# # this function assumes you've run a script that makes the values like jwstfilter, boxcar, etc
# def get_spec(x,y,d=data,e=error):
#     # doing it this way to circumvent the Big Endian pandas error
#     dat = [float(f) for f in d[:,int(y),int(x)].copy()]
#     err = [float(f) for f in e[:,int(y),int(x)].copy()]

#     spec = pd.DataFrame({'wave':wave,'flam':dat,'flamerr':err})
#     spec = convert_MJy_cgs(spec.copy())
    
#     # setting up continuum fitting things
#     spec = continuum.fit_autocont(spec.copy(),z,boxcar=boxcar,v2mask=v2mask,
#                                                 colf='flam',colcont='flam_autocont')
#     spec['flam_contsub'] = spec.flam.values - spec.flam_autocont.values
    
#     return spec.copy()


# # this script assumes you've just run lines-sgas.py to get the linemap and linefits
# # SPECIFIC FOR [OIII] RIGHT NOW
# def show_fits(x,y):
#     spec = get_spec(x,y,d=data,e=error)

#     plt.figure()

#     plt.step(spec.wave,spec.flam_contsub/scale,where='mid')
#     plt.fill_between(spec.wave,0,spec.flamerr/scale,alpha=0.4,zorder=0,color='r')
    
#     plt.step(spec.wave,gaussian(spec.wave*1e4,*linefits[0:3,y,x])/scale,color='k')
#     plt.step(spec.wave,gaussian(spec.wave*1e4,*linefits[3:6,y,x])/scale,color='k')
#     plt.step(spec.wave,gaussian(spec.wave*1e4,*linefits[6:9,y,x])/scale,color='k')

#     ratio1 = round( linemap[4,y,x] / linemap[2,y,x] ,2)
#     dratio1 = round(np.sqrt((linemap[5,y,x]/linemap[4,y,x])**2+
#                             (linemap[3,y,x]/linemap[2,y,x])**2)*ratio1,2)
#     ratio2 = round( linemap[4,y,x] / linemap[0,y,x] ,2)
#     dratio2 = round(np.sqrt((linemap[5,y,x]/linemap[4,y,x])**2+
#                             (linemap[1,y,x]/linemap[0,y,x])**2)*ratio2,2)
    
#     plt.title(f'O3 = {ratio1} $\pm$ {dratio1};   O3Hb = {ratio2} $\pm$ {dratio2}')
    
#     plt.xlim(.48*(1+z),.51*(1+z))
#     plt.xlabel(f'observed wavelength [microns] at (x,y) = {x,y}')
#     plt.ylabel('flux density [10$^{%s}$ erg/s/cm$^2$/$\AA$]'%(int(np.log10(scale))))

#     plt.tight_layout()
#     plt.show()
#     plt.close('all')