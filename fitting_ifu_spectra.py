'''
This module take an input pandas dataframe spectrum and does things with it.
My brain works in pandas dataframes, sorry. :P

The functions within this module each have summaries within them of the
inputs required (and optional) and then what the function outputs.


Last updated:  Nov 2022

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

import matplotlib.pyplot as plt
import pandas as pd


def spec_wave_range(spec,wave_range,index=False):
    '''
    INPUTS:
    >> spec  --------  a pandas dataframe set up with columns
                       "wave", "flux", "ferr"
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
                       "wave", "flux", "ferr"
    >> wave_range  --  (optional)
                       a list of 2 numbers, min & max wavelengths
                       describing the wavelength range of the line
    >> cont_sub -----  a boolean flag to denote if I want to use
                       the continuum-subtracted column ("cont_sub") 
                       for the spectrum instead of the "flux" column
                       
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
        emission = lines.cont_sub.values * cgs
    else:
        emission = lines.flux.values * cgs
    ferr = lines.ferr.values * cgs
    ferr = StdDevUncertainty(ferr)

    # making Spectrum1D object
    spectrum = Spectrum1D(spectral_axis=waves, 
                          flux=emission, 
                          uncertainty=ferr)
    
    return spectrum


def convert_MJy_cgs(spec):
    '''
    *** NOTE ****
    
        NEED TO CONVERT FROM MJY/SR TO CGS NOT MJY, UPDATE
    
    *************
    
    INPUTS:
    >> spec  --------  a pandas dataframe set up with columns
                       "wave", "flux", "ferr"; assumes wave is 
                       in microns and flux, ferr are in MJy
    OUTPUTS:
    >> spec ---------  the same pandas dataframe but in cgs
    '''
    # converting spectrum flux to cgs units first
    spec['flux'] *= 1e-23 # Jy --> erg/s/cm/Hz (fnu)
    spec['flux'] *= 2.998e14 / (spec.wave.values)**2 # fnu --> flam

    # converting spectrum error to cgs units first
    spec['ferr'] *= 1e-23 # Jy --> erg/s/cm/Hz (fnu)
    spec['ferr'] *= 2.998e14 / (spec.wave.values)**2 # fnu --> flam
    
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
                       "wave", "flux", "ferr"
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
                       continuum-subtracted flux
    '''
    # checking if exclude regions specified, if so, converting to SpectralRegion object
    if exclude != None:
        exclude = [SpectralRegion(e[0],e[1]) for e in exclude]
    
    # setting flux == 0 to NaNs
    spec['flux'][spec.query('flux == 0').index.values] = np.nan

    # making a running average to approx continuum
    avg = moving_average(spec.flux.values,n=7)
    
    # subtracting out continuum to sigma clip
    continuum_sub = spec.flux.values - avg
    
    # sigma clipping
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        continuum_sub_clipped = sigma_clip(continuum_sub,sigma=4)

    # adding continuum back in to sigma clipped spec
    clipped_spec = spec.copy()
    clipped_spec['flux'] = continuum_sub_clipped + avg

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
    continuum_sub = spec.flux.values - y_continuum_fitted.value
    spec['cont_sub'] = continuum_sub
    
    return spec.copy()
    



def fit_emission_line(spec,wave_range,x0,verbose=False,contsub=False):
    '''
    INPUTS:
    >> spec  --------  a pandas dataframe set up with columns
                       "wave", "flux", "ferr"
    >> wave_range  --  a list of 2 numbers, min & max wavelengths
                       describing the wavelength range of the line
    >> x0 -----------  a list of initial guesses (try to be fairly accurate)
    >> verbose ------  if you want it to talk to you about the results
    >> contsub ------  if you want it to fit the continuum-subtracted
                       spectrum instead of the "flux" column
                       --> used in the "make spectrum" function
                       
    OUTPUTS:
    >> results ------  a dictionary with the line fit,
                       as well as the fit values
    '''
    assert len(x0) == 3, f"Length of x0 should be 3, it's {len(x0)}"
    
    # this will feel redundant cause I make the spectrum & also
    # apply the wave cuts in the next line after these two,
    # but this is currently the only way I can retain the error
    # spectrum... will update later
    err_query = f'{wave_range[0]}<wave<{wave_range[1]}'
    err = spec.query(err_query).ferr.values
    
    # making Spectrum1D object
    spectrum = make_spectrum(spec,wave_range,contsub=contsub)
    
    cgs = u.erg * u.cm**-2 * u.s**-1 * u.AA**-1

    # setting up a Gaussian model with my initial guesses
    g_init = models.Gaussian1D(amplitude = x0[0] * cgs, 
                               mean = x0[1] * u.um, 
                               stddev = x0[2] * u.um)

    # fitting the line
    g_fit = fit_lines(spectrum, g_init)
    y_fit = g_fit(spectrum.spectral_axis)
    
    if verbose == True:
        print(f'\nAmplitude of fit: {g_fit.amplitude.value}')
        print(f'Center of fit: {g_fit.mean.value}')
        print(f'Stddev of fit: {g_fit.stddev.value}',end='\n\n')
    
    results = {'fit':y_fit, 
               'wave':spectrum.spectral_axis.value,
               'amplitude':g_fit.amplitude.value,
               'mean':g_fit.mean.value,
               'stddev':g_fit.stddev.value,
               'ferr':err}
    
    return results
    

def measure_emission_line(spec,wave_range,verbose=False,contsub=False):
    '''
    INPUTS:
    >> spec  --------  a pandas dataframe set up with columns
                       "wave", "flux", "ferr"
    >> wave_range  --  a list of 2 numbers, min & max wavelengths
                       describing the wavelength range of the line
    >> verbose ------  if you want it to talk to you about the results
    >> contsub ------  if you want it to fit the continuum-subtracted
                       spectrum instead of the "flux" column
                       --> used in the "make spectrum" function
                       
    OUTPUTS:
    >> results ------  a dictionary with the line flux & uncertainty,
                       as well as the centroid of the line
    '''
    
    # making Spectrum1D object
    emission_line = make_spectrum(spec,wave_range,contsub=contsub)
    wavemin, wavemax = wave_range
    
    # measuring both line flux and also the centroid of the line
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


def measure_ew(spec,wave_range,cont,verbose=False,contsub=False):
    '''
    INPUTS:
    >> spec  --------  a pandas dataframe set up with columns
                       "wave", "flux", "ferr"
    >> wave_range  --  a list of 2 numbers, min & max wavelengths
                       describing the wavelength range of the line
    >> cont  --------  a value (in the same units as flux) that
                       describes the continuum level
    >> verbose ------  if you want it to talk to you about the results
    >> contsub ------  if you want it to fit the continuum-subtracted
                       spectrum instead of the "flux" column
                       --> used in the "make spectrum" function
                       
    OUTPUTS:
    >> results ------  the equvalent width of the line
    '''
    
    # making Spectrum1D object
    emission_line = make_spectrum(spec,wave_range,contsub=contsub)
    wavemin, wavemax = wave_range
    
    # measuring the equivalent width of the emission line
    ew = equivalent_width(spectrum=emission_line, 
                          regions=SpectralRegion(wavemin*1e4*u.um,
                                                 wavemax*1e4*u.um),
                          continuum=cont)

    if verbose == True: print(f'\nEquivalent width: {ew}')
    
    results = ew
    
    return results

