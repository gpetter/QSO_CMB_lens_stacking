import numpy as np
import healpy as hp
import pandas as pd
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from pixell import enmap, reproject
import astropy.units as u

# number of sides to each healpix pixel
#nsides = 2048


# take in healpix map which defaults to using the UNSEEN value to denote masked pixels and return
# a masked map with NaNs instead
def set_unseen_to_nan(map):
    map[np.where(np.logical_or(map == hp.UNSEEN, np.logical_and(map < -1e30, map > -1e31)))] = np.nan
    return map


# convert a NaN scheme masked map back to the UNSEEN scheme for healpix manipulation
def set_nan_to_unseen(map):
    map[np.isnan(map)] = hp.UNSEEN
    return map


# convert UNSEEN scheme masked map to native numpy masked scheme
def set_unseen_to_mask(map):
    x = np.ma.masked_where(map == hp.UNSEEN, map)
    x.fill_value = hp.UNSEEN
    return x


# zeroes out alm amplitudes for less than a maximum l cutoff
def zero_modes(almarr, l_cutoff):
    lmax = hp.Alm.getlmax(len(almarr))
    l, m = hp.Alm.getlm(lmax=lmax)
    idxs = np.where(l < l_cutoff)
    almarr[idxs] = 0.0j
    return almarr


def wiener_filter(almarr):
    lmax = hp.Alm.getlmax(len(almarr))
    l, m = hp.Alm.getlm(lmax=lmax)

    noise_table = pd.read_csv('maps/nlkk.dat', delim_whitespace=True, header=None)
    cl_plus_nl = np.array(noise_table[2])
    nl = np.array(noise_table[1])
    cl = cl_plus_nl - nl

    wien_factor = cl/cl_plus_nl

    almarr = hp.smoothalm(almarr, beam_window=wien_factor)
    return almarr


# read in a klm fits lensing convergence map, zero l modes desired, write out map
def klm_2_map(klmname, mapname, nsides):
    # read in planck alm convergence data
    planck_lensing_alm = hp.read_alm(klmname)
    filtered_alm = zero_modes(planck_lensing_alm, 100)
    # generate map from alm data
    planck_lensing_map = hp.sphtfunc.alm2map(filtered_alm, nsides, lmax=4096)
    hp.write_map(mapname, planck_lensing_map, overwrite=True)


# smooth map with gaussian of fwhm = width arcminutes
def smooth_map(mapname, width, outname):
    map = hp.read_map(mapname)
    fwhm = width/60.*np.pi/180.
    smoothed_map = hp.sphtfunc.smoothing(map, fwhm=fwhm)

    hp.write_map(outname, smoothed_map, overwrite=True)


# mask map and remove the mean field if desired
def mask_map(map, mask, outmap):
    # read in map and mask
    importmap = hp.read_map(map)
    importmask = hp.read_map(mask).astype(np.bool)
    # set mask, invert
    masked_map = hp.ma(importmap)
    masked_map.mask = np.logical_not(importmask)
    masked_map = masked_map.filled()

    hp.write_map(outmap, masked_map, overwrite=True)


# input klm file and output final smoothed, masked map for analysis
def klm_2_product(klmname, width, maskname, nsides, lmin, subtract_mf=False, writename=None):

    # read in planck alm convergence data
    planck_lensing_alm = hp.read_alm(klmname)
    lmax = hp.Alm.getlmax(len(planck_lensing_alm))

    if subtract_mf:
        mf_alm = hp.read_alm('maps/mf_klm.fits')
        planck_lensing_alm = planck_lensing_alm - mf_alm

    # if you want to smooth with a gaussian
    if width > 0:
        # transform a gaussian of FWHM=width in real space to harmonic space
        k_space_gauss_beam = hp.gauss_beam(fwhm=width.to('radian').value, lmax=lmax)
        # if truncating small l modes
        if lmin > 0:
            # zero out small l modes in k-space filter
            k_space_gauss_beam[:lmin] = 0

        # smooth in harmonic space
        filtered_alm = hp.smoothalm(planck_lensing_alm, beam_window=k_space_gauss_beam)
    else:
        # if not smoothing with gaussian, just remove small l modes
        filtered_alm = zero_modes(planck_lensing_alm, lmin)

    planck_lensing_map = hp.sphtfunc.alm2map(filtered_alm, nsides, lmax=lmax)

    # mask map
    importmask = hp.read_map(maskname)
    if nsides < 2048:
        mask_lowres_proper = hp.ud_grade(importmask.astype(float), nside_out=1024).astype(float)
        finalmask = np.where(mask_lowres_proper == 1., True, False).astype(bool)
    else:
        finalmask = importmask.astype(np.bool)
    # set mask, invert
    smoothed_masked_map = hp.ma(planck_lensing_map)
    smoothed_masked_map.mask = np.logical_not(finalmask)

    if writename:
        hp.write_map('%s.fits' % writename, smoothed_masked_map.filled(), overwrite=True, dtype=np.single)

    return smoothed_masked_map.filled()

def make_fake_noise_map(stackedmap):
    mean, med, stdev = sigma_clipped_stats(stackedmap)
    # find best factor
    """for j in range(18, 25):
        fake_noise = np.random.normal(med, j*stdev, (300, 300))
        kernel = Gaussian2DKernel(x_stddev=15/2.355)
        scipy_conv = scipy_convolve(fake_noise, kernel, mode='same', method='direct')
        print(np.std(scipy_conv)/stdev, j)
    for j in range(1,10):
        fake_noise = np.random.normal(med, j * 0.0001, (300, 300))
        kernel = Gaussian2DKernel(x_stddev=15 / 2.355)
        scipy_conv = scipy_convolve(fake_noise, kernel, mode='same', method='direct')
        print(np.std(scipy_conv)/(j*0.0001))"""

    fake_noise = np.random.normal(med, 22.5 * stdev, (len(stackedmap), len(stackedmap)))
    kernel = Gaussian2DKernel(x_stddev=15 / 2.355)
    scipy_conv = scipy_convolve(fake_noise, kernel, mode='same', method='direct')
    return scipy_conv


def ACT_map(nside, lmax, smoothfwhm):
    #bnlensing = enmap.read_map('ACTlensing/act_planck_dr4.01_s14s15_BN_lensing_kappa_baseline.fits')
    bnlensing = enmap.read_map('ACTlensing/act_dr4.01_s14s15_BN_lensing_kappa.fits')
    bnlensing_hp = reproject.healpix_from_enmap(bnlensing, lmax=lmax, nside=nside)
    bnmask = enmap.read_map('ACTlensing/act_dr4.01_s14s15_BN_lensing_mask.fits')
    wc_bn = reproject.healpix_from_enmap(bnmask, lmax=lmax, nside=nside)
    wc_bn_mean = np.mean(np.array(bnmask) ** 2)
    #wc_bn_mean = np.mean(wc_bn**2)
    bnlensing_hp = bnlensing_hp * wc_bn_mean

    smoothbn = hp.smoothing(bnlensing_hp, fwhm=(smoothfwhm * u.arcmin.to('rad')))

    #dlensing = enmap.read_map('ACTlensing/act_planck_dr4.01_s14s15_D56_lensing_kappa_baseline.fits')
    dlensing = enmap.read_map('ACTlensing/act_dr4.01_s14s15_D56_lensing_kappa.fits')
    dlensing_hp = reproject.healpix_from_enmap(dlensing, lmax=lmax, nside=nside)
    dmask = enmap.read_map('ACTlensing/act_dr4.01_s14s15_D56_lensing_mask.fits')
    wc_d = reproject.healpix_from_enmap(dmask, lmax=lmax, nside=nside)
    wc_d_mean = np.mean(np.array(dmask) ** 2)
    #wc_d_mean = np.mean(wc_d**2)

    dlensing_hp = dlensing_hp * wc_d_mean
    smoothd = hp.smoothing(dlensing_hp, fwhm=smoothfwhm*u.arcmin.to('rad'))

    ddataidxs = np.where(wc_d > 0.8)
    combinedmask = wc_bn + wc_d

    smoothbn[ddataidxs] = smoothd[ddataidxs]
    smoothbn[np.where(combinedmask < 0.8)] = hp.UNSEEN


    return smoothbn

def change_coord(m, coord):
    """ Change coordinates of a HEALPIX map

    Parameters
    ----------
    m : map or array of maps
      map(s) to be rotated
    coord : sequence of two character
      First character is the coordinate system of m, second character
      is the coordinate system of the output map. As in HEALPIX, allowed
      coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

    Example
    -------
    The following rotate m from galactic to equatorial coordinates.
    Notice that m can contain both temperature and polarization.
    """
    # Basic HEALPix parameters
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, np.arange(npix))

    # Select the coordinate transformation
    rot = hp.Rotator(coord=reversed(coord))

    # Convert the coordinates
    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)

    return m[..., new_pix]


