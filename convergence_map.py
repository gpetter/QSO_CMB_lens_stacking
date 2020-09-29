import numpy as np
import healpy as hp
import pandas as pd

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

    for l_mode in range(len(wien_factor)):
        idxs = np.where(l == l_mode)
        almarr[idxs] *= wien_factor[l_mode]
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
def klm_2_product(klmname, width, maskname, nsides, lmin, writename):

    # read in planck alm convergence data
    planck_lensing_alm = hp.read_alm(klmname)
    lmax = hp.Alm.getlmax(len(planck_lensing_alm))

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
    importmask = hp.read_map(maskname).astype(np.bool)
    # set mask, invert
    smoothed_masked_map = hp.ma(planck_lensing_map)
    smoothed_masked_map.mask = np.logical_not(importmask)

    hp.write_map('%s.fits' % writename, smoothed_masked_map.filled(), overwrite=True, dtype=np.single)

    return smoothed_masked_map.filled()

