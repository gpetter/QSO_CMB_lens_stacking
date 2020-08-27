import numpy as np
import healpy as hp

# number of sides to each healpix pixel
nsides = 2048


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


# read in a klm fits lensing convergence map, zero l modes desired, write out map
def klm_2_map(klmname, mapname):
    # read in planck alm convergence data
    planck_lensing_alm = hp.read_alm(klmname)
    filtered_alm = zero_modes(planck_lensing_alm, 100)
    # generate map from alm data
    planck_lensing_map = hp.sphtfunc.alm2map(filtered_alm, nsides, lmax=4096)
    # write out lensing convergence map
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


# can change coordinates from ecliptic to equatorial for example, no longer needed
def change_coord(mapname, coord, outname):
    """ Change coordinates of a HEALPIX map
        
        Parameters
        ----------
        m : map or array of maps
        map(s) to be rotated
        coord : sequence of two character
        First character is the coordinate system of m, second character
        is the coordinate system of the output map. As in HEALPIX, allowed
        coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)"""
    m = hp.read_map(mapname)
    # Basic HEALPix parameters
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, np.arange(npix))
    
    # Select the coordinate transformation
    rot = hp.Rotator(coord=reversed(coord))
    
    # Convert the coordinates
    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)
    
    hp.write_map(outname, m[..., new_pix], overwrite=True)


def lamb_map(mapname, reso):
    importmap = hp.read_map(mapname, dtype=np.single)
    lambert = (hp.azeqview(importmap, xsize=12000, ysize=12000, reso=reso, lamb=True, return_projected_map=True)).filled()
    lambert[np.where(lambert == -np.inf)] = hp.UNSEEN

    lambert.astype(np.single).dump('lambert_planck.npy')

