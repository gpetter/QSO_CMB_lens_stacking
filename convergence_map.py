import numpy as np
import healpy as hp

nsides = 2048

def set_unseen_to_nan(map):
    map[np.where(map == hp.UNSEEN)[0]] = np.nan
    return(map)

def set_nan_to_unseen(map):
    map[np.isnan(map)] = hp.UNSEEN
    return(map)

def set_unseen_to_mask(map):
    x = np.ma.masked_where(map==hp.UNSEEN, map)
    x.fill_value = hp.UNSEEN
    return(x)


# zeroes out alm amplitudes for less than a maximum l cutoff
def zero_modes(almarr, l_cutoff):
    lmax = hp.Alm.getlmax(len(almarr))

    l, m = hp.Alm.getlm(lmax=lmax)
    idxs = np.where(l < l_cutoff)[0]
    almarr[idxs] = 0.0j
    return(almarr)


def read_planck_map(remove_meanfield):
    # read in planck alm convergence data
    planck_lensing_alm = hp.read_alm('dat_klm.fits')
    filtered_alm = zero_modes(planck_lensing_alm, 100)
    # generate map from alm data
    planck_lensing_map = hp.sphtfunc.alm2map(filtered_alm, nsides, lmax=4096)
    
    if remove_meanfield:
        print(np.mean(planck_lensing_map))
        mask = hp.read_map('mask.fits')
        masked_map = hp.ma(planck_lensing_map)
        masked_map.mask = np.logical_not(mask)
        masked_map = masked_map.filled()
        masked_map = set_unseen_to_nan(masked_map)
        meanfield = np.nanmean(masked_map)
        print(meanfield)
        #masked_map = masked_map - meanfield
        #masked_map = set_nan_to_unseen(masked_map)
    
    # write out lensing convergence map
    hp.write_map('planck_map.fits', planck_lensing_map, overwrite=True)





def mask_map(map, mask, remove_meanfield, outmap):
    # read in map and mask
    importmap = hp.read_map(map)
    importmask = hp.read_map(mask).astype(np.bool)
    # set mask, invert
    masked_map = hp.ma(importmap)
    masked_map.mask = np.logical_not(importmask)
    masked_map = masked_map.filled()
    
    # subtract mean field
    if remove_meanfield:
        masked_map = set_unseen_to_nan(masked_map)
        meanfield = np.nanmean(masked_map)
        masked_map = masked_map - meanfield
        masked_map = set_nan_to_unseen(masked_map)


    hp.write_map(outmap, masked_map, overwrite=True)


def masked_smoothing(mapname, width, outname):
    U = hp.read_map(mapname)
    U = set_unseen_to_nan(U)
    fwhm = width/60.*np.pi/180.
    V=U.copy()
    V[np.isnan(U)]=0
    VV=hp.smoothing(V, fwhm=fwhm)
    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=hp.smoothing(W, fwhm=fwhm)
    smoothed_map = VV/WW
    smoothed_map[np.isnan(U)] = hp.UNSEEN
    hp.write_map(outname, smoothed_map, overwrite=True)


# smooth map with gaussian of fwhm = width arcminutes
def smooth_map(mapname, width, outname):
    map = hp.read_map(mapname)
    fwhm = width/60.*np.pi/180.
    smoothed_map = hp.sphtfunc.smoothing(map, fwhm=fwhm)

    hp.write_map(outname, smoothed_map, overwrite=True)


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

