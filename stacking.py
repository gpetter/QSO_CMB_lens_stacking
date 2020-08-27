import numpy as np
import healpy as hp
import convergence_map
from astropy.io import fits
import importlib
from astropy import units as u
from astropy.coordinates import SkyCoord
import gc
import time
from math import ceil
import multiprocessing as mp
from functools import partial
importlib.reload(convergence_map)

sdss_quasar_cat = (fits.open('DR14Q_v4_4.fits'))[1].data

# select sample to match Geach+19
good_idxs = np.where((sdss_quasar_cat['Z'] <= 2.2) & (sdss_quasar_cat['Z'] >= 0.9) &
                     (sdss_quasar_cat['FIRST_MATCHED'] == 0) & (sdss_quasar_cat['MI'] <= -24))
ras = sdss_quasar_cat['RA'][good_idxs]
decs = sdss_quasar_cat['DEC'][good_idxs]


def equatorial_to_galactic(ra, dec):
    ra_decs = SkyCoord(ra, dec, unit='deg')
    ls = np.array(ra_decs.galactic.l.radian*u.rad.to('deg'))
    bs = np.array(ra_decs.galactic.b.radian*u.rad.to('deg'))
    return ls, bs


# stacks convergence map by making cutouts of a global lambert projection
def stack_lamb(mapname, boxdim, boxpix, ralist, declist, outname, galactic):
    # set resolution of pixels to stack
    # boxdim is in degrees, while resolution is expected in arcmins, multiply by 60
    # divide field width in arcmins by the number of pixels desired
    reso = boxdim * 60. / boxpix

    #convergence_map.lamb_map('smoothed_masked_planck.fits', reso)

    # read in map
    conv_map = np.load(mapname, allow_pickle=True)

    # set masked pixels to nan so they are ignored in stacking
    conv_map = convergence_map.set_unseen_to_nan(conv_map)

    # transform from (ra,dec) of quasars to (l,b) if planck map is left in galactic coords
    if galactic:
        lon, lat = equatorial_to_galactic(ralist, declist)
    else:
        lon, lat = ralist, declist

    # convert between galactic longitude and latitude to pixel coordinates in Lambert projection map
    lambproj = hp.projector.AzimuthalProj(xsize=12000, ysize=12000, reso=reso, lamb=True)
    i, j = lambproj.xy2ij(lambproj.ang2xy(lon, lat, lonlat=True))

    # half the number of pixels in image
    half_length = int(boxpix/2. - 0.5)

    # make first cutout
    avg_kappa = conv_map[i[0]-half_length:i[0]+half_length+1, j[0]-half_length:j[0]+half_length+1]

    starttime = time.time()
    for k in range(1, 1000):
        if k % 10000 == 0:
            print(k)
            print(time.time() - starttime)
        new_kappa = conv_map[i[k]-half_length:i[k]+half_length+1, j[k]-half_length:j[k]+half_length+1]
        lastavg = k / (k + 1) * avg_kappa
        newterm = new_kappa / (k + 1)
        stacked = np.dstack((lastavg, newterm))
        avg_kappa[:] = np.nansum(stacked, 2)

        gc.collect()
    print(time.time() - starttime)
    avg_kappa.dump(outname)
    print(avg_kappa)


ls, bs = equatorial_to_galactic(ras, decs)


# AzimuthalProj.projmap requires a vec2pix function for some reason, so define one where the nsides are fixed
def newvec2pix(x, y, z):
    return hp.vec2pix(nside=2048, x=x, y=y, z=z)


# for parallelization, stack lots of chunks
def stack_chunk(chunksize, nstack, lon, lat, inmap, k):
    kappa = []
    if (k*chunksize)+chunksize > nstack:
        stepsize = nstack % chunksize
    else:
        stepsize = chunksize
    for l in range(k*chunksize, (k*chunksize)+stepsize):
        azproj = hp.projector.AzimuthalProj(rot=[lon[l], lat[l]], xsize=250, reso=1.2, lamb=True)
        kappa.append(convergence_map.set_unseen_to_nan(azproj.projmap(inmap, vec2pix_func=newvec2pix)))
        # kappa.append(conv_map[i[l] - half_length:i[l] + half_length + 1, j[l] - half_length:j[l] + half_length + 1])

    return np.nanmean(np.array(kappa), axis=0)


# stack the convergence map in the lambert projection at the positions of quasars using parallel processing
def stack_mp(mapname, nstack, lons=ls, lats=bs):
    inmap = hp.read_map(mapname, dtype=np.single)
    # size of chunks to stack in. Good size is near the square root of the total number of stacks
    chunksize = 500
    # the number of chunks is the number of stacks divided by the chunk size rounded up to the nearest integer
    nchunks = ceil(nstack/chunksize)

    starttime = time.time()
    # initalize multiprocessing pool with one less core than is available for stability
    p = mp.Pool(processes=mp.cpu_count()-1)
    # fill in all arguments to stack_chunk function but the index,
    # Pool.map() requires function to only take one paramter
    stack_chunk_partial = partial(stack_chunk, chunksize, nstack, lons, lats, inmap)
    # do the stacking in chunks, map the stacks to different cores for parallel processing
    chunks = p.map(stack_chunk_partial, range(nchunks))
    # mask any pixels in any of the chunks which are NaN, as np.average can't handle NaNs
    masked_chunks = np.ma.MaskedArray(chunks, mask=np.isnan(chunks))
    # set weights for all chunks except possibly the last to be 1
    weights = np.ones(nchunks)
    if nstack % chunksize > 0:
        weights[len(weights)-1] = float(nstack % chunksize) / chunksize
    # stack the chunks, weighting by the number of stacks in each chunk
    finalstack = np.ma.average(masked_chunks, axis=0, weights=weights)
    p.close()
    p.join()
    finalstack.dump('mp_test1.npy')
    print(time.time()-starttime)



