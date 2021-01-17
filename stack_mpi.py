##########
# stack_mpi.py
# author: Grayson Petter
# code for performing a stack of many cmb projections in parallel using MPI for deployment on a cluster computer
##########

import numpy as np
import healpy as hp
import time
from functools import partial
from astropy.coordinates import SkyCoord
from math import ceil
from astropy import units as u
import convergence_map



# convert ras, decs to ls, bs
def equatorial_to_galactic(ra, dec):
	ra_decs = SkyCoord(ra, dec, unit='deg', frame='icrs')
	ls = np.array(ra_decs.galactic.l.radian*u.rad.to('deg'))
	bs = np.array(ra_decs.galactic.b.radian*u.rad.to('deg'))
	return ls, bs


# set masked pixels (which healpix sets to -1.675E-30) to NaN
def set_unseen_to_nan(map):
	map[np.where(np.logical_or(map == hp.UNSEEN, np.logical_and(map < -1e30, map > -1e31)))] = np.nan
	return map


# AzimuthalProj.projmap requires a vec2pix function for some reason, so define one where the nsides are fixed
def newvec2pix(x, y, z):
	return hp.vec2pix(nside=2048, x=x, y=y, z=z)


# for parallelization of stacking procedure, this method will stack a "chunk" of the total stack
def stack_chunk(chunksize, nstack, lon, lat, inmap, weighting, prob_weights, imsize, reso, k):
	kappa = []
	weightsum = np.zeros((imsize, imsize))
	# if this is the last chunk in the stack, the number of sources in the chunk probably won't be = chunksize
	if (k*chunksize)+chunksize > nstack:
		stepsize = nstack % chunksize
	else:
		stepsize = chunksize

	masked_fraction = []

	for l in range(k*chunksize, (k*chunksize)+stepsize):
		azproj = hp.projector.AzimuthalProj(rot=[lon[l], lat[l]], xsize=imsize, reso=reso, lamb=True)
		nanned_map = convergence_map.set_unseen_to_nan(azproj.projmap(inmap, vec2pix_func=newvec2pix))
		masked_fraction.append(np.count_nonzero(~np.isnan(nanned_map))/nanned_map.size)

		wmat = np.ones((imsize, imsize)) * weighting[l]
		wmat[np.isnan(nanned_map)] = 0
		nanned_map[np.isnan(nanned_map)] = 0

		if prob_weights is not None:
			wmat_for_sum = wmat * prob_weights[l]
		else:
			wmat_for_sum = wmat

		kappa.append(wmat * nanned_map)
		weightsum += wmat_for_sum

	return (np.sum(np.array(kappa), axis=0)/weightsum, np.mean(masked_fraction))


# stacks many projections by breaking up list into chunks and processing each chunk in parallel
def stack_mp(stackmap, ras, decs, pool, weighting=None, prob_weights=None, nstack=None, outname=None, imsize=240, chunksize=500, reso=1.5):
	if weighting is None:
		weighting = np.ones(len(ras))
	if nstack is None:
		nstack = len(ras)

	# the number of chunks is the number of stacks divided by the chunk size rounded up to the nearest integer
	nchunks = ceil(nstack/chunksize)

	lons, lats = equatorial_to_galactic(ras, decs)

	starttime = time.time()

	# fill in all arguments to stack_chunk function but the index,
	# Pool.map() requires function to only take one paramter
	stack_chunk_partial = partial(stack_chunk, chunksize, nstack, lons, lats, stackmap, weighting, prob_weights, imsize, reso)
	# do the stacking in chunks, map the stacks to different cores for parallel processing
	# use mpi processing for cluster or multiprocessing for personal computer

	chunks, chunkweights = zip(*pool.map(stack_chunk_partial, range(nchunks)))
	chunkweights = np.array(chunkweights)


	if nstack % chunksize > 0:
		chunkweights[len(chunkweights)-1] *= float(nstack % chunksize) / chunksize

	weightedchunks = np.prod(np.array([chunkweights, chunks], dtype=np.object), axis=0)
	# stack the chunks, weighting by the number of stacks in each chunk
	finalstack = np.sum(weightedchunks, axis=0) / np.sum(chunkweights)

	finalstack.dump('%s.npy' % outname)
	print(time.time()-starttime)


if __name__ == "__main__":
	import sys
	from schwimmbad import MPIPool
	from astropy.io import fits
	import glob

	# set up schwimmbad MPI pool
	pool = MPIPool()

	# if current instantiation is not the master, wait for tasks
	if not pool.is_master():
		pool.wait()
		sys.exit(0)

	sample_name = 'xdqso_specz'
	color = 'red'
	imsize = 240
	reso = 1.5
	stack_maps = True
	stack_noise = True

	cat = fits.open('QSO_cats/%s_%s.fits' % (sample_name, color))[1].data
	if (sample_name == 'xdqso') or (sample_name == 'xdqso_specz'):
		rakey, deckey = 'RA_XDQSO', 'DEC_XDQSO'
		probs = cat['PQSO']
	else:
		rakey, deckey = 'RA', 'DEC'
		probs = np.ones(len(cat))
	ras, decs = cat[rakey], cat[deckey]

	if stack_maps:
		planck_map = hp.read_map('maps/smoothed_masked_planck.fits', dtype=np.single)
		stack_mp(planck_map, ras, decs, pool, weighting=cat['weight'], prob_weights=probs, outname=('stacks/%s_%s_stack' % (sample_name, color)), imsize=imsize, reso=reso)

		if not stack_noise:
			pool.close()

	if stack_noise:
		noisemaplist = glob.glob('noisemaps/maps/*.fits')
		for j in range(len(noisemaplist)):
			noisemap = hp.read_map('noisemaps/maps/%s.fits' % j, dtype=np.single)
			stack_mp(noisemap, ras, decs, pool, weighting=cat['weight'], prob_weights=probs, outname='noise_stacks/%s_%s' % (j, color), imsize=imsize, reso=reso)

	pool.close()
