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


# sub stacking routine which worker will execute
def stack_chunk(chunksize, nstack, lon, lat, inmap, weighting, k):
	kappa = []
	if (k*chunksize)+chunksize > nstack:
		stepsize = nstack % chunksize
	else:
		stepsize = chunksize

	weights = weighting[(k*chunksize):((k*chunksize)+stepsize)]
	for l in range(k*chunksize, (k*chunksize)+stepsize):
		azproj = hp.projector.AzimuthalProj(rot=[lon[l], lat[l]], xsize=300, reso=1.2, lamb=True)
		kappa.append(set_unseen_to_nan(azproj.projmap(inmap, vec2pix_func=newvec2pix)))
	masked_kappa = np.ma.MaskedArray(np.array(kappa), mask=np.isnan(np.array(kappa)))

	return np.ma.average(masked_kappa, axis=0, weights=np.array(weights))


# stacks many projections by breaking up list into chunks and processing each chunk in parallel
def stack_mp(stackmap, nstack, outname, ras, decs, weighting, pool):
	# size of chunks to stack in. Good size is near the square root of the total number of stacks
	chunksize = 1000
	# the number of chunks is the number of stacks divided by the chunk size rounded up to the nearest integer
	nchunks = ceil(nstack/chunksize)

	lons, lats = equatorial_to_galactic(ras, decs)

	starttime = time.time()

	# fill in all arguments to stack_chunk function but the index,
	# Pool.map() requires function to only take one paramter
	stack_chunk_partial = partial(stack_chunk, chunksize, nstack, lons, lats, stackmap, weighting)
	# do the stacking in chunks, map the stacks to different cores for parallel processing
	# use mpi processing for cluster or multiprocessing for personal computer

	chunks = list(pool.map(stack_chunk_partial, range(nchunks)))

	# mask any pixels in any of the chunks which are NaN, as np.average can't handle NaNs
	masked_chunks = np.ma.MaskedArray(chunks, mask=np.isnan(chunks))
	# set weights for all chunks except possibly the last to be 1
	chunkweights = np.ones(nchunks)
	if nstack % chunksize > 0:
		chunkweights[len(chunkweights)-1] = float(nstack % chunksize) / chunksize
	# stack the chunks, weighting by the number of stacks in each chunk
	finalstack = np.ma.average(masked_chunks, axis=0, weights=chunkweights)

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

	sample_name = 'xdqso'
	stack_maps = True
	stack_noise = False

	if sample_name == 'xdqso':
		rakey, deckey = 'RA_XDQSO', 'DEC_XDQSO'
	else:
		rakey, deckey = 'RA', 'DEC'

	bluecat = fits.open('QSO_cats/%s_blue.fits' % sample_name)[1].data
	blueras, bluedecs = bluecat[rakey], bluecat[deckey]

	ctrlcat = fits.open('QSO_cats/%s_ctrl.fits' % sample_name)[1].data
	ctrlras, ctrldecs = ctrlcat[rakey], ctrlcat[deckey]

	redcat = fits.open('QSO_cats/%s_red.fits' % sample_name)[1].data
	redras, reddecs = redcat[rakey], redcat[deckey]

	if stack_maps:
		planck_map = hp.read_map('maps/smoothed_masked_planck.fits', dtype=np.single)
		stack_mp(planck_map, len(blueras), 'stacks/%s_blue_stack' % sample_name, blueras, bluedecs, bluecat['weight'], pool)
		stack_mp(planck_map, len(ctrlras), 'stacks/%s_ctrl_stack' % sample_name, ctrlras, ctrldecs, ctrlcat['weight'], pool)
		stack_mp(planck_map, len(redras), 'stacks/%s_red_stack' % sample_name, redras, reddecs, redcat['weight'], pool)
		if not stack_noise:
			pool.close()

	if stack_noise:
		noisemaplist = glob.glob('noisemaps/maps/*.fits')
		for j in range(len(noisemaplist)):
			noisemap = hp.read_map('noisemaps/maps/%s.fits' % j, dtype=np.single)
			stack_mp(noisemap, len(blueras), 'noise_stacks/%s_blue' % j, blueras, bluedecs, bluecat['weight'], pool)
			stack_mp(noisemap, len(ctrlras), 'noise_stacks/%s_ctrl' % j, ctrlras, ctrldecs, ctrlcat['weight'], pool)
			stack_mp(noisemap, len(redras), 'noise_stacks/%s_red' % j, redras, reddecs, redcat['weight'], pool)
	pool.close()
