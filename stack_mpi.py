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
import stacking
import importlib
importlib.reload(stacking)






# stacks many projections by breaking up list into chunks and processing each chunk in parallel
def stack_mp(stackmap, ras, decs, pool, weighting=None, prob_weights=None, nstack=None, outname=None, imsize=240, chunksize=500, reso=1.5):
	if weighting is None:
		weighting = np.ones(len(ras))
	if nstack is None:
		nstack = len(ras)

	# the number of chunks is the number of stacks divided by the chunk size rounded up to the nearest integer
	nchunks = ceil(nstack/chunksize)

	lons, lats = stacking.equatorial_to_galactic(ras, decs)

	starttime = time.time()

	# fill in all arguments to stack_chunk function but the index,
	# Pool.map() requires function to only take one paramter
	stack_chunk_partial = partial(stacking.stack_chunk, chunksize, nstack, lons, lats, stackmap, weighting, prob_weights, imsize, reso)
	# do the stacking in chunks, map the stacks to different cores for parallel processing
	# use mpi processing for cluster or multiprocessing for personal computer

	chunksum, chunkweightsum = zip(*pool.map(stack_chunk_partial, range(nchunks)))

	totsum = np.sum(chunksum, axis=0)
	weightsum = np.sum(chunkweightsum, axis=0)
	finalstack = totsum / weightsum

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
	imsize = 160
	reso = 1.5
	stack_maps = False
	stack_noise = False
	nbootstacks = 10
	nrandoms = 0
	temperature = False

	if temperature:
		planck_map = hp.read_map('maps/smica_masked.fits')
		outname = 'stacks/%s_temp' % sample_name
	else:
		planck_map = hp.read_map('maps/smoothed_masked_planck.fits', dtype=np.single)
		outname = 'stacks/%s_stack' % sample_name

	cat = fits.open('catalogs/derived/%s_colored.fits' % sample_name)[1].data

	if stack_maps:
		for j in range(int(np.max(cat['colorbin']))):
			colorcat = cat[np.where(cat['colorbin'] == (j+1))]
			stack_mp(planck_map, colorcat['RA'], colorcat['DEC'], pool, weighting=colorcat['weight'], prob_weights=colorcat['PQSO'], outname=(outname + '%s' % j), imsize=imsize, reso=reso)
		if not stack_noise:
			pool.close()

	if stack_noise:
		noisemaplist = glob.glob('noisemaps/maps/*.fits')
		colorcat = cat[np.where(cat['colorbin'] == 5)]
		for j in range(len(noisemaplist)):
			noisemap = hp.read_map('noisemaps/maps/%s.fits' % j, dtype=np.single)
			stack_mp(noisemap, colorcat['RA'], colorcat['DEC'], pool, weighting=colorcat['weight'], prob_weights=colorcat['PQSO'], outname='noise_stacks/map%s' % (j), imsize=imsize, reso=reso)

	if nbootstacks > 0:
		for j in range(nbootstacks):
			for i in range(int(np.max(cat['colorbin']))):
				colorcat = cat[np.where(cat['colorbin'] == (i + 1))]
				ras, decs, weights, probs = colorcat['RA'], colorcat['DEC'], colorcat['weight'], colorcat['PQSO']
				bootidxs = np.random.choice(len(colorcat), len(colorcat))
				bootras, bootdecs, bootweights, bootprobs = ras[bootidxs], decs[bootidxs], weights[bootidxs], probs[bootidxs]
				stack_mp(planck_map, bootras, bootdecs, pool, weighting=bootweights, prob_weights=bootprobs, outname='bootstacks/boot%s_%s' % (j, i), imsize=imsize, reso=reso)

	if nrandoms > 0:
		for j in range(nrandoms):
			for i in range(int(np.max(cat['colorbin']))):
				colorcat = cat[np.where(cat['colorbin'] == (i + 1))]
				randras = colorcat['RA'] + np.random.uniform(2, 4, len(colorcat))
				stack_mp(planck_map, randras, colorcat['DEC'], pool, weighting=colorcat['weight'], prob_weights=colorcat['PQSO'], outname='random_stacks/rand%s_%s' % (j, i), imsize=imsize, reso=reso)

	pool.close()
