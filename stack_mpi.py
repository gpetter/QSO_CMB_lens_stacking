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


# perform one iteration of a stack
def stack_iteration(current_sum, current_weightsum, new_cutout, weight, prob_weight, imsize):
	# create an image filled with the value of the weight, set weights to zero where the true map is masked
	wmat = np.full((imsize, imsize), weight)
	wmat[np.isnan(new_cutout)] = 0
	# the weights for summing in the denominator are multiplied by the probabilty weight to account
	# for the fact that some sources aren't quasars and contribute no signal to the stack
	wmat_for_sum = wmat * prob_weight
	# the running total sum is the sum from last iteration plus the new cutout
	new_sum = np.nansum([current_sum, new_cutout], axis=0)
	new_weightsum = np.sum([current_weightsum, wmat_for_sum], axis=0)

	return new_sum, new_weightsum


def sum_projections(lon, lat, weights, prob_weights, imsize, reso, inmap, nstack):
	running_sum, weightsum = np.zeros((imsize, imsize)), np.zeros((imsize, imsize))
	for j in range(nstack):
		azproj = hp.projector.AzimuthalProj(rot=[lon[j], lat[j]], xsize=imsize, reso=reso, lamb=True)
		# gnomproj = hp.projector.GnomonicProj(rot=[lon[j], lat[j]], xsize=imsize, reso=reso)
		new_im = weights[j] * convergence_map.set_unseen_to_nan(azproj.projmap(inmap, vec2pix_func=newvec2pix))

		running_sum, weightsum = stack_iteration(running_sum, weightsum, new_im, weights[j], prob_weights[j], imsize)

	return running_sum, weightsum




# for parallelization of stacking procedure, this method will stack a "chunk" of the total stack
def stack_chunk(chunksize, nstack, lon, lat, inmap, weighting, prob_weights, imsize, reso, k):
	# if this is the last chunk in the stack, the number of sources in the chunk probably won't be = chunksize
	if (k * chunksize) + chunksize > nstack:
		stepsize = nstack % chunksize
	else:
		stepsize = chunksize
	highidx, lowidx = ((k * chunksize) + stepsize), (k * chunksize)

	totsum, weightsum = sum_projections(lon[lowidx:highidx], lat[lowidx:highidx], weighting[lowidx:highidx], prob_weights[lowidx:highidx], imsize, reso, inmap, stepsize)

	return totsum, weightsum


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
	#color = 'ctrl'
	imsize = 160
	reso = 1.5
	stack_maps = True
	stack_noise = False
	nbootstacks = 0
	nrandoms = 0
	temperature = False

	if temperature:
		planck_map = hp.read_map('maps/smica_masked.fits')
		outname = 'stacks/%s_temp' % sample_name
	else:
		planck_map = hp.read_map('maps/smoothed_masked_planck.fits', dtype=np.single)
		outname = 'stacks/%s_stack' % sample_name

	#cat = fits.open('catalogs/derived/%s_%s.fits' % (sample_name, color))[1].data
	cat = fits.open('catalogs/derived/%s_colored.fits' % sample_name)[1].data

	"""if (sample_name == 'xdqso') or (sample_name == 'xdqso_specz'):
		probs = cat['PQSO']
	else:
		probs = np.ones(len(cat))
	ras, decs = cat['RA'], cat['DEC']
	weights = cat['weight']"""

	if stack_maps:
		for j in range(int(np.max(cat['colorbin']))):
			colorcat = cat[np.where(cat['colorbin'] == (j+1))]
			stack_mp(planck_map, colorcat['RA'], colorcat['DEC'], pool, weighting=colorcat['weight'], prob_weights=colorcat['PQSO'], outname=(outname + '%s' % j), imsize=imsize, reso=reso)
		if not stack_noise:
			pool.close()

	if stack_noise:
		noisemaplist = glob.glob('noisemaps/maps/*.fits')
		for j in range(len(noisemaplist)):
			noisemap = hp.read_map('noisemaps/maps/%s.fits' % j, dtype=np.single)
			for i in range(int(np.max(cat['colorbin']))):
				colorcat = cat[np.where(cat['colorbin'] == (i + 1))]
				stack_mp(noisemap, colorcat['RA'], colorcat['DEC'], pool, weighting=colorcat['weight'], prob_weights=colorcat['PQSO'], outname='noise_stacks/map%s_%s' % (j, i), imsize=imsize, reso=reso)

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
