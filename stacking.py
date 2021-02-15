import numpy as np
import healpy as hp
import convergence_map
from astropy.io import fits
import importlib
from astropy import units as u
from astropy.coordinates import SkyCoord
import glob
import fileinput
import sys
import time
from math import ceil
import multiprocessing as mp
from functools import partial
from astropy.table import Table

from astropy import coordinates
import urllib3
import glob
urllib3.disable_warnings()

importlib.reload(convergence_map)


# convert ras and decs to galactic l, b coordinates
def equatorial_to_galactic(ra, dec):
	ra_decs = SkyCoord(ra, dec, unit='deg', frame='icrs')
	ls = np.array(ra_decs.galactic.l.radian * u.rad.to('deg'))
	bs = np.array(ra_decs.galactic.b.radian * u.rad.to('deg'))
	return ls, bs


# given list of ras, decs, return indices of sources whose centers lie outside the masked region of the lensing map
def get_qsos_outside_mask(nsides, themap, ras, decs):
	ls, bs = equatorial_to_galactic(ras, decs)
	pixels = hp.ang2pix(nsides, ls, bs, lonlat=True)
	idxs = np.where(themap[pixels] != hp.UNSEEN)
	return idxs


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

def find_closest_cutout(l, b, fixedls, fixedbs):
	return np.argmin(hp.rotator.angdist([fixedls, fixedbs], [l, b], lonlat=True))




def stack_cutouts(ras, decs, weights, prob_weights, imsize, nstack, outname=None, bootstrap=False):
	# read in previously calculated projections covering large swaths of sky
	projectionlist = glob.glob('planckprojections/*')
	projections = np.array([np.load(filename, allow_pickle=True) for filename in projectionlist])

	if bootstrap:
		bootidxs = np.random.choice(len(ras), len(ras))
		ras, decs, weights, prob_weights = ras[bootidxs], decs[bootidxs], weights[bootidxs], prob_weights[bootidxs]
	# center longitudes/latitudes of projections
	projlons = [int(filename.split('/')[1].split('.')[0].split('_')[0]) for filename in projectionlist]
	projlats = [int(filename.split('/')[1].split('.')[0].split('_')[1]) for filename in projectionlist]
	# healpy projection objects used to create the projections
	# contains methods to convert from angular position of quasar to i,j position in projection
	projector_objects = [hp.projector.AzimuthalProj(rot=[projlons[i], projlats[i]], xsize=5000, reso=1.5, lamb=True) for i in range(len(projlons))]
	# convert ras and decs to galactic ls, bs
	lon, lat = equatorial_to_galactic(ras, decs)

	running_sum, weightsum = np.zeros((imsize, imsize)), np.zeros((imsize, imsize))
	# for each source
	for k in range(nstack):
		# choose the projection closest to the QSO's position
		cutoutidx = find_closest_cutout(lon[k], lat[k], projlons, projlats)
		cutout_to_use = projections[cutoutidx]
		projobj = projector_objects[cutoutidx]
		# find the i,j coordinates in the projection corresponding to angular positon in sky
		i, j = projobj.xy2ij(projobj.ang2xy(lon[k], lat[k], lonlat=True))
		# make cutout
		cut_at_position = cutout_to_use[int(i-imsize/2):int(i+imsize/2), int(j-imsize/2):int(j+imsize/2)]
		# stack
		running_sum, weightsum = stack_iteration(running_sum, weightsum, cut_at_position, weights[j], prob_weights[j], imsize)
	finalstack = running_sum/weightsum
	if outname is not None:
		finalstack.dump('%s.npy' % outname)

	return finalstack


def sum_projections(lon, lat, weights, prob_weights, imsize, reso, inmap, nstack):
	running_sum, weightsum = np.zeros((imsize, imsize)), np.zeros((imsize, imsize))
	for j in range(nstack):
		azproj = hp.projector.AzimuthalProj(rot=[lon[j], lat[j], np.random.randint(0, 360)], xsize=imsize, reso=reso, lamb=True)
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


# stack by computing an average iteratively. this method uses little memory but cannot be parallelized
def stack_projections(ras, decs, weights=None, prob_weights=None, imsize=240, outname=None, reso=1.5, inmap=None, nstack=None, mode='normal', chunksize=500):
	# if no weights provided, weights set to one
	if weights is None:
		weights = np.ones(len(ras))
	if prob_weights is None:
		prob_weights = np.ones(len(ras))
	# if no limit to number of stacks provided, stack the entire set
	if nstack is None:
		nstack = len(ras)

	lons, lats = equatorial_to_galactic(ras, decs)

	if mode == 'normal':
		totsum, weightsum = sum_projections(lons, lats, weights, prob_weights, imsize, reso, inmap, nstack)
		finalstack = totsum/weightsum
	elif mode == 'mp':
		# the number of chunks is the number of stacks divided by the chunk size rounded up to the nearest integer
		nchunks = ceil(nstack / chunksize)


		# initalize multiprocessing pool with one less core than is available for stability
		p = mp.Pool(5)
		# fill in all arguments to stack_chunk function but the index,
		# Pool.map() requires function to only take one paramter
		stack_chunk_partial = partial(stack_chunk, chunksize, nstack, lons, lats, inmap, weights, prob_weights,
		                              imsize, reso)
		# do the stacking in chunks, map the stacks to different cores for parallel processing
		chunksum, chunkweightsum = zip(*p.map(stack_chunk_partial, range(nchunks)))

		totsum = np.sum(chunksum, axis=0)
		weightsum = np.sum(chunkweightsum, axis=0)
		finalstack = totsum / weightsum

		p.close()
		p.join()
	else:
		return

	if outname is not None:
		finalstack.dump('%s.npy' % outname)

	return finalstack



def fast_stack(ras, decs, inmap, weights=None, prob_weights=None, nsides=2048, iterations=500):
	if weights is None:
		weights = np.ones(len(ras))

	outerkappa = []
	lons, lats = equatorial_to_galactic(ras, decs)
	inmap = convergence_map.set_unseen_to_nan(inmap)
	centerkappa = inmap[hp.ang2pix(nsides, lons, lats, lonlat=True)]
	weights[np.isnan(centerkappa)] = 0
	centerkappa[np.isnan(centerkappa)] = 0


	if prob_weights is not None:
		true_weights_for_sum = weights * np.array(prob_weights)
		weightsum = np.sum(true_weights_for_sum)
	else:
		weightsum = np.sum(weights)

	centerstack = np.sum(weights * centerkappa) / weightsum

	if iterations > 0:
		for x in range(iterations):
			outerkappa.append(np.nanmean(inmap[hp.ang2pix(nsides, (lons + np.random.uniform(2., 14.)), lats, lonlat=True)]))

		return centerstack, np.nanstd(outerkappa)
	else:
		return centerstack


# if running on local machine, can use this to stack using multiprocessing.
# Otherwise use stack_mpi.py to perform stack on a cluster computer
def stack_suite(color, sample_name, stack_map, stack_noise, reso=1.5, imsize=240, nsides=2048, mode='normal', nstack=None, bootstrap=False, temperature=False, random=False):

	if temperature:
		planck_map = hp.read_map('maps/smica_masked.fits')
		outname = 'stacks/%s_%s_temp' % (sample_name, color)
	else:
		planck_map = hp.read_map('maps/smoothed_masked_planck.fits', dtype=np.single)
		outname = 'stacks/%s_%s_stack' % (sample_name, color)

	cat = fits.open('catalogs/derived/%s_%s.fits' % (sample_name, color))[1].data
	if (sample_name == 'xdqso') or (sample_name == 'xdqso_specz'):
		probs = cat['PQSO']
	else:
		probs = np.ones(len(cat))
	ras, decs = cat['RA'], cat['DEC']

	if nstack is None:
		nstack = len(ras)

	if (color == 'complete') or (color == 'RL'):
		weights = np.ones(len(cat))
	else:
		weights = cat['weight']

	if random:
		ras = ras + np.random.uniform(2., 14., len(ras))
		outname = 'stacks/random_stack'



	if mode == 'fast':
		return fast_stack(ras, decs, planck_map, weights=weights, prob_weights=probs, nsides=nsides)
	elif mode == 'cutout':
		return stack_cutouts(ras, decs, weights, probs, imsize, nstack, outname, bootstrap)
	elif mode == 'mpi':
		for line in fileinput.input(['stack_mpi.py'], inplace=True):
			if line.strip().startswith('sample_name = '):
				line = "\tsample_name = '%s'\n" % sample_name
			if line.strip().startswith('color = '):
				line = "\tcolor = '%s'\n" % color
			if line.strip().startswith('imsize = '):
				line = '\timsize = %s\n' % imsize
			if line.strip().startswith('reso = '):
				line = '\treso = %s\n' % reso
			if line.strip().startswith('stack_maps = '):
				line = '\tstack_maps = %s\n' % stack_map
			if line.strip().startswith('stack_noise = '):
				line = '\tstack_noise = %s\n' % stack_noise
			sys.stdout.write(line)
	else:
		if stack_map:
			stack_projections(ras, decs, weights=weights, prob_weights=probs, inmap=planck_map, mode=mode,
			         outname=outname, reso=reso, imsize=imsize, nstack=nstack)

		if stack_noise:
			noisemaplist = glob.glob('noisemaps/maps/*.fits')
			for j in range(0, len(noisemaplist)):
				noisemap = hp.read_map('noisemaps/maps/%s.fits' % j)
				stack_projections(ras, decs, weights=weights, prob_weights=probs, inmap=noisemap, mode=mode,
				         outname='noise_stacks/%s_%s' % (j, color), reso=reso, imsize=imsize, nstack=nstack)




