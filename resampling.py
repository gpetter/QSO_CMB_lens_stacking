import numpy as np
import healpy as hp
import healpixhelper
import importlib
from scipy import stats
from astropy.io import fits
from astropy.table import Table
importlib.reload(healpixhelper)


def montecarlo_spearman(xs, ys, yerrs):

	#realizations = np.random.normal(loc=ys, scale=yerrs, size=(1000, len(ys)))
	#spearmanranks = stats.spearmanr(xs, realizations, axis=1)[0]

	spearmanranks = []
	for j in range(1000):
		realization = np.random.normal(ys, yerrs)
		spearmanranks.append(stats.spearmanr(xs, realization)[0])
	return np.mean(spearmanranks), np.std(spearmanranks)




def bin_on_sky(ras, decs, njackknives, nside=None):
	test_nsides = np.linspace(1, 50, 100)
	area_as_func_of_nside = hp.nside2pixarea(test_nsides, degrees=True)

	lowresdensity = healpixhelper.healpix_density_map(ras, decs, 32)
	skyfraction = len(lowresdensity[np.where(lowresdensity > 0)])/len(lowresdensity)

	footprintarea = 41252.96 * skyfraction  # deg^2

	approx_area_per_pixel = footprintarea / njackknives

	nside_for_area = int(np.interp(approx_area_per_pixel, np.flip(area_as_func_of_nside), np.flip(test_nsides)))

	if nside is not None:
		nside_for_area = nside

	pixidxs = hp.ang2pix(nside_for_area, ras, decs, lonlat=True)

	return pixidxs, nside_for_area


def bootstrap_sky_bins(ras, decs, refras, refdecs, randras, randdecs, njackknives):

	# divide sample up into subvolumes on sky
	refpixidxs, nside = bin_on_sky(refras, refdecs, njackknives)
	pixidxs, foo = bin_on_sky(ras, decs, njackknives, nside=nside)

	randpixidxs, randnside = bin_on_sky(randras, randdecs, njackknives, nside=nside)

	# Norberg+2009
	nsub_factor = 3

	# list of subvolume indices is the set of unique values of above array
	unique_pix = np.unique(refpixidxs)

	# bootstrap resample the subvolumes, oversample by factor 3
	boot_pix = np.random.choice(unique_pix, nsub_factor*len(unique_pix))

	idxs, refidxs, randidxs = [], [], []

	# for each subvolume
	for pixel in boot_pix:
		# find which objects lie in subvolume
		idxs_in_pixel = np.where(pixidxs == pixel)[0]
		idxs += list(idxs_in_pixel)

		refidxs += list(np.where(refpixidxs == pixel)[0])

		# add randoms in subvolume to list
		randidxs += list(np.where(randpixidxs == pixel)[0])

		# bootstrap resample these objects
		#booted_idxs_in_pixel = np.random.choice(idxs_in_pixel, len(idxs_in_pixel))
		# add bootstrapped objects to list
		#idxs += list(booted_idxs_in_pixel)


	# subsample
	#final_idxs = idxs
	final_idxs = np.random.choice(idxs, len(ras))
	final_refidxs = np.random.choice(refidxs, len(refras))
	final_randidxs = np.random.choice(randidxs, len(randras), replace=False)

	return final_idxs, final_refidxs, final_randidxs



def covariance_matrix(resampled_profiles, avg_profile):
	n_bins = len(resampled_profiles[0])
	n_realizations = len(resampled_profiles)
	c_ij = np.zeros((n_bins, n_bins))
	for i in range(n_bins):
		for j in range(n_bins):
			k_i = resampled_profiles[:, i]
			k_i_bar = avg_profile[i]

			k_j = resampled_profiles[:, j]
			k_j_bar = avg_profile[j]

			product = (k_i - k_i_bar) * (k_j - k_j_bar)
			sum = np.sum(product)
			c_ij[i, j] = 1 / (n_realizations - 1) * sum

	return c_ij