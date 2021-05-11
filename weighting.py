
import numpy as np
from scipy import stats



# calculate weights to control for quasar luminosity
# weight each sample down to the minimum of all samples
def lum_weights(lums_arr, minlum, maxlum, bins, bin=0):

	# the luminosities of the color in question
	lumset = lums_arr[bin]

	# calculate luminosity histograms for all samples
	hists = []
	for i in range(len(lums_arr)):
		thishist = np.histogram(lums_arr[i], bins=bins, range=(minlum, maxlum), density=True)
		hists.append(thishist[0])
		if i == 0:
			bin_edges = thishist[1]
	hists = np.array(hists)
	# the minimum in each bin
	min_in_bins = np.amin(hists, axis=0)

	# weight down to the minimum in each bin
	dist_ratio = min_in_bins / hists[bin]

	# set nans to zero
	dist_ratio[np.where(np.isnan(dist_ratio) | np.isinf(dist_ratio))] = 0


	bin_idxs = np.digitize(lumset, bin_edges) - 1
	weights = dist_ratio[bin_idxs]


	return weights


# apply weights to quasars to weight down to approximately the same redshift distribution
# indices -- 2d array, each index is an array containing the indices of a subsample of quasars
# zs_arr -- 2d array, each index is an array containing the redshifts of a subsample of quasars
# nobjects
def redshift_weights(indices, nobjects, zs_arr, bins, minz, maxz):

	hists, bin_locs = [], []

	for i in range(len(zs_arr)):
		# calculate normalized redshift distribution
		thishist = np.histogram(zs_arr[i], bins=bins, range=(minz-0.001, maxz+0.001), density=True)
		# add to list
		hists.append(thishist[0])
		# save edges of bins to later find objects within them
		if i == 0:
			bin_edges = thishist[1]
	# find minimum redshift distribution across all samples
	hists = np.array(hists)
	min_in_bins = np.amin(hists, axis=0)
	# initialize weights
	weights = np.zeros(nobjects)

	# for each subsample
	for j in range(len(zs_arr)):
		# ratio of minimum redshift distribution to the subsample's redshift distribution
		# these are the weights
		dist_ratio = min_in_bins/hists[j]
		# set divisions by zero to zero
		dist_ratio[np.where(np.isnan(dist_ratio) | np.isinf(dist_ratio))] = 0
		# find which redshift bin each object belongs in
		bin_idxs = np.digitize(zs_arr[j], bin_edges) - 1
		# apply weights to objects
		weights[indices[j]] = dist_ratio[bin_idxs]

	return weights

# weight across luminosity and redshift simultaenously
def lum_z_2d_weights(indices, number, lums_arr, zs_arr, minlum, maxlum, minz, maxz, nlumbins, nzbins):
	hists, bin_locs = [], []

	for j in range(len(lums_arr)):
		thishist = stats.binned_statistic_2d(zs_arr[j], lums_arr[j], None, statistic='count', bins=[nzbins, nlumbins], range=[[minz, maxz], [minlum, maxlum]], expand_binnumbers=True)
		normed_hist = thishist[0]/np.sum(thishist[0])
		hists.append(normed_hist)
		bin_locs.append(np.array(thishist[3]) - 1)

	hists = np.array(hists)

	min_hist = np.amin(hists, axis=0)

	weights = np.zeros(number)

	for j in range(len(lums_arr)):
		dist_ratio = min_hist / hists[j]
		# set nans to zero
		dist_ratio[np.where(np.isnan(dist_ratio) | np.isinf(dist_ratio))] = 0

		bin_idxs = bin_locs[j]
		weights[indices[j]] = dist_ratio[bin_idxs[0], bin_idxs[1]]

	return weights


# multiply luminosity weights by probability weights
def convolved_weights(pqso_weights, number, indices, lum_arr, zs_arr, minlum, maxlum, minz, maxz, nlumbins, nzbins):

	l_weights = lum_z_2d_weights(indices, number, lum_arr, zs_arr, minlum, maxlum, minz, maxz, nlumbins, nzbins)

	totweights = pqso_weights * l_weights

	return totweights
