
import numpy as np
from scipy import stats



# calculate weights to control for quasar luminosity
# weight each sample down to the minimum of all samples
def lum_weights(lums_arr, minlum, maxlum, bins, colorbin=0):

	# the luminosities of the color in question
	lumset = lums_arr[colorbin]

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
	dist_ratio = min_in_bins / hists[colorbin]

	# set nans to zero
	dist_ratio[np.where(np.isnan(dist_ratio) | np.isinf(dist_ratio))] = 0


	bin_idxs = np.digitize(lumset, bin_edges) - 1
	weights = dist_ratio[bin_idxs]


	return weights



def redshift_weights(zs_arr, colorbin, bins):
	# the luminosities of the color in question
	these_zs = zs_arr[colorbin]

	# calculate luminosity histograms for all samples
	hists = []
	for i in range(len(zs_arr)):
		thishist = np.histogram(zs_arr[i], bins=bins, range=(0, 8), density=True)
		hists.append(thishist[0])
		if i == 0:
			bin_edges = thishist[1]
	hists = np.array(hists)
	# the minimum in each bin
	min_in_bins = np.amin(hists, axis=0)

	# weight down to the minimum in each bin
	dist_ratio = min_in_bins / hists[colorbin]

	# set nans to zero
	dist_ratio[np.where(np.isnan(dist_ratio) | np.isinf(dist_ratio))] = 0

	bin_idxs = np.digitize(these_zs, bin_edges) - 1
	weights = dist_ratio[bin_idxs]

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

	#l_weights = lum_weights(lum_arr, minlum, maxlum, bins, colorbin=colorbin)
	l_weights = lum_z_2d_weights(indices, number, lum_arr, zs_arr, minlum, maxlum, minz, maxz, nlumbins, nzbins)

	totweights = pqso_weights * l_weights

	return totweights
