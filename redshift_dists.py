from astropy.io import fits
import numpy as np


def redshift_dist(zs, bin_avgs=True):

	# bin up redshift distribution of sample to integrate kappa over
	hist = np.histogram(zs, 200, density=True)
	zbins = hist[1]
	if bin_avgs:
		dz = zbins[1] - zbins[0]
		# chop off last entry which is a rightmost bound of the z distribution, find center of bins by adding dz/2
		zbins = np.resize(zbins, zbins.size - 1) + dz / 2

	dndz = hist[0]

	return zbins, dndz


def redshift_overlap(zs1, zs2, bin_avgs=True):
	# find mininum and maximum redshift across two samples
	minzs1, maxzs1 = np.min(zs1), np.max(zs1)
	minzs2, maxzs2 = np.min(zs2), np.max(zs2)
	totmin, totmax = np.min([minzs1, minzs2]) - 0.0001, np.max([maxzs1, maxzs2]) + 0.0001

	# get z distributions for both samples across same grid
	hist1 = np.histogram(zs1, 200, density=True, range=[totmin, totmax])
	hist2 = np.histogram(zs2, 200, density=True, range=[totmin, totmax])

	# z values in each bin
	zbins = hist1[1]
	if bin_avgs:
		dz = zbins[1] - zbins[0]
		# chop off last entry which is a rightmost bound of the z distribution, find center of bins by adding dz/2
		zbins = np.resize(zbins, zbins.size - 1) + dz / 2

	# redshift distribution overlap is sqrt of product of two distributions
	dndz = np.sqrt(hist1[0] * hist2[0])

	return zbins, dndz


