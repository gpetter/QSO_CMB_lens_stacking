import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import importlib
import spectrumtools
import weighting
importlib.reload(weighting)
importlib.reload(spectrumtools)

def bin_by_color(colors, zs, nbins, offset, rgb=None, nzbins=30):
	if rgb is not None:
		nbins = 20
	# make empty list of lists
	gminusibinidxs = []
	for k in range(nbins):
		gminusibinidxs.append([])

	# choose to bin up by color or by color offset
	if offset:
		offsetbins = pd.qcut(colors, nbins, retbins=True)[1]
		for j in range(nbins):
			gminusibinidxs[j] += list(
				np.where((colors < offsetbins[j + 1]) & (colors >= offsetbins[j]))[0])
	else:

		z_quantile_bins = pd.qcut(zs, nzbins, retbins=True)[1]
		z_bin_idxs = np.digitize(zs, z_quantile_bins) - 1

		# loop over redshift bins
		for i in range(len(z_quantile_bins) - 1):
			in_z_bin = (zs <= z_quantile_bins[i + 1]) & (zs >= z_quantile_bins[i])
			idxs_in_bin = np.where(in_z_bin)
			gminusi_in_bin = colors[idxs_in_bin]

			color_bins = pd.qcut(gminusi_in_bin, nbins, retbins=True)[1]
			for j in range(nbins):
				gminusibinidxs[j] += list(
					np.where((colors <= color_bins[j + 1]) & (colors >= color_bins[j]) & in_z_bin)[0])

	if rgb == 'b':
		return np.array(sum(gminusibinidxs[:3], []))
	elif rgb == 'c':
		return np.array(sum(gminusibinidxs[5:15], []))
	elif rgb == 'r':
		return np.array(sum(gminusibinidxs[17:], []))
	else:
		return gminusibinidxs

def bin_by_Av(tab, nbins):
	Avbinidxs = []
	for k in range(nbins):
		Avbinidxs.append([])
	npositivebins = int(nbins*3/4)
	nnegativebins = nbins - npositivebins
	negAvs = np.sort(-np.logspace(-2, -1, nnegativebins))
	posAvs = np.logspace(-2, 0, npositivebins)


	rel_colors = tab['deltagmini']

	Av_vals = np.concatenate([negAvs, posAvs])

	zs = tab['Z']
	zspace = np.linspace(np.min(zs), np.max(zs), 30)
	Av_fits = []
	for i in range(len(Av_vals)):
		Av_curve = []
		for j in range(len(zspace)):
			Av_curve.append(spectrumtools.relative_vdb_color(z=zspace[j], reddening=Av_vals[i], mode='Av'))
		fit = np.polyfit(zspace, Av_curve, 10)
		Av_fits.append(fit)

	for i in range(nbins):
		if i == 0:
			Avbinidxs[i] += list(
				np.where((rel_colors < np.polyval(Av_fits[i], zs)))[0])
		else:
			prevtrack = np.polyval(Av_fits[i-1], zs)
			Avbinidxs[i] += list(
				np.where((rel_colors < np.polyval(Av_fits[i], zs)) & (rel_colors >= prevtrack))[0])
	return Avbinidxs



def bin_by_bal():
	cat = fits.open('catalogs/derived/dr16_ok.fits')[1].data
	balcat = Table(cat[np.where(cat['BAL_PROB'] > -1)])
	prob_thresh = 0.5
	balidxs = np.where(balcat['BAL_PROB'] >= prob_thresh)
	nonbalidxs = np.where((balcat['BAL_PROB'] >= 0) & (balcat['BAL_PROB'] < 0.5))

	balcat['weight'] = np.zeros(len(balcat))

	balcat['weight'] = weighting.redshift_weights([balidxs, nonbalidxs], len(balcat),
	                                              [balcat['Z'][balidxs], balcat['Z'][nonbalidxs]],
	                                              20, np.min(balcat['Z']), np.max(balcat['Z']))

	balcat['bin'] = -1*np.ones(len(balcat))
	balcat['bin'][balidxs] = 1
	balcat['bin'][nonbalidxs] = 2

	balcat.write('catalogs/derived/dr16_bal.fits', format='fits', overwrite=True)



def bin_by_bh_mass(n_massbins, minz, maxz):
	cat = fits.open('catalogs/SDSS_QSOs/dr14q_spec_prop.fits')[1].data

	bhcat = Table(cat[np.where((cat['QUALITY_MBH'] == 0) & (cat['LOG_MBH_ERR'] < 0.3) & (cat['LOG_MBH'] > 7) & (cat['REDSHIFT'] <= maxz) & (cat['REDSHIFT'] >= minz))])
	mbhs = bhcat['LOG_MBH']

	massbins = pd.qcut(mbhs, n_massbins, retbins=True)[1]

	bhcat['bin'] = np.digitize(mbhs, massbins)
	indexarr = [np.where(bhcat['bin'] == j+1) for j in range(n_massbins)]
	zsarr = []
	for j in range(n_massbins):
		zsarr.append(bhcat['REDSHIFT'][indexarr[j]])

	bhcat['weight'] = weighting.redshift_weights(indexarr, len(bhcat), zsarr, 20, np.min(bhcat['REDSHIFT']), np.max(bhcat['REDSHIFT']))

	bhcat.write('catalogs/derived/bhmass/dr14_mass_binned.fits', format='fits', overwrite=True)
