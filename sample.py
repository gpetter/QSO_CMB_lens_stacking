from astropy.io import fits
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
import healpy as hp
from colossus.cosmology import cosmology
import stacking
import plotting
import importlib
from astropy.table import Table
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import spectrumtools
import wise_tools
importlib.reload(wise_tools)
importlib.reload(spectrumtools)
importlib.reload(stacking)
importlib.reload(plotting)

cosmo = cosmology.setCosmology('planck18')
astropycosmo = cosmo.toAstropy()


band_idxs = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4}
bsoftpars = [1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10]


# find which sources in a sample satisfy a certain bitmask parameter
# i.e., eBOSS CORE quasars have
def find_bitmask_matches(column, bitidx, bitvalue=1):
	maxbitmask = np.max(column)
	maxbinarybit = f'{maxbitmask:b}'
	bitmasklength = len(str(maxbinarybit))
	valueidxs = []
	binstring = '0%sb' % bitmasklength
	for j in range(len(column)):

		binarymask = format(column[j], binstring)[::-1]
		if binarymask[bitidx] == '%s' % bitvalue:
			valueidxs.append(j)
	return list(valueidxs)


# given dereddened i mags and redshifts, k correct to absolute i band magnitude Mi(z=2) a la Richards et al 2003
def k_correct_richards(dered_i_mags, zs):
	# read table
	kcorrecttab = pd.read_csv('kcorrect/table4.dat', names=['z', 'k'], delim_whitespace=True)
	# round redshift data to match to table
	roundzs = np.around(zs, 2)
	# get indices of matches
	k_idxs = np.where(roundzs.reshape(roundzs.size, 1) == np.array(kcorrecttab['z']))[1]
	# array of k correction for each QSO
	kcorrects = np.array(kcorrecttab['k'][k_idxs])
	# luminosity distance
	distances = astropycosmo.luminosity_distance(zs).to(u.pc)

	# absolute magnitude, k corrected
	i_abs_mags = dered_i_mags - 5 * np.log10(distances / (10 * u.pc)) - kcorrects

	return i_abs_mags


# remove sources from DR16 with bad redshifts
def fix_dr16():
	dr16 = Table(fits.open('catalogs/SDSS_QSOs/DR16Q_v4.fits')[1].data)
	dr16good = dr16[np.where(dr16['Z'] > 0)]
	dr16good.write('catalogs/derived/dr16_ok.fits', format='fits', overwrite=True)


def define_core_sample(sample):
	if sample == 'dr16':
		dr16 = Table(fits.open('catalogs/derived/dr16_ok.fits')[1].data)

		coreidxs = list(find_bitmask_matches(dr16['EBOSS_TARGET0'], 10)) + list(find_bitmask_matches(dr16['EBOSS_TARGET1'], 10))
		#coreidxs = list(find_bitmask_matches(dr16['EBOSS_TARGET1'], 10))
		#bosscoreidxs = list(find_bitmask_matches(dr16['BOSS_TARGET1'], 40))
		#coreidxs = coreidxs + bosscoreidxs
		coresample = dr16[coreidxs]
		coresample.write('catalogs/derived/dr16_core.fits', format='fits', overwrite=True)

	elif sample == 'dr16_superset':
		superset = Table(fits.open('catalogs/SDSS_QSOs/DR16Q_Superset_v3.fits')[1].data)
		coreidxs = list(find_bitmask_matches(superset['EBOSS_TARGET0'], 10)) + list(
			find_bitmask_matches(superset['EBOSS_TARGET1'], 10))
		bosscoreidxs = list(find_bitmask_matches(superset['BOSS_TARGET1'], 40))
		coreidxs = coreidxs + bosscoreidxs
		coresample = superset[coreidxs]
		coresample.write('catalogs/derived/dr16_superset_CORE.fits', format='fits', overwrite=True)

	elif sample == 'dr14':
		dr14 = Table(fits.open('catalogs/SDSS_QSOs/DR14Q_v4_4.fits')[1].data)

		coreidxs = list(find_bitmask_matches(dr14['EBOSS_TARGET0'], 10)) + list(
			find_bitmask_matches(dr14['EBOSS_TARGET1'], 10))

		coresample = dr14[coreidxs]
		coresample.write('catalogs/derived/dr14_core.fits', format='fits', overwrite=True)


def match_phot_qsos_to_spec_qsos(update_probs=True, core_only=False):
	dr16 = Table(fits.open('catalogs/derived/dr16_ok.fits')[1].data)
	speccoords = SkyCoord(ra=dr16['RA']*u.degree, dec=dr16['DEC']*u.degree)
	xdqso = Table(fits.open('catalogs/SDSS_QSOs/xdqso-z-cat.fits')[1].data)
	photcoords = SkyCoord(ra=xdqso['RA']*u.degree, dec=xdqso['DEC']*u.degree)

	# match XDQSO catalog with DR16 spectroscopic catalog
	xdidx, specidx, d2d, d3d = speccoords.search_around_sky(photcoords, 1*u.arcsec)
	# rip spectroscopic redshifts from DR16 to replace photo-zs from XDQSO where matched
	xdqso['Z'][xdidx] = dr16['Z'][specidx]
	xdqso['NPEAKS'][xdidx] = np.ones(len(specidx))
	xdqso['PEAKPROB'][xdidx] = np.ones(len(specidx))
	xdqso['PEAKFWHM'][xdidx] = np.zeros(len(specidx))
	xdqso['OTHERZ'][xdidx] = np.zeros((len(specidx), 6))
	xdqso['OTHERPROB'][xdidx] = np.zeros((len(specidx), 6))
	xdqso['OTHERFWHM'][xdidx] = np.zeros((len(specidx), 6))

	# optionally update probabilities based on subsequent spectroscopic observations up to DR16
	if update_probs:
		if core_only:
			# update probabilities of QSOs to 1 given a match to spectroscopic confirmation in CORE sample
			dr16core = Table(fits.open('catalogs/derived/dr16_core.fits')[1].data)
			corecoords = SkyCoord(ra=dr16core['RA']*u.degree, dec=dr16core['DEC']*u.degree)
			xdcoreidxs, speccoreidxs, d2d, d3d = corecoords.search_around_sky(photcoords, 1*u.arcsec)
			xdqso['PQSO'][xdcoreidxs] = np.ones(len(speccoreidxs))

			# update probabilites to 0 if selected in CORE sample but spectroscopically ruled out
			dr16_superset = Table(fits.open('catalogs/derived/dr16_superset_CORE.fits')[1].data)
			superset_non_QSOs = dr16_superset[np.where(dr16_superset['IS_QSO_FINAL'] < 1)]
			supercoords = SkyCoord(ra=superset_non_QSOs['RA']*u.deg, dec=superset_non_QSOs['DEC']*u.deg)
			xdidxs, nonidxs, d2d, d3d = supercoords.search_around_sky(photcoords, 1*u.arcsec)
			xdqso['PQSO'][xdidxs] = np.zeros(len(nonidxs))

		else:
			xdqso['PQSO'][xdidx] = np.ones(len(specidx))

			dr16_superset = Table(fits.open('catalogs/SDSS_QSOs/DR16Q_Superset_v3.fits')[1].data)
			superset_non_QSOs = dr16_superset[np.where(dr16_superset['IS_QSO_FINAL'] < 1)]
			supercoords = SkyCoord(ra=superset_non_QSOs['RA'] * u.deg, dec=superset_non_QSOs['DEC'] * u.deg)
			nonxdidxs, nonspecidxs, d2d, d3d = supercoords.search_around_sky(photcoords, 1*u.arcsec)

			xdqso['PQSO'][nonxdidxs] = np.zeros(len(nonspecidxs))

		xdqso = xdqso[np.where(xdqso['PQSO'] > 0.2)]



	xdqso.write('catalogs/derived/xdqso_specz.fits', format='fits', overwrite=True)


# calculate aboslute I band magnitude, g - i colors, and 1.5 micron Luminosities, writing values to file
def write_properties(sample, speczs=False):
	if sample == 'dr14':

		qso_cat = (fits.open('catalogs/SDSS_QSOs/DR14Q_v4_4.fits'))[1].data
		rakey = 'RA'
		deckey = 'DEC'
		zkey = 'Z'
		w1key = 'W1MAG'
		w2key = 'W2MAG'
		extkey = 'GAL_EXT'
		i_mags = qso_cat['psfmag'][:, band_idxs['i']]
		goodzs = ((qso_cat[zkey] > 0) & (qso_cat[zkey] < 5.494))

		extinction = qso_cat['GAL_EXT'][:, band_idxs['i']]
		extinction[np.where(extinction < 0)] = 0
		dered_i_mags = i_mags - extinction


		good_idxs = np.where((qso_cat['FIRST_MATCHED'] < 1))

	elif sample == 'dr16':
		qso_cat = Table((fits.open('catalogs/derived/dr16_core.fits'))[1].data)


		goodzs = ((qso_cat['Z'] > 0) & (qso_cat['Z'] < 5.494))
		goodmags = (qso_cat['PSFMAG'][:, band_idxs['i']] != -9999.) & (qso_cat['PSFMAG'][:, band_idxs['g']] != -9999.)
		goodextinct = (qso_cat['EXTINCTION'][:, band_idxs['i']] > 0) & (qso_cat['EXTINCTION'][:, band_idxs['g']] > 0)
		goodwise = (qso_cat['W1_FLUX'] > 0) & (qso_cat['W2_FLUX'] > 0)


		good_idxs = np.where(goodmags & goodzs & goodwise & goodextinct)

		trimmed_cat = qso_cat[good_idxs]

		dered_i_mags = trimmed_cat['PSFMAG'][:, band_idxs['i']] - trimmed_cat['EXTINCTION'][:, band_idxs['i']]
		dered_g_mags = trimmed_cat['PSFMAG'][:, band_idxs['g']] - trimmed_cat['EXTINCTION'][:, band_idxs['g']]


		i_abs_mags = k_correct_richards(dered_i_mags, trimmed_cat['Z'])
		trimmed_cat['myMI'] = i_abs_mags
		trimmed_cat['g-i'] = (dered_g_mags - dered_i_mags)
		trimmed_cat['logL1.5'] = wise_tools.rest_ir_lum(trimmed_cat['W1_FLUX'], trimmed_cat['W2_FLUX'], trimmed_cat['Z'], 1.5)
		trimmed_cat['i_mag'] = dered_i_mags
		trimmed_cat.write('catalogs/derived/dr16_new.fits', format='fits', overwrite=True)

	elif sample == 'eboss_lss':
		qso_cat = Table(fits.open('catalogs/lss/eBOSS_fullsky_phot.fits')[1].data)
		goodwise = (qso_cat['W1_FLUX'] > 0) & (qso_cat['W2_FLUX'] > 0)
		trimmed_cat = qso_cat[np.where(goodwise)]
		dered_i_mags = trimmed_cat['PSFMAG'][:, band_idxs['i']] - trimmed_cat['EXTINCTION'][:, band_idxs['i']]
		dered_g_mags = trimmed_cat['PSFMAG'][:, band_idxs['g']] - trimmed_cat['EXTINCTION'][:, band_idxs['g']]



		trimmed_cat['myMI'] = k_correct_richards(dered_i_mags, trimmed_cat['Z'])
		trimmed_cat['g-i'] = dered_g_mags - dered_i_mags
		trimmed_cat['logL1.5'] = wise_tools.rest_ir_lum(trimmed_cat['W1_FLUX'], trimmed_cat['W2_FLUX'], trimmed_cat['Z'], 1.5)
		trimmed_cat['i_mag'] = dered_i_mags
		trimmed_cat.write('catalogs/derived/eBOSS_fullsky_new.fits', format='fits', overwrite=True)

	# write properties to an XDQSO photometric catalog
	else:
		if speczs:
			qso_cat = Table(fits.open('catalogs/derived/xdqso_specz.fits')[1].data)
		else:
			qso_cat = Table(fits.open('catalogs/derived/xdqso-z-cat.fits')[1].data)

		zkey = 'Z'
		goodzs = ((qso_cat[zkey] > 0) & (qso_cat[zkey] < 5.494))
		goodprobs = (qso_cat['PQSO'] > 0.8)
		goodwise = (qso_cat['PSFFLUX'][:, 11] > 0) & (qso_cat['PSFFLUX'][:, 12] > 0)

		"""good_idxs = np.where((qso_cat['NPEAKS'] == 1) & (qso_cat['bright_star'] == False) &
			(qso_cat['bad_u'] == False) & (qso_cat['bad_field'] == False) &
			(qso_cat['wise_flagged'] == False) &
			(qso_cat['good'] == 0) & outsidemaskbool)[0]"""
		#good_idxs = np.where(outsidemaskbool & goodwise & goodzs & goodprobs)[0]
		good_idxs = np.where(goodwise & goodzs & goodprobs)[0]

		trimmed_cat = qso_cat[good_idxs]

		i_maggies = trimmed_cat['PSFFLUX'][:, band_idxs['i']] / 1e9
		i_mags = -2.5 / np.log(10.) * (
					np.arcsinh(i_maggies / (2 * bsoftpars[band_idxs['i']])) + np.log(bsoftpars[band_idxs['i']]))
		dered_i_mags = i_mags - trimmed_cat['EXTINCTION'][:, band_idxs['i']]

		g_maggies = trimmed_cat['PSFFLUX'][:, band_idxs['g']] / 1e9
		g_mags = -2.5 / np.log(10.) * (
					np.arcsinh(g_maggies / (2 * bsoftpars[band_idxs['g']])) + np.log(bsoftpars[band_idxs['g']]))
		dered_g_mags = g_mags - trimmed_cat['EXTINCTION'][:, band_idxs['g']]

		i_abs_mags = k_correct_richards(dered_i_mags, trimmed_cat[zkey])

		trimmed_cat['myMI'] = i_abs_mags
		trimmed_cat['g-i'] = (dered_g_mags - dered_i_mags)
		trimmed_cat['logL1.5'] = wise_tools.rest_ir_lum(trimmed_cat['PSFFLUX'][:, 11], trimmed_cat['PSFFLUX'][:, 12],
		                                                trimmed_cat[zkey], 1.5)
		trimmed_cat['W1_MAG'] = 22.5 - 2.5 * np.log10(trimmed_cat['PSFFLUX'][:, 11])
		trimmed_cat['W2_MAG'] = 22.5 - 2.5 * np.log10(trimmed_cat['PSFFLUX'][:, 12])
		trimmed_cat['i_mag'] = dered_i_mags

		vdb = False
		if vdb:
			compgi_binned = []
			zlist = np.linspace(0, 10, 1000)
			for j in range(len(zlist)):
				compgi_binned.append(spectrumtools.vdb_color_at_z(zlist[j]))
			interpcolors = np.interp(trimmed_cat[zkey], zlist, compgi_binned)
			trimmed_cat['deltagmini'] = trimmed_cat['g-i'] - interpcolors


		if speczs:
			trimmed_cat.write('catalogs/derived/xdqso_specz_new.fits', format='fits', overwrite=True)
		else:
			trimmed_cat.write('catalogs/derived/xdqso_new.fits', format='fits', overwrite=True)


def match_lss_dr16():
	dr16 = Table(fits.open('catalogs/SDSS_QSOs/DR16Q_v4.fits')[1].data)
	ngc = Table(fits.open('catalogs/lss/eBOSS_QSO_clustering_data-NGC-vDR16.fits')[1].data)
	photcoords = SkyCoord(ra=dr16['RA']*u.degree, dec=dr16['DEC']*u.degree)
	lsscoords = SkyCoord(ra=ngc['RA'] * u.degree, dec=ngc['DEC'] * u.degree)
	photidx, lssidx, d2d, d3d = lsscoords.search_around_sky(photcoords, 1 * u.arcsec)
	fintable = Table(ngc[lssidx])
	fintable['PSFMAG'] = dr16['PSFMAG'][photidx]
	fintable['W1_FLUX'] = dr16['W1_FLUX'][photidx]
	fintable['W2_FLUX'] = dr16['W2_FLUX'][photidx]
	fintable['EXTINCTION'] = dr16['EXTINCTION'][photidx]
	fintable.write('catalogs/lss/eBOSS_NGC_phot.fits', format='fits', overwrite=True)



def luminosity_complete_cut(qso_cat_name, lumcut, minz, maxz, plots, magcut=100, pcut=0.9, peakscut=1, apply_planck_mask=True):
	qso_cat = Table(fits.open('catalogs/derived/%s_new.fits' % qso_cat_name)[1].data)

	zkey = 'Z'

	# if a photometric catalog, make cuts on flags, probability, and number of redshift peaks
	if (qso_cat_name == 'xdqso') or (qso_cat_name == 'xdqso_specz'):
		qso_cat = qso_cat[np.where((qso_cat['GOOD'] == 0) & (qso_cat['PQSO'] >= pcut) & (qso_cat['NPEAKS'] <= peakscut) & (qso_cat['NPEAKS'] > 0))]

	# removes QSOs which fall inside mask for Planck lensing, which you can't use anyways
	if apply_planck_mask:
		map4stack = hp.read_map('maps/smoothed_masked_planck.fits', dtype=np.single)
		ls, bs = stacking.equatorial_to_galactic(qso_cat['RA'], qso_cat['DEC'])
		outsidemaskbool = (map4stack[hp.ang2pix(2048, ls, bs, lonlat=True)] != hp.UNSEEN)
		qso_cat = qso_cat[np.where(outsidemaskbool)]

	#
	i_app = qso_cat['i_mag']
	# cut on absolute or apparent magnitude, and/or redshift
	complete_cut = np.where((qso_cat['myMI'] <= lumcut) & (qso_cat['myMI'] > -100) & (qso_cat[zkey] >= minz) & (qso_cat[zkey] <= maxz) & (
					i_app < magcut))
	t = qso_cat[complete_cut]


	gminis = np.copy(t['g-i'])
	# bin redshift into 50 bins
	z_quantile_bins = pd.qcut(t[zkey], 50, retbins=True)[1]

	# loop over redshift bins
	for i in range(len(z_quantile_bins) - 1):
		in_z_bin = (t['Z'] < z_quantile_bins[i + 1]) & (t['Z'] >= z_quantile_bins[i])
		idxs_in_bin = np.where(in_z_bin)
		medcolor = np.median(gminis[idxs_in_bin])
		gminis[idxs_in_bin] = gminis[idxs_in_bin] - medcolor
	t['deltagmini'] = gminis



	if plots:
		# if using apparent magnitude cut, calculate what k-corrected absolute magnitude this corresponds as a function
		# of redshift, for plotting purposes
		if magcut < 100:
			kcorrecttab = pd.read_csv('kcorrect/table4.dat', names=['z', 'k'], delim_whitespace=True)
			zs = np.linspace(minz, maxz, 20)
			newdists = astropycosmo.luminosity_distance(zs).to(u.pc)

			roundzs = np.around(zs, 2)
			k_idxs = np.where(roundzs.reshape(roundzs.size, 1) == np.array(kcorrecttab['z']))[1]
			kcorrects = np.array(kcorrecttab['k'][k_idxs])

			limMs = magcut - 5 * np.log10(newdists / (10 * u.pc)) - kcorrects
		else:
			limMs = np.zeros(20)

		plotting.MI_vs_z(qso_cat[zkey], qso_cat['myMI'], len(complete_cut[0]), magcut, minz, maxz, lumcut, qso_cat_name, limMs)
		plotting.w1_minus_w2_plot(t['W1_MAG'], t['W2_MAG'], qso_cat_name)

	t.write('catalogs/derived/%s_complete.fits' % qso_cat_name, format='fits', overwrite=True)



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
	dist_ratio = min_in_bins / hists[0]
	# set nans to zero
	dist_ratio[np.where(np.isnan(dist_ratio) | np.isinf(dist_ratio))] = 0

	weights = np.zeros(len(lumset))

	for i in range(len(dist_ratio)):
		idxs_in_bin = np.where((lumset <= bin_edges[i + 1]) & (lumset > bin_edges[i]))
		weights[idxs_in_bin] = dist_ratio[i]

	return weights


# multiply luminosity weights by probability weights
def convolved_weights(pqso_weights, lum_arr, minlum, maxlum, bins, colorbin=0):

	l_weights = lum_weights(lum_arr, minlum, maxlum, bins, colorbin=colorbin)

	totweights = pqso_weights * l_weights

	return totweights


def remove_reddest_bin(colors, zs, nbins, offset):
	if offset:
		offsetbins = pd.qcut(colors, nbins, retbins=True)[1]
		return np.where(colors <= offsetbins[nbins - 1])
	else:
		z_quantile_bins = pd.qcut(zs, 20, retbins=True)[1]
		idcs = []
		# loop over redshift bins
		for i in range(len(z_quantile_bins) - 1):
			in_z_bin = (zs < z_quantile_bins[i + 1]) & (zs >= z_quantile_bins[i])
			idxs_in_bin = np.where(in_z_bin)
			gminusi_in_bin = colors[idxs_in_bin]

			color_bins = pd.qcut(gminusi_in_bin, nbins, retbins=True)[1]
			idcs += np.where((colors <= color_bins[nbins - 1]) & in_z_bin)[0].tolist()
		return np.array(idcs)



def bin_by_color(colors, zs, nbins, offset, rgb=None, nzbins=100):
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

		# loop over redshift bins
		for i in range(len(z_quantile_bins) - 1):
			in_z_bin = (zs < z_quantile_bins[i + 1]) & (zs >= z_quantile_bins[i])
			idxs_in_bin = np.where(in_z_bin)
			gminusi_in_bin = colors[idxs_in_bin]

			color_bins = pd.qcut(gminusi_in_bin, nbins, retbins=True)[1]
			for j in range(nbins):
				gminusibinidxs[j] += list(
					np.where((colors < color_bins[j + 1]) & (colors >= color_bins[j]) & in_z_bin)[0])
	if rgb == 'b':
		return np.array(sum(gminusibinidxs[:3], []))
	elif rgb == 'c':
		return np.array(sum(gminusibinidxs[5:15], []))
	elif rgb == 'r':
		return np.array(sum(gminusibinidxs[17:], []))
	else:
		return gminusibinidxs


def red_blue_samples(qso_cat_name, plots, offset=False, remove_reddest=False):


	if offset:
		colorkey = 'deltagmini'
	else:
		colorkey = 'g-i'
	if qso_cat_name == 'dr14':
		qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data

	elif qso_cat_name == 'dr16':

		qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data
	elif qso_cat_name == 'eboss_lss':
		qso_cat = fits.open('catalogs/derived/eboss_lss_complete.fits')[1].data

	else:
		qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data

	if remove_reddest:
		#qso_cat = qso_cat[remove_reddest_bin(qso_cat[colorkey], qso_cat['Z'], 10, offset)]
		redtailcat = Table(qso_cat[np.where(qso_cat['deltagmini'] >= 0.25)])
		redtailcat.write('catalogs/derived/%s_%s_redtail.fits', format='fits', overwrite=True)
		if plots:
			plotting.color_hist(qso_cat_name, offset=True)
		qso_cat = qso_cat[np.where(qso_cat['deltagmini'] < 0.25)]


	colors = qso_cat[colorkey]
	zs = qso_cat['Z']
	qso_tab = Table(qso_cat)
	qso_tab['colorbin'] = np.zeros(len(qso_tab))
	qso_tab['weight'] = np.zeros(len(qso_tab))

	indicesbycolor = bin_by_color(colors, zs, 10, offset)
	lumlist = []
	for j in range(len(indicesbycolor)):
		qso_tab['colorbin'][indicesbycolor[j]] = j+1
		lumlist.append(qso_tab['logL1.5'][indicesbycolor[j]])

	bluetab = qso_tab[np.where(qso_tab['colorbin'] == 1)]
	ctrltab = qso_tab[np.where((qso_tab['colorbin']==5) | (qso_tab['colorbin'] == 6))]
	redtab = qso_tab[np.where(qso_tab['colorbin'] == 10)]

	"""bluetab = Table(qso_cat[bin_by_color(colors, zs, 20, offset, rgb='b')])
	ctrltab = Table(qso_cat[bin_by_color(colors, zs, 20, offset, rgb='c')])
	redtab = Table(qso_cat[bin_by_color(colors, zs, 20, offset, rgb='r')])

	lumhistbins = 100

	if (qso_cat_name == 'dr14') or (qso_cat_name == 'dr16') or (qso_cat_name == 'eboss_lss'):
		bluetab['weight'] = lum_weights([bluetab['logL1.5'], ctrltab['logL1.5'], redtab['logL1.5']], 18, 28, lumhistbins)
		ctrltab['weight'] = lum_weights([ctrltab['logL1.5'], redtab['logL1.5'], bluetab['logL1.5']], 18, 28, lumhistbins)
		redtab['weight'] = lum_weights([redtab['logL1.5'], ctrltab['logL1.5'], bluetab['logL1.5']], 18, 28, lumhistbins)
	else:
		bluetab['weight'] = convolved_weights(bluetab['PQSO'], [bluetab['logL1.5'], ctrltab['logL1.5'], redtab['logL1.5']], 18, 28, lumhistbins)
		ctrltab['weight'] = convolved_weights(ctrltab['PQSO'], [ctrltab['logL1.5'], redtab['logL1.5'], bluetab['logL1.5']], 18, 28, lumhistbins)
		redtab['weight'] = convolved_weights(redtab['PQSO'], [redtab['logL1.5'], ctrltab['logL1.5'], bluetab['logL1.5']], 18, 28, lumhistbins)

	ctrltab.write('catalogs/derived/%s_ctrl.fits' % qso_cat_name, format='fits', overwrite=True)
	bluetab.write('catalogs/derived/%s_blue.fits' % qso_cat_name, format='fits', overwrite=True)
	redtab.write('catalogs/derived/%s_red.fits' % qso_cat_name, format='fits', overwrite=True)"""

	if plots:
		plotting.g_minus_i_plot(qso_cat_name, offset)
		if remove_reddest:
			plotting.lum_dists(qso_cat_name, 100, bluetab['logL1.5'], ctrltab['logL1.5'], redtab['logL1.5'], redtailcat['logL1.5'])
		plotting.z_dists(qso_cat_name, bluetab['Z'], ctrltab['Z'], redtab['Z'])

	for j in range(len(indicesbycolor)):
		colortab = qso_tab[indicesbycolor[j]]
		qso_tab['weight'][indicesbycolor[j]] = convolved_weights(colortab['PQSO'], lumlist, 18, 28, 100, colorbin=j)

	qso_tab.write('catalogs/derived/%s_colored.fits' % qso_cat_name, format='fits', overwrite=True)



# take coordinates from 2 surveys with different footprints and return indices of sources within the overlap of both
def match_footprints(testsample, reference_sample, nside=32):
	ras1, decs1 = testsample[0], testsample[1]
	ras2, decs2 = reference_sample[0], reference_sample[1]

	footprint1_pix = hp.ang2pix(nside=nside, theta=ras1, phi=decs1, lonlat=True)
	footprint2_pix = hp.ang2pix(nside=nside, theta=ras2, phi=decs2, lonlat=True)
	commonpix = np.intersect1d(footprint1_pix, footprint2_pix)
	commonfootprint = np.zeros(hp.nside2npix(nside))
	commonfootprint[commonpix] = 1
	idxs = np.where(commonfootprint[footprint1_pix])
	#idxs2 = np.where(commonfootprint[footprint2_pix])

	#sample1out = (ras1[idxs1], decs1[idxs1])
	#sample2out = (ras2[idxs2], decs2[idxs2])

	#return (sample1out, sample2out)

	return idxs


def first_matched(qso_cat_name):
	qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data
	firstcat = fits.open('catalogs/radio_cats/first_14dec17.fits')[1].data


	firstcoords = SkyCoord(ra=firstcat['RA'] * u.deg, dec=firstcat['DEC'] * u.deg)

	sdsscoords = SkyCoord(ra=qso_cat['RA'] * u.deg, dec=qso_cat['DEC'] * u.deg)
	firstidxs, sdssidxs, d2d, d3d = sdsscoords.search_around_sky(firstcoords, 10 * u.arcsec)

	firstmatchedcat = Table(qso_cat[sdssidxs])
	firstmatchedcat.write('catalogs/derived/%s_RL.fits' % qso_cat_name, format='fits', overwrite=True)



def radio_detect_fraction(qso_cat_name, radio_name='FIRST', lowmag=10, highmag=30, return_plot=False, bins=10, offset=False):
	qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data
	#qso_cat = fits.open('QSO_cats/dr7_bh_Nov19_2013.fits')[1].data

	if radio_name == 'FIRST':
		radrakey, raddeckey = 'RA', 'DEC'
		radio_cat = fits.open('catalogs/radio_cats/first_14dec17.fits')[1].data
	elif radio_name == 'COSMOS':
		radrakey, raddeckey = 'RAdeg', 'DEdeg'
		radio_cat = fits.open('catalogs/radio_cats/VLACOSMOS.fits')[1].data
	elif radio_name == 'LoTSS':
		radrakey, raddeckey = 'RA', 'DEC'
		radio_cat = fits.open('catalogs/radio_cats/LOFAR_DR1.fits')[1].data
	else:
		print('provide survey name')
		return

	radiocoords = SkyCoord(ra=radio_cat[radrakey] * u.deg, dec=radio_cat[raddeckey] * u.deg)


	qso_cat = qso_cat[match_footprints((qso_cat['RA'], qso_cat['DEC']), (radio_cat[radrakey], radio_cat[raddeckey]), nside=256)]
	qso_cat = qso_cat[np.where((qso_cat['i_mag'] < highmag) & (qso_cat['i_mag'] > lowmag))]


	if offset:
		colors = qso_cat['deltagmini']
	else:
		colors = qso_cat['g-i']
	zs = qso_cat['Z']

	gminusibinidxs = bin_by_color(colors, zs, bins, offset)

	radio_detect_frac = []
	for i in range(bins):
		binnedcat = qso_cat[gminusibinidxs[i]]
		coordsinbin = SkyCoord(ra=binnedcat['RA'] * u.deg, dec=binnedcat['DEC'] * u.deg)
		firstidxs, binidxs, d2d, d3d = coordsinbin.search_around_sky(radiocoords, 10 * u.arcsec)
		radio_detect_frac.append(len(firstidxs)/len(binnedcat))

	plotting.radio_detect_frac_plot(radio_detect_frac, surv_name=radio_name, return_plot=return_plot)

# bin up sample into bins of color or color offset, and stack FIRST images in each bin
# this stack can be used to estimate the median flux, median radio luminosity, or median radio loudness of each bin
def median_radio_flux_for_color(qso_cat_name, bins=10, mode='flux', remove_detections=False, minz=0, maxz=10, minL=21, maxL=26, nbootstraps=0, offset=False, remove_reddest=False):
	import first_stacking
	qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data

	# can choose to remove FIRST detections so not to bias the median stacks
	if remove_detections:
		qsocoords = SkyCoord(ra=qso_cat['RA']*u.deg, dec=qso_cat['DEC']*u.deg)
		firstcat = fits.open('catalogs/radio_cats/first_14dec17.fits')[1].data
		firstcoords = SkyCoord(ra=firstcat['RA']*u.deg, dec=firstcat['DEC']*u.deg)
		firstidx, qsoidx, d2d, d3d = qsocoords.search_around_sky(firstcoords, 10*u.arcsec)
		nondetectidxs = np.setdiff1d(np.arange(len(qso_cat)), qsoidx)
		qso_cat = qso_cat[nondetectidxs]

	# make cuts on redshift and/or bolometric luminosity if desired
	qso_cat = qso_cat[np.where((qso_cat['Z'] > minz) & (qso_cat['Z'] < maxz))]
	qso_cat = qso_cat[np.where((qso_cat['logL1.5'] > minL) & (qso_cat['logL1.5'] < maxL))]

	if offset:
		colorkey = 'deltagmini'
	else:
		colorkey = 'g-i'



	reddest_cat = qso_cat[np.where(qso_cat['deltagmini'] > 0.25)]
	qso_cat = qso_cat[np.where(qso_cat['deltagmini'] < 0.25)]

	gminusibinidxs = bin_by_color(qso_cat[colorkey], qso_cat['Z'], bins, offset)

	medcolors, median_radioflux_in_bins, medLbols, medradlum, medradloudness, boot_errs = [], [], [], [], [], []

	if remove_reddest:
		niter = bins
	else:
		niter = bins+1

	for i in range(niter):
		if i==bins:
			binnedcat = reddest_cat
		else:
			binnedcat = qso_cat[gminusibinidxs[i]]
		#binnedcat = qso_cat[gminusibinidxs[i]]
		medcolors.append(np.median(binnedcat['deltagmini']))
		stacked_flux = np.max(first_stacking.median_stack(binnedcat['OBJID_XDQSO']))
		median_radioflux_in_bins.append(stacked_flux)
		medLbol = np.median(binnedcat['logL1.5'])
		medLbols.append(medLbol)
		medzinbin = np.median(binnedcat['Z'])
		medlumdist = astropycosmo.luminosity_distance(medzinbin)
		lumnu = ((4 * np.pi * (medlumdist ** 2) * (stacked_flux * u.Jy) / ((1 + medzinbin) ** (1 - 0.5))).to(
			u.W / u.Hz)).value
		medradlum.append(lumnu)
		medradloudness.append(np.log10((1.4e9*lumnu)/(2e14*(10**(medLbol)))))

		bootmedian_radio_in_bins, bootLbols = [], []
		if nbootstraps>0:
			for j in range(nbootstraps):
				bootidxs = np.random.choice(len(binnedcat), len(binnedcat))
				bootbinnedcat = binnedcat[bootidxs]
				bootmedian_radio_in_bins.append(np.max(first_stacking.median_stack(bootbinnedcat['OBJID_XDQSO'])))
				bootLbols.append(np.median(bootbinnedcat['logL1.5']))
			bootlumnu = (
			(4 * np.pi * (medlumdist ** 2) * (bootmedian_radio_in_bins * u.Jy) / ((1 + medzinbin) ** (1 - 0.5))).to(
				u.W / u.Hz)).value
			radio_loudnesss = np.log10((1.4e9*bootlumnu)/(2e14*(10**(np.array(bootLbols)))))

			if mode == 'flux':
				boot_errs.append(np.std(bootmedian_radio_in_bins))
			elif mode == 'lum':
				boot_errs.append(np.std(bootlumnu))
			elif mode == 'loud':
				boot_errs.append(np.std(radio_loudnesss))
		else:
			boot_errs = None


	if mode == 'flux':
		np.array([medcolors, median_radioflux_in_bins, boot_errs]).dump('plotting_results/first_flux_for_color.npy')
		plotting.plot_median_radio_flux(remove_reddest=remove_reddest)
		return median_radioflux_in_bins
	elif mode == 'lum':
		np.array([medcolors, medradlum, boot_errs]).dump('plotting_results/first_lum_for_color.npy')
		plotting.plot_median_radio_luminosity(remove_reddest=remove_reddest)
	elif mode == 'loud':
		np.array([medcolors, medradloudness, boot_errs]).dump('plotting_results/first_loud_for_color.npy')
		plotting.plot_radio_loudness(remove_reddest=remove_reddest)
		return medradloudness, boot_errs



def linear_model(x, a, b):
	return a*x+b

def kappa_for_color(qso_cat_name, bins=10, offset=False, removereddest=False):
	qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data

	if offset:
		colorkey = 'deltagmini'
	else:
		colorkey = 'g-i'
	#if removereddest:
		#qso_cat = qso_cat[remove_reddest_bin(qso_cat[colorkey], qso_cat['Z'], 10, offset)]
	reddest_cat = qso_cat[np.where(qso_cat['deltagmini'] > 0.25)]
	qso_cat = qso_cat[np.where(qso_cat['deltagmini'] < 0.25)]


	gminusibinidxs = bin_by_color(qso_cat[colorkey], qso_cat['Z'], bins, offset)

	colors, kappas, errs, boots = [], [], [], []
	for i in range(bins):
		binnedcat = qso_cat[gminusibinidxs[i]]
		#colors.append(np.median(binnedcat[colorkey]))
		colors.append(np.median(binnedcat['deltagmini']))
		ras, decs = binnedcat['RA'], binnedcat['DEC']
		stackkappa = stacking.fast_stack(ras, decs, hp.read_map('maps/smoothed_masked_planck.fits'), iterations=500, bootstrap=True)
		if len(stackkappa) > 1:
			kappas.append(stackkappa[0])
			errs.append(np.std(stackkappa[1]))
			boots.append(stackkappa[1])
		else:
			kappas.append(stackkappa)
			errs.append(0)

	linfit, pcov = curve_fit(linear_model, colors, kappas, sigma=errs)
	print(linfit[0])

	boots = np.array(boots)
	slopes = []
	for i in range(len(boots[0])):
		booted = boots[:, i]
		poptboot, pcovboot = curve_fit(linear_model, colors, booted)
		slopes.append(poptboot[0])
	print(np.std(slopes))

	#spearmanranks = stats.spearmanr(colors, kappas)
	#print(spearmanranks)


	if not removereddest:
		#colors.append(np.median(reddest_cat[colorkey]))
		colors.append(np.median(reddest_cat['deltagmini']))
		kap = stacking.fast_stack(reddest_cat['RA'], reddest_cat['DEC'], hp.read_map('maps/smoothed_masked_planck.fits'), iterations=100, bootstrap=True)
		if len(kap) > 1:
			kappas.append(kap[0])
			errs.append(np.std(kap[1]))
		else:
			kappas.append(kap)
			errs.append(0)

	plotting.plot_kappa_v_color(colors, kappas, errs, offset, remove_reddest=removereddest, linfit=linfit)
	return kappas



def temp_for_color(qso_cat_name, bins=10, offset=False, removereddest=False):
	qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data

	if offset:
		colorkey = 'deltagmini'
	else:
		colorkey = 'g-i'
	# if removereddest:
	# qso_cat = qso_cat[remove_reddest_bin(qso_cat[colorkey], qso_cat['Z'], 10, offset)]
	reddest_cat = qso_cat[np.where(qso_cat['deltagmini'] > 0.25)]
	qso_cat = qso_cat[np.where(qso_cat['deltagmini'] < 0.25)]

	gminusibinidxs = bin_by_color(qso_cat[colorkey], qso_cat['Z'], bins, offset)

	colors, kappas, errs = [], [], []
	for i in range(bins):
		binnedcat = qso_cat[gminusibinidxs[i]]
		# colors.append(np.median(binnedcat[colorkey]))
		colors.append(np.median(binnedcat['deltagmini']))
		ras, decs = binnedcat['RA'], binnedcat['DEC']
		stackkappa = stacking.fast_stack(ras, decs, hp.read_map('maps/smica_masked.fits'), iterations=100,
		                                 bootstrap=True)
		if len(stackkappa) > 1:
			kappas.append(stackkappa[0])
			errs.append(stackkappa[1])
		else:
			kappas.append(stackkappa)
			errs.append(0)

	if not removereddest:
		# colors.append(np.median(reddest_cat[colorkey]))
		colors.append(np.median(reddest_cat['deltagmini']))
		kap = stacking.fast_stack(reddest_cat['RA'], reddest_cat['DEC'],
		                          hp.read_map('maps/smica_masked.fits'), iterations=100, bootstrap=True)
		if len(kap) > 1:
			kappas.append(kap[0])
			errs.append(kap[1])
		else:
			kappas.append(kap)
			errs.append(0)

	plotting.plot_temp_v_color(colors, kappas, errs, offset, removereddest)


def sed_for_color(qso_cat_name, bins=10, offset=False, removereddest=False):
	qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data

	if offset:
		colorkey = 'deltagmini'
	else:
		colorkey = 'g-i'
	# if removereddest:
	# qso_cat = qso_cat[remove_reddest_bin(qso_cat[colorkey], qso_cat['Z'], 10, offset)]

	reddest_cat = qso_cat[np.where(qso_cat['deltagmini'] > 0.25)]
	qso_cat = qso_cat[np.where(qso_cat['deltagmini'] < 0.25)]

	gminusibinidxs = bin_by_color(qso_cat[colorkey], qso_cat['Z'], bins, offset)

	colors, seds = [], []
	for i in range(bins):
		binnedcat = qso_cat[gminusibinidxs[i]]
		# colors.append(np.median(binnedcat[colorkey]))
		colors.append(np.median(binnedcat['deltagmini']))
		medians = np.median(binnedcat['PSFFLUX'], axis=0)
		meds = list(medians[:5]*(3.631*(10**(-6))))
		meds.append(medians[11]/1e9*(309.54))
		meds.append(medians[12]/1e9*(171.787))
		seds.append(np.array(meds))

	if not removereddest:
		# colors.append(np.median(reddest_cat[colorkey]))
		colors.append(np.median(reddest_cat['deltagmini']))
		medians = np.median(reddest_cat['PSFFLUX'], axis=0)
		meds = list(medians[:5] * (3.631 * (10 ** (-6))))
		meds.append(medians[11] / 1e9 * (309.54))
		meds.append(medians[12] / 1e9 * (171.787))
		seds.append(np.array(meds))

	#radiofluxes = median_radio_flux_for_color(qso_cat_name, 10, mode='flux', remove_detections=False, remove_reddest=removereddest)
	#for j in range(len(seds)):
	#	seds[j] = np.concatenate([seds[j], [radiofluxes[j]]])
	seds = np.array(seds)


	plotting.plot_sed_v_color(seds, removereddest)