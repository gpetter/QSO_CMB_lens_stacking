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
from scipy import interpolate
import pandas as pd
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
	kcorrecttab = pd.read_csv('kcorrect/table4.dat', names=['z', 'k'], delim_whitespace=True)
	roundzs = np.around(zs, 2)
	k_idxs = np.where(roundzs.reshape(roundzs.size, 1) == np.array(kcorrecttab['z']))[1]
	kcorrects = np.array(kcorrecttab['k'][k_idxs])

	distances = astropycosmo.luminosity_distance(zs).to(u.pc)

	i_abs_mags = dered_i_mags - 5 * np.log10(distances / (10 * u.pc)) - kcorrects

	return i_abs_mags



def fix_dr16():
	dr16 = Table(fits.open('QSO_cats/DR16Q_v4.fits')[1].data)
	dr16good = dr16[np.where(dr16['Z'] > 0)]
	dr16good.write('QSO_cats/dr16_ok.fits', format='fits', overwrite=True)


def define_core_sample(sample):
	if sample == 'dr16':
		dr16 = Table(fits.open('QSO_cats/dr16_ok.fits')[1].data)

		coreidxs = list(find_bitmask_matches(dr16['EBOSS_TARGET0'], 10)) + list(find_bitmask_matches(dr16['EBOSS_TARGET1'], 10))
		#bosscoreidxs = list(find_bitmask_matches(dr16['BOSS_TARGET1'], 40))
		#coreidxs = coreidxs + bosscoreidxs
		coresample = dr16[coreidxs]
		coresample.write('QSO_cats/dr16_core.fits', format='fits', overwrite=True)

	elif sample == 'dr16_superset':
		superset = Table(fits.open('QSO_cats/DR16Q_Superset_v3.fits')[1].data)
		coreidxs = list(find_bitmask_matches(superset['EBOSS_TARGET0'], 10)) + list(
			find_bitmask_matches(superset['EBOSS_TARGET1'], 10))
		bosscoreidxs = list(find_bitmask_matches(superset['BOSS_TARGET1'], 40))
		coreidxs = coreidxs + bosscoreidxs
		coresample = superset[coreidxs]
		coresample.write('QSO_cats/dr16_superset_CORE.fits', format='fits', overwrite=True)

	elif sample == 'dr14':
		dr14 = Table(fits.open('QSO_cats/DR14Q_v4_4.fits')[1].data)

		coreidxs = list(find_bitmask_matches(dr14['EBOSS_TARGET0'], 10)) + list(
			find_bitmask_matches(dr14['EBOSS_TARGET1'], 10))

		coresample = dr14[coreidxs]
		coresample.write('QSO_cats/dr14_core.fits', format='fits', overwrite=True)


def match_phot_qsos_to_spec_qsos(update_probs=True, core_only=False):
	dr16 = Table(fits.open('QSO_cats/dr16_ok.fits')[1].data)
	speccoords = SkyCoord(ra=dr16['RA']*u.degree, dec=dr16['DEC']*u.degree)
	xdqso = Table(fits.open('QSO_cats/xdqso-z-cat.fits')[1].data)
	photcoords = SkyCoord(ra=xdqso['RA_XDQSO']*u.degree, dec=xdqso['DEC_XDQSO']*u.degree)

	# match XDQSO catalog with DR16 spectroscopic catalog
	xdidx, specidx, d2d, d3d = speccoords.search_around_sky(photcoords, 1*u.arcsec)
	# rip spectroscopic redshifts from DR16 to replace photo-zs from XDQSO where matched
	xdqso['PEAKZ'][xdidx] = dr16['Z'][specidx]
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
			dr16core = Table(fits.open('QSO_cats/dr16_core.fits')[1].data)
			corecoords = SkyCoord(ra=dr16core['RA']*u.degree, dec=dr16core['DEC']*u.degree)
			xdcoreidxs, speccoreidxs, d2d, d3d = corecoords.search_around_sky(photcoords, 1*u.arcsec)
			xdqso['PQSO'][xdcoreidxs] = np.ones(len(speccoreidxs))

			# update probabilites to 0 if selected in CORE sample but spectroscopically ruled out
			dr16_superset = Table(fits.open('QSO_cats/dr16_superset_CORE.fits')[1].data)
			superset_non_QSOs = dr16_superset[np.where(dr16_superset['IS_QSO_FINAL'] < 1)]
			supercoords = SkyCoord(ra=superset_non_QSOs['RA']*u.deg, dec=superset_non_QSOs['DEC']*u.deg)
			xdidxs, nonidxs, d2d, d3d = supercoords.search_around_sky(photcoords, 1*u.arcsec)
			xdqso['PQSO'][xdidxs] = np.zeros(len(nonidxs))

		else:
			xdqso['PQSO'][xdidx] = np.ones(len(specidx))

			dr16_superset = Table(fits.open('QSO_cats/DR16Q_Superset_v3.fits')[1].data)
			superset_non_QSOs = dr16_superset[np.where(dr16_superset['IS_QSO_FINAL'] < 1)]
			supercoords = SkyCoord(ra=superset_non_QSOs['RA'] * u.deg, dec=superset_non_QSOs['DEC'] * u.deg)
			nonxdidxs, nonspecidxs, d2d, d3d = supercoords.search_around_sky(photcoords, 1*u.arcsec)
			print(len(nonxdidxs), len(nonspecidxs))
			xdqso['PQSO'][nonxdidxs] = np.zeros(len(nonspecidxs))

		xdqso = xdqso[np.where(xdqso['PQSO'] > 0.2)]



	xdqso.write('QSO_cats/xdqso_specz.fits', format='fits', overwrite=True)


def log_interp1d(xx, yy):
	logx = np.log10(xx)
	logy = np.log10(yy)
	lin_interp = interpolate.interp1d(logx, logy, fill_value='extrapolate')
	log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
	return log_interp


# calculate the rest frame 1.5 micron luminosities using log-linear interpolation
def rest_ir_lum(w1, w2, zs, ref_lambda_rest):
	w1 = w1/1e9 * 309.54
	w2 = w2/1e9 * 171.787
	w1_obs_length = 3.368
	w2_obs_length = 4.618
	ref_lambda_obs = ref_lambda_rest * (1+zs)

	ref_lums = []
	for i in range(len(w1)):
		interp_func = log_interp1d([w1_obs_length, w2_obs_length], [w1[i], w2[i]])
		ref_lums.append((1/(1 + zs[i]) * (interp_func(ref_lambda_obs[i]) * u.Jy * 4 * np.pi * (
				astropycosmo.luminosity_distance(zs[i])) ** 2).to('J').value))

	return np.log10(ref_lums)


# calculate aboslute I band magnitude, g - i colors, and 1.5 micron Luminosities, writing values to file
def write_properties(sample, speczs=False):



	if sample == 'dr14':

		qso_cat = (fits.open('QSO_cats/DR14Q_v4_4.fits'))[1].data
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
		qso_cat = Table((fits.open('QSO_cats/dr16_ok.fits'))[1].data)
		rakey = 'RA'
		deckey = 'DEC'

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
		trimmed_cat['logL1.5'] = rest_ir_lum(trimmed_cat['W1_FLUX'], trimmed_cat['W2_FLUX'], trimmed_cat['Z'], 1.5)
		trimmed_cat['i_mag'] = dered_i_mags
		trimmed_cat.write('QSO_cats/dr16_new.fits', format='fits', overwrite=True)

	elif sample == 'eboss_lss':
		qso_cat = Table(fits.open('catalogs/lss/eBOSS_NGC_phot.fits')[1].data)
		goodwise = (qso_cat['W1_FLUX'] > 0) & (qso_cat['W2_FLUX'] > 0)
		trimmed_cat = qso_cat[np.where(goodwise)]
		dered_i_mags = trimmed_cat['PSFMAG'][:, band_idxs['i']] - trimmed_cat['EXTINCTION'][:, band_idxs['i']]
		dered_g_mags = trimmed_cat['PSFMAG'][:, band_idxs['g']] - trimmed_cat['EXTINCTION'][:, band_idxs['g']]



		trimmed_cat['myMI'] = k_correct_richards(dered_i_mags, trimmed_cat['Z'])
		trimmed_cat['g-i'] = dered_g_mags - dered_i_mags
		trimmed_cat['logL1.5'] = rest_ir_lum(trimmed_cat['W1_FLUX'], trimmed_cat['W2_FLUX'], trimmed_cat['Z'], 1.5)
		trimmed_cat['i_mag'] = dered_i_mags
		trimmed_cat.write('catalogs/lss/eBOSS_NGC_new.fits', format='fits', overwrite=True)

	# write properties to an XDQSO photometric catalog
	else:
		if speczs:
			qso_cat = Table(fits.open('QSO_cats/xdqso_specz.fits')[1].data)
		else:
			qso_cat = Table(fits.open('QSO_cats/xdqso-z-cat.fits')[1].data)

		rakey = 'RA_XDQSO'
		deckey = 'DEC_XDQSO'
		zkey = 'PEAKZ'
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

		i_abs_mags = k_correct_richards(dered_i_mags, trimmed_cat['PEAKZ'])

		trimmed_cat['myMI'] = i_abs_mags
		trimmed_cat['g-i'] = (dered_g_mags - dered_i_mags)
		trimmed_cat['logL1.5'] = rest_ir_lum(trimmed_cat['PSFFLUX'][:, 11], trimmed_cat['PSFFLUX'][:, 12], trimmed_cat[zkey], 1.5)
		trimmed_cat['W1_MAG'] = 22.5 - 2.5 * np.log10(trimmed_cat['PSFFLUX'][:, 11])
		trimmed_cat['W2_MAG'] = 22.5 - 2.5 * np.log10(trimmed_cat['PSFFLUX'][:, 12])
		trimmed_cat['i_mag'] = dered_i_mags
		if speczs:
			trimmed_cat.write('QSO_cats/xdqso_specz_new.fits', format='fits', overwrite=True)
		else:
			trimmed_cat.write('QSO_cats/xdqso_new.fits', format='fits', overwrite=True)



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
	qso_cat = Table(fits.open('QSO_cats/%s_new.fits' % qso_cat_name)[1].data)

	if (qso_cat_name == 'dr14') or (qso_cat_name == 'dr16'):
		rakey, deckey = 'RA', 'DEC'
		zkey = 'Z'

	else:
		rakey, deckey = 'RA_XDQSO', 'DEC_XDQSO'
		zkey = 'PEAKZ'
		qso_cat = qso_cat[np.where((qso_cat['GOOD'] == 0) & (qso_cat['PQSO'] > pcut) & (qso_cat['NPEAKS'] <= peakscut) & (qso_cat['NPEAKS'] > 0))]



	if apply_planck_mask:
		map4stack = hp.read_map('maps/smoothed_masked_planck.fits', dtype=np.single)
		ls, bs = stacking.equatorial_to_galactic(qso_cat[rakey], qso_cat[deckey])
		outsidemaskbool = (map4stack[hp.ang2pix(2048, ls, bs, lonlat=True)] != hp.UNSEEN)
		qso_cat = qso_cat[np.where(outsidemaskbool)]

	i_app = qso_cat['i_mag']

	complete_cut = np.where((qso_cat['myMI'] <= lumcut) & (qso_cat['myMI'] > -100) & (qso_cat[zkey] >= minz) & (qso_cat[zkey] <= maxz) & (
					i_app < magcut))
	t = qso_cat[complete_cut]

	if magcut < 100:
		kcorrecttab = pd.read_csv('kcorrect/table4.dat', names=['z', 'k'], delim_whitespace=True)
		zs = np.linspace(minz, maxz, 20)
		newdists = astropycosmo.luminosity_distance(zs).to(u.pc)

		roundzs = np.around(zs, 2)
		k_idxs = np.where(roundzs.reshape(roundzs.size, 1) == np.array(kcorrecttab['z']))[1]
		kcorrects = np.array(kcorrecttab['k'][k_idxs])

		limMs = magcut - 5 * np.log10(newdists / (10 * u.pc)) - kcorrects

	if plots:
		plotting.MI_vs_z(qso_cat[zkey], qso_cat['myMI'], len(complete_cut[0]), magcut, minz, maxz, lumcut, qso_cat_name, limMs)
		plotting.w1_minus_w2_plot(t['W1_MAG'], t['W2_MAG'], qso_cat_name)


	t.write('QSO_cats/%s_complete.fits' % qso_cat_name, format='fits', overwrite=True)




def lum_weights(rb_lums, second_lums, third_lums, minlum, maxlum, bins):

	ctrlhist = np.histogram(second_lums, bins=bins, range=(minlum, maxlum), density=True)
	colorhist = np.histogram(rb_lums, bins=bins, range=(minlum, maxlum), density=True)
	othercolorhist = np.histogram(third_lums, bins=bins, range=(minlum, maxlum), density=True)
	min_in_bins = np.amin([colorhist[0], ctrlhist[0], othercolorhist[0]], axis=0)

	#dist_ratio = ctrlhist[0] / colorhist[0]
	dist_ratio = min_in_bins / colorhist[0]

	dist_ratio[np.where(np.isnan(dist_ratio) | np.isinf(dist_ratio))] = 0

	weights = np.zeros(len(rb_lums))

	for i in range(len(dist_ratio)):
		idxs_in_bin = np.where((rb_lums <= colorhist[1][i + 1]) & (rb_lums > colorhist[1][i]))
		weights[idxs_in_bin] = dist_ratio[i]

	return weights


def convolved_weights(pqso_weights, rb_lums, second_lums, third_lums, minlum, maxlum, bins):

	l_weights = lum_weights(rb_lums, second_lums, third_lums, minlum, maxlum, bins)

	totweights = pqso_weights * l_weights

	return totweights



def red_blue_samples(qso_cat_name, plots):


	if qso_cat_name == 'dr14':
		zkey = 'Z'
		sources_in_bins = 1000.
		qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data

	elif qso_cat_name == 'dr16':
		zkey = 'Z'

		sources_in_bins = 1000.
		qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data
	elif qso_cat_name == 'eboss_lss':
		zkey = 'Z'
		qso_cat = fits.open('catalogs/lss/eBOSS_NGC_new.fits')[1].data
		sources_in_bins = 1000.
	else:
		zkey = 'PEAKZ'
		qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data
		sources_in_bins = 10000.


	gminusi = qso_cat['g-i']
	zs = qso_cat[zkey]

	num_bins = int(len(zs)/sources_in_bins)
	z_quantile_bins = pd.qcut(zs, num_bins, retbins=True)[1]


	red_idxs, c_idxs, blue_idxs = [], [], []
	for i in range(len(z_quantile_bins) - 1):
		in_z_bin = (zs < z_quantile_bins[i+1]) & (zs >= z_quantile_bins[i])
		idxs_in_bin = np.where(in_z_bin)
		gminusi_in_bin = gminusi[idxs_in_bin]
		color_bins = pd.qcut(gminusi_in_bin, 20, retbins=True)[1]
		red_idxs += list(np.where((gminusi > color_bins[17]) & in_z_bin)[0])
		c_idxs += list(np.where((gminusi <= color_bins[15]) & (gminusi > color_bins[5]) & in_z_bin)[0])

		blue_idxs += list(np.where((gminusi <= color_bins[3]) & in_z_bin)[0])


	bluetab = Table(qso_cat[blue_idxs])
	ctrltab = Table(qso_cat[c_idxs])
	redtab = Table(qso_cat[red_idxs])

	#lowhistlim = floor(np.log10(np.min([np.min(bluelums), np.min(ctrllums), np.min(redlums)])))
	#highhistlim = ceil(np.log10(np.max(np.max([np.max(bluelums), np.max(ctrllums), np.max(redlums)]))))
	lumhistbins = 100

	if plots:
		plotting.g_minus_i_plot(qso_cat_name, zs[blue_idxs], gminusi[blue_idxs], zs[c_idxs], gminusi[c_idxs],
		                        zs[red_idxs], gminusi[red_idxs], zs, gminusi)
		plotting.lum_dists(qso_cat_name, lumhistbins, bluetab['logL1.5'], ctrltab['logL1.5'], redtab['logL1.5'])

	if (qso_cat_name == 'dr14') or (qso_cat_name == 'dr16') or (qso_cat_name == 'eboss_lss'):
		bluetab['weight'] = lum_weights(bluetab['logL1.5'], ctrltab['logL1.5'], redtab['logL1.5'], 18, 28, lumhistbins)
		ctrltab['weight'] = lum_weights(ctrltab['logL1.5'], redtab['logL1.5'], bluetab['logL1.5'], 18, 28, lumhistbins)
		redtab['weight'] = lum_weights(redtab['logL1.5'], ctrltab['logL1.5'], bluetab['logL1.5'], 18, 28, lumhistbins)
	else:
		bluetab['weight'] = convolved_weights(bluetab['PQSO'], bluetab['logL1.5'], ctrltab['logL1.5'], redtab['logL1.5'], 18, 28, lumhistbins)
		ctrltab['weight'] = convolved_weights(ctrltab['PQSO'], ctrltab['logL1.5'], redtab['logL1.5'], bluetab['logL1.5'], 18, 28, lumhistbins)
		redtab['weight'] = convolved_weights(redtab['PQSO'], redtab['logL1.5'], ctrltab['logL1.5'], bluetab['logL1.5'], 18, 28, lumhistbins)

	ctrltab.write('catalogs/derived/%s_ctrl.fits' % qso_cat_name, format='fits', overwrite=True)
	bluetab.write('catalogs/derived/%s_blue.fits' % qso_cat_name, format='fits', overwrite=True)
	redtab.write('catalogs/derived/%s_red.fits' % qso_cat_name, format='fits', overwrite=True)



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
	qso_cat = fits.open('QSO_cats/%s_complete.fits' % qso_cat_name)[1].data
	firstcat = fits.open('QSO_cats/first_14dec17.fits')[1].data


	firstcoords = SkyCoord(ra=firstcat['RA'] * u.deg, dec=firstcat['DEC'] * u.deg)

	sdsscoords = SkyCoord(ra=qso_cat['RA_XDQSO'] * u.deg, dec=qso_cat['DEC_XDQSO'] * u.deg)
	firstidxs, sdssidxs, d2d, d3d = sdsscoords.search_around_sky(firstcoords, 10 * u.arcsec)

	firstmatchedcat = Table(qso_cat[sdssidxs])
	firstmatchedcat.write('QSO_cats/%s_RL.fits' % qso_cat_name, format='fits', overwrite=True)



def radio_detect_fraction(qso_cat_name, radio_name='FIRST', lowmag=10, highmag=30, return_plot=False, bins=10):
	qso_cat = fits.open('QSO_cats/%s_complete.fits' % qso_cat_name)[1].data
	print(len(qso_cat))
	#qso_cat = fits.open('QSO_cats/dr7_bh_Nov19_2013.fits')[1].data

	if radio_name == 'FIRST':
		radrakey, raddeckey = 'RA', 'DEC'
		radio_cat = fits.open('QSO_cats/first_14dec17.fits')[1].data
	elif radio_name == 'COSMOS':
		radrakey, raddeckey = 'RAdeg', 'DEdeg'
		radio_cat = fits.open('QSO_cats/VLACOSMOS.fits')[1].data
	elif radio_name == 'LoTSS':
		radrakey, raddeckey = 'RA', 'DEC'
		radio_cat = fits.open('QSO_cats/LOFAR_DR1.fits')[1].data
	else:
		print('provide survey name')
		return

	radiocoords = SkyCoord(ra=radio_cat[radrakey] * u.deg, dec=radio_cat[raddeckey] * u.deg)

	if (qso_cat_name == 'xdqso') or (qso_cat_name == 'xdqso_specz'):
		rakey, deckey = 'RA_XDQSO', 'DEC_XDQSO'
		zkey = 'PEAKZ'
		sources_in_bins = 10000.
	else:
		rakey, deckey = 'RA', 'DEC'
		zkey = 'Z'
		sources_in_bins = 1000.

	qso_cat = qso_cat[match_footprints((qso_cat[rakey], qso_cat[deckey]), (radio_cat[radrakey], radio_cat[raddeckey]), nside=256)]
	qso_cat = qso_cat[np.where((qso_cat['i_mag'] < highmag) & (qso_cat['i_mag'] > lowmag))]
	print(len(qso_cat))

	gminusi = qso_cat['g-i']
	zs = qso_cat[zkey]

	num_bins = int(len(zs)/sources_in_bins)
	z_quantile_bins = pd.qcut(zs, 5, retbins=True)[1]


	# make empty list of lists
	gminusibinidxs = []
	for k in range(bins):
		gminusibinidxs.append([])

	# loop over redshift bins
	for i in range(len(z_quantile_bins) - 1):
		in_z_bin = (zs < z_quantile_bins[i+1]) & (zs >= z_quantile_bins[i])
		idxs_in_bin = np.where(in_z_bin)
		gminusi_in_bin = gminusi[idxs_in_bin]
		color_bins = pd.qcut(gminusi_in_bin, bins, retbins=True)[1]
		for j in range(bins):
			gminusibinidxs[j] += list(np.where((gminusi <= color_bins[j+1]) & (gminusi > color_bins[j]) & in_z_bin)[0])
	radio_detect_frac = []
	for i in range(bins):
		binnedcat = qso_cat[gminusibinidxs[i]]
		coordsinbin = SkyCoord(ra=binnedcat[rakey] * u.deg, dec=binnedcat[deckey] * u.deg)
		firstidxs, binidxs, d2d, d3d = coordsinbin.search_around_sky(radiocoords, 10 * u.arcsec)
		radio_detect_frac.append(len(firstidxs)/len(binnedcat))

	plotting.radio_detect_frac_plot(radio_detect_frac, surv_name=radio_name, return_plot=return_plot)


def median_radio_flux_for_color(qso_cat_name, bins=10, luminosity=False, remove_detections=False, minz=0, maxz=10, minL=21, maxL=26):
	import first_stacking
	qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data

	if remove_detections:
		qsocoords = SkyCoord(ra=qso_cat['RA_XDQSO']*u.deg, dec=qso_cat['DEC_XDQSO']*u.deg)
		firstcat = fits.open('catalogs/radio_cats/first_14dec17.fits')[1].data
		firstcoords = SkyCoord(ra=firstcat['RA']*u.deg, dec=firstcat['DEC']*u.deg)
		firstidx, qsoidx, d2d, d3d = qsocoords.search_around_sky(firstcoords, 10*u.arcsec)
		nondetectidxs = np.setdiff1d(np.arange(len(qso_cat)), qsoidx)
		qso_cat = qso_cat[nondetectidxs]


	qso_cat = qso_cat[np.where((qso_cat['PEAKZ'] > minz) & (qso_cat['PEAKZ'] < maxz))]
	qso_cat = qso_cat[np.where((qso_cat['logL1.5'] > minL) & (qso_cat['logL1.5'] < maxL))]

	gminusi = qso_cat['g-i']
	zs = qso_cat['PEAKZ']

	z_quantile_bins = pd.qcut(zs, 10, retbins=True)[1]

	# make empty list of lists
	gminusibinidxs = []
	for k in range(bins):
		gminusibinidxs.append([])


	# loop over redshift bins
	for i in range(len(z_quantile_bins) - 1):
		in_z_bin = (zs < z_quantile_bins[i + 1]) & (zs >= z_quantile_bins[i])
		idxs_in_bin = np.where(in_z_bin)
		gminusi_in_bin = gminusi[idxs_in_bin]

		color_bins = pd.qcut(gminusi_in_bin, bins, retbins=True)[1]
		for j in range(bins):
			gminusibinidxs[j] += list(
				np.where((gminusi <= color_bins[j + 1]) & (gminusi > color_bins[j]) & in_z_bin)[0])

	colors, median_radio_in_bins, Lbols = [], [], []
	for i in range(bins):
		binnedcat = qso_cat[gminusibinidxs[i]]
		colors.append(np.median(binnedcat['g-i']))
		median_radio_in_bins.append(np.max(first_stacking.median_stack(binnedcat['OBJID_XDQSO'])))
		Lbols.append(np.median(binnedcat['logL1.5']))
	print(median_radio_in_bins)


	medz = np.median(zs)
	medlumdist = astropycosmo.luminosity_distance(medz)
	lumnu = ((4 * np.pi * (medlumdist ** 2) * (median_radio_in_bins * u.Jy) / ((1 + medz) ** (1 - 0.7))).to(
		u.W / u.Hz)).value

	radioloudness = np.log10(lumnu) - Lbols

	if luminosity:

		plotting.plot_median_radio(colors, [median_radio_in_bins, lumnu], medz=medz, lum=True)
	else:
		plotting.plot_median_radio(colors, [median_radio_in_bins, radioloudness], medz)


def kappa_for_color(qso_cat_name, bins=10):
	qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data

	gminusi = qso_cat['g-i']
	zs = qso_cat['PEAKZ']

	z_quantile_bins = pd.qcut(zs, 10, retbins=True)[1]

	# make empty list of lists
	gminusibinidxs = []
	for k in range(bins):
		gminusibinidxs.append([])

	# loop over redshift bins
	for i in range(len(z_quantile_bins) - 1):
		in_z_bin = (zs < z_quantile_bins[i + 1]) & (zs >= z_quantile_bins[i])
		idxs_in_bin = np.where(in_z_bin)
		gminusi_in_bin = gminusi[idxs_in_bin]

		color_bins = pd.qcut(gminusi_in_bin, bins, retbins=True)[1]
		for j in range(bins):
			gminusibinidxs[j] += list(
				np.where((gminusi <= color_bins[j + 1]) & (gminusi > color_bins[j]) & in_z_bin)[0])

	colors, kappas = [], []
	for i in range(bins):
		binnedcat = qso_cat[gminusibinidxs[i]]
		colors.append(np.median(binnedcat['g-i']))
		ras, decs = binnedcat['RA_XDQSO'], binnedcat['DEC_XDQSO']
		kappas.append(stacking.fast_stack(ras, decs, hp.read_map('maps/smoothed_masked_planck.fits'), iterations=0))
	print(kappas)

	plotting.plot_kappa_v_color(colors, kappas)