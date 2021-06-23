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
from astropy import stats as astrostats
import astropy.constants as con
import spectrumtools
import wise_tools
import lensingModel
import fitting
import autocorrelation
import weighting
import bin_samples
import healpixhelper
importlib.reload(healpixhelper)
importlib.reload(bin_samples)
importlib.reload(weighting)
importlib.reload(autocorrelation)
importlib.reload(lensingModel)
importlib.reload(wise_tools)
importlib.reload(spectrumtools)
importlib.reload(stacking)
importlib.reload(plotting)
importlib.reload(fitting)

cosmo = cosmology.setCosmology('planck18')
astropycosmo = cosmo.toAstropy()


band_idxs = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "nuv": 5, "fuv": 6, "Y": 7, "J": 8, "H": 9, "K": 10, "W1": 11,
             "W2": 12}
bsoftpars = np.array([1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10, 1e-10, 1e-10, 3e-10, .9e-10, 5.2e-10, 1.1e-9, 1e-10,
                      1e-10])
vega_to_ab = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 309.54/3631, 171.787/3631])

def gaussian(x, mu, s1):
	gauss = 1/(np.sqrt(2*np.pi)*s1) * np.exp(-np.square(x - mu) / (2 * (s1 ** 2)))
	return gauss


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

		coreidxs = list(find_bitmask_matches(dr16['EBOSS_TARGET0'], 10)) + \
		           list(find_bitmask_matches(dr16['EBOSS_TARGET1'], 10))
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

def update_WISE(maxsep=2):
	xdqso = Table(fits.open('catalogs/SDSS_QSOs/xdqso_ALLWISE_catWISE.fits')[1].data)
	badallwise = xdqso['angDist_aw'] > maxsep
	xdqso['W4mag_aw'][badallwise] = np.nan
	xdqso['W3mag_aw'][badallwise] = np.nan
	xdqso['W2mag_aw'][badallwise] = np.nan
	xdqso['W1mag_aw'][badallwise] = np.nan
	xdqso['e_W4mag_aw'][badallwise] = np.nan
	xdqso['e_W3mag_aw'][badallwise] = np.nan
	xdqso['e_W2mag_aw'][badallwise] = np.nan
	xdqso['e_W1mag_aw'][badallwise] = np.nan



	badcatwise = xdqso['angDist_cw'] > maxsep
	xdqso['W1mag_cw'][badcatwise] = np.nan
	xdqso['e_W1mag_cw'][badcatwise] = np.nan
	xdqso['W2mag_cw'][badcatwise] = np.nan
	xdqso['e_W2mag_cw'][badcatwise] = np.nan
	xdqso['pmRA'][badcatwise] = np.nan
	xdqso['e_pmRA'][badcatwise] = np.nan
	xdqso['pmDE'][badcatwise] = np.nan
	xdqso['e_pmDE'][badcatwise] = np.nan

	goodcatwise = (xdqso['angDist_cw'] < maxsep)

	fw1 = 10**(-xdqso['W1mag_cw']/2.5) * 1e9
	fw2 = 10**(-xdqso['W2mag_cw']/2.5) * 1e9
	efw1 = xdqso['e_W1mag_cw'] * fw1
	efw2 = xdqso['e_W2mag_cw'] * fw2


	nativew1flux = xdqso['PSFFLUX'][:, 11]
	nativew2flux = xdqso['PSFFLUX'][:, 12]
	nativew1_ivar = xdqso['PSFFLUX_IVAR'][:, 11]
	nativew2_ivar = xdqso['PSFFLUX_IVAR'][:, 12]

	nativew1flux[goodcatwise] = fw1[goodcatwise]
	nativew2flux[goodcatwise] = fw2[goodcatwise]
	nativew1_ivar[goodcatwise] = 1 / np.square(efw1[goodcatwise])
	nativew2_ivar[goodcatwise] = 1 / np.square(efw2[goodcatwise])

	xdqso['PSFFLUX'][:, 11] = nativew1flux
	xdqso['PSFFLUX'][:, 12] = nativew2flux
	xdqso['PSFFLUX_IVAR'][:, 11] = nativew1_ivar
	xdqso['PSFFLUX_IVAR'][:, 12] = nativew2_ivar



	xdqso.write('catalogs/derived/xdqso_WISE.fits', format='fits', overwrite=True)



def match_phot_qsos_to_spec_qsos(update_probs=True, core_only=False):
	dr16 = Table(fits.open('catalogs/derived/dr16_ok.fits')[1].data)
	speccoords = SkyCoord(ra=dr16['RA']*u.degree, dec=dr16['DEC']*u.degree)
	xdqso = Table(fits.open('catalogs/derived/xdqso_WISE.fits')[1].data)
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
		qso_cat = Table((fits.open('catalogs/derived/dr16_ok.fits'))[1].data)


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
		trimmed_cat['logL'] = wise_tools.rest_ir_lum(trimmed_cat['W1_FLUX'], trimmed_cat['W2_FLUX'], trimmed_cat['Z'],
		                                             1.5)
		trimmed_cat['i_mag'] = dered_i_mags
		trimmed_cat.write('catalogs/derived/dr16_new.fits', format='fits', overwrite=True)

	elif sample == 'eBOSS_QSO':

		qso_cat = Table(fits.open('catalogs/lss/eBOSS_QSO/eBOSS_QSO_fullsky_phot.fits')[1].data)
		sdssflux = np.array(qso_cat['PSFFLUX'])
		empties = np.zeros((len(qso_cat), 8))
		qso_cat['PSFFLUX'] = np.hstack((sdssflux, empties))
		qso_cat['PSFFLUX'][:, 11] = qso_cat['W1_NANOMAGGIES']
		qso_cat['PSFFLUX'][:, 12] = qso_cat['W2_NANOMAGGIES']

		qso_cat['PSFFLUX_IVAR'] = np.hstack((qso_cat['PSFFLUX_IVAR'], empties))
		qso_cat['PSFFLUX_IVAR'][:, 11] = qso_cat['W1_NANOMAGGIES_IVAR']
		qso_cat['PSFFLUX_IVAR'][:, 12] = qso_cat['W2_NANOMAGGIES_IVAR']

		qso_cat['EXTINCTION'] = np.hstack((qso_cat['EXTINCTION'], empties))

		#goodwise = (qso_cat['PSFFLUX'][:, 11] > 0) & (qso_cat['PSFFLUX'][:, 12] > 0)
		#trimmed_cat = qso_cat[np.where(goodwise)]
		psfmaggies = np.array(qso_cat['PSFFLUX'] / 1e9) * vega_to_ab
		psfmags = -2.5 / np.log(10.) * (
				np.arcsinh(psfmaggies / (2 * bsoftpars)) + np.log(bsoftpars))
		deredmags = psfmags - qso_cat['EXTINCTION']

		magerrs = 1 / (qso_cat['PSFFLUX'] * np.sqrt(qso_cat['PSFFLUX_IVAR']))

		w1_nu_f_nu = np.array(
			309.54 * qso_cat['PSFFLUX'][:, 11] / 1e9 * (con.c / (3.368 * u.micron)).to('Hz').value)
		w2_nu_f_nu = np.array(
			171.787 * qso_cat['PSFFLUX'][:, 12] / 1e9 * (con.c / (4.618 * u.micron)).to('Hz').value)

		qso_cat['deredmags'] = deredmags
		qso_cat['myMI'] = k_correct_richards(deredmags[:, 3], qso_cat['Z'])
		qso_cat['g-i'] = deredmags[:, 1] - deredmags[:, 3]
		qso_cat['logL1_5'] = wise_tools.rest_ir_lum(np.array([w1_nu_f_nu, w2_nu_f_nu]),
		                                                qso_cat['Z'], 1.5)
		qso_cat['e_mags'] = np.abs(magerrs)

		qso_cat.write('catalogs/derived/eBOSS_QSO_new.fits', format='fits', overwrite=True)

	# write properties to an XDQSO photometric catalog
	else:
		if speczs:
			qso_cat = Table(fits.open('catalogs/derived/xdqso_specz.fits')[1].data)
		else:
			qso_cat = Table(fits.open('catalogs/SDSS_QSOs/xdqso-z-cat.fits')[1].data)

		zkey = 'Z'
		goodzs = ((qso_cat[zkey] > 0) & (qso_cat[zkey] < 5.494))

		goodprobs = (qso_cat['PQSO'] > 0.2)
		goodwise = (qso_cat['PSFFLUX'][:, 11] > 0) & (qso_cat['PSFFLUX'][:, 12] > 0)


		"""good_idxs = np.where((qso_cat['NPEAKS'] == 1) & (qso_cat['bright_star'] == False) &
			(qso_cat['bad_u'] == False) & (qso_cat['bad_field'] == False) &
			(qso_cat['wise_flagged'] == False) &
			(qso_cat['good'] == 0) & outsidemaskbool)[0]"""
		#good_idxs = np.where(outsidemaskbool & goodwise & goodzs & goodprobs)[0]
		good_idxs = np.where(goodwise & goodzs & goodprobs)[0]

		trimmed_cat = qso_cat[good_idxs]

		psfmaggies = np.array(trimmed_cat['PSFFLUX'] / 1e9) * vega_to_ab
		psfmags = -2.5 / np.log(10.) * (
					np.arcsinh(psfmaggies / (2 * bsoftpars)) + np.log(bsoftpars))
		deredmags = psfmags - trimmed_cat['EXTINCTION']

		magerrs = 1/(trimmed_cat['PSFFLUX'] * np.sqrt(trimmed_cat['PSFFLUX_IVAR']))

		w1_nu_f_nu = np.array(309.54 * trimmed_cat['PSFFLUX'][:, 11]/1e9 * (con.c/(3.368*u.micron)).to('Hz').value)
		w2_nu_f_nu = np.array(171.787 * trimmed_cat['PSFFLUX'][:, 12]/1e9 * (con.c/(4.618*u.micron)).to('Hz').value)


		i_abs_mags = k_correct_richards(deredmags[:, 3], trimmed_cat[zkey])

		trimmed_cat['deredmags'] = deredmags

		trimmed_cat['e_mags'] = np.abs(magerrs)

		trimmed_cat['myMI'] = i_abs_mags
		trimmed_cat['g-i'] = (deredmags[:, 1] - deredmags[:, 3])
		trimmed_cat['i-z'] = (deredmags[:, 3] - deredmags[:, 4])
		trimmed_cat['logL1_5'] = wise_tools.rest_ir_lum(np.array([w1_nu_f_nu, w2_nu_f_nu]),
		                                                trimmed_cat[zkey], 1.5)

		if speczs:
			w3_nu_f_nu = np.array(
				31.674 * 10 ** (-trimmed_cat['W3mag_aw'] / 2.5) * (con.c / (12.082 * u.micron)).to('Hz').value)
			trimmed_cat['logL6'] = wise_tools.rest_ir_lum(np.array([w1_nu_f_nu, w2_nu_f_nu, w3_nu_f_nu]),
			                                              trimmed_cat[zkey], 6)


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


def relative_colors(tab, firstband='g', secondband='i', middlemode='median'):
	zs = np.copy(tab['Z'])
	colorlist = np.copy(tab['deredmags'][:, band_idxs[firstband]] - tab['deredmags'][:, band_idxs[secondband]])
	color_errs = np.sqrt((tab['e_mags'][:, band_idxs[firstband]])**2 + (tab['e_mags'][:, band_idxs[secondband]])**2)

	# bin redshift into 50 bins
	z_quantile_bins = pd.qcut(zs, 30, retbins=True)[1]

	clip = True
	vdblist, medzbins = [], []
	if middlemode == 'median':
		# loop over redshift bins
		for i in range(len(z_quantile_bins) - 1):
			in_z_bin = (zs < z_quantile_bins[i + 1]) & (zs >= z_quantile_bins[i])
			idxs_in_bin = np.where(in_z_bin)
			if clip:
				medcolor = astrostats.sigma_clipped_stats(colorlist[idxs_in_bin], sigma_upper=0.5)[1]
			else:
				medcolor = np.median(colorlist[idxs_in_bin])
			medz = np.median(zs[in_z_bin])
			vdblist.append(spectrumtools.vdb_color_at_z(medz, 0) - medcolor)
			medzbins.append(medz)

			colorlist[idxs_in_bin] = colorlist[idxs_in_bin] - medcolor

	# calculate colors relative to the modal color at a given redshift, which should be unaffected by reddening
	# calculate mode by summing gaussians centered at observed colors, with widths given by color errors
	elif middlemode == 'mode':
		colorgrid = np.linspace(-1, 2, 1000)



		# loop over redshift bins
		for i in range(len(z_quantile_bins) - 1):
			in_z_bin = (zs < z_quantile_bins[i + 1]) & (zs >= z_quantile_bins[i])
			idxs_in_bin = np.where(in_z_bin)
			gminis_in_bin = colorlist[idxs_in_bin]

			gausses = []
			for j in range(len(gminis_in_bin)):

				gausses.append(gaussian(colorgrid, gminis_in_bin[j], color_errs[idxs_in_bin][j]))
			gauss_sum = np.sum(gausses, axis=0)
			modecolor = colorgrid[gauss_sum.argmax()]

			#roundcolors = np.around(gminis[idxs_in_bin], 3)

			#modecolor = stats.mode(roundcolors)[0]

			colorlist[idxs_in_bin] = colorlist[idxs_in_bin] - modecolor
	else:
		return 'wrong mode'
	np.array(vdblist).dump('vdbcorrection.npy')
	np.array(medzbins).dump('medzbins.npy')

	return colorlist

def select_dust_reddened_qsos(tab):

	zlinspace = np.linspace(np.min(tab['Z']), np.max(tab['Z']), 50)
	relcolors = []
	for z in zlinspace:
		relcolors.append(spectrumtools.relative_vdb_color(z, ebv=0.08)-0.05)
	fit = np.polyfit(zlinspace, relcolors, 10)
	linmod = np.polyval(fit, tab['Z'])
	tab['bin'] = np.zeros(len(tab))
	offset_diff = tab['deltagmini'] - linmod
	tab['bin'][np.where(offset_diff > 0)] = -1.
	return tab

def mateos_cut(table):
	allwise_idxs = (table['e_W1mag_cw'] < 0.2) & (table['e_W2mag_cw'] < 0.2) & (table['e_W3mag_aw'] < 0.3)
	allwise_tab = table[allwise_idxs]
	w1, w2, w3 = allwise_tab['W1mag_cw'], allwise_tab['W2mag_cw'], allwise_tab['W3mag_aw']

	x = w2 - w3
	y = w1 - w2

	lowlim = 0.315*x - 0.222
	highlim = 0.315*x + 0.796

	leftlim = -3.172 * x + 7.264

	inbox = (y <= highlim) & (y >= lowlim) & (y >= leftlim)

	boxtab = allwise_tab[inbox]
	boxtab['PQSO'] = np.ones(len(boxtab))
	return boxtab



def lum_and_z_cut(qso_cat_name, lumcut, minz, maxz, plots, magcut=100, pcut=0.9, peakscut=1, apply_planck_mask=True,
                  band='i', colorkey='g-i'):

	#qso_cat = fits.open('catalogs/derived/%s_new_uw.fits' % qso_cat_name, memmap=False)[1].data
	qso_cat = Table.read('catalogs/derived/%s_new.fits' % qso_cat_name, memmap=False)

	#qso_cat = mateos_cut(qso_cat)


	zkey = 'Z'
	firstband, secondband = colorkey.split('-')[0], colorkey.split('-')[1]
	qso_cat = qso_cat[np.where((qso_cat['e_mags'][:, band_idxs[secondband]] < 0.2) & (np.isnan(qso_cat['logL1_5']) == False))]

	# if a photometric catalog, make cuts on flags, probability, and number of redshift peaks
	if (qso_cat_name == 'xdqso') or (qso_cat_name == 'xdqso_specz'):
		qso_cat = qso_cat[np.where((qso_cat['GOOD'] == 0) & (qso_cat['PQSO'] >= pcut) & (qso_cat['BAD_FIELD'] == 0) &
		                           (qso_cat['NPEAKS'] <= peakscut) & (qso_cat['NPEAKS'] > 0)
		                            & (qso_cat['e_mags'][:, band_idxs[secondband]] < 0.2) & (qso_cat['PEAKFWHM'] < 0.75))]
	# & (qso_cat['BRIGHT_STAR'] == 0)
	#if 'xd' in qso_cat_name:
	#	qso_cat = qso_cat[np.where((qso_cat['PEAKPROB'] == 1) & (qso_cat['PEAKFWHM'] == 0))]

	#eboss_mask = pymangle.Mangle('footprints/eBOSS_QSOandLRG_fullfootprintgeometry_noveto.ply')
	#goodixs = eboss_mask.contains(qso_cat['RA'], qso_cat['DEC'])
	#qso_cat = qso_cat[goodixs]


	# removes QSOs which fall inside mask for Planck lensing, which you can't use anyways
	if apply_planck_mask:
		map4stack = hp.read_map('maps/smoothed_masked_planck.fits', dtype=np.single)
		ls, bs = stacking.equatorial_to_galactic(qso_cat['RA'], qso_cat['DEC'])
		outsidemaskbool = (map4stack[hp.ang2pix(2048, ls, bs, lonlat=True)] != hp.UNSEEN)
		qso_cat = qso_cat[np.where(outsidemaskbool)]

	qso_cat = qso_cat[np.where((qso_cat['Z'] >= minz) & (qso_cat['Z'] <= maxz))]

	#z_cut_tab['delta%smin%s' % (firstband, secondband)] = relative_colors(z_cut_tab, firstband=firstband,
	#                                                                      secondband=secondband, middlemode='mode')
	#z_cut_tab['deltagmini'] = relative_colors(z_cut_tab, firstband='g', secondband='i', middlemode='mode')

	vdb = True
	if vdb:
		compgi_binned = []
		zlist = np.linspace(0, 10, 1000)
		for j in range(len(zlist)):
			compgi_binned.append(spectrumtools.vdb_color_at_z(zlist[j]))
		interpcolors = np.interp(qso_cat[zkey], zlist, compgi_binned)
		qso_cat['deltagmini'] = qso_cat['g-i'] - interpcolors


	#z_cut_tab = select_dust_reddened_qsos(z_cut_tab)

	#
	if band == 'i':
		i_app = qso_cat['deredmags'][:, 3]
		# cut on absolute or apparent magnitude, and/or redshift
		complete_cut = np.where((qso_cat['myMI'] <= lumcut) & (qso_cat['myMI'] > -100) & (i_app < magcut))
	elif band == '1.5':
		complete_cut = np.where((qso_cat['logL1_5'] >= lumcut))
	qso_cat = qso_cat[complete_cut]




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
		if band == 'i':
			plotting.MI_vs_z(qso_cat[zkey], qso_cat['myMI'], len(complete_cut[0]), magcut, minz, maxz, lumcut,
			                 qso_cat_name, limMs)
		elif band == '1.5':
			plotting.plot_lum_vs_z(qso_cat[zkey], qso_cat['logL1_5'], len(complete_cut[0]), minz=minz, maxz=maxz,
			                       lumcut=lumcut, qso_cat_name=qso_cat_name)
		#plotting.w1_minus_w2_plot(t['W1_MAG'], t['W2_MAG'], qso_cat_name)

	qso_cat.write('catalogs/derived/%s_complete.fits' % qso_cat_name, format='fits', overwrite=True)
	#fits.PrimaryHDU(t).writeto('catalogs/derived/%s_complete.fits' % qso_cat_name,  overwrite=True)

	#if plots:
		#plotting.color_hist(qso_cat_name, colorkey=colorkey)







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


def radio_matched(qso_tab, radio_survey='FIRST'):
	qso_tab['RL'] = np.full(len(qso_tab), np.nan)
	#qso_ = Table.read('catalogs/derived/%s_complete.fits' % qso_cat_name)
	if radio_survey == 'FIRST':
		radiocat = Table.read('catalogs/radio_cats/first_14dec17.fits')
	else:
		return

	inradiofootprint = healpixhelper.match_footprints((qso_tab['RA'], qso_tab['DEC']), (radiocat['RA'],
	                                                                                    radiocat['DEC']), nside=256)
	qso_tab['RL'][inradiofootprint] = 0

	radiocoords = SkyCoord(ra=radiocat['RA'], dec=radiocat['DEC'])

	qsocoords = SkyCoord(ra=qso_tab['RA'] * u.deg, dec=qso_tab['DEC'] * u.deg)
	firstidxs, qsoidxs, d2d, d3d = qsocoords.search_around_sky(radiocoords, 10 * u.arcsec)

	qso_tab['RL'][qsoidxs] = 1
	return qso_tab
	#firstmatchedcat.write('catalogs/derived/%s_RL.fits' % qso_cat_name, format='fits', overwrite=True)



def bin_qsos(qso_cat_name, plots, nbins, binmode, offset=False, colorkey='g-i', lumkey='logL1_5'):

	if qso_cat_name == 'dr14':
		qso_cat = Table.read('catalogs/derived/%s_complete.fits' % qso_cat_name)
	elif qso_cat_name == 'dr16':
		qso_cat = Table.read('catalogs/derived/%s_complete.fits' % qso_cat_name)
	elif qso_cat_name == 'eboss_lss':
		qso_cat = Table.read('catalogs/derived/eboss_lss_complete.fits')
	elif qso_cat_name == 'gaia':
		#qso_cat = Table.read('catalogs/WISEQSOs/Gaia_unWISE_AGNs.fits')
		qso_cat = Table.read('catalogs/WISEQSOs/gaia_unwise_allwise.fits')
	else:
		qso_cat = Table.read('catalogs/derived/%s_complete.fits' % qso_cat_name)

	qso_cat['bin'] = np.zeros(len(qso_cat))

	zs = qso_cat['Z']
	minz, maxz = np.min(zs), np.max(zs)

	qso_cat['weight'] = np.zeros(len(qso_cat))

	if binmode == 'color':
		firstband, secondband = colorkey.split('-')[0], colorkey.split('-')[1]
		if qso_cat_name == 'gaia':
			qso_cat = qso_cat[
				np.where((qso_cat['Z'] > 0.75) & (qso_cat['Z'] < 2.2) & (qso_cat[firstband] < 900) & (
							qso_cat[secondband] < 900) & (qso_cat['EBV'] < 0.5) & (qso_cat['e_W4mag'] < 0.3))]
			colors = qso_cat[firstband] - qso_cat[secondband]
		else:
			colors = qso_cat['deredmags'][:, band_idxs[firstband]] - qso_cat['deredmags'][:, band_idxs[secondband]]

		indicesbybin = bin_samples.bin_by_color(colors, zs, nbins, False)
	elif binmode == 'color_offset':
		indicesbybin = bin_samples.bin_by_color(qso_cat['deltagmini'], zs, nbins, True)
	elif binmode == 'Av':
		indicesbybin = bin_samples.bin_by_Av(qso_cat['deltagmini'], zs, nbins)
	elif binmode == 'radio':
		qso_cat = radio_matched(qso_cat)
		indicesbybin = bin_samples.bin_by_radio(qso_cat['RL'])
	elif binmode == 'bal':
		indicesbybin = bin_samples.bin_by_bal(qso_cat['BAL_PROB'])
	elif binmode == 'lum':
		indicesbybin = bin_samples.bin_by_lum(qso_cat[lumkey], nbins)
	else:
		return

	lumlist, zslist = [], []
	for j in range(len(indicesbybin)):
		qso_cat['bin'][indicesbybin[j]] = j+1

		lumlist.append(qso_cat[lumkey][indicesbybin[j]])
		zslist.append(qso_cat['Z'][indicesbybin[j]])


	qso_tab = Table(qso_cat)

	if binmode == 'color':



		qso_tab['%s' % colorkey] = colors

		if 'xd' in qso_cat_name:
			qso_tab['weight'] = weighting.convolved_weights(qso_tab['PQSO'], len(qso_tab), indicesbybin, lumlist,
			                                                zslist, 40, 50, minz, maxz, 50, 20)
		elif qso_cat_name == 'gaia':
			qso_tab['weight'] = np.ones(len(qso_tab))
		else:
			qso_tab['weight'] = weighting.lum_z_2d_weights(indicesbybin, len(qso_tab), lumlist, zslist, 40, 50, minz,
			                                               maxz, 50, 20)

	elif binmode == 'bal':
		qso_tab['weight'] = weighting.lum_z_2d_weights(indicesbybin, len(qso_tab), lumlist, zslist, 40, 50, minz,
		                                               maxz, 50, 20)
	elif binmode == 'radio':
		qso_tab['weight'] = weighting.redshift_weights(indicesbybin, len(qso_tab), zslist, 20, minz, maxz)

	qso_tab.write('catalogs/derived/%s_binned.fits' % qso_cat_name, format='fits', overwrite=True)

	if binmode == 'color':
		bluetab = qso_tab[np.where(qso_tab['bin'] == 1)]
		ctrltab = qso_tab[np.where((qso_tab['bin'] == round(nbins/2)))]
		redtab = qso_tab[np.where(qso_tab['bin'] == nbins)]

		if plots:
			plotting.g_minus_i_plot(qso_cat_name, offset)
			plotting.color_v_z(qso_cat_name, colorkey)
			plotting.lum_dists(qso_cat_name, 100, bluetab[lumkey], ctrltab[lumkey], redtab[lumkey])
			plotting.z_dists(qso_cat_name, bluetab['Z'], ctrltab['Z'], redtab['Z'])








