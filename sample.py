from astropy.io import fits
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
import healpy as hp
from colossus.cosmology import cosmology
import stacking
import plotting
import importlib
from astropy.table import Table, vstack
import pandas as pd
from scipy.optimize import curve_fit
from astropy import stats as astrostats
import astropy.constants as con
import spectrumtools
import wise_tools
import lensingModel
import fitting
import autocorrelation
import weighting
import binning
importlib.reload(binning)
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


band_idxs = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "nuv": 5, "fuv": 6, "Y": 7, "J": 8, "H": 9, "K": 10, "W1": 11, "W2": 12}
bsoftpars = np.array([1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10, 1e-10, 1e-10, 3e-10, .9e-10, 5.2e-10, 1.1e-9, 1e-10, 1e-10])
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
		trimmed_cat['logL'] = wise_tools.rest_ir_lum(trimmed_cat['W1_FLUX'], trimmed_cat['W2_FLUX'], trimmed_cat['Z'], 1.5)
		trimmed_cat['i_mag'] = dered_i_mags
		trimmed_cat.write('catalogs/derived/dr16_new.fits', format='fits', overwrite=True)

	elif sample == 'eboss_lss':
		qso_cat = Table(fits.open('catalogs/lss/eBOSS_fullsky_comov_phot.fits')[1].data)
		goodwise = (qso_cat['PSFFLUX'][:, 11] > 0) & (qso_cat['PSFFLUX'][:, 12] > 0)
		trimmed_cat = qso_cat[np.where(goodwise)]
		psfmaggies = np.array(trimmed_cat['PSFFLUX'] / 1e9) * vega_to_ab
		psfmags = -2.5 / np.log(10.) * (
				np.arcsinh(psfmaggies / (2 * bsoftpars)) + np.log(bsoftpars))
		deredmags = psfmags - trimmed_cat['EXTINCTION']

		magerrs = 1 / (trimmed_cat['PSFFLUX'] * np.sqrt(trimmed_cat['PSFFLUX_IVAR']))

		w1_nu_f_nu = np.array(
			309.54 * trimmed_cat['PSFFLUX'][:, 11] / 1e9 * (con.c / (3.368 * u.micron)).to('Hz').value)
		w2_nu_f_nu = np.array(
			171.787 * trimmed_cat['PSFFLUX'][:, 12] / 1e9 * (con.c / (4.618 * u.micron)).to('Hz').value)

		trimmed_cat['deredmags'] = deredmags
		trimmed_cat['myMI'] = k_correct_richards(deredmags[:, 3], trimmed_cat['Z'])
		trimmed_cat['g-i'] = deredmags[:, 1] - deredmags[:, 3]
		trimmed_cat['logL1_5'] = wise_tools.rest_ir_lum(np.array([w1_nu_f_nu, w2_nu_f_nu]),
		                                                trimmed_cat['Z'], 1.5)
		trimmed_cat['e_mags'] = np.abs(magerrs)

		trimmed_cat.write('catalogs/derived/eboss_lss_new.fits', format='fits', overwrite=True)

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
	tab['colorbin'] = np.zeros(len(tab))
	offset_diff = tab['deltagmini'] - linmod
	tab['colorbin'][np.where(offset_diff > 0)] = -1.
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



def luminosity_complete_cut(qso_cat_name, lumcut, minz, maxz, plots, magcut=100, pcut=0.9, peakscut=1, apply_planck_mask=True, band='i', colorkey='g-i'):
	qso_cat = Table(fits.open('catalogs/derived/%s_new.fits' % qso_cat_name)[1].data)
	#qso_cat = mateos_cut(qso_cat)


	zkey = 'Z'
	firstband, secondband = colorkey.split('-')[0], colorkey.split('-')[1]

	# if a photometric catalog, make cuts on flags, probability, and number of redshift peaks
	"""if (qso_cat_name == 'xdqso') or (qso_cat_name == 'xdqso_specz'):
		qso_cat = qso_cat[np.where((qso_cat['GOOD'] == 0) & (qso_cat['PQSO'] >= pcut) & (qso_cat['BAD_FIELD'] == 0) &
		                           (qso_cat['NPEAKS'] <= peakscut) & (qso_cat['NPEAKS'] > 0) & (qso_cat['e_mags'][:, band_idxs[firstband]] < 0.33)
		                            & (qso_cat['e_mags'][:, band_idxs[secondband]] < 0.1) & (qso_cat['PEAKFWHM'] < 0.75))]"""
	# & (qso_cat['BRIGHT_STAR'] == 0)
	qso_cat = qso_cat[np.where((qso_cat['PEAKPROB'] == 1) & (qso_cat['PEAKFWHM'] == 0) & (qso_cat['e_mags'][:, band_idxs[secondband]] < 0.2))]

	#eboss_mask = pymangle.Mangle('footprints/eBOSS_QSOandLRG_fullfootprintgeometry_noveto.ply')
	#goodixs = eboss_mask.contains(qso_cat['RA'], qso_cat['DEC'])
	#qso_cat = qso_cat[goodixs]


	# removes QSOs which fall inside mask for Planck lensing, which you can't use anyways
	if apply_planck_mask:
		map4stack = hp.read_map('maps/smoothed_masked_planck.fits', dtype=np.single)
		ls, bs = stacking.equatorial_to_galactic(qso_cat['RA'], qso_cat['DEC'])
		outsidemaskbool = (map4stack[hp.ang2pix(2048, ls, bs, lonlat=True)] != hp.UNSEEN)
		qso_cat = qso_cat[np.where(outsidemaskbool)]

	z_cut_tab = qso_cat[np.where((qso_cat['Z'] >= minz) & (qso_cat['Z'] <= maxz))]

	#z_cut_tab['delta%smin%s' % (firstband, secondband)] = relative_colors(z_cut_tab, firstband=firstband,
	#                                                                      secondband=secondband, middlemode='mode')
	#z_cut_tab['deltagmini'] = relative_colors(z_cut_tab, firstband='g', secondband='i', middlemode='mode')



	#z_cut_tab = select_dust_reddened_qsos(z_cut_tab)

	#
	if band == 'i':
		i_app = z_cut_tab['deredmags'][:, 3]
		# cut on absolute or apparent magnitude, and/or redshift
		complete_cut = np.where((z_cut_tab['myMI'] <= lumcut) & (z_cut_tab['myMI'] > -100) & (i_app < magcut))
	elif band == '1.5':
		complete_cut = np.where((z_cut_tab['logL1_5'] >= lumcut))
	t = z_cut_tab[complete_cut]




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
			plotting.MI_vs_z(qso_cat[zkey], qso_cat['myMI'], len(complete_cut[0]), magcut, minz, maxz, lumcut, qso_cat_name, limMs)
		elif band == '1.5':
			plotting.plot_lum_vs_z(qso_cat[zkey], qso_cat['logL1_5'], len(complete_cut[0]), minz=minz, maxz=maxz, lumcut=lumcut, qso_cat_name=qso_cat_name)
		#plotting.w1_minus_w2_plot(t['W1_MAG'], t['W2_MAG'], qso_cat_name)

	t.write('catalogs/derived/%s_complete.fits' % qso_cat_name, format='fits', overwrite=True)

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






def red_blue_samples(qso_cat_name, plots, ncolorbins, offset=False, remove_reddest=False, colorkey='g-i', lumkey='logL1_5'):


	if offset:
		colorkey = 'deltagmini'
	else:
		colorkey = colorkey
	if qso_cat_name == 'dr14':
		qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data

	elif qso_cat_name == 'dr16':

		qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data
	elif qso_cat_name == 'eboss_lss':
		qso_cat = fits.open('catalogs/derived/eboss_lss_complete.fits')[1].data

	else:
		qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data

	if remove_reddest:
		nonred_tab = Table(qso_cat[np.where(qso_cat['colorbin'] == 0)])
		reddenedtab = Table(qso_cat[np.where(qso_cat['colorbin'] == -1)])
		# change this?
		reddenedtab['weight'] = np.ones(len(reddenedtab))
	else:
		nonred_tab = Table(qso_cat)
		nonred_tab['colorbin'] = np.zeros(len(nonred_tab))



	firstband, secondband = colorkey.split('-')[0], colorkey.split('-')[1]

	colors = nonred_tab['deredmags'][:, band_idxs[firstband]] - nonred_tab['deredmags'][:, band_idxs[secondband]]


	zs = nonred_tab['Z']
	minz, maxz = np.min(zs), np.max(zs)

	nonred_tab['weight'] = np.zeros(len(nonred_tab))


	indicesbycolor = binning.bin_by_color(colors, zs, ncolorbins, offset)
	#indicesbycolor = bin_by_Av(nonred_tab, ncolorbins)
	lumlist, zslist = [], []
	for j in range(len(indicesbycolor)):
		nonred_tab['colorbin'][indicesbycolor[j]] = j+1
		lumlist.append(nonred_tab[lumkey][indicesbycolor[j]])
		zslist.append(nonred_tab['Z'][indicesbycolor[j]])

	if remove_reddest:
		qso_tab = vstack([nonred_tab, reddenedtab])
	else:
		qso_tab = nonred_tab


	bluetab = qso_tab[np.where(qso_tab['colorbin'] == 1)]
	ctrltab = qso_tab[np.where((qso_tab['colorbin'] == round(ncolorbins/2)))]
	redtab = qso_tab[np.where(qso_tab['colorbin'] == ncolorbins)]

	qso_tab['%s' % colorkey] = colors

	if 'xd' in qso_cat_name:
		qso_tab['weight'] = weighting.convolved_weights(qso_tab['PQSO'], len(qso_tab), indicesbycolor, lumlist, zslist, 40, 50, minz, maxz, 50, 20)
	else:
		qso_tab['weight'] = weighting.lum_z_2d_weights(indicesbycolor, len(qso_tab), lumlist, zslist, 40, 50, minz, maxz, 50, 20)

	"""for j in range(len(indicesbycolor)):
		if (qso_cat_name == 'dr14') or (qso_cat_name == 'dr16'):
			qso_tab['weight'][indicesbycolor[j]] = lum_weights(lumlist, 21, 26, 100, colorbin=j)
		else:
			colortab = qso_tab[indicesbycolor[j]]
			qso_tab['weight'][indicesbycolor[j]] = convolved_weights(colortab['PQSO'], indicesbycolor, lumlist, zslist, 40, 50, minz, maxz, 30, 20, colorbin=j)
			#qso_tab['weight'][indicesbycolor[j]] = lum_z_2d_weights(lumlist, zslist, 40, 50, minz, maxz, 50, 20, colorbin=j)"""

	qso_tab.write('catalogs/derived/%s_colored.fits' % qso_cat_name, format='fits', overwrite=True)

	if plots:
		plotting.g_minus_i_plot(qso_cat_name, offset)
		plotting.color_v_z(qso_cat_name, colorkey)

		plotting.lum_dists(qso_cat_name, 100, bluetab[lumkey], ctrltab[lumkey], redtab[lumkey])
		plotting.z_dists(qso_cat_name, bluetab['Z'], ctrltab['Z'], redtab['Z'])








def first_matched(qso_cat_name):
	qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data
	firstcat = fits.open('catalogs/radio_cats/first_14dec17.fits')[1].data


	firstcoords = SkyCoord(ra=firstcat['RA'] * u.deg, dec=firstcat['DEC'] * u.deg)

	sdsscoords = SkyCoord(ra=qso_cat['RA'] * u.deg, dec=qso_cat['DEC'] * u.deg)
	firstidxs, sdssidxs, d2d, d3d = sdsscoords.search_around_sky(firstcoords, 10 * u.arcsec)

	firstmatchedcat = Table(qso_cat[sdssidxs])
	firstmatchedcat.write('catalogs/derived/%s_RL.fits' % qso_cat_name, format='fits', overwrite=True)



def radio_detect_fraction(qso_cat_name, colorkey, radio_name='FIRST', lowmag=10, highmag=30, return_plot=False, offset=False):

	#qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data
	#qso_cat = fits.open('QSO_cats/dr7_bh_Nov19_2013.fits')[1].data
	qso_cat = fits.open('catalogs/derived/%s_colored.fits' % qso_cat_name)[1].data

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


	qso_cat = qso_cat[healpixhelper.match_footprints((qso_cat['RA'], qso_cat['DEC']), (radio_cat[radrakey], radio_cat[raddeckey]), nside=256)]
	qso_cat = qso_cat[np.where((qso_cat['deredmags'][:, 3] < highmag) & (qso_cat['deredmags'][:, 3] > lowmag))]


	if offset:
		colors = qso_cat['deltagmini']
	else:
		colors = qso_cat['g-i']
	zs = qso_cat['Z']

	#gminusibinidxs = bin_by_color(colors, zs, bins, offset)

	radio_detect_frac = []
	for i in range(int(np.max(qso_cat['colorbin']))):

		binnedcat = qso_cat[np.where(qso_cat['colorbin'] == (i+1))]

		# try to match in bolometric luminosity
		# use luminosity weights as probabilities to randomly draw from each sample such that all samples should
		# represent same luminosity distribution
		normed_weights = binnedcat['weight']/np.sum(binnedcat['weight'])
		binnedcat = binnedcat[np.random.choice(len(binnedcat), len(binnedcat), p=normed_weights)]

		coordsinbin = SkyCoord(ra=binnedcat['RA'] * u.deg, dec=binnedcat['DEC'] * u.deg)
		firstidxs, binidxs, d2d, d3d = coordsinbin.search_around_sky(radiocoords, 10 * u.arcsec)
		radio_detect_frac.append(len(firstidxs)/len(binnedcat))

	plotting.radio_detect_frac_plot(radio_detect_frac, colorkey, surv_name=radio_name, return_plot=return_plot)

# bin up sample into bins of color or color offset, and stack FIRST images in each bin
# this stack can be used to estimate the median flux, median radio luminosity, or median radio loudness of each bin
def median_radio_flux_for_color(qso_cat_name, colorkey, mode='flux', remove_detections=False, minz=0, maxz=10, minL=40, maxL=50, nbootstraps=0, offset=False, remove_reddest=False):
	import first_stacking
	#qso_cat = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data
	qso_cat = fits.open('catalogs/derived/%s_colored.fits' % qso_cat_name)[1].data

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
	qso_cat = qso_cat[np.where((qso_cat['logL1_5'] > minL) & (qso_cat['logL1_5'] < maxL))]




	#reddest_cat = qso_cat[np.where(qso_cat['deltagmini'] > 0.25)]
	#qso_cat = qso_cat[np.where(qso_cat['deltagmini'] < 0.25)]

	#gminusibinidxs = bin_by_color(qso_cat[colorkey], qso_cat['Z'], bins, offset)

	median_radioflux_in_bins, medLbols, medradlum, medradloudness, boot_errs = [], [], [], [], []
	medcolors = np.linspace(0, 1, int(np.max(qso_cat['colorbin'])))


	for i in range(int(np.max(qso_cat['colorbin']))):
		#if i==bins:
		#	binnedcat = reddest_cat
		#else:
		#	binnedcat = qso_cat[gminusibinidxs[i]]
		#binnedcat = qso_cat[gminusibinidxs[i]]
		binnedcat = qso_cat[np.where(qso_cat['colorbin'] == (i+1))]
		#medcolors.append(np.median(binnedcat['deltagminz']))
		stacked_flux = np.max(first_stacking.median_stack(binnedcat['OBJID_XDQSO']))
		median_radioflux_in_bins.append(stacked_flux)
		medLbol = np.median(binnedcat['logL1_5'])
		medLbols.append(medLbol)
		medzinbin = np.median(binnedcat['Z'])
		medlumdist = astropycosmo.luminosity_distance(medzinbin)
		lumnu = ((4 * np.pi * (medlumdist ** 2) * (stacked_flux * u.Jy) / ((1 + medzinbin) ** (1 - 0.5))).to('erg')).value
		medradlum.append(lumnu)
		medradloudness.append(np.log10((1.4e9*lumnu)/(10 ** medLbol)))

		bootmedian_radio_in_bins, bootLbols = [], []
		if nbootstraps > 0:
			for j in range(nbootstraps):
				bootidxs = np.random.choice(len(binnedcat), len(binnedcat))
				bootbinnedcat = binnedcat[bootidxs]
				bootmedian_radio_in_bins.append(np.max(first_stacking.median_stack(bootbinnedcat['OBJID_XDQSO'])))
				bootLbols.append(np.median(bootbinnedcat['logL1_5']))
			bootlumnu = (
			(4 * np.pi * (medlumdist ** 2) * (bootmedian_radio_in_bins * u.Jy) / ((1 + medzinbin) ** (1 - 0.5))).to('erg')).value
			radio_loudnesss = np.log10((1.4e9*bootlumnu)/(10**(np.array(bootLbols))))

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
		plotting.plot_median_radio_flux(colorkey, remove_reddest=remove_reddest)
		return median_radioflux_in_bins
	elif mode == 'lum':
		np.array([medcolors, medradlum, boot_errs]).dump('plotting_results/first_lum_for_color.npy')
		plotting.plot_median_radio_luminosity(colorkey, remove_reddest=remove_reddest)
	elif mode == 'loud':
		np.array([medcolors, medradloudness, boot_errs]).dump('plotting_results/first_loud_for_color.npy')
		plotting.plot_radio_loudness(colorkey, remove_reddest=remove_reddest)
		return medradloudness, boot_errs



def linear_model(x, a, b):
	return a*x+b

def kappa_for_color(qso_cat_name, colorkey, bins=10, removereddest=False, dostacks=True, mission='planck', use_weights=True):


	#if removereddest:
		#qso_cat = qso_cat[remove_reddest_bin(qso_cat[colorkey], qso_cat['Z'], 10, offset)]
	#reddest_cat = qso_cat[np.where(qso_cat['colorbin'] == -1)]
	#qso_cat = qso_cat[np.where(qso_cat['deltagmini'] < 0.25)]

	qso_cat = fits.open('catalogs/derived/%s_colored.fits' % qso_cat_name)[1].data
	#qso_cat = fits.open('catalogs/derived/dr16_bal.fits')[1].data
	#gminusibinidxs = bin_by_color(qso_cat['g-i'], qso_cat['Z'], bins, offset=False)

	kappas, errs, boots = [], [], []
	colors = np.linspace(0, 1, bins)
	masses = []
	planck_kappas, act_kappas = [], []

	for i in range(bins):

		binnedcat = qso_cat[np.where(qso_cat['colorbin'] == (i+1))]
		if use_weights:
			if 'xd' in qso_cat_name:
				probweights = binnedcat['PQSO']
			else:
				probweights = None
			weights = binnedcat['weight']
		else:
			probweights, weights = None, None

		if dostacks:

			ras, decs = binnedcat['RA'], binnedcat['DEC']
			if mission == 'planck':
				planckkappa = stacking.fast_stack(ras, decs, hp.read_map('maps/smoothed_masked_planck.fits'), iterations=500, bootstrap=True, weights=weights, prob_weights=probweights)
				kappas.append(planckkappa[0])
				errs.append(np.std(planckkappa[1]))
				act_kappas = None
				planck_kappas = None
			elif mission == 'act':
				actkappa = stacking.fast_stack(ras, decs, hp.read_map('maps/both_ACT.fits'), iterations=500, bootstrap=True, weights=weights, prob_weights=probweights)
				kappas.append(actkappa[0])
				errs.append(np.std(actkappa[1]))
				act_kappas = None
				planck_kappas = None
			else:
				planckkappa = stacking.fast_stack(ras, decs, hp.read_map('maps/smoothed_masked_planck.fits'),
				                                  iterations=500, bootstrap=True, weights=weights, prob_weights=probweights)

				actkappa = stacking.fast_stack(ras, decs, hp.read_map('maps/both_ACT.fits'), iterations=500,
				                               bootstrap=True, weights=weights, prob_weights=probweights)
				planckvariance = np.var(planckkappa[1])
				actvariance = np.var(actkappa[1])
				avgd_kappa = np.average([planckkappa[0], actkappa[0]], weights=[1/planckvariance, 1/actvariance])
				kappas.append(avgd_kappa)
				planck_kappas.append(np.array([planckkappa[0], np.sqrt(planckvariance)]))
				act_kappas.append(np.array([actkappa[0], np.sqrt(actvariance)]))
				errs.append(np.sqrt(planckvariance+actvariance)/2)

			"""if len(stackkappa) > 1:
				kappas.append(stackkappa[0])
				errs.append(np.std(stackkappa[1]))
				boots.append(stackkappa[1])
			else:
				kappas.append(stackkappa)
				errs.append(0)"""

			#kappas_for_masses = np.linspace(0.0005, 0.005, 50)
			#masses_for_kappas = lensingModel.kappa_mass_relation(binnedcat['Z'], kappas_for_masses)
			#masses.append(np.interp(stackkappa[0], kappas_for_masses, masses_for_kappas))
		else:
			kap = np.load('peakkappas/%s_%s_kappa.npy' % (qso_cat_name, i), allow_pickle=True)
			kappas.append(kap[0])
			errs.append(kap[1])
			boots.append(np.random.normal(kap[0], kap[1], 500))


	#print(masses)
	#return masses


	linfit, pcov = curve_fit(linear_model, colors, kappas, sigma=errs)
	print(linfit[0])

	"""boots = np.array(boots)
	slopes = []
	for i in range(len(boots[0])):
		booted = boots[:, i]
		poptboot, pcovboot = curve_fit(linear_model, colors, booted)
		slopes.append(poptboot[0])
	print(np.std(slopes))"""

	#kappas_for_masses = np.linspace(0.001, 0.004, 50)
	#masses_for_kappas = lensingModel.kappa_mass_relation(qso_cat['Z'], kappas_for_masses)





	if not removereddest:
		colors.append(np.median(reddest_cat['deltagmini']))
		kap = stacking.fast_stack(reddest_cat['RA'], reddest_cat['DEC'], hp.read_map('maps/smoothed_masked_planck.fits'), iterations=100, bootstrap=True)
		if len(kap) > 1:
			kappas.append(kap[0])
			errs.append(np.std(kap[1]))
		else:
			kappas.append(kap)
			errs.append(0)

	#plotting.plot_kappa_v_color(kappas, errs, transforms=[kappas_for_masses, masses_for_kappas], remove_reddest=removereddest, linfit=linfit)
	plotting.plot_kappa_v_color(kappas, errs, colorkey, planck_kappas=planck_kappas, act_kappas=act_kappas,
	                transforms=None, remove_reddest=removereddest, linfit=linfit)
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

	gminusibinidxs = binning.bin_by_color(qso_cat[colorkey], qso_cat['Z'], bins, offset)

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

	gminusibinidxs = binning.bin_by_color(qso_cat[colorkey], qso_cat['Z'], bins, offset)

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

def clustering_for_color(qso_cat_name, mode, cap, bins=10, minscale=-1, maxscale=0, use_weights=False):
	n_samples = int(np.max(fits.open('catalogs/derived/%s_colored.fits' % qso_cat_name)[1].data['colorbin']))
	samples = np.arange(n_samples)
	samples = [0, n_samples - 1]
	if mode == 'ang_cross':
		for sample in samples:
			autocorrelation.cross_correlation_function_angular(qso_cat_name, sample+1, nbins=bins, nbootstraps=10, nthreads=12, minscale=minscale, maxscale=maxscale)
		plotting.plot_ang_cross_corr(qso_cat_name, bins, minscale, maxscale, samples)
	elif mode == 'spatial_cross':
		for sample in samples:
			autocorrelation.cross_corr_func_spatial(qso_cat_name, sample+1, minscale, maxscale, cap, nbins=bins, nbootstraps=3, nthreads=12, useweights=use_weights)
		plotting.plot_spatial_cross_corr(samples)
	elif mode == 'spatial':
		for j in range(n_samples):
			autocorrelation.spatial_correlation_function(qso_cat_name, j+1, bins, nbootstraps=3, nthreads=12, useweights=True, minscale=minscale, maxscale=maxscale)
		plotting.plot_spatial_correlation_function(bins, minscale, maxscale, n_samples)