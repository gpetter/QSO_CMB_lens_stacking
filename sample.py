from astropy.io import fits
import astropy.units as u
import numpy as np
import healpy as hp
from colossus.cosmology import cosmology
import stacking
import importlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import floor, ceil
from astropy.table import Table
from scipy import interpolate
import pandas as pd
importlib.reload(stacking)

cosmo = cosmology.setCosmology('planck18')
astropycosmo = cosmo.toAstropy()


band_idxs = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4}
bsoftpars = [1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10]


def luminosity_complete_cut(qso_cat_name, lumcut, minz, maxz, plots):
	map4stack = hp.read_map('maps/smoothed_masked_planck.fits', dtype=np.single)

	if qso_cat_name == 'dr14':
		qso_cat = (fits.open('QSO_cats/DR14Q_v4_4.fits'))[1].data
		rakey = 'RA'
		deckey = 'DEC'
		zkey = 'Z'
		w1key = 'W1MAG'
		w2key = 'W2MAG'
		i_mags = qso_cat['psfmag'][:, band_idxs['i']]

		extinction = qso_cat['GAL_EXT'][:, band_idxs['i']]
		extinction[np.where(extinction < 0)] = 0
		dered_i_mags = i_mags - extinction

		ls, bs = stacking.equatorial_to_galactic(qso_cat[rakey], qso_cat[deckey])
		outsidemaskbool = map4stack[hp.ang2pix(2048, ls, bs, lonlat=True)] != hp.UNSEEN

		good_idxs = np.where(outsidemaskbool & (qso_cat['FIRST_MATCHED'] == 0))

	elif qso_cat_name == 'dr16':
		qso_cat = (fits.open('QSO_cats/DR16Q_v4.fits'))[1].data
		rakey = 'RA'
		deckey = 'DEC'
		zkey = 'Z'
		w1key = 'W1_MAG'
		w2key = 'W2_MAG'
		goodzs = (qso_cat[zkey] > 0)

		i_mags = qso_cat['psfmag'][:, band_idxs['i']]
		goodimags = (i_mags != -9999.)

		goodwise = (qso_cat['W1_FLUX'] > 0) & (qso_cat['W2_FLUX'] > 0)

		dered_i_mags = i_mags - qso_cat['EXTINCTION'][:, band_idxs['i']]

		ls, bs = stacking.equatorial_to_galactic(qso_cat[rakey], qso_cat[deckey])
		outsidemaskbool = map4stack[hp.ang2pix(2048, ls, bs, lonlat=True)] != hp.UNSEEN

		good_idxs = np.where(outsidemaskbool & goodimags & goodzs & goodwise)

	else:
		qso_cat = fits.open('QSO_cats/xdqso-z-cat.fits')[1].data
		rakey = 'RA_XDQSO'
		deckey = 'DEC_XDQSO'
		zkey = 'PEAKZ'
		i_maggies = qso_cat['psfflux'][:, band_idxs['i']] / 1e9
		i_mags = -2.5 / np.log(10.) * (np.arcsinh(i_maggies / (2 * bsoftpars[band_idxs['i']])) + np.log(bsoftpars[band_idxs['i']]))
		dered_i_mags = i_mags - qso_cat['EXTINCTION'][:, band_idxs['i']]

		ls, bs = stacking.equatorial_to_galactic(qso_cat[rakey], qso_cat[deckey])
		outsidemaskbool = map4stack[hp.ang2pix(2048, ls, bs, lonlat=True)] != hp.UNSEEN

		goodwise = (qso_cat['PSFFLUX'][:, 11] > 0) & (qso_cat['PSFFLUX'][:, 12] > 0)

		"""good_idxs = np.where((qso_cat['NPEAKS'] == 1) & (qso_cat['bright_star'] == False) &
			(qso_cat['bad_u'] == False) & (qso_cat['bad_field'] == False) &
			(qso_cat['wise_flagged'] == False) &
			(qso_cat['good'] == 0) & outsidemaskbool)[0]"""
		good_idxs = np.where((qso_cat['good'] == 0) & outsidemaskbool & goodwise)[0]

	qso_cat = qso_cat[good_idxs]
	dered_i_mags = dered_i_mags[good_idxs]

	distances = astropycosmo.luminosity_distance(qso_cat[zkey]).to(u.pc)
	i_abs_mags = dered_i_mags - 5 * np.log10(distances / (10 * u.pc)) + (2.5 * 0.5 * np.log10(1 + qso_cat[zkey])) - 0.596

	complete_cut = np.where((i_abs_mags <= lumcut) & (i_abs_mags > -100) & (qso_cat[zkey] >= minz) & (qso_cat[zkey] <= maxz))

	if plots:
		plt.figure(1, (10, 10))
		plt.scatter(qso_cat[zkey], i_abs_mags, c='k', s=0.01, alpha=0.5)
		plt.xlabel('z', fontsize=20)
		plt.ylabel('$M_{i}$ (z=2)', fontsize=20)
		plt.ylim(-22, -30)
		plt.text(3, -24, 'N = %s' % len(complete_cut[0]), fontsize=20)
		ax = plt.gca()
		rect = patches.Rectangle((minz, -30), (maxz - minz), (30 + lumcut), linewidth=1, edgecolor='y', facecolor='y',
				alpha=0.3)
		ax.add_patch(rect)
		plt.savefig('plots/%s_Mi_z.png' % qso_cat_name)
		plt.close('all')

		"""plt.figure(2, (10, 10))
		plt.hist((qso_cat[complete_cut][w1key] - qso_cat[complete_cut][w2key]), bins=100)
		plt.axvline(0.8, c='k', ls='--', label='Stern+12')
		plt.xlabel('W1 - W2', fontsize=20)
		plt.legend()
		plt.savefig('plots/%s_W1-W2.png' % qso_cat_name)
		plt.close('all')"""

	t = Table(qso_cat[complete_cut])
	t.write('QSO_cats/%s_complete.fits' % qso_cat_name, format='fits', overwrite=True)


def log_interp1d(xx, yy):
	logx = np.log10(xx)
	logy = np.log10(yy)
	lin_interp = interpolate.interp1d(logx, logy, fill_value='extrapolate')
	log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
	return log_interp


def rest_ir_lum(w1, w2, zs, ref_lambda_rest):
	w1 = w1/1e9 * 309.54
	w2 = w2/1e9 * 171.787
	w1_obs_length = 3.368
	w2_obs_length = 4.618
	ref_lambda_obs = ref_lambda_rest * (1+zs)

	ref_lums = []

	for i in range(len(w1)):
		interp_func = log_interp1d([w1_obs_length, w2_obs_length], [w1[i], w2[i]])
		ref_lums.append((1 + zs[i]) * (interp_func(ref_lambda_obs[i]) * u.Jy * 4 * np.pi * (
			astropycosmo.luminosity_distance(zs[i])) ** 2).to('J').value)

	return ref_lums


def lum_weights(rb_lums, ctrl_lums, minlum, maxlum, bins):

	ctrlhist = np.histogram(ctrl_lums, bins=bins, range=(minlum, maxlum), density=True)
	colorhist = np.histogram(rb_lums, bins=bins, range=(minlum, maxlum), density=True)
	dist_ratio = ctrlhist[0] / colorhist[0]

	dist_ratio[np.where(np.isnan(dist_ratio) | np.isinf(dist_ratio))] = 0

	weights = np.zeros(len(rb_lums))

	for i in range(len(dist_ratio)):
		idxs_in_bin = np.where((rb_lums <= colorhist[1][i + 1]) & (rb_lums > colorhist[1][i]))
		weights[idxs_in_bin] = dist_ratio[i]

	return weights


def convolved_weights(qso_cat, rb_lums, ctrl_lums, minlum, maxlum, bins):

	pqso_weights = np.array(qso_cat['PQSO'])

	l_weights = lum_weights(rb_lums, ctrl_lums, minlum, maxlum, bins)

	totweights = pqso_weights * l_weights

	return totweights




def red_blue_samples(qso_cat_name, plots):
	qso_cat = fits.open('QSO_cats/%s_complete.fits' % qso_cat_name)[1].data

	if qso_cat_name == 'dr14':
		i_mags = qso_cat['psfmag'][:, band_idxs['i']]
		g_mags = qso_cat['psfmag'][:, band_idxs['g']]
		zkey = 'Z'

		sources_in_bins = 1000.

	elif qso_cat_name == 'dr16':
		i_mags = qso_cat['psfmag'][:, band_idxs['i']]
		g_mags = qso_cat['psfmag'][:, band_idxs['g']]

		dered_i_mags = i_mags - qso_cat['EXTINCTION'][:, band_idxs['i']]
		dered_g_mags = g_mags - qso_cat['EXTINCTION'][:, band_idxs['g']]

		zkey = 'Z'

		sources_in_bins = 1000.
	else:
		i_maggies = qso_cat['psfflux'][:, band_idxs['i']] / 1e9
		i_mags = -2.5 / np.log(10.) * (
					np.arcsinh(i_maggies / (2 * bsoftpars[band_idxs['i']])) + np.log(bsoftpars[band_idxs['i']]))
		g_maggies = qso_cat['psfflux'][:, band_idxs['g']] / 1e9
		g_mags = -2.5 / np.log(10.) * (
					np.arcsinh(g_maggies / (2 * bsoftpars[band_idxs['g']])) + np.log(bsoftpars[band_idxs['g']]))

		dered_i_mags = i_mags - qso_cat['EXTINCTION'][:, band_idxs['i']]
		dered_g_mags = g_mags - qso_cat['EXTINCTION'][:, band_idxs['g']]

		zkey = 'PEAKZ'

		sources_in_bins = 10000.

	gminusi = dered_g_mags - dered_i_mags
	zs = qso_cat[zkey]

	distances = astropycosmo.luminosity_distance(qso_cat[zkey]).to(u.pc)
	i_abs_mags = dered_i_mags - 5 * np.log10(distances / (10 * u.pc)) + (2.5 * 0.5 * np.log10(1 + qso_cat[zkey])) - 0.596


	num_bins = int(len(zs)/sources_in_bins)
	z_quantile_bins = pd.qcut(zs, num_bins, retbins=True)[1]

	red_idxs, c_idxs, blue_idxs = [], [], []
	for i in range(len(z_quantile_bins) - 1):
		in_z_bin = (zs <= z_quantile_bins[i+1]) & (zs > z_quantile_bins[i])
		idxs_in_bin = np.where(in_z_bin)
		gminusi_in_bin = gminusi[idxs_in_bin]
		color_bins = pd.qcut(gminusi_in_bin, 20, retbins=True)[1]
		red_idxs += list(np.where((gminusi > color_bins[15]) & in_z_bin)[0])
		c_idxs += list(np.where((gminusi <= color_bins[15]) & (gminusi > color_bins[5]) & in_z_bin)[0])
		blue_idxs += list(np.where((gminusi <= color_bins[5]) & in_z_bin)[0])

	if plots:
		plt.figure(2, (10, 10))
		plt.xlabel('z', fontsize=20)
		plt.ylabel('g - i', fontsize=20)
		plt.scatter(zs, gminusi, c='k', s=0.01, alpha=0.5)
		plt.scatter(zs[blue_idxs], gminusi[blue_idxs], c='b', s=0.05)
		plt.scatter(zs[c_idxs], gminusi[c_idxs], c='g', s=0.05)
		plt.scatter(zs[red_idxs], gminusi[red_idxs], c='r', s=0.05)
		plt.ylim(-5, 5)
		plt.savefig('plots/%s_gminusi.png' % qso_cat_name)
		plt.close('all')


		plt.figure(3, (10, 10))
		plt.hist(np.array(i_abs_mags[blue_idxs]), color='b', alpha=0.5, density=True, range=(-30, -24), bins=100)
		plt.hist(np.array(i_abs_mags[c_idxs]), color='g', alpha=0.5, density=True, range=(-30, -24), bins=100)
		plt.hist(np.array(i_abs_mags[red_idxs]), color='r', alpha=0.5, density=True, range=(-30, -24), bins=100)
		plt.xlim(-30, -22)
		plt.savefig('plots/%s_maghist.png' % qso_cat_name)
		plt.close('all')

	if (qso_cat_name == 'dr14') or (qso_cat_name == 'dr16'):
		bluelums = rest_ir_lum(qso_cat[blue_idxs]['W1_FLUX'], qso_cat[blue_idxs]['W2_FLUX'], qso_cat[blue_idxs]['Z'], 1.5)
		ctrllums = rest_ir_lum(qso_cat[c_idxs]['W1_FLUX'], qso_cat[c_idxs]['W2_FLUX'], qso_cat[c_idxs]['Z'], 1.5)
		redlums = rest_ir_lum(qso_cat[red_idxs]['W1_FLUX'], qso_cat[red_idxs]['W2_FLUX'], qso_cat[red_idxs]['Z'], 1.5)
	else:
		bluelums = rest_ir_lum(qso_cat[blue_idxs]['PSFFLUX'][:, 11], qso_cat[blue_idxs]['PSFFLUX'][:, 12], qso_cat[blue_idxs]['PEAKZ'], 1.5)
		ctrllums = rest_ir_lum(qso_cat[c_idxs]['PSFFLUX'][:, 11], qso_cat[c_idxs]['PSFFLUX'][:, 12],
		                       qso_cat[c_idxs]['PEAKZ'], 1.5)
		redlums = rest_ir_lum(qso_cat[red_idxs]['PSFFLUX'][:, 11], qso_cat[red_idxs]['PSFFLUX'][:, 12],
		                       qso_cat[red_idxs]['PEAKZ'], 1.5)

	lowhistlim = floor(np.log10(np.min([np.min(bluelums), np.min(ctrllums), np.min(redlums)])))
	highhistlim = ceil(np.log10(np.max(np.max([np.max(bluelums), np.max(ctrllums), np.max(redlums)]))))


	lumhistbins = 500

	if plots:

		plt.figure(4, (10, 10))
		plt.hist(np.log10(bluelums), color='b', alpha=0.5, density=True, bins=lumhistbins, range=(lowhistlim, highhistlim))
		plt.hist(np.log10(ctrllums), color='g', alpha=0.5, density=True, bins=lumhistbins, range=(lowhistlim, highhistlim))
		plt.hist(np.log10(redlums), color='r', alpha=0.5, density=True, bins=lumhistbins, range=(lowhistlim, highhistlim))
		plt.xlabel('log($L_{1.5\mu\mathrm{m}}$)')
		plt.savefig('plots/%s_lumhist.png' % qso_cat_name)
		plt.close('all')


	bluetab = Table(qso_cat[blue_idxs])
	ctrltab = Table(qso_cat[c_idxs])
	redtab = Table(qso_cat[red_idxs])
	if (qso_cat_name == 'dr14') or (qso_cat_name == 'dr16'):
		bluetab['weight'] = lum_weights(np.log10(bluelums), np.log10(ctrllums), lowhistlim, highhistlim, lumhistbins)
		ctrltab['weight'] = np.ones(len(ctrllums))
		redtab['weight'] = lum_weights(np.log10(redlums), np.log10(ctrllums), lowhistlim, highhistlim, lumhistbins)
	else:
		bluetab['weight'] = convolved_weights(bluetab, np.log10(bluelums), np.log10(ctrllums), lowhistlim, highhistlim, lumhistbins)
		ctrltab['weight'] = ctrltab['PQSO']
		redtab['weight'] = convolved_weights(redtab, np.log10(redlums), np.log10(ctrllums), lowhistlim, highhistlim, lumhistbins)

	ctrltab.write('QSO_cats/%s_ctrl.fits' % qso_cat_name, format='fits', overwrite=True)
	bluetab.write('QSO_cats/%s_blue.fits' % qso_cat_name, format='fits', overwrite=True)
	redtab.write('QSO_cats/%s_red.fits' % qso_cat_name, format='fits', overwrite=True)




