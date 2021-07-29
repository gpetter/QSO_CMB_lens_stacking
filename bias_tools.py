from astropy.io import fits
import numpy as np
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck18')
from colossus.lss import bias
import redshift_dists
import importlib
importlib.reload(redshift_dists)


def bias_to_mass(inputbias, z):
	masses = np.logspace(10, 14, 1000)
	biases_from_masses = bias.haloBias(M=masses, z=z, mdef='200c', model='tinker10')
	return np.interp(inputbias, biases_from_masses, masses)


# take a characteristic halo mass and calculate the resulting average bias over a given redshift distribution
def mass_to_avg_bias(m, zs, dndz):
	bh = bias.haloBias(M=m, z=zs, mdef='200c', model='tinker10')

	avg_bh = np.average(bh, weights=dndz)

	return avg_bh


# take a bias measured over a redshift distribution and calculate which characteristic halo mass would
# result in the measured bias
def avg_bias_to_mass(input_bias, zs, dndz, berr=0):

	masses = np.logspace(10, 14, 1000)
	b_avg = []
	for mass in masses:
		b_avg.append(mass_to_avg_bias(mass, zs, dndz))

	if berr > 0:
		upmass = np.interp(input_bias+berr, b_avg, masses)
		lomass = np.interp(input_bias-berr, b_avg, masses)
		return np.interp(input_bias, b_avg, masses), lomass, upmass
	else:
		return np.interp(input_bias, b_avg, masses)


# calculate bias predicted by parameterizations of quasar bias as function of redshift given by
# Croom et al 2005 or Laurent et al. 2017
def qso_bias_for_z(paper, zs, dndz=None, n_draws=100):

	if paper == 'laurent':
		a0, a1, a2 = 2.393, 0.278, 6.565
		sig_a0, sig_a1 = 0.042, 0.018
	elif paper == 'croom':
		a0, a1, a2 = 0.53, 0.289, 0
		sig_a0, sig_a1 = 0.19, 0.035
	else:
		return 'Invalid paper name. Use croom for Croom+05 or laurent for Laurent+17'

	# form of Croom and Laurent parameterization of bias with redshift
	lit_bias = a0 + a1 * ((1+zs) ** 2 - a2)

	# same as above but randomly drawn from error in parameterization n_draws times
	bias_draws = np.repeat(np.random.normal(a0, sig_a0, n_draws)[None, :], len(zs), axis=0) \
	             + np.outer(((1+zs) ** 2 - a2), np.random.normal(a1, sig_a1, n_draws))

	# if a redshift distribution given, average over using dn/dz as weight
	if dndz is not None:
		avg_b = np.average(lit_bias, weights=dndz)

		avg_b_draws = np.average(bias_draws, weights=dndz, axis=0)

		avg_b_std = np.std(avg_b_draws, axis=0)
		return avg_b, avg_b_std

	# if no redshift distribution given, just return the bias at each z, and the uncertainty at each z
	else:
		b_std = np.std(bias_draws, axis=1)
		return lit_bias, b_std







def bias_and_masses(refsample, caps='both'):
	refzs_ngc = fits.open('catalogs/lss/%s/NGC_comov.fits' % refsample)[1].data['Z']

	qso_cat = fits.open('catalogs/derived/eBOSS_QSO_binned.fits')[1].data
	qso_zs = qso_cat['Z']
	n_sample_bins = int(np.max(qso_cat['bin']))

	if caps == 'both':
		ngcbiases, sgcbiases, ngcbiaserrs, sgcbiaserrs = [], [], [], []
		for j in range(n_sample_bins):
			tmp1, tmp2 = np.load('bias/eBOSS_QSO/NGC/%s_%s.npy' % (refsample, j + 1), allow_pickle=True)

			ngcbiases.append(tmp1)
			ngcbiaserrs.append(tmp2)

		for j in range(n_sample_bins):
			tmp1, tmp2 = np.load('bias/eBOSS_QSO/SGC/%s_%s.npy' % (refsample, j + 1), allow_pickle=True)

			sgcbiases.append(tmp1)
			sgcbiaserrs.append(tmp2)

		avg_bias = np.average([ngcbiases, sgcbiases], weights=[1 / np.array(ngcbiaserrs), 1 / np.array(sgcbiaserrs)],
		                      axis=0)
		avg_err = np.sqrt(np.array(ngcbiaserrs) ** 2 + np.array(sgcbiaserrs) ** 2) / 2

		zs, dndz = redshift_dists.redshift_overlap(qso_zs, refzs_ngc)

		charmass, lomass, upmass = [], [], []
		for j in range(len(avg_bias)):
			mass = avg_bias_to_mass(avg_bias[j], zs, dndz, berr=avg_err[j])
			charmass.append(mass[0])
			lomass.append(mass[1])
			upmass.append(mass[2])

		return avg_bias, avg_err, charmass, lomass, upmass
