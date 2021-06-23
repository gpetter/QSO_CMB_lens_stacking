import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from colossus.cosmology import cosmology
from scipy.optimize import curve_fit
cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()
from colossus.lss import bias
import camb
from astropy.io import fits
import mcfit
from functools import partial
import pickle
from scipy.special import j0
import astropy.constants as const
import astropy.units as u
import abel
import plotting
import importlib
importlib.reload(plotting)

# define k space (includes little h)
kmax = 1000
k_grid = np.logspace(-5, np.log10(kmax), 1000)



def theory_projected_corr_func(rps):
	rs = np.linspace(1, 300, 1000)
	wp = []
	rstep = rs[1] - rs[0]
	for i in range(len(rps)):
		r_range = rs[np.where(rs >= rps[i])]

		integralsum = 0
		for j in range(len(r_range)):
			# radius at midpoint of r bin
			rinbin = r_range[j] + rstep/2
			# correlation function value at midpoint
			xi_in_bin = cosmo.correlationFunction(R=rinbin, z=1.5)
			# amplitude of integrand at midpoint
			amp = 2 * (xi_in_bin * rinbin)/np.sqrt(rinbin**2 - rps[i]**2)
			area = amp * rstep
			integralsum += area
		wp.append(integralsum)

	return np.array(wp)


# set CAMB cosmology, generate power spectrum at 150 redshifts, return interpolator which can estimate power spectrum
# at any given redshift
def camb_matter_power_interpolator(zs, nonlinear=True):
	pars = camb.CAMBparams()
	pars.set_cosmology(H0=cosmo.H0, ombh2=cosmo.Ombh2, omch2=(cosmo.Omh2-cosmo.Ombh2), omk=cosmo.Ok0)
	pars.InitPower.set_params(ns=cosmo.ns)
	pk_interp = camb.get_matter_power_interpolator(pars, zs=np.linspace(np.min(zs), np.max(zs), 150), kmax=kmax,
	                                               nonlinear=nonlinear)
	return pk_interp


# write out power spectra in table for speed
# run this once with your desired minz, maxz, k grid
def write_power_spectra(minz, maxz, nonlinear):
	zs = np.linspace(minz, maxz, 500)
	pk_interp = camb_matter_power_interpolator(zs, nonlinear)
	pks = pk_interp.P(zs, k_grid)
	writedict = {'zs': zs, 'ks': np.array(k_grid), 'Pk': pks}
	pickle.dump(writedict, open('power_spectra/nonlin_%s.p' % nonlinear, 'wb'))


# calculate power spectrum at given redshifts either by reading from table or using interpolator above
def power_spec_at_zs(zs, read=True, dimensionless=False):
	if read:
		pickled_powspectra = pickle.load(open('power_spectra/nonlin_True.p', 'rb'))
		if np.min(zs) < np.min(pickled_powspectra['zs']) or np.max(zs) > np.max(pickled_powspectra['zs']):
			print('Error: zs out of tabulated range')
			return
		z_idxs = np.digitize(zs, pickled_powspectra['zs']) - 1
		pks = pickled_powspectra['Pk'][z_idxs]
	else:
		pk_interp = camb_matter_power_interpolator(zs)
		pks = pk_interp.P(zs, k_grid)

	if dimensionless:
		pks = k_grid**3 / (2 * (np.pi ** 2)) * pks
	return pks


def camb_corr_func(zs, read=True):
	pks = power_spec_at_zs(zs, read=read)

	rs, cfs = mcfit.P2xi(k_grid, lowring=True)(pks, axis=1)

	return rs, cfs


def redshift_weighted_corr_func(zs, dn_dz):

	rs, cfs = camb_corr_func(zs)

	weighted_cf = np.average(cfs, axis=0, weights=dn_dz)
	return rs, weighted_cf


"""def corr_func_in_bins(zs, dn_dz, bins):
	weighted_cf = redshift_weighted_corr_func(zs, dn_dz)
	rs = np.linspace(0.01, 150, 1000)
"""

def projected_corr_func(zs, dn_dz):
	"""rp_grid = np.logspace(-0.5, 2, 1000)
	rs, weighted_cf = redshift_weighted_corr_func(zs, dn_dz)
	wp_rp = []
	for rp in rp_grid:
		goodidxs = rs > rp
		integrand = 2 * weighted_cf[goodidxs] * rs[goodidxs] / np.sqrt(rs[goodidxs]**2 - rp**2)

		wp_rp.append(np.trapz(integrand, rs[goodidxs]))"""
	rs, weighted_cf = redshift_weighted_corr_func(zs, dn_dz)
	wp_rp = abel.direct.direct_transform(weighted_cf, r=rs, direction='forward', backend='python')

	return rs, np.array(wp_rp)

def projected_corr_func_in_bins(rp_bins, zs, dn_dz, usebins):
	rps, proj_cf = projected_corr_func(zs, dn_dz)

	"""avg_wp = []
	for j in range(len(rp_bins)-1):
		bin_range = rp_bins[j+1] - rp_bins[j]
		rp_idxs_in_bin = np.where((rp_grid > rp_bins[j]) & (rp_grid <= rp_bins[j+1]))
		wp_in_bin = proj_cf[rp_idxs_in_bin]
		rp_in_bin = rp_grid[rp_idxs_in_bin]
		integral_in_bin = np.trapz(wp_in_bin, rp_in_bin)
		avg_wp.append(integral_in_bin/bin_range)"""

	avg_wp = []
	for j in range(len(rp_bins)-1):
		if usebins[j]:
			pointsinbin = proj_cf[np.where((rps < rp_bins[j+1]) & (rps > rp_bins[j]))]
			avg_wp.append(np.mean(pointsinbin))


	return np.array(avg_wp)




def biased_projected_corr_func_in_bins(rp_bins, bb, zs, dn_dz, usebins):
	return bb * projected_corr_func_in_bins(rp_bins, zs, dn_dz, usebins)

def fit_bias(rp_data, wp_data, wp_errs, zs, dn_dz, usebins):
	partialfun = partial(biased_projected_corr_func_in_bins, zs=zs, dn_dz=dn_dz, usebins=usebins)
	popt, pcov = curve_fit(partialfun, rp_data, wp_data, sigma=wp_errs)
	return popt, np.sqrt(pcov)

def fit_bias_to_cf(samplename, refname, cap, binid, rp_bins):

	if binid == 'all':
		cat = fits.open('catalogs/lss/%s/%s_comov.fits' % (samplename, cap))[1].data
		w, werr = np.load('clustering/spatial/%s/%s_all.npy' % (cap, samplename), allow_pickle=True)
		zdist = cat['Z']
		# bin up redshift distribution of sample to integrate kappa over
		hist = np.histogram(zdist, 20, density=True)
		zs = hist[1]
		dz = zs[1] - zs[0]
		# chop off last entry which is a rightmost bound of the z distribution, find center of bins by adding dz/2
		zs = np.resize(zs, zs.size - 1) + dz / 2

		dndz = hist[0]
	# if doing a cross-correlation
	else:
		cat = fits.open('catalogs/derived/eBOSS_QSO_binned.fits')[1].data
		cat = cat[np.where(cat['bin'] == binid)]

		# calculate dndz of both samples, use sqrt of product as dndz weights
		zdist = cat['Z']
		refcat = fits.open('catalogs/lss/%s/%s_comov.fits' % (refname, cap))[1].data
		refzdist = refcat['Z']
		minz, maxz = np.min(zdist), np.max(zdist)
		minrefz, maxrefz = np.min(refzdist), np.max(refzdist)
		totminz, totmaxz = np.min([minz, minrefz]), np.max([maxz, maxrefz])

		hist1 = np.histogram(zdist, 30, density=True, range=[totminz, totmaxz])
		refhist = np.histogram(refzdist, 30, density=True, range=[totminz, totmaxz])

		zs = hist1[1]
		dz = zs[1] - zs[0]
		# chop off last entry which is a rightmost bound of the z distribution, find center of bins by adding dz/2
		zs = np.resize(zs, zs.size - 1) + dz / 2

		dndz = np.sqrt(hist1[0] * refhist[0])



		avgrs, w, werr = np.load('clustering/spatial_cross/%s/%s_%s.npy' % (cap, refname, binid), allow_pickle=True)



	usebins = np.zeros(len(w))
	goodidxs = np.where((np.isfinite(werr)) & (np.isfinite(w)) & (werr > 0))


	w, werr = w[goodidxs], werr[goodidxs]
	usebins[goodidxs] = 1


	bb, sigma_bb = fit_bias(rp_bins, w, werr, zs, dndz, usebins)

	avgrps = []
	for j in range(len(rp_bins)-1):
		avgrps.append(np.mean([rp_bins[j], rp_bins[j+1]]))

	avgrps = np.array(avgrps)[goodidxs]

	if binid == 'all':

		b = np.sqrt(bb[0])
		print(b)
		sigma_b = 0.5/b * sigma_bb[0][0]

		np.array([b, sigma_b]).dump('bias/%s/%s/all.npy' % (samplename, cap))
	else:
		refb, sigma_refb = np.load('bias/%s/%s/all.npy' % (refname, cap), allow_pickle=True)

		b = bb[0]/refb


		sigma_b = b * np.sqrt(((sigma_bb[0][0]/bb[0])**2 - (sigma_refb/refb)**2))

		np.array([b, sigma_b]).dump('bias/eBOSS_QSO/%s/%s_%s.npy' % (cap, refname, binid))

	cf = biased_projected_corr_func_in_bins(rp_bins, bb[0], zs, dndz, usebins)

	plotting.plot_each_cf(avgrps, cap, w, werr, cf, binid, refname)

	return b, sigma_b, cf





# DiPompeo 2017
def angular_corr_func(thetas, zs, dn_dz_1, dn_dz_2):
	# thetas from degrees to radians
	thetas = (thetas*u.deg).to('radian').value

	# get dimensionless power spectrum
	deltasquare = power_spec_at_zs(zs, read=False, dimensionless=True)

	# power spectrum over k^2
	first_term = deltasquare / (k_grid ** 2)

	# everything inside Bessel function
	# has 3 dimensions, k, theta, and z
	# therefore need to do outer product of two arrays, then broadcast 3rd array to 3D and multiply
	besselterm = j0(cosmo.comovingDistance(np.zeros(len(zs)), zs)[:, None, None] * np.outer(k_grid, thetas))

	# Not sure if this is right
	# I think you need to convert H(z)/c from 1/Mpc to h/Mpc in order to cancel units of k, but not sure
	dz_d_chi = (apcosmo.H(zs) / const.c).to(u.littleh/u.Mpc, u.with_H0(apcosmo.H0)).value

	# product of redshift distributions, and dz/dchi
	differentials = dz_d_chi * dn_dz_1 * dn_dz_2

	# total integrand is all terms multiplied out. This is a 3D array
	integrand = np.pi * differentials * np.transpose(first_term) * np.transpose(besselterm)

	# do k integral first along k axis
	k_int = np.trapz(integrand, k_grid, axis=1)
	# then integrate along z axis
	total = np.trapz(k_int, zs, axis=1)

	return total

def mass_to_avg_bias(m, zs, dndz):

	bh = bias.haloBias(M=m, z=zs, mdef='200c', model='tinker10')

	avg_bh = np.average(bh, weights=dndz)

	return avg_bh


def avg_bias_to_mass(input_bias, zs, dndz):

	masses = np.logspace(10, 14, 100)
	b_avg = []
	for mass in masses:
		b_avg.append(mass_to_avg_bias(mass, zs, dndz))

	return np.interp(input_bias, b_avg, masses)





"""def biased_corr_func(rps, b):
	return b**2 * theory_projected_corr_func(rps)




def fit_bias(rp_data, wp_data, wp_errs):
	popt, pcov = curve_fit(biased_corr_func, rp_data, wp_data, sigma=wp_errs)
	return popt, np.sqrt(pcov)"""