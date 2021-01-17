import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import constants as const
import astropy.units as u
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import aperture_photometry
from colossus.cosmology import cosmology
from colossus.halo import concentration
from colossus.halo import profile_nfw
from colossus.lss import bias
from scipy.special import j0
from functools import partial
from scipy.optimize import curve_fit
import healpy as hp
import glob
import convergence_map
import importlib
from astropy.stats import sigma_clipped_stats
import stacking
importlib.reload(convergence_map)
importlib.reload(stacking)

cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()


# defining the critical surface density for lensing
def sigma_crit(z):
	return ((const.c ** 2) / (4. * np.pi * const.G) * (apcosmo.angular_diameter_distance(1100.) / (
		(apcosmo.angular_diameter_distance(z) * apcosmo.angular_diameter_distance_z1z2(z, 1100.))))).decompose().to(
		u.solMass * u.littleh / u.kpc ** 2, u.with_H0(apcosmo.H0))


# colossus wants inputs in units of h, so convert a mass in solar masses to solar masses per h
def mass_to_solmass_per_h(m):
	return m.to(u.solMass/u.littleh, u.with_H0(apcosmo.H0)).value


# calculate the concentration of a halo given a mass and redshift using the Ludlow+16 model
def calc_concentration(m_200, z):

	c_200 = concentration.modelLudlow16(m_200, z)
	if c_200[1]:
		return c_200[0]
	else:
		return np.nan


# converting a radial NFW profile to an angular surface mass density profile given a halo mass and redshift
def nfw_sigma(theta, m_200, z):

	# define a NFW profile in terms of the halo mass and concentration parameter c_200
	p_nfw = profile_nfw.NFWProfile(M=m_200, c=calc_concentration(m_200, z), z=z, mdef='200c')
	# angular diameter distance
	d_a = apcosmo.angular_diameter_distance(z).to(u.kpc/u.littleh, u.with_H0(apcosmo.H0)).value
	return p_nfw.surfaceDensity(r=theta*d_a)


# Takes in a redshift and halo mass and returns the convergence predicted at an angle theta due to a NFW halo
def kappa_1_halo(theta, m_200, z):
	return nfw_sigma(theta, m_200, z)/sigma_crit(z).value


# estimate lensing convergence due to correlated large scale structure to a DM halo of a given mass
def two_halo_term(theta, m_200, z):

	d_a = apcosmo.angular_diameter_distance(z).to(u.kpc/u.littleh, u.with_H0(apcosmo.H0))     # kpc/h

	# calculate halo bias through Tinker+10 model
	bh = bias.haloBias(M=m_200, z=z, mdef='200c', model='tinker10')

	# the average (matter) density of the universe
	# !!!!! i think it's okay to use rho_c instead because we are assuming flat cosmology !!!!!!!

	rho_avg = cosmo.rho_m(z)*u.solMass*(u.littleh**2)/(u.kpc**3)
	# amalgamation of constants outside the integral
	a = (rho_avg/(((1.+z)**3)*sigma_crit(z)*d_a**2))*bh/(2*np.pi)

	# scales
	ks = np.logspace(-6, 2, 1000)*u.littleh/u.Mpc
	ls = ks*(1+z)*(apcosmo.angular_diameter_distance(z).to(u.Mpc/u.littleh, u.with_H0(apcosmo.H0)))

	# do an outer product of the thetas and ls for integration
	ltheta = np.outer(theta, ls)

	# compute matter power spectrum at comoving wavenumbers k
	mps = (cosmo.matterPowerSpectrum(ks.value, z=z)*(u.Mpc/u.littleh)**3).to((u.kpc/u.littleh)**3)

	# Eq. 13 in OOguri and Hamana 2011
	integrand = a*ls*j0(ltheta)*mps
	return np.trapz(integrand, x=ls)


# integrate kappa across redshift distribution dn/dz
def int_kappa(theta, m_200, terms, zdist, zbins=100):

	# bin up redshift distribution of sample to integrate kappa over
	hist = np.histogram(zdist, zbins, density=True)

	avg_kappa = []
	zs = hist[1]
	# chop off last entry which is a rightmost bound of the z distribution
	zs = np.resize(zs, zs.size-1)

	dz = zs[1] - zs[0]
	dndz = hist[0]

	for i in range(len(dndz)):
		z = zs[i] + dz/2
		if terms == 'one':
			avg_kappa.append(kappa_1_halo(theta, m_200, z)*dndz[i])
		elif terms == 'two':
			avg_kappa.append(two_halo_term(theta, m_200, z)*dndz[i])
		elif terms == 'both':
			avg_kappa.append((kappa_1_halo(theta, m_200, z) + two_halo_term(theta, m_200, z))*dndz[i])
		else:
			return False
	avg_kappa = np.array(avg_kappa)

	return np.trapz(avg_kappa, dx=dz, axis=0)


# apply same filter applied to the map to the model
# gaussian with small l modes zeroed
def filter_model(zdist, m_per_h):
	theta_list_rad = (np.arange(0.5, 360, 0.5) * u.arcmin).to('rad').value

	bothmodel = int_kappa(theta_list_rad, m_per_h, 'both', zdist)
	kmodel = hp.beam2bl(bothmodel, theta_list_rad, lmax=4096)

	k_space_filter = hp.gauss_beam((15 * u.arcmin).to('rad').value, lmax=4096)
	k_space_filter[:100] = 0

	kconvolved = np.array(kmodel) * np.array(k_space_filter)

	return hp.bl2beam(kconvolved, theta_list_rad)


def filtered_model_at_theta(zdist, m_per_h, inputthetas):
	theta_list = np.arange(0.5, 360, 0.5)
	model = filter_model(zdist, m_per_h)
	flat_thetas = inputthetas.flatten()
	kappa_vals = []
	for j in range(len(flat_thetas)):
		kappa_vals.append(model[np.abs(theta_list - flat_thetas[j]).argmin()])

	return np.array(kappa_vals).reshape(inputthetas.shape)


def filtered_model_in_bins(zdist, obs_thetas, m_per_h):
	theta_list = np.arange(0.5, 360, 0.5)*u.arcmin
	model_full_range = filter_model(zdist, m_per_h)
	binsize = obs_thetas[1] - obs_thetas[0]

	theta_list = theta_list.value
	binned_kappa = []
	for theta in obs_thetas:
		binned_kappa.append(np.mean(model_full_range[np.where((theta_list > (theta - binsize)) & (theta_list < (theta + binsize)))]))

	return(binned_kappa)


def filtered_model_center(zdist, m_per_h):

	model_full = filter_model(zdist, m_per_h)
	return(model_full[0])


# measure the radial convergence profile from the stacked map
def measure_profile(image, step, reso=1.5):
	profile, profile_unc = [], []
	center = len(image)/2. - 0.5
	steppix = step/reso

	inner_aper = CircularAperture([center, center], steppix)
	profile.append(float(aperture_photometry(image, inner_aper)['aperture_sum']/inner_aper.area))

	i = 1
	while step*i < len(image)/2*reso:
		new_aper = CircularAnnulus([center, center], steppix*i, steppix*(i+1))
		profile.append(float(aperture_photometry(image, new_aper)['aperture_sum']/new_aper.area))

		i += 1

	return profile

"""def gauss(sigma, mean, xs):
	return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2 * ((xs-mean)/sigma)**2)

def calc_redshift_dist(color, samplename):

	zs = fits.open('QSO_cats/%s_%s.fits' % (samplename, color))[1].data['PEAKZ']

	otherzs = fits.open('QSO_cats/%s_%s.fits' % (samplename, color))[1].data['OTHERZ']

	minz = np.min(np.array([zs, np.array(otherzs).flatten()]).flatten())
	maxz = np.max(np.array([zs, np.array(otherzs).flatten()]).flatten())

	zspace = np.linspace(minz, maxz, 1000)"""


def fit_best_mass(obs_profile, zdist, p0=6E12, binsize=12, sigma=None, maxtheta=180):

	if sigma is None:
		sigma = np.ones(len(obs_profile))

	filtered_model_of_mass = partial(filtered_model_in_bins, zdist)

	obs_thetas = np.arange(binsize/2, maxtheta, binsize)

	popt, pcov = curve_fit(filtered_model_of_mass, obs_thetas, obs_profile, p0=p0, bounds=[10**11, 10**13.5], sigma=sigma)
	return np.log10(popt[0])


def fit_mass_to_peak(zdist, peakkappa, stdev=None):
	"""partialpeak = partial(filtered_model_center, zdist)

	popt, pcov = curve_fit(partialpeak, 0, peakkappa, p0=10**12)
	print(popt)"""

	masses = np.logspace(12, 13, 21)
	peakpredictions = []
	for mass in masses:
		peakpredictions.append(filtered_model_center(zdist, mass))
	bestidx = np.argmin(np.abs(np.array(peakpredictions) - peakkappa))
	if stdev is not None:
		peak_plus_noise = peakkappa + stdev
		peak_minus_noise = peakkappa - stdev

		plus_predictions, minus_predictions = [], []
		for mass in masses:
			plus_predictions.append(filtered_model_center(zdist, mass))
			minus_predictions.append(filtered_model_center(zdist, mass))

		plus_best_idx = np.argmin(np.abs(np.array(plus_predictions) - peak_plus_noise))
		minus_best_idx = np.argmin(np.abs(np.array(minus_predictions) - peak_minus_noise))

		plus_err = np.log10(masses[plus_best_idx]) - np.log10(masses[bestidx])
		minus_err = np.log10(masses[bestidx]) - np.log10(masses[minus_best_idx])
		"""errmasses = np.logspace(11, 12, 11)
		errpeakpredictions = []
		for mass in errmasses:
			errpeakpredictions.append(filtered_model_center(zdist, mass))
		errbestidx = np.argmin(np.abs(np.array(errpeakpredictions) - stdev))"""
		return np.log10(masses[bestidx]), plus_err, minus_err
	else:
		return np.log10(masses[bestidx])


def annuli_errs(color, binsize=12, reso=1.5):
	noisestacknames = sorted(glob.glob('noise_stacks/*_%s.npy' % color))

	noise_profiles = []
	for i in range(len(noisestacknames)):
		noisemap = np.load(noisestacknames[i], allow_pickle=True)
		noise_profiles.append(measure_profile(noisemap, binsize, reso))
	return np.std(noise_profiles, axis=0)


# simulate a stacked map usin
def model_stacked_map(color, samplename, m_per_h, imsize=240, reso=1.5):
	if samplename == 'xdqso' or samplename == 'xdqso_specz':
		zkey = 'PEAKZ'
	else:
		zkey = 'Z'

	zdist = fits.open('QSO_cats/%s_%s.fits' % (samplename, color))[1].data[zkey]

	center = imsize/2 - 0.5
	x_arr, y_arr = np.mgrid[0:imsize, 0:imsize]
	radii_theta = np.sqrt(((x_arr - center) * reso) ** 2 + ((y_arr - center) * reso) ** 2)
	model_kappas = filtered_model_at_theta(zdist, m_per_h, radii_theta)
	return model_kappas



def fit_mass_suite(color, samplename, plot, use_peak=False, do_stack=False, binsize=12, reso=1.5):
	if samplename == 'xdqso' or samplename == 'xdqso_specz':
		zkey = 'PEAKZ'
		rakey, deckey = 'RA_XDQSO', 'DEC_XDQSO'
	else:
		zkey = 'Z'
		rakey, deckey = 'RA', 'DEC'

	# read in the stack of lensing convergence at positions of quasars
	stacked_map = np.load('stacks/%s_%s_stack.npy' % (samplename, color), allow_pickle=True)
	# measure the profile of the stack using annular bins
	kap_profile = measure_profile(stacked_map, binsize, reso=reso)
	# calculate redshift distribution dn/dz
	zdist = fits.open('QSO_cats/%s_%s.fits' % (samplename, color))[1].data[zkey]
	# estimate uncertainty in each annulus for model fitting
	err_profile = annuli_errs(color, binsize, reso)

	maxtheta = int(len(stacked_map) / 2 * reso)

	# fit entire profile, not just the peak convergence
	if use_peak == False:
		# fit whole profile

		avg_mass = fit_best_mass(kap_profile, zdist, binsize=binsize, sigma=err_profile, maxtheta=maxtheta)
		print(avg_mass)

		# generate a noiseless stack using the best fit model
		modelmap = model_stacked_map(color, samplename, 10**avg_mass, imsize=len(stacked_map), reso=reso)

		# to estimate uncertainty on mass, add noise map to the noiseless model and refit many times
		masses = []
		profiles = []

		noisestacknames = sorted(glob.glob('noise_stacks/*_%s.npy' % color))

		for i in range(len(noisestacknames)):
			noisemap = np.load(noisestacknames[i], allow_pickle=True)
			model_plus_noise = modelmap + noisemap
			noisyprofile = measure_profile(model_plus_noise, binsize, reso=reso)
			profiles.append(noisyprofile)
			noisymass = fit_best_mass(noisyprofile, zdist, p0=10**avg_mass, binsize=binsize, sigma=err_profile, maxtheta=maxtheta)
			print(noisymass)
			masses.append(noisymass)
		masses = np.array(masses)
		highermasses = masses[np.where(masses > avg_mass)]
		higher_std = np.sqrt(1/(len(highermasses)-1)*np.sum(np.square(highermasses - avg_mass)))
		lowermasses = masses[np.where(masses < avg_mass)]
		lower_std = np.sqrt(1/(len(lowermasses)-1)*np.sum(np.square(lowermasses - avg_mass)))

		#print(higherstd, lowerstd)

		kap_errs = np.std(profiles, axis=0)
		mass_uncertainty = np.std(masses)
		#np.array([avg_mass, mass_uncertainty]).dump('masses/%s_%s_mass.npy' % (samplename, color))


		plt.figure()
		plt.hist(masses)
		plt.savefig('plots/masshist.png')
		plt.close('all')

		if plot:
			if color == 'blue':
				ckey = 'b'
			elif color == 'red':
				ckey = 'r'
			else:
				ckey = 'g'
			obs_theta = np.arange(binsize/2, maxtheta, binsize)
			theta_range = np.arange(0.5, maxtheta, 0.5)
			best_mass_profile = filtered_model_at_theta(zdist, 10**(avg_mass), theta_range)
			#lowest_mass_profile = filter_model(zdist, theta_range, 10**(lowmass))
			binned_model = filtered_model_in_bins(zdist, obs_theta, 10**(avg_mass))
			plt.figure(0, (10, 8))
			plt.scatter(obs_theta, kap_profile, c=ckey)
			plt.errorbar(obs_theta, kap_profile, yerr=kap_errs, c=ckey, fmt='none')
			plt.scatter(obs_theta, binned_model, marker='s', facecolors='none', edgecolors='k', s=50)
			plt.plot(theta_range, best_mass_profile, c='k')
			#plt.plot(theta_range, lowest_mass_profile, c='k')
			#plt.scatter(obs_theta, filtered_model_in_bins(zdist, obs_theta, 10**(lowmass)), marker='s')
			#plt.scatter(obs_theta, lowprofile)
			plt.ylim(-0.0005, 0.003)
			plt.ylabel(r'$ \langle \kappa \rangle$', fontsize=20)
			plt.xlabel(r'$\theta$ (arcminutes)', fontsize=20)
			#if error:
			plt.text(50, 0.0015, 'log$_{10}(M/h^{-1} M_{\odot}$) = %s $\pm$ %s' % (round(avg_mass, 1), round(mass_uncertainty, 1)), fontsize=20)
			#else:
			#plt.text(50, 0.0015, 'log$_{10}(M/h^{-1} M_{\odot}$) = %s' % round(avg_mass, 3), fontsize=20)
			plt.savefig('plots/profile_%s.png' % color)
			plt.close('all')
	else:
		if do_stack:
			cat = fits.open('QSO_cats/%s_%s.fits' % (samplename, color))[1].data
			ras, decs = cat[rakey], cat[deckey]
			planck_map = hp.read_map('maps/smoothed_masked_planck.fits', dtype=np.single)
			peak_k, background_std = stacking.fast_stack(ras, decs, planck_map, iterations=1000)

		else:
			background_avg, background_med, background_std = sigma_clipped_stats(stacked_map)
			peak_k = np.max(stacked_map)
		print(peak_k, background_std)
		avg_mass, higher_std, lower_std = fit_mass_to_peak(zdist, peak_k, stdev=background_std)

	np.array([avg_mass, higher_std, lower_std]).dump('masses/%s_%s_mass.npy' % (samplename, color))

