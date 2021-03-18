import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import aperture_photometry
from colossus.cosmology import cosmology
from functools import partial
from scipy.optimize import curve_fit
import glob
import convergence_map
import importlib
import astropy.units as u
from astropy.stats import sigma_clipped_stats
import stacking
import plotting
import lensingModel
from scipy.optimize import minimize
importlib.reload(convergence_map)
importlib.reload(stacking)
importlib.reload(plotting)
importlib.reload(lensingModel)

cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()





# measure the radial convergence profile from the stacked map
def measure_profile(image, step, reso=1.5, maxtheta=180):
	profile, profile_unc = [], []
	center = len(image)/2. - 0.5
	steppix = step/reso

	inner_aper = CircularAperture([center, center], steppix)
	profile.append(float(aperture_photometry(image, inner_aper)['aperture_sum']/inner_aper.area))

	i = 1
	while step*i < maxtheta:
		new_aper = CircularAnnulus([center, center], steppix*i, steppix*(i+1))
		profile.append(float(aperture_photometry(image, new_aper)['aperture_sum']/new_aper.area))

		i += 1

	return np.array(profile)


def fit_best_mass(obs_profile, zdist, p0=6E12, binsize=12, sigma=None, maxtheta=180):

	if sigma is None:
		sigma = np.ones(len(obs_profile))

	filtered_model_of_mass = partial(lensingModel.filtered_model_in_bins, zdist, binsize=binsize, maxtheta=maxtheta)

	obs_thetas = np.arange(binsize/2, maxtheta, binsize)
	popt, pcov = curve_fit(filtered_model_of_mass, obs_thetas, obs_profile, p0=p0, bounds=[10**11, 10**14], sigma=sigma)
	return np.log10(popt[0])


def fit_mass_to_peak(zdist, peakkappa, stdev=None):
	partialpeak = partial(lensingModel.filtered_model_center, zdist)

	popt, pcov = curve_fit(partialpeak, [0], [peakkappa], p0=[6e12])
	return popt[0]



def annuli_errs(color, mode, binsize=12, reso=1.5, maxtheta=180):
	if mode == 'bootstrap':
		names = sorted(glob.glob('bootstacks/*_%s.npy' % color))
	elif mode == 'noise':
		names = sorted(glob.glob('noise_stacks/*_%s.npy' % color))
	elif mode == 'random':
		names = sorted(glob.glob('random_stacks/*_%s.npy' % color))

	noise_profiles = []
	for i in range(len(names)):
		noisemap = np.load(names[i], allow_pickle=True)
		noise_profiles.append(measure_profile(noisemap, binsize, reso, maxtheta=maxtheta))

	return np.std(noise_profiles, axis=0)




def covariance_matrix(color, mode, binsize=12, reso=1.5, maxtheta=180):
	if mode == 'bootstrap':
		names = sorted(glob.glob('bootstacks/*_%s.npy' % color))
	elif mode == 'noise':
		names = sorted(glob.glob('noise_stacks/*_%s.npy' % color))
	elif mode == 'random':
		names = sorted(glob.glob('random_stacks/*_%s.npy' % color))

	noise_profiles = []
	for i in range(len(names)):
		noisemap = np.load(names[i], allow_pickle=True)
		noise_profiles.append(measure_profile(noisemap, binsize, reso, maxtheta=maxtheta))
	noise_profiles = np.array(noise_profiles)
	avg_profile = np.mean(noise_profiles, axis=0)

	n_bins = len(noise_profiles[0])
	n_realizations = len(noise_profiles)
	c_ij = np.zeros((n_bins, n_bins))
	for i in range(n_bins):
		for j in range(n_bins):
			k_i = noise_profiles[:,i]
			k_i_bar = avg_profile[i]

			k_j = noise_profiles[:, j]
			k_j_bar = avg_profile[j]

			product = (k_i - k_i_bar) * (k_j - k_j_bar)
			sum = np.sum(product)
			c_ij[i, j] = 1/(n_realizations - 1) * sum
	plotting.plot_cov_matrix(color, c_ij)
	return c_ij

def model_variance_weights(zdist, binsize=12, reso=1.5, imsize=240, maxtheta=180):
	masses = np.logspace(11.5, 13.5, 21)
	models = []
	for mass in masses:
		models.append(lensingModel.filtered_model_in_bins(zdist, 1, mass, binsize=binsize, reso=reso, imsize=imsize, maxtheta=maxtheta))
	return np.std(models, axis=0)


def likelihood_for_mass(observed_profile, covar_mat, zdist, obs_thetas, m_per_h):

	filtered_model_of_mass = lensingModel.filtered_model_in_bins(zdist, obs_thetas, m_per_h)

	residual = observed_profile - filtered_model_of_mass
	residual = residual[:, np.newaxis]

	chi_square = np.dot(residual.T, np.dot(np.linalg.inv(covar_mat), residual))[0][0]
	return 0.5 * chi_square


def fit_mc(color, samplename, plot=False, binsize=12, reso=1.5, mode='noise'):
	# read in the stack of lensing convergence at positions of quasars
	stacked_map = np.load('stacks/%s_%s_stack.npy' % (samplename, color), allow_pickle=True)
	maxtheta = int(len(stacked_map) / 2 * reso)

	# measure the profile of the stack using annular bins
	kap_profile = measure_profile(stacked_map, binsize, reso=reso)
	# calculate redshift distribution dn/dz
	zdist = fits.open('catalogs/derived/%s_%s.fits' % (samplename, color))[1].data['Z']

	covar = covariance_matrix(color, mode, binsize, reso)
	obs_thetas = np.arange(binsize / 2, maxtheta, binsize)
	partiallike = partial(likelihood_for_mass, kap_profile, covar, zdist, obs_thetas)
	mss = np.logspace(12.5, 13, 10)
	for m in mss:
		print(partiallike(m))

	#likelihood = log_likelihood(kap_profile, filtered_model_of_mass, covar)
	#result = minimize(partiallike, )
	#print(result)



def fit_mass_suite(color, samplename, plot, binsize=12, reso=1.5, mode='noise'):


	# read in the stack of lensing convergence at positions of quasars
	stacked_map = np.load('stacks/%s_%s_stack.npy' % (samplename, color), allow_pickle=True)
	maxtheta = int(len(stacked_map) / 2 * reso)
	maxtheta = 180

	# measure the profile of the stack using annular bins
	kap_profile = measure_profile(stacked_map, binsize, reso=reso, maxtheta=maxtheta)
	# calculate redshift distribution dn/dz
	zdist = fits.open('catalogs/derived/%s_%s.fits' % (samplename, color))[1].data['Z']
	# estimate uncertainty in each annulus for model fitting
	err_profile = annuli_errs(color, 'noise', binsize, reso, maxtheta=maxtheta)
	print(err_profile)
	mod_weights = model_variance_weights(zdist, binsize=binsize, reso=reso, maxtheta=maxtheta)
	err_profile = err_profile / mod_weights
	print(err_profile)
	#print(err_profile)




	#err_profile = covariance_matrix(color, 'noise', binsize=binsize, reso=reso, maxtheta=maxtheta)
	#print(err_profile)
	#print(np.sqrt(np.diagonal(err_profile)))
	#print(err_profile)



	# fit entire profile, not just the peak convergence
	#if use_peak == False:
	# fit whole profile

	avg_mass = fit_best_mass(kap_profile, zdist, binsize=binsize, sigma=err_profile, maxtheta=maxtheta)
	print(avg_mass)

	# generate a noiseless stack using the best fit model
	modelmap = lensingModel.model_stacked_map(zdist, 10**avg_mass, imsize=len(stacked_map), reso=reso)

	# to estimate uncertainty on mass, add noise map to the noiseless model and refit many times
	masses = []
	profiles = []

	if mode == 'bootstrap':
		bootnames = sorted(glob.glob('bootstacks/*_%s.npy' % color))
		#for i in range(len(bootnames)):
		for i in range(len(bootnames)):
			bootmap = np.load(bootnames[i], allow_pickle=True)
			bootprofile = measure_profile(bootmap, binsize, reso=reso, maxtheta=maxtheta)
			profiles.append(bootprofile)
			bootmass = fit_best_mass(bootprofile, zdist, p0=10**avg_mass, binsize=binsize, sigma=err_profile, maxtheta=maxtheta)
			print(bootmass)
			masses.append(bootmass)
	elif mode == 'noise':
		noisestacknames = sorted(glob.glob('noise_stacks/*_%s.npy' % color))


		#for i in range(len(noisestacknames)):
		for i in range(10):
			noisemap = np.load(noisestacknames[i], allow_pickle=True)

			model_plus_noise = modelmap + noisemap
			noisyprofile = measure_profile(model_plus_noise, binsize, reso=reso, maxtheta=maxtheta)
			profiles.append(noisyprofile)
			noisymass = fit_best_mass(noisyprofile, zdist, p0=10**avg_mass, binsize=binsize, sigma=err_profile, maxtheta=maxtheta)
			print(noisymass)
			masses.append(noisymass)
	elif mode == 'random':
		randnames = sorted(glob.glob('random_stacks/*_%s.npy' % color))
		for i in range(len(randnames)):
			randmap = np.load(randnames[i], allow_pickle=True)
			model_plus_noise = modelmap + randmap
			noisyprofile = measure_profile(model_plus_noise, binsize, reso=reso)
			profiles.append(noisyprofile)
			noisymass = fit_best_mass(noisyprofile, zdist, p0=10 ** avg_mass, binsize=binsize, sigma=err_profile,
			                          maxtheta=maxtheta)
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
		#maxtheta=180
		#kap_profile = measure_profile(stacked_map, binsize, reso=reso, maxtheta=maxtheta)
		#kap_errs = np.zeros(len(kap_profile))
		obs_theta = np.arange(binsize / 2, maxtheta, binsize)
		theta_range = np.arange(0.5, maxtheta, 0.5)
		best_mass_profile = lensingModel.filtered_model_at_theta(zdist, 10 ** (avg_mass), theta_range)
		# lowest_mass_profile = filter_model(zdist, theta_range, 10**(lowmass))
		binned_model = lensingModel.filtered_model_in_bins(zdist, obs_theta, 10 ** (avg_mass), binsize, reso, maxtheta=maxtheta)
		theta_range = np.arange(0.5, maxtheta, 0.5) * u.arcmin.to('rad')
		oneterm = lensingModel.int_kappa(theta_range, 10**avg_mass, 'one', zdist)
		twoterm = lensingModel.int_kappa(theta_range, 10**avg_mass, 'two', zdist)
		plotting.plot_kappa_profile(color, kap_profile, kap_errs, binsize, maxtheta, best_mass_profile, binned_model, oneterm, twoterm)
	"""else:
		if do_stack:
			cat = fits.open('catalogs/derived/%s_%s.fits' % (samplename, color))[1].data
			ras, decs = cat['RA'], cat['DEC']
			planck_map = hp.read_map('maps/smoothed_masked_planck.fits', dtype=np.single)
			peak_k, background_std = stacking.fast_stack(ras, decs, planck_map, iterations=1000)

		else:
			background_avg, background_med, background_std = sigma_clipped_stats(stacked_map)
			peak_k = np.max(stacked_map)
		print(peak_k, background_std)
		avg_mass, higher_std, lower_std = fit_mass_to_peak(zdist, peak_k, stdev=background_std)"""

	np.array([avg_mass, higher_std, lower_std]).dump('masses/%s_%s_mass.npy' % (samplename, color))

def fit_mass_to_cutouts(sample_id, color, binsize=12, nbootstraps=0):
	# read in the stack of lensing convergence at positions of quasars
	stacked_map = stacking.stack_suite(color, sample_id, True, False, mode='cutout')
	print(np.max(stacked_map))
	kap_profile = measure_profile(stacked_map, binsize, reso=1.5)
	# calculate redshift distribution dn/dz
	zdist = fits.open('catalogs/derived/%s_%s.fits' % (sample_id, color))[1].data['Z']
	avgmass = fit_best_mass(kap_profile, zdist, binsize=binsize, maxtheta=180)
	print(avgmass)

	if nbootstraps > 0:
		mass_bootstraps = []
		for j in range(nbootstraps):
			bootmap = stacking.stack_suite(color, sample_id, True, False, mode='cutout', bootstrap=True)
			bootprofile = measure_profile(bootmap, binsize, reso=1.5)

			mass_bootstraps.append(fit_best_mass(bootprofile, zdist, binsize=binsize, maxtheta=180))
		mass_errs = np.std(mass_bootstraps, axis=0)
	np.array([avgmass, mass_errs, mass_errs]).dump('masses/%s_%s_mass.npy' % (sample_id, color))

	# !!!!! need to figure out weight for each bin for optimal fit

def gaussian(x, a1, b1, s1):
	gauss = a1 * np.exp(-np.square(x - b1) / (2 * (s1 ** 2)))
	return gauss

def fit_gauss_hist_one_sided(data, xs, nbins):

	histnp = np.histogram(data, bins=nbins)
	histvals = histnp[0]
	histbins = histnp[1]
	histbins2 = histbins[:len(histbins) - 1]
	lefthist = histvals[:histvals.argmax() + int(nbins/50)]
	leftbins = histbins[:histvals.argmax() + int(nbins/50)]
	maxval = np.max(histvals)
	peakcolor = histbins[histvals.argmax()]

	popt, pcov = curve_fit(gaussian, leftbins, lefthist, p0=[maxval, peakcolor, 0.1])
	return gaussian(xs, popt[0], popt[1], popt[2])