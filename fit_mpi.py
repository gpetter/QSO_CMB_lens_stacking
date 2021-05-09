import numpy as np
import lensingModel
import fitting
from astropy.io import fits
import glob
from functools import partial
import sys
from schwimmbad import MPIPool
import astropy.units as u
import importlib
import plotting
importlib.reload(fitting)
importlib.reload(lensingModel)
importlib.reload(plotting)

binsize = 12
reso = 1.5


def fit_mass_suite(samplename, colorbin):

	# read in the stack of lensing convergence at positions of quasars
	stacked_map = np.load('stacks/%s_stack%s.npy' % (samplename, colorbin), allow_pickle=True)
	maxtheta = int(len(stacked_map) / 2 * reso)
	imsize = len(stacked_map)


	# measure the profile of the stack using annular bins
	kap_profile = fitting.measure_profile(stacked_map, binsize, reso=reso, maxtheta=maxtheta)

	# calculate redshift distribution dn/dz
	cat = fits.open('catalogs/derived/%s_colored.fits' % samplename)[1].data
	colorcat = cat[np.where(cat['colorbin'] == (colorbin+1))]
	zdist = colorcat['Z']

	# estimate uncertainty in each annulus for model fitting
	err_profile = fitting.annuli_errs(binsize=binsize, reso=reso, maxtheta=maxtheta)

	mod_weights = fitting.model_variance_weights(zdist, binsize=binsize, reso=reso, maxtheta=maxtheta)
	err_profile = err_profile / mod_weights


	avg_mass = fitting.fit_best_mass(kap_profile, zdist, binsize=binsize, sigma=err_profile, maxtheta=maxtheta)
	print(avg_mass)
	peakkap = lensingModel.filtered_model_center(zdist, 0, 10**avg_mass, reso=reso, imsize=imsize)


	# to estimate uncertainty on mass, add noise map to the noiseless model and refit many times
	masses, profiles, peakkappas = [], [], []

	bootnames = sorted(glob.glob('bootstacks/boot*_%s.npy' % colorbin))

	for i in range(len(bootnames)):
		bootmap = np.load(bootnames[i], allow_pickle=True)
		bootprofile = fitting.measure_profile(bootmap, binsize, reso=reso, maxtheta=maxtheta)
		profiles.append(bootprofile)
		bootmass = fitting.fit_best_mass(bootprofile, zdist, p0=10**avg_mass, binsize=binsize, sigma=err_profile, maxtheta=maxtheta)
		masses.append(bootmass)
		peakkappas.append(lensingModel.filtered_model_center(zdist, 0, 10**bootmass, reso=reso, imsize=imsize))

	masses = np.array(masses)
	highermasses = masses[np.where(masses > avg_mass)]
	higher_std = np.sqrt(1/(len(highermasses)-1)*np.sum(np.square(highermasses - avg_mass)))
	lowermasses = masses[np.where(masses < avg_mass)]
	lower_std = np.sqrt(1/(len(lowermasses)-1)*np.sum(np.square(lowermasses - avg_mass)))
	kappa_std = np.std(peakkappas)
	kap_errs = np.std(profiles, axis=0)

	plot=True

	if plot:
		obs_theta = np.arange(binsize / 2, maxtheta, binsize)
		theta_range = np.arange(0.5, maxtheta, 0.5)
		best_mass_profile = lensingModel.filtered_model_at_theta(zdist, 10 ** (avg_mass), theta_range)
		# lowest_mass_profile = filter_model(zdist, theta_range, 10**(lowmass))
		binned_model = lensingModel.filtered_model_in_bins(zdist, obs_theta, 10 ** (avg_mass), binsize, reso,
		                                                   maxtheta=maxtheta)
		theta_range = np.arange(0.5, maxtheta, 0.5) * u.arcmin.to('rad')
		oneterm = lensingModel.int_kappa(theta_range, 10 ** avg_mass, 'one', zdist)
		twoterm = lensingModel.int_kappa(theta_range, 10 ** avg_mass, 'two', zdist)
		plotting.plot_kappa_profile(colorbin, kap_profile, kap_errs, binsize, maxtheta, best_mass_profile, binned_model,
		                            oneterm, twoterm)


	np.array([avg_mass, higher_std, lower_std]).dump('masses/%s_%s_mass.npy' % (samplename, colorbin))
	np.array([peakkap, kappa_std]).dump('peakkappas/%s_%s_kappa.npy' % (samplename, colorbin))


if __name__ == "__main__":

	# set up schwimmbad MPI pool
	pool = MPIPool()

	# if current instantiation is not the master, wait for tasks
	if not pool.is_master():
		pool.wait()
		sys.exit(0)

	sample_name = 'xdqso_specz'

	partial_suite = partial(fit_mass_suite, sample_name)

	pool.map(partial_suite, np.arange(5))

	pool.close()