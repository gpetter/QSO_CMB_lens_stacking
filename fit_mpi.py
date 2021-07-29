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


def fit_mass_suite(samplename, bin_no, mode='mass'):

	# read in the stack of lensing convergence at positions of quasars
	stacked_map = np.load('stacks/%s_stack%s.npy' % (samplename, bin_no), allow_pickle=True)
	maxtheta = int(len(stacked_map) / 2 * reso)
	imsize = len(stacked_map)


	# measure the profile of the stack using annular bins
	kap_profile = fitting.measure_profile(stacked_map, binsize, reso=reso, maxtheta=maxtheta)

	# calculate redshift distribution dn/dz
	cat = fits.open('catalogs/derived/%s_binned.fits' % samplename)[1].data
	colorcat = cat[np.where(cat['bin'] == (bin_no+1))]
	zdist = colorcat['Z']

	# estimate uncertainty in each annulus for model fitting
	err_profile = fitting.annuli_errs(binsize=binsize, reso=reso, maxtheta=maxtheta)

	mod_weights = fitting.model_variance_weights(zdist, binsize=binsize, reso=reso, maxtheta=maxtheta)
	err_profile = err_profile / mod_weights


	avg_mass_or_bias = fitting.fit_best_bias_or_mass(kap_profile, zdist, binsize=binsize, sigma=err_profile,
	                                                 maxtheta=maxtheta, mode=mode)

	#peakkap = lensingModel.filtered_model_center(zdist, 0, 10**avg_mass, reso=reso, imsize=imsize)


	# to estimate uncertainty on mass, add noise map to the noiseless model and refit many times
	masses_or_biases, profiles, peakkappas = [], [], []

	bootnames = sorted(glob.glob('stacks/bootstacks/boot*_%s.npy' % bin_no))
	if mode == 'mass':
		p0 = 10**avg_mass_or_bias
	elif mode == 'bias':
		p0 = avg_mass_or_bias
	else:
		return


	for i in range(len(bootnames)):
		bootmap = np.load(bootnames[i], allow_pickle=True)
		bootprofile = fitting.measure_profile(bootmap, binsize, reso=reso, maxtheta=maxtheta)

		profiles.append(bootprofile)

		bootmass_or_bias = fitting.fit_best_bias_or_mass(bootprofile, zdist, p0=p0, binsize=binsize,
		                                sigma=err_profile, maxtheta=maxtheta, mode=mode)
		masses_or_biases.append(bootmass_or_bias)
		#peakkappas.append(lensingModel.filtered_model_center(zdist, 0, 10**bootmass, reso=reso, imsize=imsize))

	if mode == 'mass':
		masses = np.array(masses_or_biases)
		highermasses = masses[np.where(masses > avg_mass_or_bias)]
		higher_std = np.sqrt(1 / (len(highermasses) - 1) * np.sum(np.square(highermasses - avg_mass_or_bias)))
		lowermasses = masses[np.where(masses < avg_mass_or_bias)]
		lower_std = np.sqrt(1 / (len(lowermasses) - 1) * np.sum(np.square(lowermasses - avg_mass_or_bias)))
		param = 10 ** avg_mass_or_bias
		np.array([avg_mass_or_bias, higher_std, lower_std]).dump('masses/%s_%s_mass.npy' % (samplename, bin_no))
	elif mode == 'bias':
		param = avg_mass_or_bias
		np.array([avg_mass_or_bias, np.std(masses_or_biases)]).dump('bias/%s/%s.npy' % (samplename, bin_no))
	else:
		return

	#kappa_std = np.std(peakkappas)
	kap_errs = np.std(profiles, axis=0)

	plot=True

	if plot:



		obs_theta = np.arange(binsize / 2, maxtheta, binsize)
		theta_range = np.arange(0.5, maxtheta, 0.5)
		best_mass_profile = lensingModel.filtered_model_at_theta(zdist, param, theta_range, mode=mode)
		# lowest_mass_profile = filter_model(zdist, theta_range, 10**(lowmass))
		binned_model = lensingModel.filtered_model_in_bins(zdist, obs_theta, param, binsize, reso,
		                                                   maxtheta=maxtheta, mode=mode)
		theta_range = np.arange(0.5, maxtheta, 0.5) * u.arcmin.to('rad')
		oneterm = lensingModel.int_kappa(theta_range, param, 'one', zdist, mode=mode)
		twoterm = lensingModel.int_kappa(theta_range, param, 'two', zdist, mode=mode)
		plotting.plot_kappa_profile(bin_no, kap_profile, kap_errs, binsize, maxtheta, best_mass_profile, binned_model,
		                            oneterm, twoterm)



	#np.array([peakkap, kappa_std]).dump('peakkappas/%s_%s_kappa.npy' % (samplename, bin_no))


if __name__ == "__main__":

	# set up schwimmbad MPI pool
	pool = MPIPool()

	# if current instantiation is not the master, wait for tasks
	if not pool.is_master():
		pool.wait()
		sys.exit(0)

	sample_name = 'xdqso_specz'

	partial_suite = partial(fit_mass_suite, sample_name, mode='bias')

	pool.map(partial_suite, np.arange(7))

	pool.close()