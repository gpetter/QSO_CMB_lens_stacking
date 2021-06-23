

from astropy.io import fits
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
import healpy as hp
import stacking
import plotting
import importlib
from scipy.optimize import curve_fit
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



def radio_detect_fraction(qso_cat_name, colorkey, radio_name='FIRST', lowmag=10, highmag=30, return_plot=False,
                          offset=False):

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
	for i in range(int(np.max(qso_cat['bin']))):

		binnedcat = qso_cat[np.where(qso_cat['bin'] == (i+1))]

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
	medcolors = np.linspace(0, 1, int(np.max(qso_cat['bin'])))


	for i in range(int(np.max(qso_cat['bin']))):
		#if i==bins:
		#	binnedcat = reddest_cat
		#else:
		#	binnedcat = qso_cat[gminusibinidxs[i]]
		#binnedcat = qso_cat[gminusibinidxs[i]]
		binnedcat = qso_cat[np.where(qso_cat['bin'] == (i+1))]
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

def kappa_for_bin(qso_cat_name, colorkey, removereddest=False, dostacks=True, mission='planck', use_weights=True,
                  mode='color'):


	#if removereddest:
		#qso_cat = qso_cat[remove_reddest_bin(qso_cat[colorkey], qso_cat['Z'], 10, offset)]
	#reddest_cat = qso_cat[np.where(qso_cat['colorbin'] == -1)]
	#qso_cat = qso_cat[np.where(qso_cat['deltagmini'] < 0.25)]

	"""if mode == 'color':
		qso_cat = fits.open('catalogs/derived/%s_colored.fits' % qso_cat_name)[1].data
	elif mode == 'bal':
		qso_cat = fits.open('catalogs/derived/dr16_bal.fits')[1].data
	elif mode == 'bhmass':
		qso_cat = fits.open('catalogs/derived/bhmass/dr14_mass_binned.fits')[1].data
	else:
		print('specify mode')
		return"""
	qso_cat = fits.open('catalogs/derived/%s_binned.fits' % qso_cat_name)[1].data

	#gminusibinidxs = bin_by_color(qso_cat['g-i'], qso_cat['Z'], bins, offset=False)

	kappas, errs, boots = [], [], []

	masses = []
	planck_kappas, act_kappas = [], []
	bins = int(np.max(qso_cat['bin']))

	for i in range(bins):

		binnedcat = qso_cat[np.where(qso_cat['bin'] == (i+1))]
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
				planckkappa = stacking.fast_stack(ras, decs, hp.read_map('maps/smoothed_masked_planck.fits'),
				                                  iterations=100, bootstrap=True, weights=weights, prob_weights=probweights)
				kappas.append(planckkappa[0])
				errs.append(np.std(planckkappa[1]))
				act_kappas = None
				planck_kappas = None
			elif mission == 'act':
				actkappa = stacking.fast_stack(ras, decs, hp.read_map('maps/both_ACT.fits'), iterations=100,
				                               bootstrap=True, weights=weights, prob_weights=probweights)
				kappas.append(actkappa[0])
				errs.append(np.std(actkappa[1]))
				act_kappas = None
				planck_kappas = None
			else:
				planckkappa = stacking.fast_stack(ras, decs, hp.read_map('maps/smoothed_masked_planck.fits'),
				                                  iterations=200, bootstrap=True, weights=weights,
				                                  prob_weights=probweights)

				actkappa = stacking.fast_stack(ras, decs, hp.read_map('maps/both_ACT.fits'), iterations=200,
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

	colors = range(1, bins+1)
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
		kap = stacking.fast_stack(reddest_cat['RA'], reddest_cat['DEC'],
		                          hp.read_map('maps/smoothed_masked_planck.fits'), iterations=100, bootstrap=True)
		if len(kap) > 1:
			kappas.append(kap[0])
			errs.append(np.std(kap[1]))
		else:
			kappas.append(kap)
			errs.append(0)

	#plotting.plot_kappa_v_color(kappas, errs, transforms=[kappas_for_masses, masses_for_kappas], remove_reddest=removereddest, linfit=linfit)
	plotting.plot_kappa_v_color(kappas, errs, colorkey, planck_kappas=planck_kappas, act_kappas=act_kappas,
	                transforms=None, remove_reddest=removereddest, linfit=None, mode=mode)
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

	gminusibinidxs = bin_samples.bin_by_color(qso_cat[colorkey], qso_cat['Z'], bins, offset)

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

	gminusibinidxs = bin_samples.bin_by_color(qso_cat[colorkey], qso_cat['Z'], bins, offset)

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


def clustering_for_bin(qso_cat_name, clustermode, cap, refsample, bins=10, minscale=-1, maxscale=0, use_weights=False,
                       pimax=50):
	n_samples = int(np.max(fits.open('catalogs/derived/%s_binned.fits' % qso_cat_name)[1].data['bin']))
	samples = np.arange(n_samples)
	#samples = [0, 4, 9]
	if clustermode == 'ang_cross':
		for sample in samples:
			autocorrelation.cross_correlation_function_angular(qso_cat_name, sample+1, nbins=bins,
			                                                   nbootstraps=10, nthreads=12, minscale=minscale,
			                                                   maxscale=maxscale)
		plotting.plot_ang_cross_corr(qso_cat_name, bins, minscale, maxscale, samples)
	elif clustermode == 'spatial_cross':
		for sample in samples:
			autocorrelation.cross_corr_func_spatial(qso_cat_name, sample+1, minscale, maxscale, cap, refsample,
			                                        nbins=bins, nbootstraps=10, nthreads=1,
			                                        useweights=use_weights, pimax=pimax)
		plotting.plot_spatial_cross_corr(samples, cap)
	elif clustermode == 'spatial':
		for j in range(n_samples):
			autocorrelation.spatial_correlation_function(qso_cat_name, j+1, cap, bins, nbootstraps=3, nthreads=12,
			                                             useweights=use_weights, minscale=minscale,
			                                             maxscale=maxscale, pimax=pimax)
		plotting.plot_spatial_correlation_function(qso_cat_name, cap, bins, minscale, maxscale, n_samples)