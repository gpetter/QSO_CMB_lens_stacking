import convergence_map
import importlib
import stacking
import fitting
import astropy.units as u
import glob
import sample
import autocorrelation
import first_stacking
import plotting
importlib.reload(convergence_map)
importlib.reload(plotting)
importlib.reload(stacking)
importlib.reload(sample)
importlib.reload(fitting)
importlib.reload(autocorrelation)
importlib.reload(first_stacking)



nsides = 2048
lmin = 100
smooth_fwhm = 15. * u.arcmin

reso = 1.5
imsize = 240

color = 'blue'
sample_id = 'xdqso_specz'
plots = True
nbins = 5
colorkey = 'r-W2'
mission = 'both'
lumkey = 'logL1_5'


# prepare all data for stacking
# can convert lensing klm file to smoothed masked map
# can do the same for all noise realizations
# can take a quasar catalog and define a luminosity complete sample
# then can split sample into red, control and blue quasars with the same redshift distribution
# will weight across bolometric luminosity to account for luminosity effects
def prepare_data(process_map, process_noise, process_sample, select_sample, sample_name):
	if process_map:
		convergence_map.klm_2_product('maps/dat_klm.fits', smooth_fwhm, 'maps/mask.fits', nsides, lmin, writename='maps/smoothed_masked_planck')

	if process_noise:
		noisemaplist = glob.glob('noisemaps/klm/sim*')
		for j in range(len(noisemaplist)):
			convergence_map.klm_2_product(noisemaplist[j], smooth_fwhm, 'maps/mask.fits', nsides, lmin, subtract_mf=True, writename='noisemaps/maps/%s' % int(noisemaplist[j].split('.')[0].split('_')[2]))

	if process_sample:
		#sample.fix_dr16()
		#sample.define_core_sample(sample_name)
		sample.update_WISE()
		if sample_name == 'xdqso_specz':
			sample.match_phot_qsos_to_spec_qsos()
			sample.write_properties(sample_name, speczs=True)
		else:
			sample.write_properties(sample_name, speczs=False)

	if select_sample:
		sample.luminosity_complete_cut(sample_name, -21, 0.5, 2.5, plots=plots, magcut=100, pcut=0.95, peakscut=1, apply_planck_mask=True, band='i', colorkey=colorkey)
		sample.red_blue_samples(sample_name, plots=plots, ncolorbins=nbins, offset=False, remove_reddest=False, colorkey=colorkey, lumkey=lumkey)


prepare_data(False, False, False, True, sample_id)
sample.kappa_for_color(sample_id, colorkey, bins=nbins, removereddest=True, dostacks=True, mission=mission, use_weights=False)
#sample.clustering_for_color(sample_id, mode='spatial_cross', cap='both', minscale=0.4, maxscale=1.3, bins=5, use_weights=False)
#sample.clustering_for_color(sample_id, mode='ang_cross', minscale=-2, maxscale=0, bins=6)
#autocorrelation.angular_correlation_function(sample_id, 10, nbootstraps=3, nthreads=12)

#print(stacking.stack_suite(color, sample_id, True, False, mode='fast', reso=reso, imsize=imsize, nsides=nsides, temperature=False, bootstrap=True))
#stacking.stack_suite(color, sample_id, True, False, mode='cutout')
#fitting.fit_mass_suite(color, sample_id, plot=True, reso=reso, binsize=10, mode='bootstrap')
#sample.radio_detect_fraction(qso_cat_name=sample_id, colorkey=colorkey, radio_name='FIRST')
#sample.median_radio_flux_for_color(sample_id, colorkey, offset=False, mode='lum', nbootstraps=3, remove_reddest=True)
#first_stacking.download_first_cutouts_mp()
#first_stacking.stack_first(color, sample_id)
#autocorrelation.angular_correlation_function(sample_id, color, nbootstraps=3, nthreads=12, nbins=5, useweights=False)
#autocorrelation.cross_correlation_function_angular(sample_id, 1, nbins=10, nthreads=12, nbootstraps=3)
#autocorrelation.spatial_correlation_function(sample_id, 'all', 'NGC', nthreads=12, nbins=40, nbootstraps=3, useweights=False, minscale=0, maxscale=2.5)
#plotting.plot_spatial_correlation_function(40, 0, 2.5, 1)
#sample.temp_for_color(sample_id, bins=10, offset=False, removereddest=False)
#fitting.fit_mass_to_cutouts(sample_id, color, nbootstraps=5)
#print(stacking.stack_suite(color, sample_id, True, False, reso=1.5, mode='fast', temperature=True))
#stacking.stack_suite(color, sample_id, True, True, temperature=False, reso=reso, mode='mpi')
#autocorrelation.cross_corr_func_spatial(color, nbins=10, nthreads=12, useweights=True)
#sample.sed_for_color(sample_id, 10, False)
#first_stacking.median_first_stack(color, sample_id, write=True)