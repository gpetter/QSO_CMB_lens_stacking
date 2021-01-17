import convergence_map
import importlib
import stacking
import fitting
import astropy.units as u
import glob
import sample
import autocorrelation
import first_stacking
importlib.reload(convergence_map)
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
		if sample_name == 'xdqso_specz':
			#sample.match_phot_qsos_to_spec_qsos()
			sample.write_properties(sample_name, speczs=True)
		else:
			sample.write_properties(sample_name, speczs=False)

	if select_sample:
		#sample.luminosity_complete_cut(sample_name, -20, 0.5, 2.5, plots=plots, magcut=21.5, pcut=0.95, peakscut=1, apply_planck_mask=True)
		sample.red_blue_samples(sample_name, plots=plots)


#repare_data(False, False, False, True, sample_id)

#sample.first_fraction(sample_id, 'first')

#stacking.stack_suite(color, sample_id, True, False, mode='cutout', reso=reso, imsize=imsize, nsides=nsides)

#fitting.fit_mass_suite(color, sample_id, plot=True, use_peak=False, do_stack=False, reso=reso)

#sample.radio_detect_fraction(qso_cat_name=sample_id, radio_name='LoTSS', bins=10)
#sample.median_radio_flux_for_color(sample_id)
#first_stacking.download_first_cutouts_mp()
#first_stacking.stack_first(color, sample_id)

#autocorrelation.angular_correlation_function(sample_id, color, nbootstraps=3, nthreads=10, nbins=20, useweights=False)
#autocorrelation.spatial_correlation_function(sample_id, color, nthreads=10, nbins=10)
sample.median_radio_flux_for_color(sample_id, luminosity=True)
#sample.kappa_for_color(sample_id)
#plotting.plot_ang_correlation_function()