import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import units as u
import astropy.constants as con
from dust_extinction.averages import G03_SMCBar
from scipy.optimize import curve_fit
from scipy.optimize import minimize


ext = G03_SMCBar()

# https://astronomy.stackexchange.com/questions/16286/how-can-i-convolve-a-template-spectrum-with-a-photometric-filter-response-spectr
def integrate_over_filter_wavelength(lambdas, fluxes, responses):
	responded_flux = fluxes * responses * lambdas
	numerator = np.trapz(responded_flux, lambdas)
	denominator = np.trapz(responses/lambdas, lambdas)
	return numerator / denominator

def flux_in_band(band, redshifted_wavelengths, f_lambda):
	if band == 'g':
		band_table = fits.open('filter_curves.fits')[2].data
	else:
		band_table = fits.open('filter_curves.fits')[4].data
	obs_wavelengths = band_table['wavelength']
	response = band_table['respt']
	obs_fluxes = np.interp(obs_wavelengths, redshifted_wavelengths, f_lambda)

	int_flux = integrate_over_filter_wavelength(obs_wavelengths, obs_fluxes, response)
	return int_flux

def gminusi_color(redshifted_wavelengths, f_lambda):
	int_g_flux = flux_in_band('g', redshifted_wavelengths, f_lambda)
	int_i_flux = flux_in_band('i', redshifted_wavelengths, f_lambda)
	gminus_i = -2.5 * (np.log10(int_g_flux / int_i_flux))
	return gminus_i

# simulate what g-i color would observe from QSO at redshift z, using Vanden Berk 2001 template
# can redden spectrum in rest-frame if given a E(B-V) value
def vdb_color_at_z(z, reddening=0, mode='Ebv'):

	vdb_table = pd.read_csv('composite_spectra/vandenberk01_comp_QSO_spec.txt', delim_whitespace=True)
	wavelengths = vdb_table['lamb']
	f_lambda = vdb_table['flux']

	goodidxs = np.where(wavelengths > 1100.)[0]
	wavelengths = wavelengths[goodidxs]
	f_lambda = f_lambda[goodidxs]
	if mode == 'Ebv':
		f_lambda = f_lambda*ext.extinguish(x=np.array(wavelengths)*u.Angstrom, Ebv=reddening)
	else:
		f_lambda = f_lambda * ext.extinguish(x=np.array(wavelengths) * u.Angstrom, Av=reddening)


	redshifted_wavelengths = wavelengths * (1+z)

	gminus_i = gminusi_color(redshifted_wavelengths, f_lambda)

	return gminus_i


def relative_vdb_color(z, reddening, mode='Ebv'):
	unreddened = vdb_color_at_z(z, reddening=0, mode=mode)
	reddened = vdb_color_at_z(z, reddening=reddening, mode=mode)
	return reddened - unreddened

def simple_gminusi_color(observed_wavelengths, f_lambda):
	g_wavelength = 4686
	i_wavelength = 7480

	g_obs_flux = np.interp(g_wavelength, observed_wavelengths, f_lambda)
	i_obs_flux = np.interp(i_wavelength, observed_wavelengths, f_lambda)

	return -2.5 * np.log10(g_obs_flux/i_obs_flux)


def power_law_spectrum(alpha, wavelengths):
	fluxes = 1e10*wavelengths**(-alpha)
	return fluxes


def power_law_gminusi_color(alpha, z=0, Av=0):
	optical_wavelengths = np.linspace(1300., 5000., 100)
	spect = power_law_spectrum(alpha, optical_wavelengths)
	if Av > 0:
		spect = spect * ext.extinguish(x=optical_wavelengths * u.Angstrom, Av=Av)
	redshifted_wavelengths = optical_wavelengths * (1+z)

	simple = True
	if simple:
		gminusi = simple_gminusi_color(redshifted_wavelengths, spect)
	else:
		gminusi = gminusi_color(redshifted_wavelengths, spect)
	return gminusi


def power_law_redden_dist(min, max, alpha, size=1):
	"""Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
	r = np.random.random(size=size)
	ag, bg = min**alpha, max**alpha
	return (ag + (bg - ag)*r)**(1./alpha)


def model_color_dist(dummy, center_alpha, sigma_alpha, Av_alpha):
	alphas = np.random.normal(loc=center_alpha, scale=sigma_alpha, size=5000)
	Avs = power_law_redden_dist(0.05, 2.0, Av_alpha, size=5000)

	refcolor = power_law_gminusi_color(center_alpha)
	nzbins = 30

	obs_zs = fits.open('catalogs/derived/xdqso_specz_complete.fits')[1].data['Z']
	counts, bin_edges = np.histogram(obs_zs, bins=nzbins, density=True)
	counts *= (np.max(bin_edges) - np.min(bin_edges))/nzbins
	bincenters = (bin_edges[1:] + bin_edges[:-1]) / 2


	colors = []
	for j in range(len(alphas)):
		z = np.random.choice(bincenters, p=counts)
		colors.append(power_law_gminusi_color(alphas[j], z=z, Av=Avs[j]) - refcolor)
	return colors

	#deltagminusi = fits.open('catalogs/derived/xdqso_specz_complete.fits')[1].data['deltagmini']
	#data_hist, edges = np.histogram(deltagminusi, bins=100, range=(-0.5, 1.5), density=True)
	modhist = np.histogram(colors, bins=100, range=(-0.5, 1.5), density=True)[0]

	return modhist



	#chi = ((data_hist - modhist)**2)/data_hist

	#return np.sum(chi)



def fit_hist():
	deltagminusi = fits.open('catalogs/derived/xdqso_specz_complete.fits')[1].data['deltagmini']
	data_hist, edges = np.histogram(deltagminusi, bins=100, range=(-0.5, 1.5), density=True)
	colorcenters = (edges[1:] + edges[:-1]) / 2
	"""alphas = np.random.normal(0.3, 0.5, size=30)
	sigmas = np.random.uniform(0.05, 0.5, size=30)
	inds = np.random.normal(-1.5, 0.5, size=30)

	chis = []
	for j in range(len(alphas)):
		chis.append(model_color_dist(0, alphas[j], sigmas[j], inds[j]))
	argdfjs = np.where(chis == np.nanmin(chis))
	print(chis)"""


	#return alphas[argdfjs], sigmas[argdfjs], inds[argdfjs]
	popt, pcov = curve_fit(model_color_dist, colorcenters, data_hist, p0=[0.3, 0.2, -1.5], sigma=np.sqrt(data_hist))
	#popt = minimize(model_color_dist, x0=np.array([0, 0.3, 0.2, -1.5]))
	return popt



# Doi et al 2010
def integrate_over_filter_log_frequency(logfreq, Lnus, responses):
	numerator_integrand = Lnus * responses
	numerator = np.trapz(numerator_integrand, logfreq)
	denominator = np.trapz(responses, logfreq)
	return numerator / denominator

def richards_color_at_z(z):
	richardstable = fits.open('composite_spectra/Richards06_compositeQSOspectrum.fits')[1].data
	freqs = 10**richardstable['LogFreq']

	Lnus = (10**richardstable['LogLall'])/freqs
	redshifted_freqs = freqs/(1+z)

	g_band_table = fits.open('filter_curves.fits')[2].data
	i_band_table = fits.open('filter_curves.fits')[4].data

	g_obs_wavelengths = g_band_table['wavelength']*u.Angstrom
	g_response = np.flip(g_band_table['respt'])
	g_obs_freqs = np.flip((con.c / g_obs_wavelengths).to(u.Hz).value)
	log_g_obs_freqs = np.log10(g_obs_freqs)


	i_obs_wavelengths = i_band_table['wavelength'] * u.Angstrom
	i_response = np.flip(i_band_table['respt'])
	i_obs_freqs = np.flip((con.c / i_obs_wavelengths).to(u.Hz).value)
	log_i_obs_freqs = np.log10(i_obs_freqs)

	g_obs_lums = np.interp(g_obs_freqs, redshifted_freqs, Lnus)
	i_obs_lums = np.interp(i_obs_freqs, redshifted_freqs, Lnus)


	int_g_lum = integrate_over_filter_log_frequency(log_g_obs_freqs, g_obs_lums, g_response)
	int_i_lum = integrate_over_filter_log_frequency(log_i_obs_freqs, i_obs_lums, i_response)

	return -2.5 * (np.log10(int_g_lum/int_i_lum))