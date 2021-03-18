import healpy as hp
import numpy as np
from astropy import constants as const
import astropy.units as u
from colossus.halo import concentration
from colossus.halo import profile_nfw
from colossus.cosmology import cosmology
from colossus.lss import bias
from scipy.special import j0
from astropy.io import fits
import fitting
import importlib
importlib.reload(fitting)

cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()

# defining the critical surface density for lensing
def sigma_crit(z):
	return ((const.c ** 2) / (4. * np.pi * const.G) * (apcosmo.angular_diameter_distance(1100.) / (
		(apcosmo.angular_diameter_distance(z) * apcosmo.angular_diameter_distance_z1z2(z, 1100.))))).decompose().to(
		u.solMass * u.littleh / u.kpc ** 2, u.with_H0(apcosmo.H0))


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

	rho_avg = cosmo.rho_m(z)*u.solMass*(u.littleh**2)/(u.kpc**3)
	# amalgamation of constants outside the integral
	a = (rho_avg/(((1.+z)**3)*sigma_crit(z)*d_a**2))*bh/(2*np.pi)

	# scales
	ks = np.logspace(-5, 0, 200)*u.littleh/u.Mpc
	ls = ks*(1+z)*(apcosmo.angular_diameter_distance(z).to(u.Mpc/u.littleh, u.with_H0(apcosmo.H0)))

	# do an outer product of the thetas and ls for integration
	ltheta = np.outer(theta, ls)

	# compute matter power spectrum at comoving wavenumbers k
	mps = (cosmo.matterPowerSpectrum(ks.value, z=z)*(u.Mpc/u.littleh)**3).to((u.kpc/u.littleh)**3)

	# Eq. 13 in OOguri and Hamana 2011
	integrand = a*ls*j0(ltheta)*mps
	return np.trapz(integrand, x=ls)


# integrate kappa across redshift distribution dn/dz
def int_kappa(theta, m_200, terms, zdist, zbins=20):

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
	theta_list_rad = (np.arange(0.1, 360, 0.1) * u.arcmin).to('rad').value

	bothmodel = int_kappa(theta_list_rad, m_per_h, 'both', zdist)
	kmodel = hp.beam2bl(bothmodel, theta_list_rad, lmax=4096)

	k_space_filter = hp.gauss_beam((15 * u.arcmin).to('rad').value, lmax=4096)
	k_space_filter[:100] = 0

	kconvolved = np.array(kmodel) * np.array(k_space_filter)

	return hp.bl2beam(kconvolved, theta_list_rad)


def filtered_model_at_theta(zdist, m_per_h, inputthetas):
	theta_list = np.arange(0.1, 360, 0.1)
	model = filter_model(zdist, m_per_h)
	flat_thetas = inputthetas.flatten()
	kappa_vals = []
	for j in range(len(flat_thetas)):
		kappa_vals.append(model[np.abs(theta_list - flat_thetas[j]).argmin()])

	return np.array(kappa_vals).reshape(inputthetas.shape)









# simulate a stacked map usin
def model_stacked_map(zdist, m_per_h, imsize=240, reso=1.5):

	center = imsize/2 - 0.5
	x_arr, y_arr = np.mgrid[0:imsize, 0:imsize]
	radii_theta = np.sqrt(((x_arr - center) * reso) ** 2 + ((y_arr - center) * reso) ** 2)
	model_kappas = filtered_model_at_theta(zdist, m_per_h, radii_theta)
	return model_kappas

"""# calculate average model value in same bins as measured
def filtered_model_in_bins(zdist, obs_thetas, m_per_h):
	theta_list = np.arange(0.1, 360, 0.1)*u.arcmin
	model_full_range = filter_model(zdist, m_per_h)
	binsize = obs_thetas[1] - obs_thetas[0]

	theta_list = theta_list.value
	binned_kappa = []
	for theta in obs_thetas:

		binned_kappa.append(np.mean(model_full_range[np.where((theta_list > (theta - binsize/2)) & (theta_list < (theta + binsize/2)))]))

	return(binned_kappa)
"""



def filtered_model_center(zdist, obs_theta, m_per_h, reso=1.5, imsize=240):

	modelmap = model_stacked_map(zdist, m_per_h, reso=reso, imsize=imsize)
	return modelmap[int(imsize/2), int(imsize/2)]

# calculate average model value in same bins as measured
def filtered_model_in_bins(zdist, obs_thetas, m_per_h, binsize=12, reso=1.5, imsize=240, maxtheta=180):

	modelmap = model_stacked_map(zdist, m_per_h, reso=reso, imsize=imsize)

	profile = fitting.measure_profile(modelmap, binsize, reso, maxtheta=maxtheta)
	return profile

def lensing_kernel(zs):
	wk = 3/2*(apcosmo.Om0)*((apcosmo.H0/const.c)**2)*(1+zs)*\
	     (apcosmo.comoving_distance(zs))*(apcosmo.comoving_distance(1100)-apcosmo.comoving_distance(zs))/\
	     (apcosmo.comoving_distance(1100))

	return wk

def mass_to_avg_bias(m_per_h, zs):
	medz = np.median(zs)
	bh = bias.haloBias(M=m_per_h, z=medz, mdef='200c', model='tinker10')
	return bh