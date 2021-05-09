import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import interpolate
from astroquery.irsa import Irsa
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck18')
astropycosmo = cosmo.toAstropy()
import multiprocessing as mp
from functools import partial



def log_interp1d(xx, yy):
	logx = np.log10(xx)
	logy = np.log10(yy)
	lin_interp = interpolate.interp1d(logx, logy, fill_value='extrapolate')
	log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
	return log_interp

def log_interp_numpy(xx, yy):
	logx = np.log10(xx)
	logy = np.log10(yy)
	pfit = np.polyfit(logx, logy, 1)

	log_interp = lambda zz: np.power(10.0, np.polyval(pfit, np.log10(zz)))
	return log_interp

# calculate the rest frame 1.5 micron luminosities using log-linear interpolation
def rest_ir_lum(fluxes, zs, ref_lambda_rest):
	# should i apply a color correction to W3?
	if len(fluxes) == 2:

		obswavelengths = np.array([3.368, 4.618])
	else:

		obswavelengths = np.array([3.368, 4.618, 12.082])

	ref_lambda_obs = ref_lambda_rest * (1+zs)



	ref_lums = []
	for i in range(len(fluxes[0])):
		if np.isnan(fluxes[:, i]).any():
			ref_lums.append(np.nan)
		else:
			interp_func = log_interp1d(obswavelengths, fluxes[:, i])

			ref_lums.append(((interp_func(ref_lambda_obs[i]) * u.Jy * 4 * np.pi * (
					astropycosmo.luminosity_distance(zs[i])) ** 2).to('erg').value))

	return np.log10(ref_lums)