import numpy as np
#import astroML.correlation as cor
import healpy as hp
from astropy.io import fits
from Corrfunc.mocks import *
from Corrfunc.utils import convert_3d_counts_to_cf, convert_rp_pi_counts_to_wp
import pymangle
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()
import plotting


def healpix_density_map(ras, decs, nsides):

	pix_of_sources = hp.ang2pix(nsides, ras, decs, lonlat=True)
	npix = hp.nside2npix(nsides)
	density_map = np.bincount(pix_of_sources, minlength=npix)

	return density_map


# draw points randomly on the sky
def uniform_sphere(npoints):
	randlons = np.random.uniform(0, 360, npoints)
	randdecs = np.arcsin(np.random.uniform(-1, 1, npoints)) * 180./np.pi
	return randlons, randdecs

#def uniform_contiguous():


def generate_random_points_from_mangle(manglename, npoints):
	manglemask = pymangle.Mangle('mangle/%s_footprint.ply' % manglename)
	ra_rand, dec_rand = manglemask.genrand(npoints)
	return ra_rand, dec_rand

def gen_rand_points_from_healpix(ras, decs, npoints, nside=32):
	pix_with_sources = hp.ang2pix(nside, ras, decs, lonlat=True)
	hpmap = np.zeros(hp.nside2npix(nside))
	hpmap[pix_with_sources] = 1

	# calculate how many points needed distributed over whole sky to result in npoints over observed region
	npoints_whole_sky = round(npoints * len(hpmap)/len(np.where(hpmap)[0]))

	randras, randdecs = uniform_sphere(npoints_whole_sky)
	# get those which fall inside observed region
	idxs_in_mask = (hpmap[hp.ang2pix(nside, randras, randdecs, lonlat=True)] == 1)
	randras, randdecs = randras[idxs_in_mask], randdecs[idxs_in_mask]

	return randras, randdecs

"""def angular_correlation_ML(sample_name, color, bootstrap=False):
	if (sample_name == 'xdqso') or (sample_name == 'xdqso_specz'):
		rakey, deckey = 'RA_XDQSO', 'DEC_XDQSO'
		zkey = 'PEAKZ'
	else:
		rakey, deckey = 'RA', 'DEC'
		zkey = 'Z'
	cat = fits.open('QSO_cats/%s_%s.fits' % (sample_name, color))[1].data
	ras, decs = cat[rakey], cat[deckey]
	bins = np.logspace(-1, 1, 20)
	if bootstrap:
		twoptang = cor.bootstrap_two_point_angular(ras, decs, bins, Nbootstraps=5)
		np.array(twoptang[0]).dump('clustering/%s_%s.npy' % (sample_name, color))
		np.array(twoptang[1]).dump('clustering/%s_%s_err.npy' % (sample_name, color))
	else:
		np.array(cor.two_point_angular(ras, decs, bins)).dump('clustering/%s_%s.npy' % (sample_name, color))"""


def angular_corr_from_coords(ras, decs, randras, randdecs, weights=None, randweights=None, nthreads=1, nbins=10):

	bins = np.logspace(-2, 1, (nbins + 1))

	# autocorrelation of catalog
	DD_counts = DDtheta_mocks(1, nthreads, bins, ras, decs, weights1=weights)

	# cross correlation between data and random catalog
	DR_counts = DDtheta_mocks(0, nthreads, bins, ras, decs, RA2=randras, DEC2=randdecs, weights1=weights, weights2=randweights)

	# autocorrelation of random points
	RR_counts = DDtheta_mocks(1, nthreads, bins, randras, randdecs, weights1=randweights)

	wtheta = convert_3d_counts_to_cf(len(ras), len(ras), len(randras), len(randras), DD_counts, DR_counts, DR_counts, RR_counts)

	return wtheta

def spatial_corr_from_coords(ras, decs, cz, randras, randdecs, randcz, weights=None, randweights=None, nthreads=1, nbins=10):

	bins = np.logspace(1, 2, nbins + 1)

	pimax = 100

	DD_counts = DDrppi_mocks(1, 2, nthreads, pimax, bins, ras, decs, cz, weights, is_comoving_dist=True)

	DR_counts = DDrppi_mocks(0, 2, nthreads, pimax, bins, ras, decs, cz, RA2=randras, DEC2=randdecs, CZ2=randcz, weights1=weights, weights2=randweights, is_comoving_dist=True)

	RR_counts = DDrppi_mocks(1, 2, nthreads, pimax, bins, randras, randdecs, randcz, randweights, is_comoving_dist=True)

	wp = convert_rp_pi_counts_to_wp(len(ras), len(ras), len(randras), len(randras), DD_counts, DR_counts, DR_counts, RR_counts, nbins, pimax)

	return wp/bins[:len(bins)-1]





def angular_correlation_function(sample_name, color, nbins=10, nbootstraps=0, nthreads=1, useweights=False):

	if (sample_name == 'xdqso') or (sample_name == 'xdqso_specz'):
		rakey, deckey = 'RA_XDQSO', 'DEC_XDQSO'

	else:
		rakey, deckey = 'RA', 'DEC'

	cat = fits.open('catalogs/derived/%s_%s.fits' % (sample_name, color))[1].data

	ras, decs = cat[rakey], cat[deckey]


	randcat = fits.open('catalogs/lss/eBOSS_randoms_fullsky.fits')[1].data
	# subsample huge random catalog

	bidxs = np.random.choice(len(randcat), 5*len(cat), replace=False)
	randcat = randcat[bidxs]
	randras, randdecs = randcat['RA'], randcat['DEC']


	#randras, randdecs = gen_rand_points_from_healpix(ras, decs, 5 * len(ras))

	if useweights:
		totweights = cat['WEIGHT_SYSTOT'] * cat['WEIGHT_CP'] * cat['WEIGHT_NOZ']
		randweights = randcat['WEIGHT_SYSTOT'] * randcat['WEIGHT_CP'] * randcat['WEIGHT_NOZ']
	else:
		totweights = None
		randweights = None

	wtheta = angular_corr_from_coords(ras, decs, randras, randdecs, weights=totweights, randweights=randweights, nthreads=nthreads, nbins=nbins)
	print(wtheta)

	if nbootstraps > 0:
		wtheta_realizations = []
		for j in range(nbootstraps):
			bootidxs = np.random.randint(0, len(ras), len(ras))
			bootras = ras[bootidxs]
			bootdecs = decs[bootidxs]

			# should i also regenerate random catalog?
			boot_wtheta = angular_corr_from_coords(bootras, bootdecs, randras, randdecs, nthreads=nthreads, nbins=nbins)
			wtheta_realizations.append(boot_wtheta)
		wtheta_std = np.std(wtheta_realizations, axis=0)

		amp_and_err = np.array([wtheta, wtheta_std])
		amp_and_err.dump('clustering/%s_%s.npy' % (sample_name, color))
	else:
		amp_and_err = np.array([wtheta, np.zeros(nbins)])
		amp_and_err.dump('clustering/%s_%s.npy' % (sample_name, color))


def spatial_correlation_function(sample_name, color, nbins=10, nbootstraps=0, nthreads=1, useweights=False):
	if (sample_name == 'xdqso') or (sample_name == 'xdqso_specz'):
		rakey, deckey = 'RA_XDQSO', 'DEC_XDQSO'

	else:
		rakey, deckey = 'RA', 'DEC'

	cat = fits.open('catalogs/derived/%s_%s.fits' % (sample_name, color))[1].data
	ras, decs = cat[rakey], cat[deckey]
	zs = cat['Z']
	cmdists = apcosmo.angular_diameter_distance(zs)

	randcat = fits.open('catalogs/lss/eBOSS_randoms_fullsky.fits')[1].data
	# subsample huge random catalog

	bidxs = np.random.choice(len(randcat), 15 * len(cat), replace=False)
	randcat = randcat[bidxs]
	randras, randdecs = randcat['RA'], randcat['DEC']
	randcmdists = apcosmo.angular_diameter_distance(randcat['Z'])

	if useweights:
		totweights = cat['WEIGHT_SYSTOT'] * cat['WEIGHT_CP'] * cat['WEIGHT_NOZ']
		randweights = randcat['WEIGHT_SYSTOT'] * randcat['WEIGHT_CP'] * randcat['WEIGHT_NOZ']
	else:
		totweights = None
		randweights = None

	wr = spatial_corr_from_coords(ras, decs, cmdists, randras, randdecs, randcmdists, weights=totweights, randweights=randweights, nthreads=nthreads, nbins=nbins)

	print(wr)

	if nbootstraps > 0:
		wr_realizations = []
		for j in range(nbootstraps):
			bootidxs = np.random.choice(len(ras), len(ras))
			bootras = ras[bootidxs]
			bootdecs = decs[bootidxs]
			bootdists = apcosmo.angular_diameter_distance(zs[bootidxs])

			# should i also regenerate random catalog?
			boot_wr = spatial_corr_from_coords(bootras, bootdecs, bootdists, randras, randdecs, randcmdists, nthreads=nthreads, nbins=nbins)
			wr_realizations.append(boot_wr)
		wtheta_std = np.std(wr_realizations, axis=0)

		amp_and_err = np.array([wr, wtheta_std])
		amp_and_err.dump('clustering/%s_%s.npy' % (sample_name, color))
	else:
		amp_and_err = np.array([wr, np.zeros(nbins)])
		amp_and_err.dump('clustering/%s_%s.npy' % (sample_name, color))

	plotting.plot_spatial_correlation_function()