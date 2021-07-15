import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.table import Table

import myCorrfunc
import random_catalogs
import importlib
import resampling
import twoPointCFs
import redshift_dists
importlib.reload(resampling)
importlib.reload(redshift_dists)
importlib.reload(twoPointCFs)
importlib.reload(myCorrfunc)
importlib.reload(random_catalogs)


from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()
import plotting

# calculate comoving distances for a catalog and write out new catalog
# enables much faster correlation functions than just passing redshifts
def write_comoving_distances():
	randcat = fits.open('catalogs/lss/eBOSS_QSO_clustering_random-NGC-vDR16.fits')[1].data
	randzs = randcat['Z']
	rand_comov_dists = cosmo.comovingDistance(np.zeros(len(randzs)), randzs)
	randtab = Table(randcat)
	randtab['comov_dist'] = rand_comov_dists
	randtab.write('catalogs/lss/eBOSS_randoms_NGC_comov.fits', format='fits', overwrite=True)


# compute angular autocorrelation function
def angular_correlation_function(sample_name, colorbin, nbins=10, nbootstraps=0, nthreads=1, useweights=False):

	cat = fits.open('catalogs/derived/%s_binned.fits' % sample_name)[1].data

	colorcat = cat[np.where(cat['colorbin'] == colorbin)]
	ras, decs = colorcat['RA'], colorcat['DEC']




	"""randcat = fits.open('catalogs/lss/eBOSS_QSO_clustering_random-NGC-vDR16.fits')[1].data
	# subsample huge random catalog

	bidxs = np.random.choice(len(randcat), 5*len(cat), replace=False)
	randcat = randcat[bidxs]
	randras, randdecs = randcat['RA'], randcat['DEC']"""


	randras, randdecs = random_catalogs.gen_rand_points_from_healpix(ras, decs, 10 * len(ras))

	if useweights:
		totweights = cat['WEIGHT_SYSTOT'] * cat['WEIGHT_CP'] * cat['WEIGHT_NOZ']
		randweights = randcat['WEIGHT_SYSTOT'] * randcat['WEIGHT_CP'] * randcat['WEIGHT_NOZ']
	else:
		totweights = None
		randweights = None

	wtheta = twoPointCFs.angular_corr_from_coords(ras, decs, randras, randdecs, weights=totweights,
	                                              randweights=randweights, nthreads=nthreads, nbins=nbins)
	print(wtheta)

	if nbootstraps > 0:
		wtheta_realizations = []
		for j in range(nbootstraps):
			bootidxs = np.random.choice(len(ras), len(ras))
			bootras = ras[bootidxs]
			bootdecs = decs[bootidxs]

			randbootidx = np.random.choice(len(randras), len(randras))
			bootrandras = randras[randbootidx]
			bootranddecs = randdecs[randbootidx]


			boot_wtheta = twoPointCFs.angular_corr_from_coords(bootras, bootdecs, bootrandras, bootranddecs,
			                                                   nthreads=nthreads, nbins=nbins)
			wtheta_realizations.append(boot_wtheta)
		wtheta_std = np.std(wtheta_realizations, axis=0)

		amp_and_err = np.array([wtheta, wtheta_std])
		amp_and_err.dump('angclustering/%s_%s.npy' % (sample_name, colorbin))
	else:
		amp_and_err = np.array([wtheta, np.zeros(nbins)])
		amp_and_err.dump('angclustering/%s_%s.npy' % (sample_name, colorbin))

	plotting.plot_ang_correlation_function(sample_name, nbins)


# 3D autocorrelation
def spatial_correlation_function(sample_name, colorbin, cap, nbins=10, nbootstraps=0, nthreads=1,
                                 useweights=False, minscale=0, maxscale=1.5, pimax=50, twoD=False):

	# if doing an autocorrelation of an entire eBOSS LSS tracer catalog
	if colorbin == 'all':
		if cap == 'both':
			cat = fits.open('catalogs/lss/%s/fullsky_comov.fits' % sample_name)[1].data
		else:
			cat = fits.open('catalogs/lss/%s/%s_comov.fits' % (sample_name, cap))[1].data

	else:
		cat = fits.open('catalogs/derived/%s_binned.fits' % sample_name)[1].data
		cat = cat[np.where(cat['bin'] == colorbin)]

	#cat = cat[np.where(cat['Z'] > 0.8)]

	if cap == 'both':
		randcat = random_catalogs.fullsky_eboss_randoms(5)
	else:
		if cap == 'NGC':
			cat = cat[np.where((cat['RA'] > 90) & (cat['RA'] < 270))]
		else:
			cat = cat[np.where((cat['RA'] < 90) | (cat['RA'] > 270))]

		randcat = fits.open('catalogs/lss/%s/randoms_%s_comov.fits' % (sample_name, cap))[1].data
		#randcat = randcat[np.where(randcat['Z'] > 0.8)]
		#if useweights:
		#else:
		#	randcat = fits.open('catalogs/lss/%s/randoms_%s_subsampled.fits' % (sample_name, cap))[1].data
		#randcat = randcat[:10*len(cat)]
		#randcat = randcat[np.random.choice(np.arange(len(randcat)), 5*len(cat), replace=False)]

	ras, decs, cmdists = cat['RA'], cat['DEC'], cat['comov_dist']
	randras, randdecs, randcmdists = randcat['RA'], randcat['DEC'], randcat['comov_dist']

	if twoD:
		bins = np.linspace(0.1, 10**maxscale, int(10**maxscale)+1)
		pimax = int(10**maxscale)
	else:
		bins = np.logspace(minscale, maxscale, nbins + 1)

	# apply weights to only autocorrelate tracers within the redshift range that overlaps with QSOs

	if sample_name == 'eBOSS_QSO':
		data_overlap_weights = np.ones(len(cat))
		random_overlap_weights = np.ones(len(randcat))
	else:
		qso_zs = fits.open('catalogs/lss/eBOSS_QSO/%s_comov.fits' % cap)[1].data['Z']
		overlap_zs, overlap_dndz = redshift_dists.redshift_overlap(cat['Z'], qso_zs, bin_avgs=False)

		data_overlap_weights = overlap_dndz[np.digitize(cat['Z'], overlap_zs) - 1]
		random_overlap_weights = overlap_dndz[np.digitize(randcat['Z'], overlap_zs) - 1]


	if useweights:
		totweights = cat['WEIGHT_SYSTOT'] * cat['WEIGHT_CP'] * cat['WEIGHT_NOZ'] * data_overlap_weights
		randweights = randcat['WEIGHT_SYSTOT'] * randcat['WEIGHT_CP'] * randcat['WEIGHT_NOZ'] * random_overlap_weights
	else:
		totweights = None
		randweights = None

	avgrbin = []
	for j in range(len(bins)-1):
		avgrbin.append(np.mean([bins[j], bins[j + 1]]))
	avgrbin = np.array(avgrbin)

	wr = twoPointCFs.spatial_corr_from_coords(ras, decs, cmdists, randras, randdecs, randcmdists, bins,
	                                          weights=totweights, randweights=randweights, nthreads=nthreads,
	                                          pimax=pimax, twoD=twoD)
	if twoD:
		plotting.plot_2d_corr_func(wr, '%s_auto' % sample_name)
		return

	wr_realizations = []


	if nbootstraps > 0:


		for j in range(nbootstraps):
			#bootidxs, randbootidx = np.random.choice(len(ras), len(ras)), np.random.choice(len(randras), len(randras))
			bootidxs, randbootidx = resampling.bootstrap_sky_bins(ras, decs, randras, randdecs, 10)
			bootras, bootdecs, bootdists = ras[bootidxs], decs[bootidxs], cmdists[bootidxs]
			bootrandras, bootranddecs, bootranddist = randras[randbootidx], randdecs[randbootidx], randcmdists[randbootidx]


			if useweights:
				bootweights, bootrandweights = totweights[bootidxs], randweights[randbootidx]
			else:
				bootweights, bootrandweights = None, None

			boot_wr = twoPointCFs.spatial_corr_from_coords(bootras, bootdecs, bootdists, bootrandras, bootranddecs,
			                                               bootranddist, bins, weights=bootweights,
			                                               randweights=bootrandweights, nthreads=nthreads,
			                                               pimax=pimax)
			wr_realizations.append(boot_wr)
		wtheta_std = np.std(wr_realizations, axis=0)

		amp_and_err = np.array([avgrbin, wr, wtheta_std])
		amp_and_err.dump('clustering/spatial/%s/%s_%s.npy' % (cap, sample_name, colorbin))
		np.array(wr_realizations).dump('clustering/spatial/%s/%s_%s_reals.npy' % (cap, sample_name, colorbin))


	else:
		amp_and_err = np.array([avgrbin, wr, np.zeros(nbins)])
		amp_and_err.dump('clustering/spatial/%s/%s_%s.npy' % (cap, sample_name, colorbin))

	return amp_and_err


# angular cross correlation
def cross_correlation_function_angular(sample_name, colorbin, nbins=10, nbootstraps=0, nthreads=1, useweights=False,
                                       minscale=-1, maxscale=0):
	cat = fits.open('catalogs/derived/%s_binned.fits' % sample_name)[1].data
	colorcat = cat[np.where(cat['colorbin'] == colorbin)]

	colorras, colordecs = colorcat['RA'], colorcat['DEC']

	#ctrlcat = cat[np.where((cat['colorbin'] >= 4) & (cat['colorbin'] <= 7))]
	ctrlcat = fits.open('catalogs/derived/%s_complete.fits' % sample_name)[1].data
	ctrlras, ctrldecs = ctrlcat['RA'], ctrlcat['DEC']


	#randcat = fits.open('random_catalogs/ebossrands.fits')[1].data
	#randcat = fits.open('catalogs/lss/eBOSS_fullsky_randoms_shuffled_comov.fits')[1].data
	randcat = fits.open('random_catalogs/dr8rands.fits')[1].data
	#randcat = randcat[:10*len(ctrlras)]
	randras, randdecs = randcat['RA'], randcat['DEC']
	#randras, randdecs = random_catalogs.gen_rand_points_from_healpix(ctrlras, ctrldecs, 10 * len(ctrlras))
	#randras, randdecs = uniform_sphere(10*len(ctrlras))


	crosscorr = twoPointCFs.ang_cross_corr_from_coords(colorras, colordecs, ctrlras, ctrldecs, randras, randdecs,
	                                                   minscale, maxscale, nthreads=nthreads, nbins=nbins)

	if nbootstraps > 0:
		wtheta_realizations = []
		for j in range(nbootstraps):
			bootidxs = np.random.choice(len(colorras), len(colorras))
			bootras = colorras[bootidxs]
			bootdecs = colordecs[bootidxs]

			refbootidx = np.random.choice(len(ctrlras), len(ctrlras))
			bootctrlras, bootctrldecs = ctrlras[refbootidx], ctrldecs[refbootidx]

			randbootidx = np.random.choice(len(randras), len(randras))
			bootrandras = randras[randbootidx]
			bootranddecs = randdecs[randbootidx]

			boot_wtheta = twoPointCFs.ang_cross_corr_from_coords(bootras, bootdecs, bootctrlras, bootctrldecs,
			                                                     bootrandras, bootranddecs, minscale, maxscale,
			                                                     nthreads=nthreads, nbins=nbins)
			wtheta_realizations.append(boot_wtheta)
		wtheta_std = np.std(wtheta_realizations, axis=0)

		amp_and_err = np.array([crosscorr, wtheta_std])
		amp_and_err.dump('clustering/ang_cross/%s_%s.npy' % (sample_name, colorbin))
	else:
		amp_and_err = np.array([crosscorr, np.zeros(nbins)])
		amp_and_err.dump('clustering/ang_cross/%s_%s.npy' % (sample_name, colorbin))


# 3D cross correlation
def cross_corr_func_spatial(bin, qso_cat_name, minscale, maxscale, cap, refsample, nbins=10, nbootstraps=0, nthreads=1,
                            useweights=False, pimax=50, twoD=False):
	#
	cat = fits.open('catalogs/derived/%s_binned.fits' % qso_cat_name)[1].data


	if cap == 'both':
		refcat = fits.open('catalogs/lss/eBOSS_QSO/eBOSS_fullsky_comov.fits')[1].data
		#refcat = cat[np.where((cat['colorbin'] > 1) & (cat['colorbin'] < 10))]
		randcat = random_catalogs.fullsky_eboss_randoms(10)
	else:
		if cap == 'NGC':
			cat = cat[np.where((cat['RA'] > 90) & (cat['RA'] < 270))]
		else:
			cat = cat[np.where((cat['RA'] < 90) | (cat['RA'] > 270))]
		refcat = fits.open('catalogs/lss/%s/%s_comov.fits' % (refsample, cap))[1].data
		randcat = fits.open('catalogs/lss/%s/randoms_%s_comov.fits' % (refsample, cap))[1].data

	randcat = randcat[:10 * len(refcat)]

	colorcat = cat[np.where(cat['bin'] == bin)]


	ras, decs, cmdists = colorcat['RA'], colorcat['DEC'], colorcat['comov_dist']
	refras, refdecs, refcmdists = refcat['RA'], refcat['DEC'], refcat['comov_dist']
	randras, randdecs, randcmdists = randcat['RA'], randcat['DEC'], randcat['comov_dist']



	if useweights:
		#weights = colorcat['WEIGHT_SYSTOT'] * colorcat['WEIGHT_CP'] * colorcat['WEIGHT_NOZ'] * colorcat['WEIGHT_FKP']
		weights = None
		refweights = refcat['WEIGHT_SYSTOT'] * refcat['WEIGHT_CP'] * refcat['WEIGHT_NOZ']
		randweights = randcat['WEIGHT_SYSTOT'] * randcat['WEIGHT_CP'] * randcat['WEIGHT_NOZ']
	else:
		weights, refweights, randweights = None, None, None

	if twoD:
		bins = np.linspace(0.1, 10 ** maxscale, int(10 ** maxscale) + 1)
		pimax = int(10 ** maxscale)
	else:
		bins = np.logspace(minscale, maxscale, nbins + 1)

	avgrbin = []
	for j in range(len(bins) - 1):
		avgrbin.append(np.mean([bins[j], bins[j + 1]]))
	avgrbin = np.array(avgrbin)

	xcorr = twoPointCFs.spatial_cross_corr_from_coords(bins, ras, decs, cmdists, refras, refdecs, refcmdists, randras,
	                                       randdecs, randcmdists, weights=weights, refweights=refweights,
	                                       randweights=randweights, nthreads=nthreads, nbins=nbins, pimax=pimax,
	                                                   twoD=twoD)
	if twoD:
		plotting.plot_2d_corr_func(xcorr, 'QSO_cross_%s' % refsample)
		return

	if nbootstraps > 0:
		realizations = []
		for j in range(nbootstraps):
			bootidxs = np.random.choice(len(ras), len(ras))
			bootras, bootdecs, bootdists = ras[bootidxs], decs[bootidxs], cmdists[bootidxs]

			refbootidx = np.random.choice(len(refras), len(refras))
			bootctrlras, bootctrldecs, bootrefdists = refras[refbootidx], refdecs[refbootidx], refcmdists[refbootidx]

			randbootidx = np.random.choice(len(randras), len(randras))
			bootrandras, bootranddecs, bootranddists = randras[randbootidx], randdecs[randbootidx], \
			                                           randcmdists[randbootidx]

			if useweights:
				bootweights, refbootweights, randbootweights = None, refweights[refbootidx], randweights[randbootidx]
			else:
				bootweights, refbootweights, randbootweights = None, None, None

			boot_xcorr = twoPointCFs.spatial_cross_corr_from_coords(bins, bootras, bootdecs, bootdists, bootctrlras,
														bootctrldecs, bootrefdists, bootrandras, bootranddecs,
														bootranddists, weights=bootweights, refweights=refweights,
														randweights=randweights, nthreads=nthreads, nbins=nbins,
														pimax=pimax)
			realizations.append(boot_xcorr)
		xcorr_std = np.std(realizations, axis=0)

		amp_and_err = np.array([avgrbin, xcorr, xcorr_std])
		amp_and_err.dump('clustering/spatial_cross/%s/%s_%s.npy' % (cap, refsample, bin))
		np.array(realizations).dump('clustering/spatial_cross/%s/%s_%s_reals.npy' % (cap, refsample, bin))
	else:
		amp_and_err = np.array([avgrbin, xcorr, np.zeros(nbins)])
		amp_and_err.dump('clustering/spatial_cross/%s/%s_%s.npy' % (cap, refsample, bin))

