import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.table import Table
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf, convert_rp_pi_counts_to_wp
import myCorrfunc
import random_catalogs
import importlib
importlib.reload(myCorrfunc)
importlib.reload(random_catalogs)


from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()
import plotting


def write_comoving_distances():
	randcat = fits.open('catalogs/lss/eBOSS_QSO_clustering_random-NGC-vDR16.fits')[1].data
	randzs = randcat['Z']
	rand_comov_dists = cosmo.comovingDistance(np.zeros(len(randzs)), randzs)
	randtab = Table(randcat)
	randtab['comov_dist'] = rand_comov_dists
	randtab.write('catalogs/lss/eBOSS_randoms_NGC_comov.fits', format='fits', overwrite=True)






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

# angular cross correlation
def ang_cross_corr_from_coords(ras, decs, refras, refdecs, randras, randdecs, minscale, maxscale, weights=None, refweights=None, randweights=None, nthreads=1, nbins=10):
	# set up logarithimically spaced bins in units of degrees
	bins = np.logspace(minscale, maxscale, (nbins + 1))

	# count pairs between sample and control sample
	DD_counts = DDtheta_mocks(0, nthreads, bins, ras, decs, RA2=refras, DEC2=refdecs, weights1=weights, weights2=refweights)

	# extract number counts
	dd = []
	for j in range(nbins):
		dd.append(DD_counts[j][3])


	# cross correlation between sample and random catalog
	DR_counts = np.array(DDtheta_mocks(0, nthreads, bins, ras, decs, RA2=randras, DEC2=randdecs, weights1=weights,
							  weights2=randweights))
	dr = []
	for j in range(nbins):
		dr.append(DR_counts[j][3])



	wtheta = np.array(dd)/np.array(dr) * (float(len(randras))) / float(len(refras)) - 1

	return wtheta


def spatial_corr_from_coords(ras, decs, cz, randras, randdecs, randcz, bins, weights=None, randweights=None, nthreads=1, comoving=True, estimator='LS'):

	pimax = 100

	if weights is None:
		weighttype = None
	else:
		weighttype = 'pair_product'

	DD_counts = DDrppi_mocks(1, 2, nthreads, pimax, bins, ras, decs, cz, weights, weight_type=weighttype, is_comoving_dist=comoving)


	DR_counts = DDrppi_mocks(0, 2, nthreads, pimax, bins, ras, decs, cz, RA2=randras, DEC2=randdecs, CZ2=randcz, weights1=weights, weights2=randweights, weight_type=weighttype, is_comoving_dist=comoving)
	if estimator == 'LS':
		RR_counts = DDrppi_mocks(1, 2, nthreads, pimax, bins, randras, randdecs, randcz, randweights, weight_type=weighttype, is_comoving_dist=comoving)

		#cf = myCorrfunc.convert_counts_to_cf(len(ras), len(ras), len(randras), len(randras), DD_counts, DR_counts, DR_counts, RR_counts, estimator='LS')
		#plotting.plot_2d_corr_func(cf)

		wp = myCorrfunc.convert_counts_to_wp(len(ras), len(ras), len(randras), len(randras), DD_counts, DR_counts, DR_counts, RR_counts, len(bins)-1, pimax, estimator='LS')

	else:
		wp = myCorrfunc.convert_counts_to_wp(len(ras), 1, len(randras), 1, DD_counts, DR_counts, DR_counts, DR_counts, len(bins)-1, pimax)

	return wp


def spatial_cross_corr_from_coords(bins, ras, decs, dists, refras, refdecs, refdists, randras, randdecs, randdists, weights=None, refweights=None, randweights=None, nthreads=1, nbins=10):

	if weights is None:
		weighttype = None
	else:
		weighttype = 'pair_product'


	pimax = 40

	DD_counts = DDrppi_mocks(0, 2, nthreads, pimax, bins, ras, decs, dists, RA2=refras, DEC2=refdecs, CZ2=refdists, weights1=weights, weights2=refweights, weight_type=weighttype, is_comoving_dist=True)

	DR_counts = DDrppi_mocks(0, 2, nthreads, pimax, bins, ras, decs, dists, RA2=randras, DEC2=randdecs, CZ2=randdists, weights1=weights, weights2=randweights, weight_type=weighttype, is_comoving_dist=True)

	cf = myCorrfunc.convert_counts_to_wp(len(refras), 1, len(randras), 1, DD_counts, DR_counts, DR_counts, DR_counts, nbins, pimax)

	return cf




def angular_correlation_function(sample_name, colorbin, nbins=10, nbootstraps=0, nthreads=1, useweights=False):


	cat = fits.open('catalogs/derived/%s_colored.fits' % (sample_name))[1].data

	colorcat = cat[np.where(cat['colorbin'] == colorbin)]
	ras, decs = colorcat['RA'], colorcat['DEC']




	"""randcat = fits.open('catalogs/lss/eBOSS_QSO_clustering_random-NGC-vDR16.fits')[1].data
	# subsample huge random catalog

	bidxs = np.random.choice(len(randcat), 5*len(cat), replace=False)
	randcat = randcat[bidxs]
	randras, randdecs = randcat['RA'], randcat['DEC']"""


	randras, randdecs = gen_rand_points_from_healpix(ras, decs, 10 * len(ras))

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
			bootidxs = np.random.choice(len(ras), len(ras))
			bootras = ras[bootidxs]
			bootdecs = decs[bootidxs]

			randbootidx = np.random.choice(len(randras), len(randras))
			bootrandras = randras[randbootidx]
			bootranddecs = randdecs[randbootidx]


			boot_wtheta = angular_corr_from_coords(bootras, bootdecs, bootrandras, bootranddecs, nthreads=nthreads, nbins=nbins)
			wtheta_realizations.append(boot_wtheta)
		wtheta_std = np.std(wtheta_realizations, axis=0)

		amp_and_err = np.array([wtheta, wtheta_std])
		amp_and_err.dump('angclustering/%s_%s.npy' % (sample_name, colorbin))
	else:
		amp_and_err = np.array([wtheta, np.zeros(nbins)])
		amp_and_err.dump('angclustering/%s_%s.npy' % (sample_name, colorbin))

	plotting.plot_ang_correlation_function(sample_name, nbins)


def spatial_correlation_function(sample_name, colorbin, cap, nbins=10, nbootstraps=0, nthreads=1, useweights=False, minscale=0, maxscale=1.5):

	if colorbin == 'all':
		if cap == 'both':
			colorcat = fits.open('catalogs/lss/eBOSS_fullsky_comov.fits')[1].data
		else:
			colorcat = fits.open('catalogs/lss/eBOSS_%s_comov.fits' % cap)[1].data

	else:
		cat = fits.open('catalogs/derived/%s_colored.fits' % sample_name)[1].data
		colorcat = cat[np.where(cat['colorbin'] == colorbin)]


	ras, decs, cmdists = colorcat['RA'], colorcat['DEC'], colorcat['comov_dist']

	"""randras, randdecs = gen_rand_points_from_healpix(ras, decs, 10 * len(ras))

	z_hist = np.histogram(zs, bins=20, density=True)
	histvals, histbins = z_hist[0], z_hist[1]

	minz, maxz = np.min(zs), np.max(zs)

	uniform_zs = np.random.uniform(minz, maxz, len(randras))

	bins_of_randzs = np.digitize(uniform_zs, histbins) - 1
	weightsforsampling = histvals[bins_of_randzs]
	weightsforsampling = weightsforsampling/np.sum(weightsforsampling)


	randzs = np.random.choice(uniform_zs, len(randras), p=weightsforsampling)
	"""

	if cap == 'both':
		randcat = random_catalogs.fullsky_eboss_randoms(5)
	else:
		randcat = fits.open('catalogs/lss/eBOSS_randoms_%s_comov.fits' % cap)[1].data
		randcat = randcat[:5*len(colorcat)]

	randras, randdecs, randcmdists = randcat['RA'], randcat['DEC'], randcat['comov_dist']

	bins = np.logspace(minscale, maxscale, nbins + 1)
	#bins = np.linspace(0.1, 200, 201)


	if useweights:
		totweights = colorcat['WEIGHT_SYSTOT'] * colorcat['WEIGHT_CP'] * colorcat['WEIGHT_NOZ'] * colorcat['WEIGHT_FKP']
		randweights = randcat['WEIGHT_SYSTOT'] * randcat['WEIGHT_CP'] * randcat['WEIGHT_NOZ'] * randcat['WEIGHT_FKP']
	else:
		totweights = None
		randweights = None

	avgrbin = []
	for j in range(len(bins)-1):
		avgrbin.append(np.mean([bins[j], bins[j + 1]]))
	avgrbin = np.array(avgrbin)

	wr = spatial_corr_from_coords(ras, decs, cmdists, randras, randdecs, randcmdists, bins, weights=totweights, randweights=randweights, nthreads=nthreads)/avgrbin


	if nbootstraps > 0:
		wr_realizations = []
		for j in range(nbootstraps):
			bootidxs = np.random.choice(len(ras), len(ras))
			bootras = ras[bootidxs]
			bootdecs = decs[bootidxs]
			bootdists = cmdists[bootidxs]



			randbootidx = np.random.choice(len(randras), len(randras))
			bootrandras = randras[randbootidx]
			bootranddecs = randdecs[randbootidx]
			bootranddist = randcmdists[randbootidx]

			if useweights:
				bootweights = totweights[bootidxs]
				bootrandweights = randweights[randbootidx]
			else:
				bootweights, bootrandweights = None, None

			boot_wr = spatial_corr_from_coords(bootras, bootdecs, bootdists, bootrandras, bootranddecs, bootranddist, bins, weights=bootweights, randweights=bootrandweights, nthreads=nthreads)/avgrbin
			wr_realizations.append(boot_wr)
		wtheta_std = np.std(wr_realizations, axis=0)

		amp_and_err = np.array([wr, wtheta_std])
		amp_and_err.dump('clustering/spatial/%s_%s.npy' % (sample_name, colorbin))
	else:
		amp_and_err = np.array([wr, np.zeros(nbins)])
		amp_and_err.dump('clustering/spatial/%s_%s.npy' % (sample_name, colorbin))



def cross_correlation_function_angular(sample_name, colorbin, nbins=10, nbootstraps=0, nthreads=1, useweights=False, minscale=-1, maxscale=0):
	cat = fits.open('catalogs/derived/%s_colored.fits' % sample_name)[1].data
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


	crosscorr = ang_cross_corr_from_coords(colorras, colordecs, ctrlras, ctrldecs, randras, randdecs, minscale, maxscale, nthreads=nthreads, nbins=nbins)

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

			boot_wtheta = ang_cross_corr_from_coords(bootras, bootdecs, bootctrlras, bootctrldecs, bootrandras, bootranddecs, minscale, maxscale, nthreads=nthreads, nbins=nbins)
			wtheta_realizations.append(boot_wtheta)
		wtheta_std = np.std(wtheta_realizations, axis=0)

		amp_and_err = np.array([crosscorr, wtheta_std])
		amp_and_err.dump('clustering/ang_cross/%s_%s.npy' % (sample_name, colorbin))
	else:
		amp_and_err = np.array([crosscorr, np.zeros(nbins)])
		amp_and_err.dump('clustering/ang_cross/%s_%s.npy' % (sample_name, colorbin))


def cross_corr_func_spatial(qso_cat_name, colorbin, minscale, maxscale, cap, nbins=10, nbootstraps=0, nthreads=1, useweights=False):
	cat = fits.open('catalogs/derived/%s_colored.fits' % qso_cat_name)[1].data

	#else:
	#	colorcat = cat[np.where(cat['colorbin'] == colorbin)]


	#colorcat = cat
	if cap == 'both':
		refcat = fits.open('catalogs/lss/eBOSS_fullsky_comov.fits')[1].data
		#refcat = cat[np.where((cat['colorbin'] > 1) & (cat['colorbin'] < 10))]
		randcat = random_catalogs.fullsky_eboss_randoms(10)
	else:
		if cap == 'NGC':
			cat = cat[np.where((cat['RA'] > 90) & (cat['RA'] < 270))]
		else:
			cat = cat[np.where((cat['RA'] < 90) | (cat['RA'] > 270))]
		refcat = fits.open('catalogs/lss/eBOSS_%s_comov.fits' % cap)[1].data
		randcat = fits.open('catalogs/lss/eBOSS_randoms_%s_comov.fits' % cap)[1].data
		randcat = randcat[:5 * len(refcat)]

	colorcat = cat[np.where(cat['colorbin'] == colorbin)]
	#if colorbin == 5:
	#	colorcat = cat[np.where((cat['colorbin'] > 1) & (cat['colorbin'] < 10))]


	ras, decs, cmdists = colorcat['RA'], colorcat['DEC'], colorcat['comov_dist']
	refras, refdecs, refcmdists = refcat['RA'], refcat['DEC'], refcat['comov_dist']
	randras, randdecs, randcmdists = randcat['RA'], randcat['DEC'], randcat['comov_dist']



	if useweights:
		weights = colorcat['WEIGHT_SYSTOT'] * colorcat['WEIGHT_CP'] * colorcat['WEIGHT_NOZ'] * colorcat['WEIGHT_FKP']
		refweights = refcat['WEIGHT_SYSTOT'] * refcat['WEIGHT_CP'] * refcat['WEIGHT_NOZ'] * refcat['WEIGHT_FKP']
		randweights = randcat['WEIGHT_SYSTOT'] * randcat['WEIGHT_CP'] * randcat['WEIGHT_NOZ'] * randcat['WEIGHT_FKP']
	else:
		weights, refweights, randweights = None, None, None

	bins = np.logspace(minscale, maxscale, nbins + 1)

	avgrbin = []
	for j in range(len(bins) - 1):
		avgrbin.append(np.mean([bins[j], bins[j + 1]]))
	avgrbin = np.array(avgrbin)

	xcorr = spatial_cross_corr_from_coords(bins, ras, decs, cmdists, refras, refdecs, refcmdists, randras,
	                                       randdecs, randcmdists, weights=weights, refweights=refweights,
	                                       randweights=randweights, nthreads=nthreads, nbins=nbins)

	if nbootstraps > 0:
		realizations = []
		for j in range(nbootstraps):
			bootidxs = np.random.choice(len(ras), len(ras))
			bootras, bootdecs, bootdists = ras[bootidxs], decs[bootidxs], cmdists[bootidxs]

			refbootidx = np.random.choice(len(refras), len(refras))
			bootctrlras, bootctrldecs, bootrefdists = refras[refbootidx], refdecs[refbootidx], refcmdists[refbootidx]

			randbootidx = np.random.choice(len(randras), len(randras))
			bootrandras, bootranddecs, bootranddists = randras[randbootidx], randdecs[randbootidx], randcmdists[randbootidx]

			if useweights:
				bootweights, refbootweights, randbootweights = weights[bootidxs], refweights[refbootidx], randweights[randbootidx]
			else:
				bootweights, refbootweights, randbootweights = None, None, None

			boot_xcorr = spatial_cross_corr_from_coords(bins, bootras, bootdecs, bootdists, bootctrlras,
														bootctrldecs, bootrefdists, bootrandras, bootranddecs,
														bootranddists, weights=bootweights, refweights=refweights,
														randweights=randweights, nthreads=nthreads, nbins=nbins)
			realizations.append(boot_xcorr)
		xcorr_std = np.std(realizations, axis=0)

		amp_and_err = np.array([avgrbin, xcorr, xcorr_std])
		amp_and_err.dump('clustering/spatial_cross/%s.npy' % (colorbin))
	else:
		amp_and_err = np.array([avgrbin, xcorr, np.zeros(nbins)])
		amp_and_err.dump('clustering/spatial_cross/%s.npy' % (colorbin))



def theory_projected_corr_func(rps):
	rs = np.linspace(1, 300, 1000)
	wp = []
	rstep = rs[1] - rs[0]
	for i in range(len(rps)):
		r_range = rs[np.where(rs >= rps[i])]

		integralsum = 0
		for j in range(len(r_range)):
			# radius at midpoint of r bin
			rinbin = r_range[j] + rstep/2
			# correlation function value at midpoint
			xi_in_bin = cosmo.correlationFunction(R=rinbin, z=1.5)
			# amplitude of integrand at midpoint
			amp = 2 * (xi_in_bin * rinbin)/np.sqrt(rinbin**2 - rps[i]**2)
			area = amp * rstep
			integralsum += area
		wp.append(integralsum)

	return np.array(wp)/rps

