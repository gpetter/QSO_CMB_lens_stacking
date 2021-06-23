
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf, convert_rp_pi_counts_to_wp
import myCorrfunc
import numpy as np
import importlib
importlib.reload(myCorrfunc)



def angular_corr_from_coords(ras, decs, randras, randdecs, weights=None, randweights=None, nthreads=1, nbins=10):

	bins = np.logspace(-2, 1, (nbins + 1))

	# autocorrelation of catalog
	DD_counts = DDtheta_mocks(1, nthreads, bins, ras, decs, weights1=weights)

	# cross correlation between data and random catalog
	DR_counts = DDtheta_mocks(0, nthreads, bins, ras, decs, RA2=randras, DEC2=randdecs, weights1=weights,
	                          weights2=randweights)

	# autocorrelation of random points
	RR_counts = DDtheta_mocks(1, nthreads, bins, randras, randdecs, weights1=randweights)

	wtheta = convert_3d_counts_to_cf(len(ras), len(ras), len(randras), len(randras), DD_counts, DR_counts,
	                                 DR_counts, RR_counts)

	return wtheta


# angular cross correlation
def ang_cross_corr_from_coords(ras, decs, refras, refdecs, randras, randdecs, minscale, maxscale, weights=None,
                               refweights=None, randweights=None, nthreads=1, nbins=10):
	# set up logarithimically spaced bins in units of degrees
	bins = np.logspace(minscale, maxscale, (nbins + 1))

	# count pairs between sample and control sample
	DD_counts = DDtheta_mocks(0, nthreads, bins, ras, decs, RA2=refras, DEC2=refdecs, weights1=weights,
	                          weights2=refweights)

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


def spatial_corr_from_coords(ras, decs, cz, randras, randdecs, randcz, bins, weights=None, randweights=None,
                             nthreads=1, comoving=True, estimator='LS', pimax=50):

	# to use weights Corrfunc needs keyword weight_type set to 'pair_product'
	if weights is None:
		weighttype = None
	else:
		weighttype = 'pair_product'

	# data data pair counts
	DD_counts = DDrppi_mocks(1, 2, nthreads, pimax, bins, ras, decs, cz, weights, weight_type=weighttype,
	                         is_comoving_dist=comoving)

	# data random pair counts
	DR_counts = DDrppi_mocks(0, 2, nthreads, pimax, bins, ras, decs, cz, RA2=randras, DEC2=randdecs, CZ2=randcz,
	                         weights1=weights, weights2=randweights, weight_type=weighttype, is_comoving_dist=comoving)
	# if using Landy-Szalay estimator
	if estimator == 'LS':
		# random random pair counts
		RR_counts = DDrppi_mocks(1, 2, nthreads, pimax, bins, randras, randdecs, randcz, randweights,
		                         weight_type=weighttype, is_comoving_dist=comoving)

		#cf = myCorrfunc.convert_counts_to_cf(len(ras), len(ras), len(randras), len(randras), DD_counts, DR_counts, DR_counts, RR_counts, estimator='LS')
		#plotting.plot_2d_corr_func(cf)

		wp = myCorrfunc.convert_counts_to_wp(len(ras), len(ras), len(randras), len(randras), DD_counts, DR_counts,
		                                     DR_counts, RR_counts, len(bins)-1, pimax, estimator='LS')

	else:
		wp = myCorrfunc.convert_counts_to_wp(len(ras), 1, len(randras), 1, DD_counts, DR_counts, DR_counts,
		                                     DR_counts, len(bins)-1, pimax)

	return wp


def spatial_cross_corr_from_coords(bins, ras, decs, dists, refras, refdecs, refdists, randras, randdecs, randdists,
                                   weights=None, refweights=None, randweights=None, nthreads=1, nbins=10, pimax=50):

	if refweights is None:
		weighttype = None
	else:
		weighttype = 'pair_product'


	DD_counts = DDrppi_mocks(0, 2, nthreads, pimax, bins, ras, decs, dists, RA2=refras, DEC2=refdecs, CZ2=refdists,
	                         weights1=weights, weights2=refweights, weight_type=weighttype, is_comoving_dist=True)

	DR_counts = DDrppi_mocks(0, 2, nthreads, pimax, bins, ras, decs, dists, RA2=randras, DEC2=randdecs, CZ2=randdists,
	                         weights1=weights, weights2=randweights, weight_type=weighttype, is_comoving_dist=True)

	cf = myCorrfunc.convert_counts_to_wp(len(refras), 1, len(randras), 1, DD_counts, DR_counts, DR_counts, DR_counts,
	                                     nbins, pimax)

	return cf