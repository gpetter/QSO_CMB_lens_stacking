import autocorrelation
import importlib
from multiprocessing import Pool
import numpy as np
from functools import partial
importlib.reload(autocorrelation)

# run cross correlations on multiple samples in parallel
def mpxcorr(qso_cat_name, minscale, maxscale, cap, refsample, bins, nbootstraps, nthreads, use_weights, pimax,
            n_sample_bins):
	# list of bin numbers to cross correlate
	samples = np.arange(n_sample_bins)
	# set up number of threads, my computer doesn't like more than about 10
	if n_sample_bins > 8:
		mpthreads = int(n_sample_bins/int(n_sample_bins/8))
	else:
		mpthreads = n_sample_bins
	# partial of function to cross correlate. returns function which only takes single parameter, the bin number
	partialxcorr = partial(autocorrelation.cross_corr_func_spatial, qso_cat_name=qso_cat_name, minscale=minscale,
				                                        maxscale=maxscale, cap=cap,
				                                        refsample=refsample,
				                                        nbins=bins, nbootstraps=nbootstraps, nthreads=nthreads,
				                                        useweights=use_weights, pimax=pimax)

	# cross correlate various samples in parallel
	with Pool(mpthreads) as p:
		p.map(partialxcorr, samples + 1)


def mpautocorr(n_bootstraps, refsample, samplebin, galcap, n_scale_bins, useweights, minscale, maxscale, pimax_i,
               twodcfs):

	mpthreads = 5

	bootstraps = np.arange(n_bootstraps+1)

	partialautocorr = partial(autocorrelation.spatial_correlation_function, sample_name=refsample, colorbin=samplebin,
	                          cap=galcap, nthreads=1, nbins=n_scale_bins, useweights=useweights, minscale=minscale,
				                maxscale=maxscale, pimax=pimax_i, twoD=twodcfs)

	with Pool(mpthreads) as p:
		outputs = p.map(partialautocorr, bootstraps)
	avgrs = outputs[0][0]
	cf = outputs[0][1]
	realizations = outputs[1:]
	cf_std = np.std(realizations, axis=0)
	np.array([avgrs, cf, cf_std]).dump('clustering/spatial/%s/%s_%s.npy' % (galcap, refsample, samplebin))
	np.array(realizations).dump('clustering/spatial/%s/%s_%s_reals.npy' % (galcap, refsample, samplebin))