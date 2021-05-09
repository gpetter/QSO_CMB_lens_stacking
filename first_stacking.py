import numpy as np
from astroquery.image_cutouts.first import First
import healpy as hp
from astropy.io import fits
from astropy import units as u
from astropy import coordinates
import autocorrelation
from math import ceil
import multiprocessing as mp
from functools import partial
import glob



def download_chunk(ras, decs, cutoutsize_arcsec, chunksize, nstack, k):

	if (k * chunksize) + chunksize > nstack:
		stepsize = nstack % chunksize
	else:
		stepsize = chunksize
	highidx, lowidx = ((k * chunksize) + stepsize), (k * chunksize)
	newras, newdecs = ras[lowidx:highidx], decs[lowidx:highidx]

	cutoutlist = []
	for j in range(stepsize):
		try:
			cutoutlist.append(np.array(First.get_images(coordinates.SkyCoord(newras[j] * u.deg, newdecs[j] * u.deg, frame='icrs'),
			                         image_size=(cutoutsize_arcsec * u.arcsec))[0].data, dtype=np.single))
		except:
			cutoutlist.append(np.nan)
		if j%100==0:
			print(j)
	np.array(cutoutlist).dump('first_cutouts/xdqso_complete_cutouts_%s.npy' % k)
	print('%s done' % k)
	return 0


def download_first_cutouts_mp(cutoutsize_arcsec=63, usemp=True, combine=True):
	objids = fits.open('catalogs/derived/xdqso_specz_complete.fits')[1].data['OBJID_XDQSO']
	ras = fits.open('catalogs/derived/xdqso_specz_complete.fits')[1].data['RA_XDQSO']
	decs = fits.open('catalogs/derived/xdqso_specz_complete.fits')[1].data['DEC_XDQSO']

	firstras = fits.open('catalogs/radio_cats/first_14dec17.fits')[1].data['RA']
	firstdecs = fits.open('catalogs/radio_cats/first_14dec17.fits')[1].data['DEC']

	firstdensmap = autocorrelation.healpix_density_map(firstras, firstdecs, 64)
	qsopix = hp.ang2pix(64, ras, decs, lonlat=True)
	idxs = np.where(firstdensmap[qsopix] > 0)

	"""idxs = np.where(
		((ras > 105) & (ras < 262.5) & (decs > -8) & (decs < 57.6)) | (((ras > 306) & (decs > -11.5) & (decs < 12.5)) | (
					(ras < 60) & (decs > -11.5) & (decs < 12.5))))"""
	objids, ras, decs = objids[idxs], ras[idxs], decs[idxs]

	"""if usemp:

		chunksize = 10000

		nchunks = ceil(len(ras) / chunksize)

		# initalize multiprocessing pool with one less core than is available for stability
		p = mp.Pool(mp.cpu_count() - 1)

		partialdown = partial(download_chunk, ras, decs, cutoutsize_arcsec, chunksize, len(ras))

		pmap = p.map(partialdown, range(nchunks))
		p.close()
		p.join()

	else:
		cutoutlist = []
		for j in range(len(ras)):
			try:
				cutoutlist.append(
					np.array(First.get_images(coordinates.SkyCoord(ras[j] * u.deg, decs[j] * u.deg, frame='icrs'),
					                          image_size=(cutoutsize_arcsec * u.arcsec))[0].data, dtype=np.single))
			except:
				cutoutlist.append(np.nan)
			print(j)
			if ((j % 10000 == 0) and (j > 0)):
				np.array(cutoutlist).dump('first_cutouts/xdqso_complete_cutouts_%s.npy' % int(j / 10000))
				cutoutlist = []"""


	if combine:
		fulllist, fullidlist = [], []
		chunklist = glob.glob('first_cutouts/xdqso_complete_cutouts_*')
		for j in range(len(chunklist)):
			chunkarr = np.load('first_cutouts/xdqso_complete_cutouts_%s.npy' % j, allow_pickle=True)
			for k in range(len(chunkarr)):
				if not np.isnan(chunkarr[k]).all():
					fulllist.append(chunkarr[k])
					fullidlist.append(objids[(j*10000)+k])


		with open('first_cutouts/all_cutouts.npy', 'wb') as f:
			np.save(f, fulllist)


		np.array(fullidlist).dump('first_cutouts/IDs.npy')





def stack_first(color, sample_name, weights=None, prob_weights=None, outname=None):
	
	
	cat = fits.open('catalogs/derived/%s_%s.fits' % (sample_name, color))[1].data

	if (sample_name == 'xdqso') or (sample_name == 'xdqso_specz'):
		rakey, deckey = 'RA_XDQSO', 'DEC_XDQSO'
		probs = cat['PQSO']
	else:
		rakey, deckey = 'RA', 'DEC'
		probs = np.ones(len(cat))
	ids = cat['OBJID_XDQSO']

	# load in list of cutouts and reference IDs
	cutoutlist = np.load('first_cutouts/all_cutouts.npy', allow_pickle=True)
	refidlist = np.load('first_cutouts/IDs.npy', allow_pickle=True)
	# if no weights provided, weights set to one
	if weights is None:
		weights = np.ones(len(ids))
	if prob_weights is None:
		prob_weights = np.ones(len(ids))


	"""imsize = 35

	running_sum = np.zeros((imsize, imsize))
	wsum = np.zeros((imsize, imsize))
	for j in range(len(ids)):
		# find where the reference ID matches the current source ID
		cutidx = np.where(refidlist == ids[j])[0]
		# if there's a match
		if len(cutidx) == 1:
			#
			new_im = cutoutlist[cutidx][0]

			# if source falls outside FIRST footprint, the cutout will just be a single nan value, dont add to stack
			if np.isnan(new_im).all():
				continue
		else:
			continue


		wmat = np.full((imsize, imsize), weights[j])
		wmat[np.isnan(new_im)] = 0
		new_im[np.isnan(new_im)] = 0

		wmat_for_sum = wmat * prob_weights[j]

		running_sum = np.nansum([running_sum, new_im], axis=0)
		wsum = np.nansum([wsum, wmat_for_sum], axis=0)

	if outname is not None:
		running_sum.dump('%s.npy' % outname)"""

	# return running_sum/wsum
	return np.nanmean(cutoutlist[np.isin(refidlist, ids)], axis=0)

def median_stack(ids):
	"""stacked = np.empty((imsize, imsize))
	for i in range(imsize):
		for j in range(imsize):
			pixlist = []
			for k in range(len(ids)):
				cutidx = np.where(refidlist == ids[k])[0]
				if len(cutidx) == 1:
					new_im = cutoutlist[cutidx][0]

					if np.isnan(new_im).all():
						continue
					pixlist.append(new_im[i, j])
				else:
					continue
			stacked[i, j] = np.nanmedian(pixlist)"""
	cutoutlist = np.load('first_cutouts/all_cutouts.npy')
	refidlist = np.load('first_cutouts/IDs.npy', allow_pickle=True)
	cutouts = cutoutlist[np.isin(refidlist, ids)]
	stacked = np.nanmedian(cutouts, axis=0)
	return stacked

def median_first_stack(color, sample_name, weights=None, prob_weights=None, write=False, nbootstraps=0):
	cat = fits.open('catalogs/derived/%s_%s.fits' % (sample_name, color))[1].data

	if (sample_name == 'xdqso') or (sample_name == 'xdqso_specz'):
		rakey, deckey = 'RA_XDQSO', 'DEC_XDQSO'
		probs = cat['PQSO']
	else:
		rakey, deckey = 'RA', 'DEC'
		probs = np.ones(len(cat))
	ids = cat['OBJID_XDQSO']


	# if no weights provided, weights set to one
	if weights is None:
		weights = np.ones(len(ids))
	if prob_weights is None:
		prob_weights = np.ones(len(ids))

	# White et al. 2007 found snapshot bias, correct for this
	bias_corrected_stack = 1.4 * median_stack(ids)
	stackrealizations = np.zeros(nbootstraps)
	for j in range(nbootstraps):
		bootidxs = np.random.choice(len(ids), len(ids))
		newids = ids[bootidxs]
		stackrealizations[j] = 1.4*median_stack(newids)
	stackerr = np.std(stackrealizations)

	if write:
		bias_corrected_stack.dump('radio_stacks/%s_%s_first_stack.npy' % (sample_name, color))
	return bias_corrected_stack