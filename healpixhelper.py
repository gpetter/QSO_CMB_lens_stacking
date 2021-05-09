import healpy as hp
import numpy as np
from astropy.table import Table
from astropy.io import fits

# for a given source list with ras and decs, create a healpix map of source density for a given pixel size
def healpix_density_map(ras, decs, nsides):
	# convert coordinates to healpix pixels
	pix_of_sources = hp.ang2pix(nsides, ras, decs, lonlat=True)
	# number of pixels for healpix map with nsides
	npix = hp.nside2npix(nsides)
	# count number of sources in each pixel
	density_map = np.bincount(pix_of_sources, minlength=npix)

	return density_map


# input two survey catalogs and return subset of the first catalog where their footrprints overlap
def match_footprints(cat1, cat2, outname=None, nside=64):
	# coordinates
	cat1coords = cat1['RA'], cat1['DEC']
	cat2coords = cat2['RA'], cat2['DEC']
	# create density maps for each catalog
	densmap1 = healpix_density_map(cat1coords[0], cat1coords[1], nside)
	densmap2 = healpix_density_map(cat2coords[0], cat2coords[1], nside)
	# multiply density maps together, where density is zero in either map, product will be zero
	combinedmap = densmap1 * densmap2
	# indices in catalog 1 where both catalogs have sources
	goodixs1 = np.where(combinedmap[hp.ang2pix(nside, cat1coords[0], cat1coords[1], lonlat=True)] > 0)
	# write the subset of the first catalog which
	Table(cat1[goodixs1]).write('catalogs/derived/%s.fits' % outname, format='fits', overwrite=True)

# take coordinates from 2 surveys with different footprints and return indices of sources within the overlap of both
def match_footprints2(testsample, reference_sample, nside=32):
	ras1, decs1 = testsample[0], testsample[1]
	ras2, decs2 = reference_sample[0], reference_sample[1]

	footprint1_pix = hp.ang2pix(nside=nside, theta=ras1, phi=decs1, lonlat=True)
	footprint2_pix = hp.ang2pix(nside=nside, theta=ras2, phi=decs2, lonlat=True)
	commonpix = np.intersect1d(footprint1_pix, footprint2_pix)
	commonfootprint = np.zeros(hp.nside2npix(nside))
	commonfootprint[commonpix] = 1
	idxs = np.where(commonfootprint[footprint1_pix])
	#idxs2 = np.where(commonfootprint[footprint2_pix])

	#sample1out = (ras1[idxs1], decs1[idxs1])
	#sample2out = (ras2[idxs2], decs2[idxs2])

	#return (sample1out, sample2out)

	return idxs


def change_coord(m, coord):
	""" Change coordinates of a HEALPIX map

	Parameters
	----------
	m : map or array of maps
	  map(s) to be rotated
	coord : sequence of two character
	  First character is the coordinate system of m, second character
	  is the coordinate system of the output map. As in HEALPIX, allowed
	  coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

	Example
	-------
	The following rotate m from galactic to equatorial coordinates.
	Notice that m can contain both temperature and polarization.
	"""
	# Basic HEALPix parameters
	npix = m.shape[-1]
	nside = hp.npix2nside(npix)
	ang = hp.pix2ang(nside, np.arange(npix))

	# Select the coordinate transformation
	rot = hp.Rotator(coord=reversed(coord))

	# Convert the coordinates
	new_ang = rot(*ang)
	new_pix = hp.ang2pix(nside, *new_ang)

	return m[..., new_pix]


def masked_temp_map():
	tempmap = hp.read_map('maps/COM_CMB_IQU-smica_2048_R3.00_full.fits', field=0, hdu=1, partial=False, nest=False)
	tempmask = hp.read_map('maps/COM_CMB_IQU-smica_2048_R3.00_full.fits', field=3, hdu=1, partial=False, nest=False)
	tempmap[np.where(tempmask == 0)] = hp.UNSEEN
	hp.write_map('maps/smica_masked.fits', tempmap, overwrite=True)
