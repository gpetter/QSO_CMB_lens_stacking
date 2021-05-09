import numpy as np
# edited code from Corrfunc code to allow Peebles estimator of correlation function


def convert_counts_to_cf(ND1, ND2, NR1, NR2,
							D1D2, D1R2, D2R1, R1R2,
							estimator='Peebles'):

	pair_counts = dict()

	fields = ['D1D2', 'D1R2', 'D2R1', 'R1R2']
	arrays = [D1D2, D1R2, D2R1, R1R2]
	for (field, array) in zip(fields, arrays):
		try:

			if np.max(array['weightavg']) > 0:
				npairs = array['npairs'] * array['weightavg']

			else:
				npairs = array['npairs']
			pair_counts[field] = npairs

		except IndexError:
			pair_counts[field] = array

	nbins = len(pair_counts['D1D2'])
	if (nbins != len(pair_counts['D1R2'])) or \
	   (nbins != len(pair_counts['D2R1'])) or \
	   (nbins != len(pair_counts['R1R2'])):
		msg = 'Pair counts must have the same number of elements (same bins)'
		raise ValueError(msg)

	nonzero = pair_counts['R1R2'] > 0
	if 'LS' in estimator or 'Landy' in estimator:
		fN1 = np.float(NR1) / np.float(ND1)
		fN2 = np.float(NR2) / np.float(ND2)
		cf = np.zeros(nbins)
		cf[:] = np.nan
		cf[nonzero] = (fN1 * fN2 * pair_counts['D1D2'][nonzero] -
					   fN1 * pair_counts['D1R2'][nonzero] -
					   fN2 * pair_counts['D2R1'][nonzero] +
					   pair_counts['R1R2'][nonzero]) / pair_counts['R1R2'][nonzero]
		if len(cf) != nbins:
			msg = 'Bug in code. Calculated correlation function does not '\
				  'have the same number of bins as input arrays. Input bins '\
				  '={0} bins in (wrong) calculated correlation = {1}'.format(
					  nbins, len(cf))
			raise RuntimeError(msg)
	elif estimator == 'Peebles':
		fN1 = np.float(NR1) / np.float(ND1)

		cf = np.zeros(nbins)
		cf[:] = np.nan
		cf[nonzero] = (fN1 * pair_counts['D1D2'][nonzero]) / pair_counts['D1R2'][nonzero] - 1
	else:
		return

	return cf


def convert_counts_to_wp(ND1, ND2, NR1, NR2,
							   D1D2, D1R2, D2R1, R1R2,
							   nrpbins, pimax, dpi=1.0,
							   estimator='Peebles'):


	import numpy as np
	if dpi <= 0.0:
		msg = 'Binsize along the line of sight (dpi) = {0}'\
			  'must be positive'.format(dpi)
		raise ValueError(msg)

	xirppi = convert_counts_to_cf(ND1, ND2, NR1, NR2,
									 D1D2, D1R2, D2R1, R1R2,
									 estimator=estimator)
	wp = np.empty(nrpbins)
	npibins = len(xirppi) // nrpbins
	if ((npibins * nrpbins) != len(xirppi)):
		msg = 'Number of pi bins could not be calculated correctly.'\
			  'Expected to find that the total number of bins = {0} '\
			  'would be the product of the number of pi bins = {1} '\
			  'and the number of rp bins = {2}'.format(len(xirppi),
													   npibins,
													   nrpbins)
		raise ValueError(msg)

	# Check that dpi/pimax/npibins are consistent
	# Preventing issue #96 (https://github.com/manodeep/Corrfunc/issues/96)
	# where npibins would be calculated incorrectly, and the summation would
	# be wrong.
	if (dpi*npibins != pimax):
		msg = 'Pimax = {0} should be equal to the product of '\
			  'npibins = {1} and dpi = {2}. Check your binning scheme.'\
			  .format(pimax, npibins, dpi)
		raise ValueError(msg)

	for i in range(nrpbins):
		wp[i] = 2.0 * dpi * np.sum(xirppi[i * npibins:(i + 1) * npibins])

	return wp
