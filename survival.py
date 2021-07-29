import lifelines

def km_median(values, censored, censorship='upper'):
	kmf = lifelines.KaplanMeierFitter()
	if censorship == 'upper':
		kmf.fit_left_censoring(values, censored)
		return kmf.median_survival_time_
	elif censorship == 'lower':
		kmf.fit(values, censored)
		return kmf.median_survival_time_
	else:
		print('error')
		return
