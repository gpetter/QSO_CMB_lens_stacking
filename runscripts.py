import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from astropy.io import fits
import pandas as pd
import convergence_map
import importlib
importlib.reload(convergence_map)
import stacking
importlib.reload(stacking)



"""convergence_map.read_planck_map(False)

# smooth first then mask
convergence_map.smooth_map('planck_map.fits', 15., 'smoothed_planck.fits')

convergence_map.mask_map('smoothed_planck.fits', 'mask.fits', False, 'smoothed_masked_planck.fits')"""



sdss_quasar_cat = (fits.open('DR14Q_v4_4.fits'))[1].data

good_idxs = np.where((sdss_quasar_cat['Z'] <= 2.2) & (sdss_quasar_cat['Z'] >= 0.9) & (sdss_quasar_cat['FIRST_MATCHED']==0) & (sdss_quasar_cat['MI'] <= -24))[0]
ras = sdss_quasar_cat['RA'][good_idxs]
decs = sdss_quasar_cat['DEC'][good_idxs]
perm = np.random.permutation(len(ras))
newras = ras[perm]
newdecs = decs[perm]

stacking.stack_convergence('smoothed_masked_planck.fits', 5., 250., newras, newdecs, ['stacked_1.npy', 'signoise_1.npy'], True)


"""#stacking.stack_convergence('smoothed_masked_planck_map.fits', 2., 100., ras, decs, ['kappa_extra.npy', 'signoise_extra.npy'], True)

a = []
for i in range(30):
    perm = np.random.permutation(len(ras))
    print(perm)
    newras = ras[perm]
    newdecs = decs[perm]
    b = stacking.stack_convergence('smoothed_masked_planck_eq.fits', 2., 100., newras, newdecs, ['kappa_2.npy', 'signoise_2.npy'], False)
    a.append(b[1])
c = np.mean(np.array(a), axis=0)
c.dump('avg_signoise')


print('yo')

convergence_map.mask_map('planck_map.fits', 'mask.fits', False, -1)

convergence_map.smooth_map('masked_planck_map.fits', 15., 'smoothed_masked_planck_map.fits')
                             
convergence_map.change_coord('smoothed_masked_planck_map.fits', ['G', 'C'], 'smoothed_masked_planck_eq.fits')
                             
stacking.stack_convergence('smoothed_masked_planck_eq.fits', 2., 100., ras, decs, ['kappa_3.npy', 'signoise_3.npy'], False)
                             
print('fam')

convergence_map.smooth_map('planck_map.fits', 15., 'smoothed_first.fits')
                             
convergence_map.mask_map('smoothed_first.fits', 'mask.fits', False, -1)
                             

                             
stacking.stack_convergence('smoothed_first_eq.fits', 2., 100., ras, decs, ['kappa_4.npy', 'signoise_4.npy'], False)

convergence_map.mask_map('smoothed_first.fits', 'mask.fits', True, -1)
                             
convergence_map.change_coord('masked_planck_map.fits', ['G', 'C'], 'smoothed_first_mean_eq.fits')
                             
stacking.stack_convergence('smoothed_first_mean_eq.fits', 2., 100., ras, decs, ['kappa_5.npy', 'signoise_5.npy'], False)

print('cuz')
                             
convergence_map.mask_map('planck_map.fits', 'mask.fits', True, -1)
                             
convergence_map.change_coord('masked_planck_map.fits', ['G', 'C'], 'not_smoothed.fits')
                             
stacking.stack_convergence('not_smoothed.fits', 2., 100., ras, decs, ['kappa_6.npy', 'signoise_6.npy'], False)"""
