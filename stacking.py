import numpy as np
import healpy as hp
import convergence_map
import importlib
importlib.reload(convergence_map)
from astropy import units as u
from astropy.coordinates import SkyCoord
import gc



def equatorial_to_galactic(ra, dec):
    ra_decs = SkyCoord(ra, dec, unit='deg')
    ls = np.array(ra_decs.galactic.l.radian*u.rad.to('deg'))
    bs = np.array(ra_decs.galactic.b.radian*u.rad.to('deg'))
    return(ls, bs)




def stack_convergence(mapname, boxdim, boxpix, ralist, declist, outname, galactic):
    # read in map
    conv_map = hp.read_map(mapname)
    # set masked pixels to nan so they are ignored in stacking
    conv_map = convergence_map.set_unseen_to_nan(conv_map)
    # set resolution of pixels to stack
    # boxdim is in degrees, while resolution is expected in arcmins, multiply by 60
    # divide field width in arcmins by the number of pixels desired
    reso = boxdim*60./boxpix
    
    # transform from (ra,dec) of quasars to (l,b) if planck map is left in galactic coords
    if galactic:
        lon, lat = equatorial_to_galactic(ralist, declist)
    else:
        lon, lat = ralist, declist
    
    # center index is half number of pixels - 1
    middle_idx = int(boxpix/2 - 1)
    # make first gnomonic projection
    avg_kappa = hp.gnomview(conv_map, reso=reso, xsize=boxpix, rot=[lon[0], lat[0]], no_plot=True, return_projected_map=True)
    signoise = [avg_kappa[middle_idx][middle_idx]/np.nanstd(avg_kappa)]

    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)


    
    for i in range(1, 150000):
        new_kappa = hp.gnomview(conv_map, reso=reso, xsize=boxpix, rot=[lon[i], lat[i]], no_plot=True, return_projected_map=True)
        lastavg = i/(i+1)*avg_kappa
        newterm = new_kappa/(i+1)
        stacked = np.dstack((lastavg, newterm))
        avg_kappa[:] = np.nansum(stacked,2)
        signoise.append(avg_kappa[middle_idx][middle_idx]/np.nanstd(avg_kappa))
        
        gc.collect()
        
        if i%500 == 0:
            tr = tracker.SummaryTracker()
            tr.print_diff()
        
        
        if i%10000 == 0:
            print(i)
    
    avg_kappa.dump(outname[0])

    with open(outname[1], 'wb') as f:
        np.save(f, np.array(signoise))
    return(avg_kappa, signoise)
