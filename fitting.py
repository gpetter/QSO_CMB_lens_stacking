import numpy as np
from astropy.io import fits
from scipy.stats import iqr
import matplotlib.pyplot as plt
# change to Planck 18!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
from astropy import constants as const
import astropy.units as u
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import aperture_photometry
from colossus.cosmology import cosmology
from colossus.halo import concentration
from colossus.halo import profile_nfw
from colossus.lss import peaks
from colossus.lss import bias
from scipy.special import j0
from scipy.ndimage import gaussian_filter

colcosmo = cosmology.setCosmology('planck18')

sdss_quasar_cat = (fits.open('DR14Q_v4_4.fits'))[1].data

good_idxs = np.where((sdss_quasar_cat['Z'] <= 2.2) & (sdss_quasar_cat['Z'] >= 0.9) & (sdss_quasar_cat['FIRST_MATCHED']==0) & (sdss_quasar_cat['MI'] <= -24))[0]
zs = sdss_quasar_cat['Z'][good_idxs]


# use Freedman–Diaconis rule to calculate number of bins to make histogram of data
def Freedman_Diaconis(data):
    width = 2.*iqr(data)*(len(data))**(-1./3.)
    return int(np.ptp(data)/width)


hist = np.histogram(zs, 100, density=True)


def Sigma_crit(z):
    return (((const.c)**2)/(4.*np.pi*const.G)*(cosmo.angular_diameter_distance(1100.)/((cosmo.angular_diameter_distance(z)*cosmo.angular_diameter_distance_z1z2(z, 1100.))))).decompose().to(u.solMass*u.littleh/(u.kpc)**2, u.with_H0(cosmo.H0))

def mass_to_solmass_per_h(M):
    return M.to(u.solMass/u.littleh, u.with_H0(cosmo.H0))

def calc_concentration(M_200, z):
    M_200_per_h = mass_to_solmass_per_h(M_200)
    c_200 = concentration.modelLudlow16(M_200_per_h.value, z)
    if c_200[1]:
        return c_200[0]
    else:
        return(np.nan)

def NFW_sigma(theta, M_200, z):
    m = mass_to_solmass_per_h(M_200)
    p_nfw = profile_nfw.NFWProfile(M=m.value, c=calc_concentration(M_200, z), z=z, mdef='200c')
    d_a = ((colcosmo.angularDiameterDistance(z)*u.Mpc/u.littleh).to(u.kpc/u.littleh)).value
    R = theta*d_a
    sigma_R = p_nfw.surfaceDensity(r=R)
    return sigma_R

def NFW_1_halo(theta, M_200, z):
    return NFW_sigma(theta, M_200, z)/Sigma_crit(z).value



theta_list = (np.linspace(1., 180., 180)*u.arcmin).to('radian').value


def two_halo_term(theta, M_200, z):
    m = mass_to_solmass_per_h(M_200) #solMass/h
    d_a = 1000.*u.kpc/u.littleh*colcosmo.angularDiameterDistance(z) #kpc/h

    bh = bias.haloBias(M=m.value, z=z, mdef='200c', model='tinker10')
    rho_avg = colcosmo.rho_c(z)*u.solMass*(u.littleh**2)/(u.kpc**3)
    a = (rho_avg/(((1.+z)**3)*Sigma_crit(z)*(d_a)**2))*bh/(2*np.pi)
    
    ks = np.logspace(-5, 1, 1000)*u.littleh/u.Mpc
    ls = ks*(1+z)*(colcosmo.angularDiameterDistance(z)*u.Mpc/u.littleh)
    

    
    ltheta = np.outer(theta, ls)

    
    mps = (colcosmo.matterPowerSpectrum(ks.value, z=z)*(u.Mpc/u.littleh)**3).to((u.kpc/u.littleh)**3)
    integrand = a*ls*j0(ltheta)*mps
    integral = np.trapz(integrand, x=ls)
    return(integral)

def int_kappa(theta, M_200, terms):
    
    avg_kappa = []
    zs = hist[1]
    zs = np.resize(zs, zs.size-1)

    dz = zs[1] - zs[0]
    dndz = hist[0]

    for i in range(len(dndz)):
        z = zs[i] + dz/2
        if terms == 'one':
            avg_kappa.append(NFW_1_halo(theta, M_200, z)*dndz[i])
        elif terms == 'two':
            avg_kappa.append(two_halo_term(theta, M_200, z)*dndz[i])
        elif terms == 'both':
            avg_kappa.append((NFW_1_halo(theta, M_200, z) + two_halo_term(theta, M_200, z))*dndz[i])
        else:
            return(False)
    avg_kappa = np.array(avg_kappa)

    return np.trapz(avg_kappa, dx=dz, axis=0)




one_halo_model = int_kappa(theta_list, 6E12*u.solMass, 'one')
one_halo_model.dump('one_term.npy')

"""two_model = int_kappa(theta_list, 6E12*u.solMass, 'two')
two_model.dump('two_term.npy')

bothmodel = int_kappa(theta_list, 6E12*u.solMass, 'both')
bothsmoothed = gaussian_filter(bothmodel, sigma=15.)
bothsmoothed.dump('both_terms.npy')"""



def measure_profile(image, step):
    profile = []
    center = int(len(image)/2. - 1)
    reso = 1.2  # arcmin/pixel
    steppix = step/reso
    inner_aper = CircularAperture([center, center], steppix)
    profile.append(float(aperture_photometry(image, inner_aper)['aperture_sum']/inner_aper.area))
    i = 1
    while step*(i+1) < 180.:
        new_aper = CircularAnnulus([center, center], steppix*(i), steppix*(i+1))
        profile.append(float(aperture_photometry(image, new_aper)['aperture_sum']/new_aper.area))
        i += 1
    return(profile)

kap = np.load('stacked_1.npy', allow_pickle=True)

meas = np.array(measure_profile(kap, 10.))

meas.dump('measured.npy')
