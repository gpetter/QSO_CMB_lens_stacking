import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import pandas as pd
from scipy.interpolate import interpn
from astropy.io import fits
import healpy as hp
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.cm as cm
#import fitting
import importlib

#importlib.reload(fitting)


#plt.style.use('default')
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['figure.dpi'] = 400
mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'figure.autolayout': True})

def show_planck_map():
    plt.figure(493, (10, 5))
    hp.mollview(hp.read_map('maps/smoothed_masked_planck.fits'))
    plt.savefig('plots/k_map.png')
    plt.close('all')
#show_planck_map()

# stolen from https://stackoverflow.com/a/53865762/3015186
def density_scatter(x, y, cobar=False, ax=None, fig=None, sort=True, bins=20, cutcolorbar=False, vmaxin=None, vminin=None, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x,y]).T, method="splinef2d", bounds_error=False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    if cutcolorbar:
        if vmaxin is None:
            ax.scatter(x, y, c=z, vmin=-2*np.max(z), vmax=np.max(z), **kwargs)
        else:
            ax.scatter(x, y, c=z, vmin=-2*vmaxin, vmax=1.5*vmaxin, **kwargs)
    else:
        ax.scatter(x, y, c=z, **kwargs)

    if cobar:
        norm = Normalize(vmin=np.min(z), vmax=np.max(z))
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
        cbar.ax.set_ylabel('Density')

    return ax


def wiener_filter():
    noise_table = pd.read_csv('maps/nlkk.dat', delim_whitespace=True, header=None)
    cl_plus_nl = np.array(noise_table[2])
    nl = np.array(noise_table[1])
    cl = cl_plus_nl - nl

    wien_factor = cl / cl_plus_nl
    print(wien_factor[:35])
    plt.figure(0, (10, 10))


    plt.plot(np.arange(4097), wien_factor, c='k')
    plt.savefig('plots/wiener_filter.png')
    plt.close('all')

"""def plot_profiles():
    plt.figure(43, (10, 8))
    oneterm = int_kappa(np.arange(1, 180, 1) * u.arcmin.to('rad'), 6E12, 'one', [0.5, 1.0, 0.75])
    twoterm = int_kappa(np.arange(1, 180, 1) * u.arcmin.to('rad'), 6E12, 'two', [0.5, 1.0, 0.75])
    bothterm = filter_model([0.5, 1.0, 0.75], np.arange(1, 180, 1) * u.arcmin, 6E12)
    plt.plot(np.arange(1, 180, 1), oneterm, label='1-halo term', c='k', ls=':')
    plt.plot(np.arange(1, 180, 1), twoterm, label='2-halo term', c='k', ls='--')
    plt.plot(np.arange(1, 180, 1), bothterm, label='Filtered total', c='k', ls='-')
    plt.legend(fontsize=15)
    plt.savefig('plots/models.png')
    plt.close('all')"""

def plot_stacks(samplename):
    bluestack = np.load('stacks/%s_blue_stack.npy' % samplename, allow_pickle=True)*1000
    ctrlstack = np.load('stacks/%s_ctrl_stack.npy' % samplename, allow_pickle=True)*1000
    redstack = np.load('stacks/%s_red_stack.npy' % samplename, allow_pickle=True)*1000
    minval = np.min(np.array([bluestack.flatten(), redstack.flatten(), ctrlstack.flatten()]))
    maxval = np.max(np.array([bluestack.flatten(), redstack.flatten(), ctrlstack.flatten()]))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), dpi=300)
    ax1.imshow(bluestack, vmin=minval, vmax=maxval, cmap='inferno', extent=[-180., 180, -180., 180])
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel(r'$\theta$ (arcmin)', fontsize=15)
    ax1.text(-50, 200, 'Blue', c='mediumblue', fontsize=25)
    im = ax2.imshow(ctrlstack, vmin=minval, vmax=maxval, cmap='inferno', extent=[-180., 180, -180., 180])
    ax2.text(-80, 200, 'Control', c='darkgreen', fontsize=25)
    ax2.axis('off')
    ax3.imshow(redstack, vmin=minval, vmax=maxval, cmap='inferno', extent=[-180., 180, -180., 180])
    ax3.axis('off')
    ax3.text(-50, 200, 'Red', c='firebrick', fontsize=25)
    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('$10^{3} \ \kappa$', fontsize=20)
    #fig.colorbar(im)
    plt.subplots_adjust(wspace=0.05)
    plt.savefig('plots/%s_stacks.pdf' % samplename, bbox_inches='tight')
    plt.close('all')
#plot_stacks('xdqso_specz')

def plot_masses(samplename):
    bluecolor = np.median(fits.open('QSO_cats/%s_blue.fits' % samplename)[1].data['g-i'])
    #ctrlcolor = np.median(fits.open('QSO_cats/%s_ctrl.fits' % samplename)[1].data['g-i'])
    redcolor = np.median(fits.open('QSO_cats/%s_red.fits' % samplename)[1].data['g-i'])

    bluestd = np.std(fits.open('QSO_cats/%s_blue.fits' % samplename)[1].data['g-i'])
    ctrlstd = np.std(fits.open('QSO_cats/%s_ctrl.fits' % samplename)[1].data['g-i'])
    redstd = np.std(fits.open('QSO_cats/%s_red.fits' % samplename)[1].data['g-i'])

    bluemass = np.load('masses/%s_blue_mass.npy' % samplename, allow_pickle=True)
    #ctrlmass = np.load('masses/%s_ctrl_mass.npy' % samplename, allow_pickle=True)
    redmass = np.load('masses/%s_red_mass.npy' % samplename, allow_pickle=True)

    plt.close('all')
    plt.figure(4, (10, 8), dpi=300)
    plt.xlabel(r'$\langle g - i \rangle$', fontsize=20)
    plt.ylabel('log$_{10}(M/h^{-1} M_{\odot}$)', fontsize=20)
    #plt.errorbar(ctrlcolor, ctrlmass[0], yerr=[[ctrlmass[2]], [ctrlmass[1]]], ecolor='k')
    plt.errorbar(redcolor, redmass[0], yerr=[[redmass[2]], [redmass[1]]], ecolor='k')
    plt.errorbar(bluecolor, bluemass[0], yerr=[[bluemass[2]], [bluemass[1]]], ecolor='k')
    #plt.errorbar([bluecolor, ctrlcolor, redcolor], [bluemass[0], ctrlmass[0], redmass[0]], yerr=[[bluemass[1], bluemass[2]], [ctrlmass[1], ctrlmass[2]], [redmass[1], redmass[2]]], xerr=[bluestd, ctrlstd, redstd], ls='none', ecolor='k')

    plt.scatter(bluecolor, bluemass[0], c='b', s=100)
    #plt.scatter(ctrlcolor, ctrlmass[0], c='g', s=100)
    plt.scatter(redcolor, redmass[0], c='r', s=100)

    plt.savefig('plots/%s_color_mass.pdf' % samplename)
    plt.close('all')

#plot_masses('xdqso_specz')


def MI_vs_z(zlist, MIlist, nsources, magcut, minz, maxz, lumcut, qso_cat_name, limMs):

    plt.figure(35235, (10, 10))
    ax = plt.gca()
    # plt.scatter(qso_cat[zkey], i_abs_mags, c='k', s=0.01, alpha=0.5)

    density_scatter(zlist, MIlist, bins=[30, 30], ax=ax, s=0.05, cmap='magma', rasterized=True)
    zs = np.linspace(minz, maxz, 20)

    ax.text(0.9, -29, 'N = %s' % nsources, fontsize=20)
    if magcut == 100:
        rect = patches.Rectangle((minz, -30), (maxz - minz), (30 + lumcut), linewidth=1, edgecolor='y', facecolor='y',
                                 alpha=0.3)
        ax.add_patch(rect)
    else:
        plt.fill_between(zs, -30, limMs, color='y', alpha=0.3)
    plt.xlabel('$z$', fontsize=30)
    plt.ylabel('$M_{i} (z=2)$', fontsize=30)
    plt.ylim(-21, -30)
    plt.savefig('plots/%s_Mi_z.pdf' % qso_cat_name)
    plt.close('all')
    plt.close('all')

def w1_minus_w2_plot(w1mag, w2mag, qso_cat_name):
    plt.close('all')
    plt.figure(2534, (10, 10))
    plt.hist((w1mag - w2mag), bins=100)
    plt.xlim(-3, 3)
    plt.axvline(0.8, c='k', ls='--', label='Stern+12')
    plt.xlabel('W1 - W2', fontsize=20)
    plt.legend()
    plt.savefig('plots/%s_W1-W2.pdf' % qso_cat_name)
    plt.close('all')

def g_minus_i_plot(qso_cat_name, bluezs, bluegs, czs, cgs, redzs, redgs, allzs, allgs):
    plt.close('all')

    fig = plt.figure(12351, (10, 10))
    ax = plt.gca()

    bins = 100
    ps = 0.03
    data, x_e, y_e = np.histogram2d(allzs, allgs, bins=[bins, bins], density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([allzs, allgs]).T, method="splinef2d",
                bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    density_scatter(allzs, allgs, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Greys', rasterized=True,
                    cutcolorbar=True, alpha=0.3, vminin=np.min(z), vmaxin=np.max(z))
    density_scatter(czs, cgs, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Greens', rasterized=True,
                    cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))
    density_scatter(redzs, redgs, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Reds', rasterized=True,
                    cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))
    density_scatter(bluezs, bluegs, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Blues', rasterized=True,
                    cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))


    plt.xlabel('$z$', fontsize=30)
    plt.ylabel('$g - i$', fontsize=30)
    #plt.scatter(zs, gminusi, c='k', s=0.01, alpha=0.5)
    #plt.scatter(bluezs, bluegs, c='b', s=0.05, rasterized=True)
    #plt.scatter(czs, cgs, c='g', s=0.05, rasterized=True)
    #plt.scatter(redzs, redgs, c='r', s=0.05, rasterized=True)
    plt.ylim(-2, 6)
    plt.savefig('plots/%s_gminusi.pdf' % qso_cat_name)
    plt.close('all')

def lum_dists(qso_cat_name, lumhistbins, blueLs, cLs, redLs):
    plt.close('all')
    plt.figure(4, (10, 10))
    plt.hist(blueLs, color='b', alpha=0.8, density=True, bins=lumhistbins, range=(21, 26), histtype='step', linewidth=2)
    plt.hist(cLs, color='g', alpha=0.8, density=True, bins=lumhistbins, range=(21, 26), histtype='step', linewidth=2)
    plt.hist(redLs, color='r', alpha=0.8, density=True, bins=lumhistbins, range=(21, 26), histtype='step', linewidth=2)
    plt.xlabel(r'log($\frac{L_{1.5\mu\mathrm{m}}}{\mathrm{W \ Hz}^{-1}}$)', fontsize=30)
    plt.ylabel('N', fontsize=30)
    plt.savefig('plots/%s_lumhist.pdf' % qso_cat_name)
    plt.close('all')

def radio_detect_frac_plot(fracs, surv_name='FIRST', return_plot=False):
    xs = np.arange(1, len(fracs)+1)*1/len(fracs) - 1/(2*len(fracs))
    plt.figure(153, (8, 6))
    colors = cm.rainbow(np.linspace(0, 1, len(fracs)))
    for j in range(len(fracs)):
        plt.scatter(xs[j], fracs[j], color=colors[j])
    plt.xlim(0, 1)
    plt.ylabel('$f_{\mathrm{%s}}$' % surv_name, fontsize=20)
    plt.xlabel('$g - i$ bin', fontsize=20)
    if return_plot:
        plt.show()
    else:
        plt.savefig('plots/radio_fraction.pdf')
    plt.close('all')





def kappa_vs_mass():
    plt.close('all')
    masses = 10**np.arange(11.5, 13.5, 0.1)
    print(masses)
    zdist = fits.open('QSO_cats/xdqso_specz_complete.fits')[1].data['PEAKZ']
    kappas = []
    for j in range(len(masses)):
        kappas.append(fitting.filtered_model_center(zdist, masses[j]))

    plt.figure(341, (8, 6))
    plt.scatter(np.log10(masses), kappas)
    plt.savefig('kappa_vs_mass.pdf')
    plt.close('all')

def plot_ang_correlation_function():
    bins = np.logspace(-2, 1, 20)
    redw, redw_err = np.load('clustering/eboss_lss_red.npy', allow_pickle=True)
    bluew, bluew_err = np.load('clustering/eboss_lss_blue.npy', allow_pickle=True)


    plt.figure(424, (8, 6))
    plt.scatter(bins, redw, c='r')
    plt.errorbar(bins, redw, yerr=redw_err, fmt='none', ecolor='r')
    plt.scatter(bins, bluew, c='b')
    plt.errorbar(bins, bluew, yerr=bluew_err, fmt='none', ecolor='b')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\theta$ (degrees)', fontsize=20)
    plt.ylabel(r'$w(\theta)$', fontsize=20)
    plt.savefig('plots/ang_clustering.pdf')
    plt.close('all')

def plot_spatial_correlation_function():
    bins = np.logspace(0.5, 1.5, 10)
    redw, redw_err = np.load('clustering/eboss_lss_red.npy', allow_pickle=True)
    bluew, bluew_err = np.load('clustering/eboss_lss_blue.npy', allow_pickle=True)
    plt.close('all')
    plt.figure(91, (8, 6))
    plt.scatter(bins, redw, c='r')
    plt.errorbar(bins, redw, yerr=redw_err, fmt='none', ecolor='r')
    plt.scatter(bins, bluew, c='b')
    plt.errorbar(bins, bluew, yerr=bluew_err, fmt='none', ecolor='b')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$r$ (Mpc)', fontsize=20)
    plt.ylabel(r'$w_{p}(r_{p})/r_{p}$', fontsize=20)
    plt.savefig('plots/spatial_clustering.pdf')
    plt.close('all')

def plot_median_radio(colors, median_fluxes, lum=False):
    plt.figure(5343, (10,8))
    fluxes = np.array(median_fluxes[0])*10**6

    if lum:
        fig, ax = plt.subplots()
        ax.scatter(colors, fluxes)
        ax.set_ylabel(r'$S_{1.4 \mathrm{GHz}} (\mu \mathrm{Jy})$', fontsize=20)
        ax.set_xlabel('$g - i$', fontsize=20)
        ax2 = ax.twinx()
        ax2.scatter(colors, median_fluxes[1], alpha=0)
        ax2.set_ylabel(r'$\mathrm{log}_{10}(L_{1.4 \mathrm{GHz}}(z=1.7))$ (W Hz$^{-1}$)', fontsize=20)
    else:
        plt.scatter(colors, median_fluxes)

    plt.savefig('plots/median_flux_by_color.pdf')
    plt.close('all')

def plot_kappa_v_color(colors, kappas):
    plt.figure(531, (8, 6))
    plt.scatter(colors, kappas)
    plt.xlabel('$g-i$', fontsize=20)
    plt.ylabel('$\kappa$', fontsize=20)
    plt.savefig('plots/kappa_v_color.pdf')
    plt.close('all')

#plot_ang_correlation_function()
#kappa_vs_mass()