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
import spectrumtools
#import fitting
import importlib
from matplotlib.ticker import AutoMinorLocator
import fitting
import clusteringModel
import redshift_dists
importlib.reload(redshift_dists)
importlib.reload(clusteringModel)
importlib.reload(fitting)
importlib.reload(spectrumtools)

#importlib.reload(fitting)


#plt.style.use('default')
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams['xtick.labelsize'] = 'xx-large'
mpl.rcParams['ytick.labelsize'] = 'xx-large'
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
            ax.scatter(x, y, c=z, vmin=-1.5*np.max(z), vmax=1.5*np.max(z), **kwargs)
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



def plot_stacks(samplename):
    bluestack = np.load('stacks/%s_stack0.npy' % samplename, allow_pickle=True)*1000
    ctrlstack = np.load('stacks/%s_stack2.npy' % samplename, allow_pickle=True)*1000
    redstack = np.load('stacks/%s_stack4.npy' % samplename, allow_pickle=True)*1000
    bluestd, ctrlstd, redstd = np.std(bluestack), np.std(ctrlstack), np.std(redstack)

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

def plot_radio_stacks(samplename):
    bluestack = np.load('radio_stacks/%s_blue_first_stack.npy' % samplename, allow_pickle=True) * 1e6
    ctrlstack = np.load('radio_stacks/%s_ctrl_first_stack.npy' % samplename, allow_pickle=True) * 1e6
    redstack = np.load('radio_stacks/%s_red_first_stack.npy' % samplename, allow_pickle=True) * 1e6
    bluestd, ctrlstd, redstd = np.std(bluestack), np.std(ctrlstack), np.std(redstack)

    minval = np.min(np.array([bluestack.flatten(), redstack.flatten(), ctrlstack.flatten()]))
    maxval = np.max(np.array([bluestack.flatten(), redstack.flatten(), ctrlstack.flatten()]))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), dpi=300)
    halfwidth = 31.5
    ax1.imshow(bluestack, vmin=minval, vmax=maxval, cmap='inferno', extent=[-halfwidth, halfwidth, -halfwidth, halfwidth])
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel(r'$\theta$ (arcseconds)', fontsize=15)

    ax1.text(-halfwidth/4, halfwidth*1.25, 'Blue', c='mediumblue', fontsize=25)
    im = ax2.imshow(ctrlstack, vmin=minval, vmax=maxval, cmap='inferno', extent=[-halfwidth, halfwidth, -halfwidth, halfwidth])
    ax2.text(-halfwidth/3, halfwidth*1.25, 'Control', c='darkgreen', fontsize=25)

    ax2.axis('off')

    ax3.imshow(redstack, vmin=minval, vmax=maxval, cmap='inferno', extent=[-halfwidth, halfwidth, -halfwidth, halfwidth])

    ax3.axis('off')
    ax3.text(-halfwidth/4, halfwidth*1.25, 'Red', c='firebrick', fontsize=25)

    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('$F_{1.4 \ \mathrm{GHz}} (\mu$Jy)', fontsize=20)
    # fig.colorbar(im)
    plt.subplots_adjust(wspace=0.05)
    plt.savefig('plots/%s_first_stacks.pdf' % samplename, bbox_inches='tight')
    plt.close('all')

#plot_radio_stacks('xdqso_specz')

def plot_temp_stacks(samplename):
    bluestack = np.load('stacks/%s_blue_temp.npy' % samplename, allow_pickle=True)
    ctrlstack = np.load('stacks/%s_ctrl_temp.npy' % samplename, allow_pickle=True)
    redstack = np.load('stacks/%s_red_temp.npy' % samplename, allow_pickle=True)
    minval = np.min(np.array([bluestack.flatten(), redstack.flatten(), ctrlstack.flatten()]))
    maxval = np.max(np.array([bluestack.flatten(), redstack.flatten(), ctrlstack.flatten()]))

    extent = 120

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), dpi=300)
    ax1.imshow(bluestack, vmin=minval, vmax=maxval, cmap='magma', extent=[-extent, extent, -extent, extent])
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel(r'$\theta$ (arcmin)', fontsize=15)
    ax1.text(-extent/4, extent*1.25, 'Blue', c='mediumblue', fontsize=25)
    im = ax2.imshow(ctrlstack, vmin=minval, vmax=maxval, cmap='magma', extent=[-extent, extent, -extent, extent])
    ax2.text(-extent/3, extent*1.25, 'Control', c='darkgreen', fontsize=25)
    ax2.axis('off')
    ax3.imshow(redstack, vmin=minval, vmax=maxval, cmap='magma', extent=[-extent, extent, -extent, extent])
    ax3.axis('off')
    ax3.text(-extent/4, extent*1.25, 'Red', c='firebrick', fontsize=25)
    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('$\Delta T$', fontsize=20)
    # fig.colorbar(im)
    plt.subplots_adjust(wspace=0.05)
    plt.savefig('plots/%s_temp_stacks.pdf' % samplename, bbox_inches='tight')
    plt.close('all')
#plot_temp_stacks('xdqso_specz')

def plot_masses(samplename):
    bluecolor = np.median(fits.open('catalogs/derived/%s_blue.fits' % samplename)[1].data['deltagmini'])
    ctrlcolor = np.median(fits.open('catalogs/derived/%s_ctrl.fits' % samplename)[1].data['deltagmini'])
    redcolor = np.median(fits.open('catalogs/derived/%s_red.fits' % samplename)[1].data['deltagmini'])

    bluestd = np.std(fits.open('catalogs/derived/%s_blue.fits' % samplename)[1].data['g-i'])
    ctrlstd = np.std(fits.open('catalogs/derived/%s_ctrl.fits' % samplename)[1].data['g-i'])
    redstd = np.std(fits.open('catalogs/derived/%s_red.fits' % samplename)[1].data['g-i'])

    bluemass = np.load('masses/%s_blue_mass.npy' % samplename, allow_pickle=True)
    ctrlmass = np.load('masses/%s_ctrl_mass.npy' % samplename, allow_pickle=True)
    redmass = np.load('masses/%s_red_mass.npy' % samplename, allow_pickle=True)

    plt.close('all')
    plt.figure(4, (10, 10), dpi=300)
    plt.xlabel(r'$\langle \Delta (g - i) \rangle$', fontsize=30)
    plt.ylabel('log$_{10}(M/h^{-1} M_{\odot}$)', fontsize=30)
    plt.errorbar(ctrlcolor, ctrlmass[0], yerr=[[ctrlmass[2]], [ctrlmass[1]]], ecolor='k')
    plt.errorbar(redcolor, redmass[0], yerr=[[redmass[2]], [redmass[1]]], ecolor='k')
    plt.errorbar(bluecolor, bluemass[0], yerr=[[bluemass[2]], [bluemass[1]]], ecolor='k')
    #plt.errorbar([bluecolor, ctrlcolor, redcolor], [bluemass[0], ctrlmass[0], redmass[0]], yerr=[[bluemass[1], bluemass[2]], [ctrlmass[1], ctrlmass[2]], [redmass[1], redmass[2]]], xerr=[bluestd, ctrlstd, redstd], ls='none', ecolor='k')

    plt.scatter(bluecolor, bluemass[0], c='b', s=100)
    plt.scatter(ctrlcolor, ctrlmass[0], c='g', s=100)
    plt.scatter(redcolor, redmass[0], c='r', s=100)

    plt.savefig('plots/%s_color_mass.pdf' % samplename)
    plt.close('all')

#plot_masses('xdqso_specz')



def plot_kappa_profile(colorbin, kap_profile, kap_errs, binsize, maxtheta, best_mass_profile, binned_model, oneterm=None, twoterm=None):


    obs_theta = np.arange(binsize / 2, maxtheta, binsize)

    theta_range = np.arange(0.5, maxtheta, 0.5)
    plt.figure(0, (10, 8))
    plt.scatter(obs_theta, kap_profile, c='k')
    plt.errorbar(obs_theta, kap_profile, yerr=kap_errs, c='k', fmt='none')
    plt.scatter(obs_theta, binned_model, marker='s', facecolors='none', edgecolors='k', s=50)
    plt.plot(theta_range, oneterm, label='1-halo term', c='k', ls=':')
    plt.plot(theta_range, twoterm, label='2-halo term', c='k', ls='--')
    plt.plot(theta_range, best_mass_profile, label='Filtered Total', c='k', ls='-')
    # plt.plot(theta_range, lowest_mass_profile, c='k')
    # plt.scatter(obs_theta, filtered_model_in_bins(zdist, obs_theta, 10**(lowmass)), marker='s')
    # plt.scatter(obs_theta, lowprofile)
    plt.ylim(-0.0005, np.max(best_mass_profile)+0.001)
    plt.ylabel(r'$ \langle \kappa \rangle$', fontsize=20)
    plt.xlabel(r'$\theta$ (arcminutes)', fontsize=20)
    plt.legend(fontsize=20)
    # if error:
    #plt.text(50, 0.0015,
    #         'log$_{10}(M/h^{-1} M_{\odot}$) = %s $\pm$ %s' % (round(avg_mass, 1), round(mass_uncertainty, 1)),
    #         fontsize=20)
    # else:
    # plt.text(50, 0.0015, 'log$_{10}(M/h^{-1} M_{\odot}$) = %s' % round(avg_mass, 3), fontsize=20)
    plt.savefig('plots/profiles/profile_%s.png' % colorbin)
    plt.close('all')

#plot_masses('xdqso_specz')

def plot_cov_matrix(color, matrix):
    plt.figure(7562, (8, 8))
    plt.imshow(matrix)
    plt.savefig('plots/%s_covariance.pdf' % color)
    plt.close('all')


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
    plt.xlabel('$\mathrm{Redshift}$', fontsize=30)
    ax.tick_params(axis='both', labelsize=20)
    plt.ylabel('$M_{i} (z=2)$', fontsize=30)
    plt.ylim(-21, -30)
    plt.savefig('plots/%s_Mi_z.pdf' % qso_cat_name)
    plt.close('all')
    plt.close('all')

def plot_lum_vs_z(zlist, lumlist, nsources, minz, maxz, lumcut, qso_cat_name):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    # plt.scatter(qso_cat[zkey], i_abs_mags, c='k', s=0.01, alpha=0.5)

    density_scatter(zlist, lumlist, bins=[30, 30], ax=ax, s=0.05, cmap='magma', rasterized=True)
    plt.xlabel('$\mathrm{Redshift}$', fontsize=30)
    ax.tick_params(axis='both', labelsize=20)
    plt.ylabel('$log(L_{1.5 \ \mu m})$', fontsize=30)
    plt.ylim(21, 26)
    plt.savefig('plots/%s_lum_z.pdf' % qso_cat_name)
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

def g_minus_i_plot(qso_cat_name, offset):
    plt.close('all')

    allcat = fits.open('catalogs/derived/%s_binned.fits' % qso_cat_name)[1].data
    bluecat = allcat[np.where(allcat['bin'] == 1)]
    ctrlcat = allcat[np.where(allcat['bin'] == round(np.max(allcat['bin'])/2+0.1))]
    redcat = allcat[np.where(allcat['bin'] == np.max(allcat['bin']))]

    allzs, czs, redzs, bluezs = allcat['Z'], ctrlcat['Z'], redcat['Z'], bluecat['Z']

    hists = False

    if offset:
        colkey = 'deltagmini'
    else:
        colkey = 'g-i'

    allcolors, ccolors, redcolors, bluecolors = allcat[colkey], ctrlcat[colkey], redcat[colkey], bluecat[colkey]


    ebossandxd = True

    if ebossandxd:
        fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 8))
        fig.supxlabel('$\mathrm{Redshift}$', fontsize=30)
        ebosscat = fits.open('catalogs/derived/eBOSS_QSO_binned.fits')[1].data
        xdcat = fits.open('catalogs/derived/xdqso_specz_binned.fits')[1].data

        blueebosscat = ebosscat[np.where(ebosscat['bin'] == 1)]
        ctrlebosscat = ebosscat[np.where(ebosscat['bin'] == round(np.max(ebosscat['bin']) / 2 + 0.1))]
        redebosscat = ebosscat[np.where(ebosscat['bin'] == np.max(ebosscat['bin']))]

        bluexdcat = xdcat[np.where(xdcat['bin'] == 1)]
        ctrlxdcat = xdcat[np.where(xdcat['bin'] == round(np.max(xdcat['bin']) / 2 + 0.1))]
        redxdcat = xdcat[np.where(xdcat['bin'] == np.max(xdcat['bin']))]

        allxdzs, cxdzs, redxdzs, bluexdzs = xdcat['Z'], ctrlxdcat['Z'], redxdcat['Z'], bluexdcat['Z']
        allebosszs, cebosszs, redebosszs, blueebosszs = ebosscat['Z'], ctrlebosscat['Z'], redebosscat['Z'], blueebosscat['Z']

        allebosscolors, cebosscolors, redebosscolors, blueebosscolors = ebosscat[colkey], ctrlebosscat[colkey], \
                                                                        redebosscat[colkey], blueebosscat[colkey]
        allxdcolors, cxdcolors, redxdcolors, bluexdcolors = xdcat[colkey], ctrlxdcat[colkey], \
                                                                        redxdcat[colkey], bluexdcat[colkey]

        bins = 100
        ps = 0.03
        data, x_e, y_e = np.histogram2d(allebosszs, allebosscolors, bins=[bins, bins], density=True)
        z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([allebosszs, allebosscolors]).T,
                    method="splinef2d",
                    bounds_error=False)

        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

        density_scatter(allebosszs, allebosscolors, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Greys', rasterized=True,
                        cutcolorbar=True, alpha=0.3, vminin=np.min(z), vmaxin=np.max(z))
        density_scatter(cebosszs, cebosscolors, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Greens', rasterized=True,
                        cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))
        density_scatter(redebosszs, redebosscolors, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Reds', rasterized=True,
                        cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))
        density_scatter(blueebosszs, blueebosscolors, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Blues', rasterized=True,
                        cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))

        data, x_e, y_e = np.histogram2d(allxdzs, allxdcolors, bins=[bins, bins], density=True)
        z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data,
                    np.vstack([allebosszs, allebosscolors]).T,
                    method="splinef2d",
                    bounds_error=False)

        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0
        density_scatter(allxdzs, allxdcolors, bins=[bins, bins], ax=ax2, fig=fig, s=ps, cmap='Greys',
                        rasterized=True,
                        cutcolorbar=True, alpha=0.3, vminin=np.min(z), vmaxin=np.max(z))
        density_scatter(cxdzs, cxdcolors, bins=[bins, bins], ax=ax2, fig=fig, s=ps, cmap='Greens', rasterized=True,
                        cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))
        density_scatter(redxdzs, redxdcolors, bins=[bins, bins], ax=ax2, fig=fig, s=ps, cmap='Reds',
                        rasterized=True,
                        cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))
        density_scatter(bluexdzs, bluexdcolors, bins=[bins, bins], ax=ax2, fig=fig, s=ps, cmap='Blues',
                        rasterized=True,
                        cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))
        ax.set_ylim(-0.5, 2)
        ax2.set_ylim(-0.5, 2)

        ax.title.set_text('eBOSS QSOs')
        ax2.title.set_text('XDQSOz QSOs')
        ax.title.set_size(25)
        ax2.title.set_size(25)

        ax.set_ylabel('$g-i$', fontsize=30)

        plt.savefig('plots/gminusi_both.pdf')
        plt.close('all')
        return

    else:
        fig = plt.figure(figsize=(10, 9))





    if hists:
        offset=True
        left, width = 0.12, 0.7
        #bottom, height = 0.1, 0.65
        bottom, height = 0.1, 0.85
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        #rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.15, height]

        ax = fig.add_axes(rect_scatter)
        #ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)

        nbins = 500

        ax_histy.hist(allcolors, bins=nbins, orientation='horizontal', histtype='step', color='k')

        ax_histy.tick_params(axis="y", labelleft=False)
        ax_histy.axhline(0.25, linestyle='--', c='k')
        xcolors = np.linspace(-1, 1, 1000)
        gaussfit = fitting.fit_gauss_hist_one_sided(allcolors, xcolors, nbins)
        ax_histy.plot(gaussfit, xcolors, c='grey')
        ax_histy.set_xlabel('N', fontsize=30)
        #ax_histx.plot(np.linspace(0.5, 2.5, 20), lensingModel.lensing_kernel(np.linspace(0.5, 2.5, 20)))
    else:
        ax = plt.gca()



    bins = 100
    ps = 0.03
    data, x_e, y_e = np.histogram2d(allzs, allcolors, bins=[bins, bins], density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([allzs, allcolors]).T, method="splinef2d",
                bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    density_scatter(allzs, allcolors, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Greys', rasterized=True,
                    cutcolorbar=True, alpha=0.3, vminin=np.min(z), vmaxin=np.max(z))
    density_scatter(czs, ccolors, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Greens', rasterized=True,
                    cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))
    density_scatter(redzs, redcolors, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Reds', rasterized=True,
                    cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))
    density_scatter(bluezs, bluecolors, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Blues', rasterized=True,
                    cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))

    if offset:
        ax.set_ylabel('$\Delta (g - i)$', fontsize=30)
    else:
        ax.set_ylabel('$g - i$', fontsize=30)
    ax.set_xlabel('$\mathrm{Redshift}$', fontsize=30)

    zs = np.linspace(np.min(allzs), np.max(allzs), 30)
    modcols = []
    for z in zs:
        modcols.append(spectrumtools.vdb_color_at_z(z))
    plt.plot(zs, modcols, c='k')

    #plt.scatter(zs, gminusi, c='k', s=0.01, alpha=0.5)
    #plt.scatter(bluezs, bluegs, c='b', s=0.05, rasterized=True)
    #plt.scatter(czs, cgs, c='g', s=0.05, rasterized=True)
    #plt.scatter(redzs, redgs, c='r', s=0.05, rasterized=True)
    ax.set_ylim(-1, 1.5)
    plt.savefig('plots/%s_gminusi.pdf' % qso_cat_name)
    plt.close('all')

def color_v_z(qso_cat_name, colorkey):
    allcat = fits.open('catalogs/derived/%s_binned.fits' % qso_cat_name)[1].data
    bluecat = allcat[np.where(allcat['bin'] == 1)]
    ctrlcat = allcat[np.where(allcat['bin'] == round(np.max(allcat['bin']) / 2 + 0.1))]
    redcat = allcat[np.where(allcat['bin'] == np.max(allcat['bin']))]

    allzs = allcat['Z']
    czs = ctrlcat['Z']
    redzs = redcat['Z']
    bluezs = bluecat['Z']

    fig = plt.figure(12351, (10, 9))

    ax = plt.gca()

    allcolors = allcat[colorkey]
    bluecolors = bluecat[colorkey]
    ccolors = ctrlcat[colorkey]
    redcolors = redcat[colorkey]




    bins = 100
    ps = 0.03
    data, x_e, y_e = np.histogram2d(allzs, allcolors, bins=[bins, bins], density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([allzs, allcolors]).T,
                method="splinef2d",
                bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    density_scatter(allzs, allcolors, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Greys', rasterized=True,
                    cutcolorbar=True, alpha=0.3, vminin=np.min(z), vmaxin=np.max(z))
    density_scatter(czs, ccolors, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Greens', rasterized=True,
                    cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))
    density_scatter(redzs, redcolors, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Reds', rasterized=True,
                    cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))
    density_scatter(bluezs, bluecolors, bins=[bins, bins], ax=ax, fig=fig, s=ps, cmap='Blues', rasterized=True,
                    cutcolorbar=True, vminin=np.min(z), vmaxin=np.max(z))


    ax.set_ylabel('$%s_{AB}$' % colorkey, fontsize=30)
    ax.set_xlabel('$\mathrm{Redshift}$', fontsize=30)



    ax.set_ylim(-1, 5)
    plt.savefig('plots/%s_color_z.pdf' % qso_cat_name)
    plt.close('all')

#g_minus_i_plot('xdqso_specz', False)

def color_hist(qso_cat_name, colorkey='g-i'):

    fig = plt.figure(figsize=(8,10))
    allzs = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data['Z']
    firstband, secondband = colorkey.split('-')[0], colorkey.split('-')[1]

    allcolors = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data['delta%smin%s' % (firstband, secondband)]



    nbins = 1000
    sepax=True
    if sepax:
        left, width = 0.14, 0.82
        # bottom, height = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]


        ax = fig.add_axes(rect_scatter)
        ax.set_ylabel('$z$', fontsize=30)
        # ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histx, sharex=ax)

        ax_histy.hist(allcolors, bins=nbins, orientation='vertical', histtype='step', color='k')

        ax_histy.tick_params(axis="x", labelbottom=False)
        #ax_histy.axvline(0.25, linestyle='--', c='k', label='Red-tail cutoff')
        xcolors = np.linspace(-1, 1, 1000)
        gaussfit = fitting.fit_gauss_hist_one_sided(allcolors, xcolors, nbins)
        calcgauss = fitting.gaussian(xcolors, gaussfit[0], gaussfit[1], gaussfit[2])
        #ax_histy.hist(spectrumtools.model_color_dist(0, 1, 0.2, -1), bins=200, orientation='vertical', histtype='step', color='b', density=True)

        ax_histy.plot(xcolors, calcgauss, c='grey')
        ax_histy.set_ylabel('N', fontsize=30)
        #ax_histy.legend(loc='upper right')

    else:
        ax = plt.gca()


    #plt.hist(allcolors, bins=nbins, histtype='step', color='k', density=True)


    density_scatter(allcolors, allzs, bins=[20, 20], ax=ax, fig=fig, s=0.05, cmap='Greys', rasterized=True, cutcolorbar=True)

    pltzs = np.linspace(np.min(allzs), np.max(allzs), 50)
    #pltzs = np.load('medzbins.npy', allow_pickle=True)
    ebvs = [-0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32]
    lses = ['dashdot', 'solid', 'dotted', 'dashdot', 'solid', 'dotted', 'dashdot', 'solid', 'dotted', 'dashdot']
    #correction = np.load('vdbcorrection.npy', allow_pickle=True)

    for j in range(len(ebvs)):
        reddened_cols = []
        for z in pltzs:
            reddened_cols.append(spectrumtools.relative_vdb_color(z, ebvs[j]))
        ax.plot(reddened_cols, pltzs, ls='solid', c='k', label='E(B-V) = %s' % ebvs[j])
    ax.set_xlabel('$\Delta (%s)$' % colorkey, fontsize=30)
    ax.legend(loc='lower right')


    plt.xlim(-1, 2)

    plt.savefig('plots/%s_colorhist.pdf' % qso_cat_name)
    plt.close('all')

def color_color():
    plt.close('all')
    catpath = 'catalogs/derived/xdqso_specz_complete.fits'
    firstcolor = fits.open(catpath)[1].data['deltagmini']
    secondcolor = fits.open(catpath)[1].data['deltaiminz']
    zs = fits.open(catpath)[1].data['Z']

    plt.figure(figsize=(10, 10))
    #plt.scatter(secondcolor, firstcolor, rasterized=True)
    density_scatter(secondcolor, firstcolor, ax=plt.gca(), bins=(10, 10), cutcolorbar=True, rasterized=True, s=0.01)

    plt.ylim((-.5, 1))
    plt.xlim(-1, 1)

    plt.savefig('plots/color_color.pdf')
    plt.close('all')

#def plot_mateos_cut(table):


#color_color()

#color_hist('xdqso_specz')
def quickhist():
    allcolors = fits.open('catalogs/derived/xdqso_specz_complete.fits')[1].data['deltagmini']
    plt.figure(figsize=(10, 8))
    plt.hist(allcolors, density=True, range=(-1, 2), bins=100, histtype='step', color='k')
    modcols = spectrumtools.model_color_dist(0, 0.4, 0.2, -1.5)
    plt.hist(modcols, density=True, range=(-1, 2), bins=100, histtype='step', color='b')
    plt.savefig('plots/quickhist.pdf')
    plt.close('all')
#quickhist()

def z_dists(qso_cat_name, bzs, czs, rzs):
    plt.close('all')
    plt.figure(747, (8, 6))
    plt.hist(rzs, color='r', alpha=0.8, density=True, bins=20, histtype='step', linewidth=2)
    plt.hist(bzs, color='b', alpha=0.8, density=True, bins=20, histtype='step', linewidth=2)
    plt.hist(czs, color='g', alpha=0.8, density=True, bins=20, histtype='step', linewidth=2)
    plt.xlabel('$\mathrm{Redshift}$', fontsize=20)
    plt.ylabel('$\mathrm{Frequency}$', fontsize=20)
    plt.savefig('plots/%s_zdists.pdf' % qso_cat_name)
    plt.close('all')

def lum_dists(qso_cat_name, lumhistbins, blueLs, cLs, redLs, tailLs=None):
    plt.close('all')
    plt.figure(figsize=(10, 10))
    plt.hist(blueLs, color='b', alpha=0.8, density=True, bins=lumhistbins, range=(42, 48), histtype='step', linewidth=2)
    plt.hist(cLs, color='g', alpha=0.8, density=True, bins=lumhistbins, range=(42, 48), histtype='step', linewidth=2)
    plt.hist(redLs, color='r', alpha=0.8, density=True, bins=lumhistbins, range=(42, 48), histtype='step', linewidth=2)
    if tailLs is not None:
        plt.hist(tailLs, color='grey', alpha=0.5, density=True, bins=lumhistbins, range=(21,26), histtype='step', linewidth=2)
    plt.xlabel(r'log$\left( \frac{ \nu L_{\nu} (1.5\mu\mathrm{m})}{\mathrm{erg \ s}^{-1}} \right)$', fontsize=30)
    plt.ylabel('$\mathrm{Frequency}$', fontsize=30)
    plt.savefig('plots/%s_lumhist.pdf' % qso_cat_name)
    plt.close('all')

def radio_detect_frac_plot(fracs, colorkey, surv_name='FIRST', return_plot=False):
    xs = np.arange(1, len(fracs)+1)*1/len(fracs) - 1/(2*len(fracs))
    plt.figure(153, (8, 6))
    colors = cm.rainbow(np.linspace(0, 1, len(fracs)))
    for j in range(len(fracs)):
        plt.scatter(xs[j], fracs[j], color=colors[j])
    plt.xlim(0, 1)
    plt.ylabel('$f_{\mathrm{%s}}$' % surv_name, fontsize=20)
    plt.xlabel('$%s$ bin' % colorkey, fontsize=20)
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

def plot_ang_correlation_function(sample_id, nbins):
    bins = np.logspace(-2, 1, nbins)
    redw, redw_err = np.load('angclustering/%s_10.npy' % sample_id, allow_pickle=True)
    bluew, bluew_err = np.load('angclustering/%s_1.npy' % sample_id, allow_pickle=True)
    #ctrlw, ctrl_err = np.load('clustering/%s_ctrl.npy' % sample_id, allow_pickle=True)


    plt.figure(424, (8, 6))
    plt.scatter(bins, redw, c='r')
    plt.errorbar(bins, redw, yerr=redw_err, fmt='none', ecolor='r')
    plt.scatter(bins, bluew, c='b')
    plt.errorbar(bins, bluew, yerr=bluew_err, fmt='none', ecolor='b')
    #plt.scatter(bins, ctrlw, c='g')
    #plt.errorbar(bins, ctrlw, yerr=ctrl_err, fmt='none', ecolor='g')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\theta$ (degrees)', fontsize=20)
    plt.ylabel(r'$w(\theta)$', fontsize=20)
    plt.savefig('plots/ang_clustering.pdf')
    plt.close('all')

def plot_ang_cross_corr(sample_id, nbins, minscale, maxscale, samples):
    plt.close('all')
    bins = np.logspace(minscale, maxscale, nbins)


    colors = cm.rainbow(np.linspace(0, 1, len(samples)))

    plt.figure(figsize=(8, 6))

    for j, sample in enumerate(samples):
        w, werr = np.load('clustering/ang_cross/%s_%s.npy' % (sample_id, sample+1), allow_pickle=True)
        plt.scatter(bins, w, color=colors[j])
        plt.errorbar(bins, w, yerr=werr, fmt='none', ecolor=colors[j], alpha=0.3)


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\theta$ (degrees)', fontsize=20)
    plt.ylabel(r'$w(\theta)$', fontsize=20)


    plt.savefig('plots/ang_cross_corr.pdf')
    plt.close('all')

def plot_spatial_cross_corr(samples, cap):
    plt.close('all')
    #bins = np.logspace(minscale, maxscale, nbins)



    colors = cm.rainbow(np.linspace(0, 1, len(samples)))


    plt.figure(figsize=(8, 6))

    for j, sample in enumerate(samples):
        avg_rs, w, werr = np.load('clustering/spatial_cross/%s/%s.npy' % (cap, sample+1), allow_pickle=True)
        #linfit = np.polyfit(np.log(avg_rs), np.log(w), 1)
        #print(linfit)

        plt.scatter(avg_rs, w, color=colors[j])
        plt.errorbar(avg_rs, w, yerr=werr, fmt='none', ecolor=colors[j], alpha=0.3)




    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$r_{p} (h^{-1}$ Mpc)', fontsize=20)
    plt.ylabel(r'$w_{p}(r_{p})$', fontsize=20)


    plt.savefig('plots/%s_spatial_cross_corr.pdf' % cap)
    plt.close('all')
    if len(w) == 1:
        plt.figure(figsize=(8, 6))
        """avg_rs, redw, rederr = np.load('clustering/spatial_cross/%s.npy' % int(np.max(samples)+1), allow_pickle=True)
        avg_rs, bluew, bluerr = np.load('clustering/spatial_cross/1.npy', allow_pickle=True)
        plt.scatter(avg_rs, redw/bluew)
        errs = redw/bluew * np.sqrt((rederr/redw)**2+(bluerr/bluew)**2)
        plt.errorbar(avg_rs, redw/bluew, yerr=errs, fmt='none')"""

        refr, refw, referr = np.load('clustering/spatial_cross/%s/1.npy' % cap, allow_pickle=True)

        ratios, ratio_errs = [], []
        for j, sample in enumerate(samples):
            avg_rs, w, werr = np.load('clustering/spatial_cross/%s/%s.npy' % (cap, sample+1), allow_pickle=True)
            ratios.append(w/refw)
            ratio_errs.append((w/refw * np.sqrt((werr/w)**2+(referr/refw)**2))[0])



        plt.scatter(np.array(samples)+1, ratios)
        plt.errorbar(np.array(samples)+1, ratios, yerr=ratio_errs, fmt='none')
        plt.xlabel('bin', fontsize=30)
        plt.ylabel('bias (bin) / bias (bin = 1)', fontsize=30)

        plt.savefig('plots/cross_ratio.pdf')
        plt.close('all')

def plot_spatial_correlation_function(qso_cat_name, cap, nbins, minscale, maxscale, n_samples, cf=None):

    residuals = False


    bins = np.logspace(minscale, maxscale, nbins)
    colors = cm.rainbow(np.linspace(0, 1, n_samples))

    #bluew, bluew_err = np.load('clustering/xdqso_specz_1.npy', allow_pickle=True)
    #ctrlw, ctrlw_err = np.load('clustering/eboss_lss_ctrl.npy', allow_pickle=True)

    plt.close('all')
    if residuals:
        fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8,6))

    else:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    ax.set_yscale('log')
    ax.set_ylabel(r'$w_{p}(r_{p})$', fontsize=20)

    if n_samples > 1:
        for j in range(n_samples):
            w, werr = np.load('clustering/spatial/%s/%s_%s.npy' % (cap, qso_cat_name, j+1), allow_pickle=True)
            ax.scatter(bins, w, color=colors[j])
            ax.errorbar(bins, w, yerr=werr, fmt='none', ecolor=colors[j], alpha=0.3)
    else:
        #w, werr = np.load('clustering/spatial/eboss_lss_all.npy', allow_pickle=True)
        w, werr = np.load('clustering/spatial/%s/%s_all.npy' % (cap, qso_cat_name), allow_pickle=True)
        ax.scatter(bins, w, color=colors[0])
        ax.errorbar(bins, w, yerr=werr, fmt='none', ecolor=colors[0], alpha=0.3)


    if cf is not None:
        ax.plot(bins, cf)
        if residuals:
            ax2.scatter(bins, cf-w)
            ax2.set_ylabel('Residual', fontsize=20)

    plt.xscale('log')

    plt.xlabel('$r_{p} (h^{-1}$ Mpc)', fontsize=20)
    plt.savefig('plots/spatial_clustering_%s.pdf' % cap)
    plt.close('all')

def plot_bias_by_bin(refnames, n_sample_bins):

    plt.close('all')
    fig, axs = plt.subplots(len(refnames), 1, sharex=True, figsize=(10, 7*len(refnames)))



    tab = fits.open('catalogs/derived/eBOSS_QSO_binned.fits')[1].data

    shift = 0.007

    nboots = 5
    medcolors, mederrs = [], []
    for j in range(n_sample_bins):
        bintab = tab[np.where(tab['bin'] == j + 1)]
        medcolors.append(np.median(bintab['deltagmini']))
        bootmeds = []
        for k in range(nboots):
            boottab = bintab[np.random.choice(len(bintab), len(bintab))]
            bootmeds.append(np.median(boottab['deltagmini']))
        mederrs.append(np.std(bootmeds))

    medcolors = np.array(medcolors)


    for i, refname in enumerate(refnames):

        ngcbiases, sgcbiases, ngcbiaserrs, sgcbiaserrs = [], [], [], []
        for j in range(n_sample_bins):
            tmp1, tmp2 = np.load('bias/eBOSS_QSO/NGC/%s_%s.npy' % (refname, j+1), allow_pickle=True)

            ngcbiases.append(tmp1)
            ngcbiaserrs.append(tmp2)




        axs[i].scatter(medcolors - shift, ngcbiases, c='slategray', alpha=0.2, label='NGC')
        axs[i].errorbar(medcolors - shift, ngcbiases, yerr=ngcbiaserrs, fmt='none', c='slategray', alpha=0.2)

        for j in range(n_sample_bins):
            tmp1, tmp2 = np.load('bias/eBOSS_QSO/SGC/%s_%s.npy' % (refname, j + 1), allow_pickle=True)

            sgcbiases.append(tmp1)
            sgcbiaserrs.append(tmp2)
        axs[i].scatter(medcolors + shift, sgcbiases, c='saddlebrown', alpha=0.2, label='SGC')
        axs[i].errorbar(medcolors + shift, sgcbiases, yerr=ngcbiaserrs, fmt='none', c='saddlebrown', alpha=0.2)

        avg_bias = np.average([ngcbiases, sgcbiases], weights=[1/np.array(ngcbiaserrs), 1/np.array(sgcbiaserrs)], axis=0)
        avg_err = np.sqrt(np.array(ngcbiaserrs)**2 + np.array(sgcbiaserrs)**2)/2


        minb, maxb = np.min([np.min(ngcbiases), np.min(sgcbiases)]), np.max([np.max(ngcbiases), np.max(sgcbiases)])
        b_grid = np.linspace(minb, maxb, 20)
        zs, dndz = redshift_dists.redshift_dist(tab)
        masses = []
        for bb in b_grid:
            masses.append(clusteringModel.avg_bias_to_mass(bb, zs, dndz))

        transforms = [b_grid, np.log10(masses)]

        if transforms is not None:
            def forward(x):
                return np.interp(x, transforms[0], transforms[1])

            def inverse(x):
                return np.interp(x, transforms[1], transforms[0])

            secax = axs[i].secondary_yaxis('right', functions=(forward, inverse))
            secax.yaxis.set_minor_locator(AutoMinorLocator())

            secax.set_ylabel(r'$ \mathrm{log}_{10}(M_h/h^{-1} M_{\odot})$', fontsize=30, labelpad=20)
            secax.tick_params(axis='y', which='major', labelsize=25)



        axs[i].scatter(medcolors, avg_bias, c='k', label='Mean')
        axs[i].errorbar(medcolors, avg_bias, yerr=avg_err, fmt='none', c='k')
        axs[i].set_ylabel('$b_Q$', fontsize=35, rotation=0, labelpad=25)
        axs[i].legend(fontsize=20)
        axs[i].tick_params(axis='both', which='major', labelsize=25)
        #axs[i].set_title(refname, fontsize=20)

    plt.xlabel(r'$\langle \Delta (g-i) \rangle$', fontsize=35)

    plt.savefig('plots/rel_bias.pdf')
    plt.close('all')

def plot_each_cf(rps, cap, w, werr, cf, binid, refname):
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8,6))
    #rps = np.logspace(0.3, 1.4, 10)

    ax.scatter(rps, w, c='k')
    ax.errorbar(rps, w, yerr=werr, c='k', fmt='none')
    ax.plot(rps, cf, ls='--', c='Grey')
    ax2.scatter(rps, w-cf, c='k')
    ax2.errorbar(rps, w-cf, yerr=werr, c='k', fmt='none')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel('$r_{p} (h^{-1}$ Mpc)', fontsize=20)
    ax.set_ylabel('$w_p(r_p)$', fontsize=20)
    ax2.set_ylabel('Residual', fontsize=20)
    plt.savefig('plots/cf_fits/%s_%s_%s.pdf' % (cap, refname, binid))
    plt.close('all')



def plot_2d_corr_func(cf):
    ara = np.arange(200)
    xx, yy = np.meshgrid(ara, ara, sparse=True)
    rs = np.sqrt(xx**2 + yy**2)
    cf2d = np.reshape(cf, (-1, 200)) * rs
    plt.figure(figsize=(10, 8))
    #plt.imshow(cf, aspect='auto', origin='lower')
    plt.contourf(ara, ara, cf2d)
    plt.colorbar(label=r'$\xi(r_{p}, \pi)$')
    plt.xlabel('$\pi_{max}$', fontsize=30)
    plt.ylabel('$r_{p}$', fontsize=30)
    plt.legend()
    plt.savefig('plots/2dCF.pdf')
    plt.close('all')


def plot_median_radio_flux(colorkey, remove_reddest=False):
    colors, median_fluxes, fluxerrs = np.load('plotting_results/first_flux_for_color.npy', allow_pickle=True)
    plt.figure(5343, (8, 6))
    fluxes = np.array(median_fluxes) * 10 ** 6

    if not remove_reddest:
        lastidx = len(colors)-1
        redcolor = colors[lastidx]
        redflux = fluxes[lastidx]
        print(redflux)

        plt.scatter(redcolor, redflux, edgecolors='grey', marker='s', facecolors='none')
        if fluxerrs is not None:
            rederr = fluxerrs[lastidx]
            plt.errorbar(redcolor, redflux, yerr=rederr, c='grey', fmt='none')
            fluxerrs = fluxerrs[:lastidx]


        colors = colors[:lastidx]
        fluxes = fluxes[:lastidx]


    plt.scatter(colors, fluxes, c='k')
    if fluxerrs is not None:
        plt.errorbar(colors, fluxes, yerr=fluxerrs, c='k', fmt='none')
    plt.ylabel(r'$S_{1.4 \mathrm{GHz}} (\mu \mathrm{Jy})$', fontsize=20)

    plt.xlabel('$%s$ bin' % colorkey, fontsize=20)

    plt.savefig('plots/median_flux_by_color.pdf')
    plt.close('all')

def plot_median_radio_luminosity(colorkey, remove_reddest=False):
    plt.figure(964, (8, 6))

    colors, median_lums, lumerrs = np.load('plotting_results/first_lum_for_color.npy', allow_pickle=True)

    if not remove_reddest:
        lastidx = len(colors)-1
        redcolor = colors[lastidx]
        redlums = median_lums[lastidx]

        plt.scatter(redcolor, redlums, edgecolors='grey', marker='s', facecolors='none')
        if lumerrs is not None:
            rederr = lumerrs[lastidx]
            plt.errorbar(redcolor, redlums, yerr=rederr, c='grey', fmt='none')
            lumerrs = lumerrs[:lastidx]

        colors = colors[:lastidx]
        median_lums = median_lums[:lastidx]

    plt.scatter(colors, median_lums, c='k')
    if lumerrs is not None:
        plt.errorbar(colors, median_lums, yerr=lumerrs, c='k', fmt='none')
    plt.ylabel(r'$L_{1.4 \mathrm{GHz}}$ (erg s$^{-1}$ Hz$^{-1}$)', fontsize=20)
    plt.axhline(1.5e30, linestyle='--', c='k', label='SFR $ = 100 M_{\odot} yr ^ {-1}$')
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci')

    plt.xlabel('$%s$ bin' % colorkey, fontsize=20)

    plt.savefig('plots/median_lum_by_color.pdf')
    plt.close('all')




def plot_radio_loudness(colorkey, remove_reddest=False):
    plt.figure(34334, (8, 8))
    colors, loudness, louderrs = np.load('plotting_results/first_loud_for_color.npy', allow_pickle=True)

    if not remove_reddest:
        lastidx = len(colors)-1
        redcolor = colors[lastidx]
        redloud = loudness[lastidx]


        plt.scatter(redcolor, redloud, edgecolors='grey', marker='s', facecolors='none')
        if louderrs is not None:
            rederr = louderrs[lastidx]
            plt.errorbar(redcolor, redloud, yerr=rederr, c='grey', fmt='none')
            louderrs = louderrs[:lastidx]

        colors = colors[:lastidx]
        loudness = loudness[:lastidx]


    plt.xlabel('$%s$ bin' % colorkey, fontsize=30)

    plt.scatter(colors, loudness, c='k')
    if louderrs is not None:
        plt.errorbar(colors, loudness, yerr=louderrs, c='k', fmt='none')
    plt.ylabel(r'$R = \mathrm{log}_{10}(L_{1.4 \mathrm{GHz}}/L_{\mathrm{bol}})$', fontsize=30)
    plt.savefig('plots/radio_loudness_by_color.pdf')
    plt.close('all')

def plot_kappa_v_color(kappas, errs, colorkey, planck_kappas=None, act_kappas=None, transforms=None, remove_reddest=False, linfit=None, mode='color'):
    fig, ax1 = plt.subplots(figsize=(8, 7))

    #colors = np.arange(1, len(kappas)+1)*1/len(kappas) - 1/(2*len(kappas))
    colors = range(1, len(kappas)+1)



    if not remove_reddest:
        lastidx = len(colors)-1
        redcolor = colors[lastidx]
        redkappa = kappas[lastidx]
        rederr = errs[lastidx]

        plt.scatter(redcolor, redkappa, edgecolors='grey', marker='s', facecolors='none')
        plt.errorbar(redcolor, redkappa, yerr=rederr, fmt='none', ecolor='grey')

        colors = colors[:lastidx]
        kappas = kappas[:lastidx]
        errs = errs[:lastidx]

    ax1.scatter(colors, kappas, c='k')
    ax1.errorbar(colors, kappas, yerr=errs, fmt='none', ecolor='k')
    if linfit is not None:
        linmod = linfit[0] * np.array(colors) + linfit[1]
        ax1.plot(colors, linmod, c='k', linestyle='--')
    #if offset:
    #ax1.set_xlabel(r'$\langle \Delta (g-i) \rangle$', fontsize=20)
    if mode == 'color':
        ax1.set_xlabel('$%s$ bin' % colorkey, fontsize=20)
    elif mode == 'bal':
        ax1.set_xlabel('BAL bin', fontsize=20)
    elif mode == 'bhmass':
        ax1.set_xlabel('BH Mass bin', fontsize=20)
    #else:
        #plt.xlabel('$g-i$', fontsize=20)
    ax1.set_ylabel(r'$\langle \kappa_{\mathrm{peak}} \rangle$', fontsize=25)
    if planck_kappas is not None:
        planck_kappas = np.array(planck_kappas)
        ax1.scatter(colors, planck_kappas[:, 0], c='cyan', label='Planck')
        ax1.errorbar(colors, planck_kappas[:, 0], yerr=planck_kappas[:, 1], fmt='none', ecolor='cyan', alpha=0.3)
    if act_kappas is not None:
        act_kappas = np.array(act_kappas)
        ax1.scatter(colors, act_kappas[:, 0], c='pink', label='ACT')
        ax1.errorbar(colors, act_kappas[:, 0], yerr=act_kappas[:, 1], ecolor='pink', fmt='none', alpha=0.3)
        plt.legend()


    if transforms is not None:
        def forward(x):
            return np.interp(x, transforms[0], transforms[1])

        def inverse(x):
            return np.interp(x, transforms[1], transforms[0])

        secax = ax1.secondary_yaxis('right', functions=(forward, inverse))
        secax.yaxis.set_minor_locator(AutoMinorLocator())

        secax.set_ylabel(r'$\langle \mathrm{log}_{10}(M/h^{-1} M_{\odot})\rangle$', fontsize=20)

    plt.savefig('plots/kappa_v_%s.pdf' % mode)
    plt.close('all')



def plot_temp_v_color(colors, temps, errs, offset, remove_reddest=False):
    plt.figure(531, (8, 6))
    if not remove_reddest:
        lastidx = len(colors)-1
        redcolor = colors[lastidx]
        redtemp = temps[lastidx]
        rederr = errs[lastidx]

        plt.scatter(redcolor, redtemp, edgecolors='grey', marker='s', facecolors='none')
        plt.errorbar(redcolor, redtemp, yerr=rederr, fmt='none', ecolor='grey')

        colors = colors[:lastidx]
        temps = temps[:lastidx]
        errs = errs[:lastidx]



    plt.scatter(colors, temps, c='k')
    plt.errorbar(colors, temps, yerr=errs, fmt='none', ecolor='k')
    #if offset:
    plt.xlabel(r'$\langle \Delta (g-i) \rangle$', fontsize=20)
    #else:
        #plt.xlabel('$g-i$', fontsize=20)
    plt.ylabel('$T$', fontsize=20)
    plt.savefig('plots/temp_v_color.pdf')
    plt.close('all')

def plot_sed_v_color(seds, remove_reddest=False):
    plt.figure(531, (8, 6))
    lambdas = (10 ** (-7)) * np.array([3.551, 4.686, 6.166, 7.48, 8.932, 34, 46])#, 2.1e5])

    nus = 3E8 / lambdas
    i_band_med = np.median(seds[:,3])

    ratios = seds[:,3]/i_band_med

    for j in range(len(seds)):
        seds[j] = seds[j] * 1/ratios[j]
    if not remove_reddest:
        lastidx = len(seds)-1

        redsed = seds[lastidx]

        plt.scatter((10**6)*lambdas, nus*redsed, edgecolors='grey', marker='s', facecolors='none')

        seds = seds[:lastidx]

    plottingcolors = ['blue', 'royalblue', 'dodgerblue', 'cyan', 'aquamarine', 'forestgreen', 'orange',
                      'orangered', 'red', 'firebrick']

    for j in range(len(seds)):
        plt.plot((10**6)*lambdas, nus*seds[j], c=plottingcolors[j])


    #if offset:
    plt.xlabel(r'$\lambda_{obs}(\mu$m)', fontsize=20)
    #else:
        #plt.xlabel('$g-i$', fontsize=20)
    plt.ylabel(r'$\nu F_{\nu}$ (Jy Hz)', fontsize=20)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('plots/sed_v_color.pdf')
    plt.close('all')

#plot_ang_correlation_function()
#kappa_vs_mass()