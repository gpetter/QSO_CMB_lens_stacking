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
import fitting
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



def plot_stacks(samplename):
    bluestack = np.load('stacks/%s_stack0.npy' % samplename, allow_pickle=True)*1000
    ctrlstack = np.load('stacks/%s_stack5.npy' % samplename, allow_pickle=True)*1000
    redstack = np.load('stacks/%s_stack9.npy' % samplename, allow_pickle=True)*1000
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



def plot_kappa_profile(color, kap_profile, kap_errs, binsize, maxtheta, best_mass_profile, binned_model, oneterm, twoterm):
    if color == 'blue':
        ckey = 'b'
    elif color == 'red':
        ckey = 'r'
    else:
        ckey = 'g'
    obs_theta = np.arange(binsize / 2, maxtheta, binsize)

    theta_range = np.arange(0.5, maxtheta, 0.5)
    plt.figure(0, (10, 8))
    plt.scatter(obs_theta, kap_profile, c=ckey)
    plt.errorbar(obs_theta, kap_profile, yerr=kap_errs, c=ckey, fmt='none')
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
    plt.savefig('plots/profile_%s.png' % color)
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
    plt.xlabel('$z$', fontsize=30)
    ax.tick_params(axis='both', labelsize=20)
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

def g_minus_i_plot(qso_cat_name, offset):
    plt.close('all')
    allzs = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data['Z']
    czs = fits.open('catalogs/derived/%s_ctrl.fits' % qso_cat_name)[1].data['Z']
    redzs = fits.open('catalogs/derived/%s_red.fits' % qso_cat_name)[1].data['Z']
    bluezs = fits.open('catalogs/derived/%s_blue.fits' % qso_cat_name)[1].data['Z']
    hists = True
    fig = plt.figure(12351, (10, 9))

    offset=True


    if offset:
        allcolors = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data['deltagmini']
        ccolors = fits.open('catalogs/derived/%s_ctrl.fits' % qso_cat_name)[1].data['deltagmini']
        redcolors = fits.open('catalogs/derived/%s_red.fits' % qso_cat_name)[1].data['deltagmini']
        bluecolors = fits.open('catalogs/derived/%s_blue.fits' % qso_cat_name)[1].data['deltagmini']

    else:
        allcolors = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data['g-i']
        ccolors = fits.open('catalogs/derived/%s_ctrl.fits' % qso_cat_name)[1].data['g-i']
        redcolors = fits.open('catalogs/derived/%s_red.fits' % qso_cat_name)[1].data['g-i']
        bluecolors = fits.open('catalogs/derived/%s_blue.fits' % qso_cat_name)[1].data['g-i']


    if hists:
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
    ax.set_xlabel('$z$', fontsize=30)


    """zs = np.linspace(0.5, 2.5, 100)
    modcols = []
    for z in zs:
        modcols.append(spectrumtools.vdb_color_at_z(z))
    plt.plot(zs, modcols, c='k')"""

    #plt.scatter(zs, gminusi, c='k', s=0.01, alpha=0.5)
    #plt.scatter(bluezs, bluegs, c='b', s=0.05, rasterized=True)
    #plt.scatter(czs, cgs, c='g', s=0.05, rasterized=True)
    #plt.scatter(redzs, redgs, c='r', s=0.05, rasterized=True)
    ax.set_ylim(-1, 1.5)
    plt.savefig('plots/%s_gminusi.pdf' % qso_cat_name)
    plt.close('all')

def color_hist(qso_cat_name, offset=True):

    plt.figure(5946, (8,8))

    if offset:
        allcolors = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data['deltagmini']
        plt.xlabel('$\Delta (g - i)$', fontsize=30)
    else:
        allcolors = fits.open('catalogs/derived/%s_complete.fits' % qso_cat_name)[1].data['g-i']
        plt.xlabel('$g - i$', fontsize=30)
    plt.ylabel('N', fontsize=30)

    nbins=1000

    plt.hist(allcolors, bins=nbins, histtype='step', color='k')
    xcolors = np.linspace(-1, 1, 1000)
    gaussfit = fitting.fit_gauss_hist_one_sided(allcolors, xcolors, nbins)
    plt.plot(xcolors, gaussfit, c='grey')

    #plt.yscale('log')
    plt.axvline(0.25, c='k', ls='--')
    plt.xlim(-1, 2)
    plt.savefig('plots/colorhist.pdf')
    plt.close('all')

#color_hist('xdqso_specz', offset=True)

def z_dists(qso_cat_name, bzs, czs, rzs):
    plt.close('all')
    plt.figure(747, (8, 6))
    plt.hist(rzs, color='r', alpha=0.8, density=True, bins=20, histtype='step', linewidth=2)
    plt.hist(bzs, color='b', alpha=0.8, density=True, bins=20, histtype='step', linewidth=2)
    plt.hist(czs, color='g', alpha=0.8, density=True, bins=20, histtype='step', linewidth=2)
    plt.xlabel('$z$', fontsize=20)
    plt.ylabel('$N$', fontsize=20)
    plt.savefig('plots/%s_zdists.pdf' % qso_cat_name)
    plt.close('all')

def lum_dists(qso_cat_name, lumhistbins, blueLs, cLs, redLs, tailLs=None):
    plt.close('all')
    plt.figure(4, (10, 10))
    plt.hist(blueLs, color='b', alpha=0.8, density=True, bins=lumhistbins, range=(21, 26), histtype='step', linewidth=2)
    plt.hist(cLs, color='g', alpha=0.8, density=True, bins=lumhistbins, range=(21, 26), histtype='step', linewidth=2)
    plt.hist(redLs, color='r', alpha=0.8, density=True, bins=lumhistbins, range=(21, 26), histtype='step', linewidth=2)
    if tailLs is not None:
        plt.hist(tailLs, color='grey', alpha=0.5, density=True, bins=lumhistbins, range=(21,26), histtype='step', linewidth=2)
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

def plot_ang_correlation_function(sample_id, nbins):
    bins = np.logspace(-2, 1, nbins)
    redw, redw_err = np.load('clustering/%s_red.npy' % sample_id, allow_pickle=True)
    bluew, bluew_err = np.load('clustering/%s_blue.npy' % sample_id, allow_pickle=True)
    ctrlw, ctrl_err = np.load('clustering/%s_ctrl.npy' % sample_id, allow_pickle=True)


    plt.figure(424, (8, 6))
    plt.scatter(bins, redw, c='r')
    plt.errorbar(bins, redw, yerr=redw_err, fmt='none', ecolor='r')
    plt.scatter(bins, bluew, c='b')
    plt.errorbar(bins, bluew, yerr=bluew_err, fmt='none', ecolor='b')
    plt.scatter(bins, ctrlw, c='g')
    plt.errorbar(bins, ctrlw, yerr=ctrl_err, fmt='none', ecolor='g')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\theta$ (degrees)', fontsize=20)
    plt.ylabel(r'$w(\theta)$', fontsize=20)
    plt.savefig('plots/ang_clustering.pdf')
    plt.close('all')

def plot_ang_cross_corr(sample_id, nbins):
    plt.close('all')
    bins = np.logspace(-2, -0.5, nbins)

    redw, redw_err = np.load('clustering/%s_cross_red.npy' % sample_id, allow_pickle=True)
    bluew, bluew_err = np.load('clustering/%s_cross_blue.npy' % sample_id, allow_pickle=True)
    plt.figure(929, (8, 6))

    plt.scatter(bins, redw, c='r')
    plt.errorbar(bins, redw, yerr=redw_err, fmt='none', ecolor='r')
    plt.scatter(bins, bluew, c='b')
    plt.errorbar(bins, bluew, yerr=bluew_err, fmt='none', ecolor='b')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\theta$ (degrees)', fontsize=20)
    plt.ylabel(r'$w(\theta)$', fontsize=20)


    plt.savefig('plots/ang_cross_corr.pdf')
    plt.close('all')

def plot_spatial_correlation_function(bins):
    import autocorrelation
    redw, redw_err = np.load('clustering/eboss_lss_red.npy', allow_pickle=True)
    bluew, bluew_err = np.load('clustering/eboss_lss_blue.npy', allow_pickle=True)
    ctrlw, ctrlw_err = np.load('clustering/eboss_lss_ctrl.npy', allow_pickle=True)

    plt.figure(91, (8, 6))
    plt.scatter(bins, bluew, c='b')
    plt.errorbar(bins, bluew, yerr=bluew_err, fmt='none', ecolor='b', elinewidth=1, capsize=3)

    plt.scatter(bins, redw, c='r')
    plt.errorbar(bins, redw, yerr=redw_err, fmt='none', ecolor='r', elinewidth=1, capsize=3)

    plt.scatter(bins, ctrlw, c='g')
    plt.errorbar(bins, ctrlw, yerr=ctrlw_err, fmt='none', ecolor='g', elinewidth=1, capsize=3)
    plt.plot(bins, 6 * autocorrelation.theory_projected_corr_func(bins), c='k', label='Dark Matter (b=2.45 at z=1.5)')
    plt.legend()

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$r$ (Mpc)', fontsize=20)
    plt.ylabel(r'$w_{p}(r_{p})/r_{p}$', fontsize=20)
    plt.savefig('plots/spatial_clustering.pdf')
    plt.close('all')


def plot_median_radio_flux(remove_reddest=False):
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

    plt.xlabel(r'$\langle \Delta (g-i) \rangle$', fontsize=20)

    plt.savefig('plots/median_flux_by_color.pdf')
    plt.close('all')

def plot_median_radio_luminosity(remove_reddest=False):
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
    plt.ylabel(r'$L_{1.4 \mathrm{GHz}}$ (W Hz$^{-1}$)', fontsize=20)
    plt.axhline(1.57e23, linestyle='--', c='k')
    plt.text(0.6, 1.5e23, 'SFR $ = 100 M_{\odot} yr ^ {-1}$')

    plt.xlabel(r'$\langle \Delta (g-i) \rangle$', fontsize=20)

    plt.savefig('plots/median_lum_by_color.pdf')
    plt.close('all')




def plot_radio_loudness(remove_reddest=False):
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


    plt.xlabel(r'$\langle \Delta (g-i) \rangle$', fontsize=30)

    plt.scatter(colors, loudness, c='k')
    if louderrs is not None:
        plt.errorbar(colors, loudness, yerr=louderrs, c='k', fmt='none')
    plt.ylabel(r'$R = \mathrm{log}_{10}(L_{1.4 \mathrm{GHz}}/L_{\mathrm{bol}})$', fontsize=30)
    plt.savefig('plots/radio_loudness_by_color.pdf')
    plt.close('all')

def plot_kappa_v_color(colors, kappas, errs, offset, remove_reddest=False, linfit=None):
    plt.figure(531, (8, 6))
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


    linmod = linfit[0]*np.array(colors) + linfit[1]


    plt.scatter(colors, kappas, c='k')
    plt.errorbar(colors, kappas, yerr=errs, fmt='none', ecolor='k')
    if linfit is not None:
        plt.plot(colors, linmod, c='k', linestyle='--')
    #if offset:
    plt.xlabel(r'$\langle \Delta (g-i) \rangle$', fontsize=20)
    #else:
        #plt.xlabel('$g-i$', fontsize=20)
    plt.ylabel('$\kappa_{\mathrm{peak}}$', fontsize=20)
    plt.savefig('plots/kappa_v_color.pdf')
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