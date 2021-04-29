import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.interpolate import UnivariateSpline, interp1d
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.stats import spearmanr, kendalltau, pearsonr

matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['legend.fontsize'] = 13   # could be smaller
matplotlib.rcParams['axes.titlesize'] = 20   # could be smaller

def limits(data, low=0.001, high=0.999, bins=200, **kwargs):
    if len(data) == 0:
        return [0, 0]
    if min(data) == max(data):
        return [0, 0]
    if 'weights' in kwargs:
        hist = np.histogram(data, bins=bins, range=[min(data), max(data)], weights=kwargs.get('weights'))[0]
    else:
        hist = np.histogram(data, bins=bins, range=[min(data), max(data)])[0]
    xrange = np.linspace(min(data), max(data), bins+1)
    xcentre = 0.5*(xrange[1:]+xrange[:-1])
    # print(hist, np.cumsum(hist))
    cumhist = np.cumsum(hist)/max(np.cumsum(hist))
    cumhist_int = interp1d(cumhist, xcentre, fill_value=min(data), bounds_error=False)
    [lim_min, lim_max] = cumhist_int([low, high])
    return [lim_min, lim_max]


def parse_limits(lim_input, data):
    if not isinstance(lim_input, str):
        real_lim = lim_input
    else:
        if '%' in lim_input:
            if lim_input.split('%')[-1] == '':
                low = float(lim_input.split('%')[0])
                percents = [low/100, 1-low/100]
            else:
                percents = [float(elem)/100 for elem in lim_input.split('%')]
        else:
            percents = [0.01, 0.99]

        if len(data) == 1:
            real_lim = limits(data[0], low=percents[0], high=percents[-1])
        else:
            real_lim = [limits(data[0], low=percents[0], high=percents[-1]),
                        limits(data[1], low=percents[0], high=percents[-1])]
    return real_lim


def density_plot(x, y, xlabel='x', ylabel='z', title='stellar disk', clabel='Number of points per pixel', name='xz.png',
                 figsize=(12, 8), cbar_flag=True, **kwargs):

    if ('clim' in kwargs) and ('c' in kwargs):
        climits = parse_limits(kwargs.get('clim'), [kwargs.get('c')])
    elif 'clim' in kwargs:
        climits = kwargs.get('clim')
    else:
        climits = [0, 750]

    if 'axlim' in kwargs:
        lim = parse_limits(kwargs.get('axlim'), [x, y])
        if np.shape(lim) == (2,):
            xlimits = lim
            ylimits = lim
        else:
            xlimits, ylimits = lim
    else:
        xlimits = limits(np.ravel(x))
        ylimits = limits(np.ravel(y))

    if 'stretch' in kwargs:
        if kwargs.get('stretch') == 'log':
            if climits[0] == 0:
                climits[0] += 1
            norm = colors.LogNorm(vmin=climits[0], vmax=climits[1])
        elif kwargs.get('stretch') == 'lin':
            norm = colors.Normalize(vmin=climits[0], vmax=climits[1])
    else:
        norm = colors.LogNorm(vmin=climits[0], vmax=climits[1])

    if 'cmap' in kwargs:
        cmap=plt.get_cmap(kwargs.get('cmap'))
    else:
        cmap=plt.get_cmap('Blues')

    if 'bins' in kwargs:
        bins = kwargs.get('bins')  # should be different at x and y?
    else:
        bins = 1024

    fig = plt.figure()
    fig.subplots_adjust(left=0.2)
    ax = fig.add_subplot(1, 1, 1)
    # fontsize = 14
    ax.set_xlabel(xlabel) #, fontsize=fontsize)
    ax.set_ylabel(ylabel) #, fontsize=fontsize)
    ax.set_title(title)
    # ax.set_axis_bgcolor('black')
    if 'c' in kwargs:
        if np.shape(x) == (6,):
            fig.clear()
            fig, ax = plt.subplots(nrows=1, ncols=5, figsize=figsize, sharex=True, sharey=True)
            ax[0].set_ylabel(ylabel) #, fontsize=fontsize)
            fig.suptitle(title)
            # hist_all = np.histogram2d(np.ravel(x), np.ravel(y), bins=bins, range=[[xlimits[0], xlimits[1]],[ylimits[0], ylimits[1]]])
            labels = ['t < 10', '10 < t < 8', '8 < t < 6', '6 < t < 4', ' 2 < t < 4', 't < 2']
            for xi, yi, cmapi, i in zip(x, y, [plt.get_cmap('Purples_r'), plt.get_cmap('Blues_r'), plt.get_cmap('Greens_r'), plt.get_cmap('Oranges_r'), plt.get_cmap('Reds_r')], range(5)):
                ax[i].set_xlabel(xlabel) #, fontsize=fontsize)
                hist_weights = np.histogram2d(xi, yi, bins=bins, range=[[xlimits[0], xlimits[1]], [ylimits[0], ylimits[1]]],
                                              weights=kwargs.get('c')[i])
                hist = np.histogram2d(xi, yi, bins=bins, range=[[xlimits[0], xlimits[1]], [ylimits[0], ylimits[1]]])
                # dens_all = ax[i].imshow(hist_all[0].T, origin='lower', extent=[xlimits[0], xlimits[1], ylimits[0], ylimits[1]], norm=norm, cmap=plt.get_cmap('bone_r'), aspect='auto', alpha=0.5)
                dens = ax[i].imshow((hist_weights[0]/hist[0]).T, origin='lower', extent=[xlimits[0], xlimits[1], ylimits[0], ylimits[1]], norm=norm, cmap=cmap, aspect='auto', label=labels[i])
                ax[i].legend()
        else:
            hist = np.histogram2d(x, y, bins=bins, range=[[xlimits[0], xlimits[1]],[ylimits[0], ylimits[1]]])
            hist_weights = np.histogram2d(x, y, bins=bins, range=[[xlimits[0], xlimits[1]], [ylimits[0], ylimits[1]]],
                                      weights=kwargs.get('c'))
            dens = ax.imshow((hist_weights[0]/hist[0]).T, origin='lower',
                             extent=[xlimits[0], xlimits[1], ylimits[0], ylimits[1]], cmap=cmap, norm=norm, aspect='auto')
    else:
        if np.shape(x) == (6,):
            fig.clear()
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize, sharex=True, sharey=True)
            ax = ax.ravel()
            ax[0].set_ylabel(ylabel) #, fontsize=fontsize)
            fig.suptitle(title)
            # hist_all = np.histogram2d(np.ravel(x), np.ravel(y), bins=bins, range=[[xlimits[0], xlimits[1]],[ylimits[0], ylimits[1]]])
            labels = ['t < 10', '10 < t < 8', '8 < t < 6', '6 < t < 4', ' 2 < t < 4', 't < 2']
            for xi, yi, cmapi, i in zip(x, y, [plt.get_cmap('Purples_r'), plt.get_cmap('Blues_r'), plt.get_cmap('Greens_r'), plt.get_cmap('Oranges_r'), plt.get_cmap('Reds_r')], range(5)):
                ax[i].set_xlabel(xlabel) #, fontsize=fontsize)
                hist = np.histogram2d(xi, yi, bins=bins, range=[[xlimits[0], xlimits[1]], [ylimits[0], ylimits[1]]])
                # dens_all = ax[i].imshow(hist_all[0].T, origin='lower', extent=[xlimits[0], xlimits[1], ylimits[0], ylimits[1]], norm=norm, cmap=plt.get_cmap('bone_r'), aspect='auto', alpha=0.5)
                dens = ax[i].imshow(hist[0].T, origin='lower', extent=[xlimits[0], xlimits[1], ylimits[0], ylimits[1]], norm=norm, cmap=cmap, aspect='auto', label=labels[i])
                ax[i].legend()
        else:
            hist = np.histogram2d(x, y, bins=bins, range=[[xlimits[0], xlimits[1]],[ylimits[0], ylimits[1]]])
            dens = ax.imshow(hist[0].T, origin='lower', extent=[xlimits[0], xlimits[1], ylimits[0], ylimits[1]], cmap=cmap,
                         norm=norm, aspect='auto')

    if 'overplot' in kwargs:
        plt.plot(np.linspace(0, 14, len(kwargs.get('overplot'))), kwargs.get('overplot'), color='k', lw=3., ls='dashed')
    # fig.colorbar(dens, label=clabel)
    if cbar_flag:
        cbar = plt.colorbar(dens, orientation='vertical', fraction=0.1, aspect=60, label=clabel)
        cbar.ax.tick_params(labelsize=8)
    fig.savefig(name, dpi=320) # , figsize=figsize)
    # fig.show()
    plt.close()


def density_mult_plot(x, y, xlabel='x', ylabel='z', title='stellar disk', clabel='Number of points per pixel', name='xz.png',
                 figsize=(12, 12), nrows=2, ncols=3, **kwargs):

    n_plot = np.shape(x)[0]

    if 'axlim' in kwargs:
        lim = parse_limits(kwargs.get('axlim'), [x, y])
        if np.shape(lim) == (2,):
            xlimits = lim
            ylimits = lim
        else:
            xlimits, ylimits = lim
    else:
        xlimits = limits(np.ravel(x))
        ylimits = limits(np.ravel(y))

    fontsize = 20
    if 'bins' in kwargs:
        bins = kwargs['bins']
    else:
        bins=1024
    cmap = plt.get_cmap('gist_earth')
    labels = ['t < 10', '10 < t < 8', '8 < t < 6', '6 < t < 4', ' 2 < t < 4', 't < 2']

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 12), sharex=True, sharey=True)

    fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=fontsize)
    fig.text(0.07, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=fontsize)

    fig.suptitle(title, fontsize=fontsize)

    for xi, yi, i in zip(x, y, range(n_plot)):
        hist = np.histogram2d(xi, yi, bins=bins, range=[[xlimits[0], xlimits[1]], [ylimits[0], ylimits[1]]])
        hist_plot = hist[0]
        if 'c' in kwargs:
            hist_weights = np.histogram2d(xi, yi, bins=bins, range=[[xlimits[0], xlimits[1]], [ylimits[0], ylimits[1]]],
                                          weights=kwargs['c'][i])
            hist_plot = hist_weights[0]/hist_plot
            # print(np.shape(kwargs.get('c')))
            # print(len(kwargs.get('c')[i]))
            climits = parse_limits(kwargs.get('clim'), [kwargs.get('c')[i]])
            norm = colors.Normalize(vmin=climits[0], vmax=climits[1])
        else:
            climits = kwargs['clim']
            norm = colors.LogNorm(vmin=climits[0], vmax=climits[1])
        dens = ax.ravel()[i].imshow(hist_plot.T, origin='lower',
                            extent=[xlimits[0], xlimits[1], ylimits[0], ylimits[1]], norm=norm, cmap=cmap,
                            aspect='auto', label=labels[i])
        ax.ravel()[i].tick_params(labelsize=15)
        if 'c' in kwargs:
            cbar = fig.colorbar(dens, ax=ax.ravel()[i], orientation='horizontal', pad=-0.2, shrink=0.5)
            cbar.ax.tick_params(labelsize=12) 
            cbar.set_ticks([climits[0], climits[1]])
            cbar.set_ticklabels([np.round(climits[0], 2), np.round(climits[1], 2)])
            cbar.ax.invert_xaxis()
        ax.ravel()[i].legend()
        fig.savefig(name, dpi=300, figsize=figsize)
        # fig.show()
        plt.close()


def ang_mom(r, v):
    '''
    :param r: vector of coordinates
    :param v: vector of velocities
    :return: vector of angular momentum
    '''
    jz = r[0]*v[1] - r[1]*v[0]
    jy = -r[0]*v[2] + r[2]*v[0]
    jx = r[1]*v[2] - r[2]*v[1]
    return [jx, jy, jz]


def ang_mom_circ(r, v, phi, rmax=200, bins=100):
    '''
    :param r: radius
    :param v: total velocity
    :param phi: potential
    :return: j_circ(E)
    '''
    r_phi = np.array(sorted([x for x in zip(r, phi)]))
    r_edges = np.linspace(0, rmax, bins+1)  # check what is max(r); should I apply limits?
    r_bin = 0.5*(r_edges[1:] + r_edges[:-1])
    phi_bin = np.zeros_like(r_bin)

    for i in range(len(r_bin)-1):
        idxs_bin = np.where((r_phi.T[0] > r_edges[i]) & (r_phi.T[0] < r_edges[i+1]))
        phi_bin[i] = np.mean(r_phi.T[1][idxs_bin])

    phi_spl = UnivariateSpline(r_bin, phi_bin)
    phi_der = phi_spl.derivative()

    # j(E) interpolation
    r = np.linspace(0, rmax, 1000)
    j_circ_E_spl = interp1d(0.5 * r * phi_der(r) + phi_spl(r), np.sqrt(r * phi_der(r)) * r,
                        fill_value=0, bounds_error=False)

    j_circ_E = j_circ_E_spl(0.5 * v ** 2 + phi)

    return j_circ_E


def calc_sfh(data, r, bins=100, rmin=0, rmax=14):
    redges = np.linspace(rmin, rmax, bins+1)
    dr = (redges[1] - redges[0])*1e9
    hist = np.zeros(bins)
    for i in range(bins):
        idxs = np.where((r > redges[i]) & (r < redges[i+1]))
        hist[i] = np.sum(data[idxs])/dr
    return [hist, 0.5*(redges[1:]+redges[:-1])]


def plot_sfh(data, r, xlabel, ylabel, title, name, stddev=10, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_axis_bgcolor('white')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if kwargs.get('convolve'):
        g = Gaussian1DKernel(stddev=stddev)
        z = convolve(data, g)
        ax.plot(r, data, color='navy', alpha=0.4)
        ax.plot(r, z, color='crimson')
    else:
        ax.plot(r, data, color='navy')
        ax.axvline(4., linestyle='-', linewidth=1., color='k')
        ax.axvline(4+r[1]-r[0], linestyle='-', linewidth=1., color='k')
    ax.grid(True)
    fig.savefig(name, dpi=300)
    # fig.show()
    plt.close()


def calc_conc(fe, ofe, age, dt=0.2, n_width=0.68, w_width=0.999, stddev=10, **kw):
    dt = dt

    g = Gaussian1DKernel(stddev=stddev)

    age_space = np.arange(0, 14, 2 * dt)
    age_space_cent = 0.5 * (age_space[1:] + age_space[:-1])
    width_fe_n = np.zeros_like(age_space_cent)
    width_fe_w = np.zeros_like(age_space_cent)
    width_ofe_n = np.zeros_like(age_space_cent)
    width_ofe_w = np.zeros_like(age_space_cent)
    conc_fe = np.zeros_like(age_space_cent)
    conc_ofe = np.zeros_like(age_space_cent)
    disp_sfh = np.zeros_like(age_space_cent)
    limits_fe_n = np.zeros((2, len(age_space_cent)))
    limits_fe_w = np.zeros((2, len(age_space_cent)))
    limits_ofe_n = np.zeros((2, len(age_space_cent)))
    limits_ofe_w = np.zeros((2, len(age_space_cent)))
    # limits_sfh = np.zeros((2, len(age_space_cent)))

    for t, i in zip(age_space_cent, range(len(age_space_cent))):
        idxs_age = np.where(abs(age - t) < dt)

        # limits_fe_n[:, i] = limits(fe[idxs_age], low=0.5-n_width/2., high=0.5+n_width/2., bins=500)[:]
        # limits_fe_w[:, i] = limits(fe[idxs_age], low=0.5-w_width/2., high=0.5+w_width/2., bins=500)[:]
        try:
            limits_fe_n[:, i] = np.percentile(fe[idxs_age], [50.-n_width*50., 50.+n_width*50.])
            limits_fe_w[:, i] = np.percentile(fe[idxs_age], [50.-w_width*50., 50.+w_width*50.])
        except:
            limits_fe_n[:, i] = [0., 0.]
            limits_fe_w[:, i] = [0., 0.]
        width_fe_n[i] = limits_fe_n[1, i] - limits_fe_n[0, i]
        width_fe_w[i] = limits_fe_w[1, i] - limits_fe_w[0, i]
        conc_fe[i] = width_fe_n[i]/width_fe_w[i]

        # limits_ofe_n[:, i] = limits(ofe[idxs_age], low=0.5-n_width/2., high=0.5+n_width/2., bins=500)[:]
        # limits_ofe_w[:, i] = limits(ofe[idxs_age], low=0.5-w_width/2., high=0.5+w_width/2., bins=500)[:]
        try:
            limits_ofe_n[:, i] = np.percentile(ofe[idxs_age], [50.-n_width*50., 50.+n_width*50.])
            limits_ofe_w[:, i] = np.percentile(ofe[idxs_age], [50.-w_width*50., 50.+w_width*50.])
        except:
            limits_ofe_n[:, i] = [0., 0.]
            limits_ofe_w[:, i] = [0., 0.]
        width_ofe_n[i] = limits_ofe_n[1, i] - limits_ofe_n[0, i]
        width_ofe_w[i] = limits_ofe_w[1, i] - limits_ofe_w[0, i]
        conc_ofe[i] = width_ofe_n[i]/width_ofe_w[i]

        if 'sfh' in kw:
            z = convolve(kw['sfh'], g)
            idxs_sfh = np.where(abs(kw['sfh_age'] - t) < dt)
        # limits_sfh[:, i] = limits(sfh[idxs_sfh], low=0.1, high=0.9, bins=500)[:]
            disp_sfh[i] = np.std(kw['sfh'][idxs_sfh]-z[idxs_sfh])  #/np.mean(z[idxs_sfh])

    if 'sfh' in kw:
        return conc_fe, conc_ofe, disp_sfh, z, limits_fe_n[1,:], (limits_ofe_n[0,:]+limits_ofe_n[1,:])/2., width_fe_w, width_ofe_w

    else: 
        return conc_fe, conc_ofe, limits_fe_n[1,:], (limits_ofe_n[0,:]+limits_ofe_n[1,:])/2., width_fe_w, width_ofe_w


    # plt.figure()
    # plt.plot(age_space_cent, limits_fe_n[0,:], label='n0')
    # plt.plot(age_space_cent, limits_fe_n[1,:], label='n1')
    # plt.plot(age_space_cent, limits_fe_w[0,:], label='w0')
    # plt.plot(age_space_cent, limits_fe_w[1,:], label='w1')
    # plt.plot(age_space_cent, (limits_fe_w[0,:]+limits_fe_w[1,:])/2., label='mean w')
    # plt.legend()
    # plt.savefig('/bighome/hparul/m12m/check_w999_FeH_{}.png'.format(dt))


    # return conc_fe, conc_ofe, disp_sfh, z, limits_fe_n[1,:], (limits_ofe_n[0,:]+limits_ofe_n[1,:])/2., width_fe_w, width_ofe_w


def plot_conc_mult(disp_sfh, conc_fe, conc_ofe, name, title, **kwargs):
    fig, ax = plt.subplots(3, 1, sharex=True)
    if 'ylabel' in kwargs:
        ylabel = kwargs.get('ylabel')
    else:
        ylabel = ['$std(SFH)$', '$w_{68}/w_{95} [Fe/H]$', '$w_{68}/w_{95} [O/Fe]$']
    age_space_cent = np.linspace(0, 14, len(conc_fe))
    if 'sfh' in kwargs:
        sfh_age = np.linspace(0, 14, len(kwargs.get('sfh')))
        ax[0].plot(sfh_age, kwargs.get('sfh'), color='navy', alpha=0.3)
        ax[0].set_ylabel(ylabel[0])
    elif 'sfh_conv' in kwargs:
        sfh_age = np.linspace(0, 14, len(kwargs.get('sfh_conv')))
        ax[0].plot(sfh_age, kwargs.get('sfh_conv'), color='crimson')
        ax[0].set_ylabel(ylabel[0])
    else:
        sfh_age = np.linspace(0, 14, len(disp_sfh))
        ax[0].plot(sfh_age, disp_sfh)
    ax[1].plot(age_space_cent, conc_fe)
    ax[2].plot(age_space_cent, conc_ofe)
    ax[2].set_xlabel('$age$')
    ax[0].set_ylabel(ylabel[0])
    ax[1].set_ylabel(ylabel[1])
    ax[2].set_ylabel(ylabel[2])
    ax[0].set_title(title)
    fig.savefig(name, dpi=300)
    plt.close()


def plot_corr_mult(disp_sfh, conc_fe, conc_ofe, name, title, **kwargs):
    if 'ylabel' in kwargs:
        ylabel = kwargs.get('ylabel')
    else:
        ylabel = ['$w_{68}/w_{95} [Fe/H]$', '$w_{68}/w_{95} [O/Fe]$']
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].scatter(disp_sfh, conc_fe, color='crimson', s=3)
    ax[0].set_ylabel(ylabel[0])
    ax[0].set_title(title)
    ax[1].scatter(disp_sfh, conc_ofe, color='crimson', s=3)
    ax[1].set_ylabel(ylabel[1])
    ax[1].set_xlabel('$std(SFH)$')
    fig.savefig(name, dpi=300)
    plt.close()


def plot_corr_mult_color(disp_sfh, conc_fe, conc_ofe, name, title, ylim, xlim, clabel='age', **kwargs):
    age_space_cent = np.linspace(0, 14, np.shape(conc_fe.T)[0])
    if 'ylabel' in kwargs:
        ylabel = kwargs.get('ylabel')
    else:
        ylabel = ['$w_{68}/w_{95} [Fe/H]$', '$w_{68}/w_{95} [O/Fe]$']


    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
    else:
        xlabel = '$\\sigma_{SFH}$'


    if 'c' in kwargs:
        carray = kwargs.get('c')
    else:
        carray = age_space_cent


    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8.5, 5), sharex=True)

    fe = axes.flat[0].scatter(disp_sfh, conc_fe, s=10, c=carray, cmap=plt.get_cmap('gist_rainbow'), linewidth=0, edgecolors='None')
    ofe = axes.flat[1].scatter(disp_sfh, conc_ofe, s=10, c=carray, cmap=plt.get_cmap('gist_rainbow'), linewidth=0, edgecolors='None')

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.05, hspace=0.05)

    axes.flat[0].set_ylabel(ylabel[0])
    axes.flat[0].set_title(title)
    axes.flat[1].set_ylabel(ylabel[1])
    axes.flat[1].set_xlabel(xlabel)
    axes.flat[1].set_xlim(xlim)

    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(fe, cax=cb_ax, label=clabel)

    fig.savefig(name, dpi=300)
    plt.close()


def plot_conc_3(disp_sfh, conc_fe, conc_ofe, name):
    fig, ax = plt.subplots(3, 1, sharex=True)
    age_space_cent = np.linspace(0, 14, np.shape(conc_fe.T)[0])
    sfh_age = np.linspace(0, 14, np.shape(disp_sfh.T)[0])

    ax[0].plot(sfh_age, disp_sfh[0], color='crimson', label='m12f')
    ax[0].plot(sfh_age, disp_sfh[1], color='mediumblue', label='m12i')
    ax[0].plot(sfh_age, disp_sfh[2], color='springgreen', label='m12m')

    ax[1].plot(age_space_cent, conc_fe[0], color='crimson', label='m12f')
    ax[1].plot(age_space_cent, conc_fe[1], color='mediumblue', label='m12i')
    ax[1].plot(age_space_cent, conc_fe[2], color='springgreen', label='m12m')

    ax[2].plot(age_space_cent, conc_ofe[0], color='crimson', label='m12f')
    ax[2].plot(age_space_cent, conc_ofe[1], color='mediumblue', label='m12i')
    ax[2].plot(age_space_cent, conc_ofe[2], color='springgreen', label='m12m')

    ax[2].set_xlabel('$age$')
    ax[0].set_ylabel('$std(SFH)$')
    ax[1].set_ylabel('$w_{68}/w_{95} [Fe/H]$')
    ax[2].set_ylabel('$w_{68}/w_{95} [O/Fe]$')
    ax[0].legend()
    fig.savefig(name, dpi=300)
    plt.close()


def plot_corr_3(disp_sfh, conc_fe, conc_ofe, name, clabel='age', **kwargs):
    fig, ax = plt.subplots(2, 1, sharex=True)
    if 'c' in kwargs:
        colors = kwargs.get('c')
    else:
        colors = ['crimson', 'mediumblue', 'springgreen']

    ax[0].scatter(disp_sfh[0], conc_fe[0], color=colors[0], label='m12f', s=3)
    ax[0].scatter(disp_sfh[1], conc_fe[1], color=colors[1], label='m12i', s=3)
    ax[0].scatter(disp_sfh[2], conc_fe[2], color=colors[2], label='m12m', s=3)
    ax[0].set_ylabel('$w_{68}/w_{95} [Fe/H]$')
    ax[0].legend()
    ax[1].scatter(disp_sfh[0], conc_ofe[0], color=colors[0], label='m12f', s=3)
    ax[1].scatter(disp_sfh[1], conc_ofe[1], color=colors[1], label='m12i', s=3)
    ax[1].scatter(disp_sfh[2], conc_ofe[2], color=colors[2], label='m12m', s=3)
    ax[1].set_ylabel('$w_{68}/w_{95} [O/Fe]$')
    ax[1].set_xlabel('$std(SFH)$')

    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(fe, cax=cb_ax, label=clabel)

    fig.savefig(name, dpi=300)
    plt.close()


# def plot_corr_3_color(disp_sfh, conc_fe, conc_ofe, name):
#     age_space_cent = np.linspace(0, 14, np.shape(conc_fe.T)[0])
#     fig, ax = plt.subplots(2, 1, sharex=True)
#     ax[0].scatter(disp_sfh[0], conc_fe[0], label='m12f', s=3, c=age_space_cent, cmap=plt.get_cmap('viridis'))
#     ax[0].scatter(disp_sfh[1], conc_fe[1], label='m12i', s=3, c=age_space_cent, cmap=plt.get_cmap('plasma'))
#     ax[0].scatter(disp_sfh[2], conc_fe[2], label='m12m', s=3, c=age_space_cent, cmap=plt.get_cmap('inferno'))
#     ax[0].set_ylabel('$w_{68}/w_{95} [Fe/H]$')
#     ax[0].legend()
#     ax[1].scatter(disp_sfh[0], conc_ofe[0], label='m12f', s=3, cmap=plt.get_cmap('viridis'))
#     ax[1].scatter(disp_sfh[1], conc_ofe[1], label='m12i', s=3, cmap=plt.get_cmap('plasma'))
#     ax[1].scatter(disp_sfh[2], conc_ofe[2], label='m12m', s=3, cmap=plt.get_cmap('inferno'))
#     ax[1].set_ylabel('$w_{68}/w_{95} [O/Fe]$')
#     ax[1].set_xlabel('$std(SFH)$')
#     fig.savefig(name, dpi=300)
#     plt.close()


def shift(disp_sfh, conc_fe, conc_ofe, name, shift_len=21, name_order=['f', 'i', 'm'], return_shift=False, image=True):
    age_space_cent = np.linspace(0, 14, np.shape(conc_fe.T)[0])

    if disp_sfh.ndim > 1:
        shift_fe = [[] for i in range(np.shape(disp_sfh)[0])]
        shift_ofe = [[] for i in range(np.shape(disp_sfh)[0])]
    else:
        shift_fe = [[]]
        shift_ofe = [[]]

    colors = ['crimson', 'mediumblue', 'springgreen']
    length = shift_len

    for k in range(np.shape(shift_fe)[0]):
        shift_fe[k].append(spearmanr(disp_sfh[k][np.where(~np.isnan(conc_fe[k]))[0][:]], conc_fe[k][np.where(~np.isnan(conc_fe[k]))[0][:]])[0])
        shift_ofe[k].append(spearmanr(disp_sfh[k][np.where(~np.isnan(conc_ofe[k]))[0][:]], conc_ofe[k][np.where(~np.isnan(conc_ofe[k]))[0][:]])[0])
        for i in range(1, length):
            shift_fe[k].append(spearmanr(disp_sfh[k][np.where(~np.isnan(conc_fe[k]))[0][i:]], conc_fe[k][np.where(~np.isnan(conc_fe[k]))[0][:-i]])[0])
            shift_ofe[k].append(spearmanr(disp_sfh[k][np.where(~np.isnan(conc_ofe[k]))[0][i:]], conc_ofe[k][np.where(~np.isnan(conc_ofe[k]))[0][:-i]])[0])
   
    dt = age_space_cent[1]-age_space_cent[0]

    if image:
        fig, ax = plt.subplots(2, 1, sharex=True)
        for i in range(np.shape(shift_fe)[0]):
            ax[0].scatter(range(length)*dt, shift_fe[i], color=colors[i], label='m12{}'.format(name_order[i]), s=3)
            ax[1].scatter(range(length)*dt, shift_ofe[i], color=colors[i], label='m12f{}'.format(name_order[i]), s=3)
        ax[0].set_ylabel('$r_s, [Fe/H]$')
        ax[0].set_xlim(-dt, dt*shift_len)
        ax[0].legend()
        ax[1].set_ylabel('$r_s [O/Fe]$')
        ax[1].set_xlabel('$time shift$')
        ax[1].set_xlim(-dt, dt*shift_len)
        fig.savefig(name, dpi=300)
        plt.close()

    if return_shift:
        return shift_fe, shift_ofe