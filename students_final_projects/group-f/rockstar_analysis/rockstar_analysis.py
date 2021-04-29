'''
Analyze Rockstar halo catalog.

@author: Andrew Wetzel <arwetzel@gmail.com>

Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    position [kpc comoving]
    distance, radius [kpc physical]
    velocity [km / s]
    time [Gyr]
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import numpy as np
from numpy import Inf
from matplotlib import pyplot as plt
# local ----
import utilities as ut


#===================================================================================================
# utility
#===================================================================================================
def print_properties(hal, hal_indices, properties=None, digits=3):
    '''
    Print useful properties of halo[s].

    Parameters
    ----------
    hal : dict : catalog of halos
    hal_indices : int or array : index[s] of halo[s]
    properties : str or list : name[s] of properties to print
    digits : int : number of digits after period
    '''
    Say = ut.io.SayClass(print_properties)

    hal_indices = ut.array.arrayize(hal_indices)

    if properties:
        # print input properties
        if properties == 'default':
            properties = [
                'id', 'mass', 'mass.vir', 'mass.200c', 'vel.circ.max',
                'spin.bullock', 'spin.peebles', 'position',
            ]

        for hi in hal_indices:
            print('halo index = {}'.format(hi))
            for prop in properties:
                string = ut.io.get_string_from_numbers(hal[prop][hi], digits)
                print('{} = {}'.format(prop, string))
            print()

    else:
        # print detailed (galaxy) properties
        for hi in hal_indices:
            Say.say('host distance = {:.1f} kpc'.format(hal.prop('host.distance.total', hi)))
            print()

            Say.say('halo:')
            Say.say('  M_total = {} Msun'.format(
                ut.io.get_string_from_numbers(hal.prop('mass', hi), 2)))
            Say.say('  M_bound/M_total = {}'.format(
                ut.io.get_string_from_numbers(hal.prop('mass.bound/mass', hi), 3)))
            Say.say('  V_circ,max = {} km/s'.format(
                ut.io.get_string_from_numbers(hal.prop('vel.circ.max', hi), 1)))
            Say.say('  V_std = {} km/s'.format(
                ut.io.get_string_from_numbers(hal.prop('vel.std', hi), 1)))
            Say.say('  R_halo = {} kpc'.format(
                ut.io.get_string_from_numbers(hal.prop('radius', hi), 1)))
            print()

            if 'star.mass' in hal and np.nanmax(hal['star.mass']) > 0:
                Say.say('star:')
                Say.say('  N_star = {:d}'.format(hal.prop('star.number', hi)))
                Say.say('  M_star = {} M_sun'.format(
                    ut.io.get_string_from_numbers(hal.prop('star.mass', hi), 2)))
                #Say.say('star mass: rockstar = {:.2e}, mine = {:.2e}, ratio = {:.2f}'.format(
                #        hal.prop('star.mass.rockstar', hi), hal.prop('star.mass', hi),
                #        hal.prop('star.mass/star.mass.rockstar', hi)))
                Say.say('  M_star/M_bound = {}'.format(
                    ut.io.get_string_from_numbers(hal.prop('star.mass/mass.bound', hi), 4)))
                Say.say('  R_50 = {}, R_90 = {} kpc'.format(
                    ut.io.get_string_from_numbers(hal.prop('star.radius.50', hi), 2),
                    ut.io.get_string_from_numbers(hal.prop('star.radius.90', hi), 2)))
                Say.say('  density(R_50) = {} M_sun/kpc^3'.format(
                    ut.io.get_string_from_numbers(
                        hal.prop('star.density.50', hi), 2, exponential=True)))
                Say.say('  V_std = {}, V_std(R_50) = {} km/s'.format(
                    ut.io.get_string_from_numbers(hal.prop('star.vel.std', hi), 1),
                    ut.io.get_string_from_numbers(hal.prop('star.vel.std.50', hi), 1)))

                print()

                Say.say(
                    '  age: 50% = {:.2f}, 100% = {:.2f}, 68% dif = {:.3f} Gyr'.format(
                        hal.prop('star.form.time.50.lookback', hi),
                        hal.prop('star.form.time.100.lookback', hi),
                        hal.prop('star.form.time.dif.68', hi)))

                try:
                    Say.say('  metallicity total = {:.3f}, [Fe/H] = {:.3f}'.format(
                            hal.prop('star.metallicity.total', hi),
                            hal.prop('star.metallicity.iron', hi)))
                except KeyError:
                    Say.say('  metallicity total = {:.3f},'.format(
                            hal.prop('star.metallicity.total', hi)))

                print()

                Say.say('star v dark:')
                distance = ut.coordinate.get_distances(
                    hal.prop('star.position', hi), hal.prop('position', hi),
                    hal.info['box.length'], total_distance=True)
                Say.say('  position offset = {:.0f} pc, {:0.2f} R_50'.format(
                    distance * 1000, distance / hal.prop('star.radius.50', hi)))
                velocity = ut.coordinate.get_velocity_differences(
                    hal.prop('star.velocity', hi), hal.prop('velocity', hi),
                    hal.prop('star.position', hi), hal.prop('position', hi), hal.info['box.length'],
                    hal.snapshot['scalefactor'], hal.snapshot['time.hubble'],
                    total_velocity=True)
                Say.say('  velocity offset = {:.1f} km/s, {:0.2f} V_std(R_50)'.format(
                    velocity, velocity / hal.prop('star.vel.std.50', hi)))
                print()

            if 'gas.mass' in hal and np.max(hal['gas.mass']) > 0:
                try:
                    Say.say('gas mass: rockstar = {:.2e}, mine = {:.2e}, ratio = {:.2f}'.format(
                            hal.prop('gas.mass.rockstar')[hi], hal.prop('gas.mass')[hi],
                            hal.prop('gas.mass/gas.mass.rockstar')[hi]))
                except KeyError:
                    Say.say('gas mass = {:.3e}'.format(hal.prop('gas.mass', hi)))

                Say.say('gas/star mass = {:.3f}'.format(hal.prop('gas.mass/star.mass', hi)))

                try:
                    Say.say('neutral hydrogen: mass = {:.3e}, gas/star mass = {:.3f}'.format(
                            hal.prop('gas.mass.neutral', hi),
                            hal.prop('gas.mass.neutral/star.mass', hi)))
                except KeyError:
                    pass

            #Say.say('position = {:.2f}, {:.2f}, {:.2f} kpc'.format(
            #        hal.prop('star.position')[hi, 0], hal.prop('star.position')[hi, 1],
            #        hal.prop('star.position')[hi, 2]))
            print()


#===================================================================================================
# diagnostic
#===================================================================================================
def get_indices_diffuse(
    hals, star_mass_limits=[1e5, 1e6], star_radius_limits=[2, Inf], star_vel_std_limits=[3, 15]):
    '''
    .
    '''
    Say = ut.io.SayClass(get_indices_diffuse)

    if isinstance(hals, dict):
        hals = [hals]

    for hal_i, hal in enumerate(hals):
        his = hal.get_indices(hal)
        his = ut.array.get_indices(hal['star.mass'], star_mass_limits, his)
        his = ut.array.get_indices(hal['star.radius.50'], star_radius_limits, his)
        his = ut.array.get_indices(hal['star.vel.std.50'], star_vel_std_limits, his)
        if his.size:
            Say.say('catalog {}: {}'.format(hal_i, hal.info['simulation.name']))
            for hi in his:
                string = 'index {} | star N {:3d} | star M {:.1e} | R_50 {:.1f} | sigma {:.1f}'
                Say.say(string.format(
                    hi, hal['star.number'][hi], hal['star.mass'][hi], hal['star.radius.50'][hi],
                    hal['star.vel.std.50'][hi]))
            print()


def test_host_catalog_v_tree(hals, halt, host_rank=0):
    '''
    Test differences in primary host assignment between halo catalogs and halo merger trees.
    '''
    host_name = ut.catalog.get_host_name(host_rank)

    for hal in hals:
        if len(hal) and len(hal['mass']):
            # get real (non-phantom) halos at this snapshot in trees
            halt_indices = np.where(
                (halt['am.phantom'] == 0) * (halt['snapshot'] == hal.snapshot['index']))[0]
            if halt_indices.size:
                halt_host_index = halt[host_name + 'index'][halt_indices[0]]
                hal_host_index_halt = halt['catalog.index'][halt_host_index]
                hal_host_index = hal[host_name + 'index'][0]
                if hal_host_index_halt != hal_host_index:
                    print('snapshot {}: {} has {:.3e}, {} has {:.3e}'.format(
                        hal.snapshot['index'],
                        halt_host_index, halt['mass'][halt_host_index],
                        hal_host_index, hal['mass'][hal_host_index]))


def test_halo_jump(
    halt, jump_prop_name='position', jump_prop_value=100,
    select_prop_name='vel.circ.max', select_prop_limits=[]):
    '''
    Test jumps in halo properties across adjacent snapshots in halo merger trees.
    '''
    Say = ut.io.SayClass(test_halo_jump)

    snapshot_index_max = halt['snapshot'].max()
    snapshot_index_min = halt['snapshot'].min()
    snapshot_indices = np.arange(snapshot_index_min + 1, snapshot_index_max + 1)

    Say.say('halo tree progenitor -> child {} jump > {}'.format(jump_prop_name, jump_prop_value))

    jump_number = 0
    total_number = 0

    for snapshot_index in snapshot_indices:
        hindices = np.where(halt['snapshot'] == snapshot_index)[0]
        if select_prop_name and select_prop_limits is not None and len(select_prop_limits):
            hindices = ut.array.get_indices(halt[select_prop_name], select_prop_limits, hindices)
        desc_hindices = hindices[np.where(halt['progenitor.main.index'][hindices] >= 0)[0]]
        prog_hindices = halt['progenitor.main.index'][desc_hindices]

        if jump_prop_name == 'position':
            # position difference [kpc comoving]
            position_difs = ut.coordinate.get_distances(
                halt['position'][prog_hindices], halt['position'][desc_hindices],
                halt.info['box.length'], total_distance=True)
            hiis_jump = np.where(position_difs > jump_prop_value)[0]
        elif jump_prop_name in ['vel.circ.max', 'mass']:
            # V_c,max or mass jump
            prop_ratios = halt[jump_prop_name][desc_hindices] / halt[jump_prop_name][prog_hindices]
            hiis_jump = np.where(prop_ratios > jump_prop_value)[0]
            #hiis_jump = np.where(velcircmax_ratios < 1 / jump_value)[0]
            #hiis_jump = np.logical_or(
            #    velcircmax_ratios > jump_value, velcircmax_ratios < (1 / jump_value))

        total_number += prog_hindices.size

        if hiis_jump.size:
            Say.say('snapshot {:3d} to {:3d}:  {} (of {})'.format(
                snapshot_index, snapshot_index + 1, hiis_jump.size, desc_hindices.size))

            jump_number += hiis_jump.size

            #print(desc_hindices[hiis_jump])

            # check how many descendants skip a snapshot
            prog_snapshot_indices = halt['snapshot'][prog_hindices[hiis_jump]]
            hiis_skip_snapshot = np.where(prog_snapshot_indices != snapshot_index - 1)[0]
            if hiis_skip_snapshot.size:
                Say.say('  {} descendants skip a snapshot'.format(hiis_skip_snapshot.size))

        Say.say('across all snapshots:  {} (of {})'.format(jump_number, total_number))


def test_halo_lowres_mass(hals, lowres_mass_name='dark2.mass', lowres_mass_frac_max=8):
    '''
    Test low-res dark2 mass assigned to halos in star_*.hdf5.
    '''
    for snapshot_i, hal in enumerate(hals):
        if len(hal) and lowres_mass_name in hal and len(hal[lowres_mass_name]):
            lowres_mass_fracs = hal.prop(lowres_mass_name + ' / mass')
            if np.nanmax(lowres_mass_fracs) > lowres_mass_frac_max:
                print('s_i {:3d} | max = {:.1e} | n(>{}x) = {}'.format(
                    snapshot_i, np.nanmax(lowres_mass_fracs), lowres_mass_frac_max,
                    np.sum(lowres_mass_fracs > lowres_mass_frac_max)))


#===================================================================================================
# mass function
#===================================================================================================
def plot_number_v_mass(
    hals=None, gal=None,
    mass_kinds='mass', mass_limits=[], mass_width=0.2, mass_scaling='log',
    host_distance_limitss=[[1, 350]], halo_kind='halo',
    hal_indicess=None, gal_indices=None,
    func_kind='number.cum', axis_y_limits=None, axis_y_scaling='log', include_above_limits=True,
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot mass function, that is, number (cumulative or differential) v mass_kind.

    Parameters
    ----------
    hals : dict or list : catalog[s] of halos at snapshot
    gal : dict : catalog of galaxies to compare against
    mass_kind : str : halo mass kind to plot
    mass_limits : list : min and max limits for mass_kind
    mass_width : float : width of mass_kind bin
    mass_scaling : str : 'log' or 'linear'
    host_distance_limitss : list or list of lists :
        min and max limits of distance to host [kpc physical]
    halo_kind : str : shortcut for halo kind to plot:
        'halo', 'galaxy', 'cluster' and/or 'satellite', 'isolated'
    hal_indicess : array or list of arrays : halo indices to plot
    gal_indices : array : galaxy indices to plot
    func_kind : str : mass function kind to plot: 'number',  'number.dif', 'number.cum'
    axis_y_limits : list : min and max limits to impose on y-axis
    axis_y_scaling : str : scaling along y-axis: 'log', 'linear'
    include_above_limits : bool : whether to include mass_kind values above limits for cumulative
    write_plot : bool : whether to write figure to file
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    if hals is None:
        hals = []
    if isinstance(hals, dict):
        hals = [hals]
    if np.isscalar(mass_kinds):
        mass_kinds = [mass_kinds]
    if len(mass_kinds) == 1 and len(hals) > 1:
        mass_kinds = [mass_kinds[0] for _ in hals]

    mass_kind_default = mass_kinds[0]

    if host_distance_limitss is not None:
        host_distance_limitss = np.array(host_distance_limitss)
        if np.ndim(host_distance_limitss) == 1:
            host_distance_limitss = np.array([host_distance_limitss])
        host_distance_bin_number = host_distance_limitss.shape[0]
    else:
        host_distance_bin_number = 1

    if not isinstance(hal_indicess, list):
        hal_indicess = [hal_indicess for _ in hals]

    assert func_kind in ['number', 'number.dif', 'number.cum']

    MassBin = ut.binning.BinClass(mass_limits, mass_width, include_max=True, scaling=mass_scaling)

    hal_number_values, hal_number_errs = np.zeros(
        [2, len(hals), host_distance_bin_number, MassBin.number])

    # get mass function for halos
    for hal_i, hal in enumerate(hals):
        mass_kind = mass_kinds[hal_i]

        if hal_indicess[hal_i] is None or not len(hal_indicess[hal_i]):
            hal_indices = ut.array.get_arange(hal.prop(mass_kind))
        else:
            hal_indices = hal_indicess[hal_i]

        for dist_i, host_distance_limits in enumerate(host_distance_limitss):
            his_d = hal.get_indices(
                halo_kind=halo_kind, host_distance_limits=host_distance_limits,
                hal_indices=hal_indices)

            if len(his_d):
                hal_number_d = MassBin.get_distribution(
                    hal.prop(mass_kind, his_d), False, include_above_limits)
                hal_number_values[hal_i, dist_i] = hal_number_d[func_kind]
                hal_number_errs[hal_i, dist_i] = hal_number_d[func_kind + '.err']

    # get mass function for observed galaxies
    host_names = ['MW', 'M31']
    gal_mass_kind = mass_kind_default.replace('.part', '')
    if gal is not None and gal_mass_kind in gal:
        gal_number_values, gal_number_errs = np.zeros(
            [2, 2, host_distance_bin_number, MassBin.number])

        for dist_i, host_distance_limits in enumerate(host_distance_limitss):
            gis = ut.array.get_indices(
                gal['host.distance.total'], host_distance_limits, gal_indices)

            for host_i, host_name in enumerate(host_names):
                gis_h = gis[gal['host.name'][gis] == host_name.encode()]

                gal_number_h = MassBin.get_distribution(
                    gal[gal_mass_kind][gis_h], False, include_above_limits)
                gal_number_values[host_i, dist_i] = gal_number_h[func_kind]
                gal_number_errs[host_i, dist_i] = gal_number_h[func_kind + '.err']

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    mass_funcs_all = []
    if hal_number_values.size:
        mass_funcs_all.append(hal_number_values)
    if gal is not None:
        mass_funcs_all.append(gal_number_values)

    ut.plot.set_axes_scaling_limits(
        subplot, mass_scaling, mass_limits, None, axis_y_scaling, axis_y_limits, mass_funcs_all)

    """
    if mass_scaling == 'linear':
        minor_locator = AutoMinorLocator(5)
        subplot.xaxis.set_minor_locator(minor_locator)

    if axis_y_scaling == 'linear':
        minor_locator = AutoMinorLocator(5)
        subplot.yaxis.set_minor_locator(minor_locator)
    """

    axis_x_label = ut.plot.Label.get_label(gal_mass_kind)
    subplot.set_xlabel(axis_x_label)

    mass_label = ut.plot.Label.get_label(gal_mass_kind, get_units=False).strip('$')
    if 'dif' in func_kind:
        axis_y_label = '${{\\rm d}}n / {{\\rm d}}log({})$'.format(mass_label)
    elif 'cum' in func_kind:
        if len(host_distance_limitss) and len(host_distance_limitss[0]):
            axis_y_label = '$N_{{\\rm satellite}}(> {})$'.format(mass_label)
        else:
            axis_y_label = '$N(> {})$'.format(mass_label)
    else:
        axis_y_label = '$N({})$'.format(mass_label)
    subplot.set_ylabel(axis_y_label)

    colors = ut.plot.get_colors(len(hals))
    line_styles = ut.plot.get_line_styles(host_distance_bin_number)

    x_values = MassBin.get_bin_values(func_kind)

    # plot observed galaxies
    host_label_dict = {
        'MW': {'color': 'black', 'linestyle': '--'},
        'M31': {'color': 'black', 'linestyle': ':'}
    }

    if gal is not None and gal_mass_kind in gal:
        for dist_i, host_distance_limits in enumerate(host_distance_limitss):
            for host_i, host_name in enumerate(host_names):
                label = host_name.replace('MW', 'Milky Way').replace('M31', 'M31 (Andromeda)')
                subplot.plot(x_values, gal_number_values[host_i, dist_i],
                             # gal_number_errs[hal_i],
                             color=host_label_dict[host_name]['color'],
                             linestyle=host_label_dict[host_name]['linestyle'],
                             linewidth=3.0, alpha=0.8, label=label)

    # plot halos
    for hal_i, hal in enumerate(hals):
        for dist_i, host_distance_limits in enumerate(host_distance_limitss):
            linewidth = 3.0
            alpha = 0.9
            color = colors[hal_i]

            label = hal.info['simulation.name']
            #if 'm12i r13' in label:
            #    label = 'Latte simulation'  # \ngalaxy $\sigma_{\\rm velocity,star}$'
            #if len(host_distance_limits) > len(hals):
            #    label = ut.plot.get_label_distance('host.distance.total', host_distance_limits)

            # ensure n = 1 is clear on log plot
            y_values = hal_number_values[hal_i, dist_i]
            if 'log' in axis_y_scaling:
                y_values = np.clip(y_values, 0.5, Inf)

            """
            if '57000' in hal.info['simulation.name']:
                #label = 'Latte low-res'
                label = None
                linewidth = 1.5
                #if 'star.mass' in mass_kind_default:
                #    y_values[x_values < 3e7] = np.nan
                #elif 'star.vel.std' in mass_kind_default:
                #    y_values[x_values < 9] = np.nan
                #color = colors[0]
                color = ut.plot.get_color('blue.lite')
            """

            if ('star' in mass_kinds[0] and 'vel.circ.max' in mass_kinds[hal_i] or
                    'vel.std' in mass_kinds[hal_i] and hal_i > 0 and 'star.mass' not in hal):
                linewidth = 1.6
                alpha = 0.35
                label = 'subhalo $V_{\\rm circ,max}$'
                color = ut.plot.get_color('blue.lite')

            if ('star' in mass_kinds[0] and 'vel' in mass_kinds[hal_i] and
                    'dm' in hal.info['simulation.name']):
                linewidth = 1.6
                alpha = 0.35
                #label = '\ndark matter only\nsubhalo $V_{\\rm circ,max}$'
                label = '\nDMO simulation\nsubhalo $V_{\\rm circ,max}$'
                color = ut.plot.get_color('orange.mid')

            subplot.plot(x_values, y_values,
                         # hal_number_errs[hal_i],
                         color=color, linestyle=line_styles[dist_i], linewidth=linewidth,
                         alpha=alpha, label=label)

    if len(hals) > 1 or gal is not None or len(host_distance_limitss) > 1:
        legend = subplot.legend(loc='best')
        legend.get_frame()

    redshift_label = ''
    galaxy_label = ''
    if len(hals):
        redshift_label = ut.plot.get_time_name('redshift', hals[0].snapshot)
    if gal is not None:
        galaxy_label = '_lg'
    if not len(hals) and '.part' in mass_kind_default:
        mass_kind_default = mass_kind_default.replace('.part', '')
    plot_name = '{}_v_{}{}{}'.format(func_kind, mass_kind_default, galaxy_label, redshift_label)
    ut.plot.parse_output(write_plot, plot_name, plot_directory)


def plot_has_baryon_fraction_v_mass(
    hals, baryon_mass_kind='star.mass',
    mass_kind='mass', mass_limits=[], mass_width=0.2, mass_scaling='log',
    host_distance_limitss=[[0, 350]],
    axis_y_limits=[0, 1], axis_y_scaling='linear',
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot .

    Parameters
    ----------
    hals : dict or list : catalog[s] of halos at snapshot
    baryon_mass_kind : str : baryon species to use: 'star', 'gas', 'baryon'
    mass_kind : str : halo mass kind
    mass_limits : list : min and max limits for mass_kind
    mass_width : float : width of mass_kind bin
    mass_scaling : str : 'log' or 'linear'
    host_distance_limitss : list or list of lists :
        min and max limits of distance to host [kpc physical]
    axis_y_limits : list : min and max limits for y-axis
    axis_y_scaling : str : scaling along y-axis: 'log', 'linear'
    write_plot : bool : whether to write figure to file
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    Fraction = ut.math.FractionClass(error_kind='beta')

    if isinstance(hals, dict):
        hals = [hals]

    if host_distance_limitss is not None:
        host_distance_limitss = np.array(host_distance_limitss)
        if np.ndim(host_distance_limitss) == 1:
            host_distance_limitss = np.array([host_distance_limitss])
        host_distance_bin_number = host_distance_limitss.shape[0]

    MassBin = ut.binning.BinClass(mass_limits, mass_width, scaling=mass_scaling)

    has_baryon_frac = {
        'value': np.zeros([len(hals), host_distance_bin_number, MassBin.number]),
        'error': np.zeros([len(hals), host_distance_bin_number, 2, MassBin.number]),
        'number': np.zeros([len(hals), host_distance_bin_number, MassBin.number]),
    }

    for hal_i, hal in enumerate(hals):
        his = ut.array.get_indices(hal.prop(mass_kind), mass_limits)

        for dist_i, host_distance_limits in enumerate(host_distance_limitss):
            his_d = hal.get_indices(
                halo_kind='halo', host_distance_limits=host_distance_limits, hal_indices=his)

            his_d_baryon = ut.array.get_indices(hal.prop(baryon_mass_kind), [1, Inf], his_d)

            halo_numbers_d = MassBin.get_histogram(hal.prop(mass_kind, his_d), normed=False)
            baryon_numbers_d = MassBin.get_histogram(
                hal.prop(mass_kind, his_d_baryon), normed=False)

            print(halo_numbers_d)
            print(baryon_numbers_d)

            has_baryon_frac['number'][hal_i, dist_i] = halo_numbers_d

            has_baryon_frac['value'][hal_i, dist_i], has_baryon_frac['error'][hal_i, dist_i] = \
                Fraction.get_fraction(baryon_numbers_d, halo_numbers_d)

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)
    #, left=0.17, right=0.96, top=0.96, bottom=0.14)

    ut.plot.set_axes_scaling_limits(
        subplot, mass_scaling, mass_limits, None, axis_y_scaling, axis_y_limits,
        has_baryon_frac['value'])

    subplot.set_xlabel(ut.plot.Label.get_label(mass_kind))
    label = 'fraction with $M_{{\\rm {}}}$'.format(
        baryon_mass_kind.replace('.mass', '').replace('.part', ''))
    subplot.set_ylabel(label, fontsize=30)

    colors = ut.plot.get_colors(len(hals))
    line_styles = ut.plot.get_line_styles(host_distance_bin_number)

    for hal_i in range(len(hals)):
        for dist_i, host_distance_limits in enumerate(host_distance_limitss):
            label = None
            if host_distance_limits is not None and len(host_distance_limits):
                label = ut.plot.Label.get_label('host.distance.total', host_distance_limits)
            #if dist_i == 0:
            #    label = hals[hal_i].info['simulation.name']

            pis = ut.array.get_indices(has_baryon_frac['number'][hal_i, dist_i], [1, Inf])
            subplot.plot(MassBin.mids[pis], has_baryon_frac['value'][hal_i, dist_i, pis],
                         # frac_errs[hal_i, dist_i, :, pis],
                         color=colors[hal_i], linestyle=line_styles[dist_i], label=label, alpha=0.7)

    legend = subplot.legend(loc='best')
    legend.get_frame()

    plot_name = 'has.{}.fraction_v_{}'.format(baryon_mass_kind.replace('.part', ''), mass_kind)
    plot_name += ut.plot.get_time_name('redshift', hals[0].snapshot)
    ut.plot.parse_output(write_plot, plot_name, plot_directory)


#===================================================================================================
# number v distance
#===================================================================================================
def plot_number_v_distance(
    hals=None, gal=None,
    mass_kind='mass', mass_limitss=[[]],
    distance_limits=[1, 1000], distance_width=0.1, distance_scaling='log',
    halo_kind='halo',
    hal_indicess=None, gal_indices=None,
    gal_host_names=['MW', 'M31'],
    func_kind='sum', axis_y_limits=None, axis_y_scaling='log',
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot mass function, that is, number (cumulative or differential) v mass_kind.

    Parameters
    ----------
    hals : dict or list : catalog[s] of halos at snapshot
    gal : dict : catalog of galaxies to compare against
    mass_kind : str : halo mass kind to plot
    mass_limitss : list or list of lists : min and max limits of halo mass
    distance_limits : list : min and max distance from host [kpc physical]
    distance_width : float : width of distance bin
    distance_scaling : str : 'log' or 'linear'
    halo_kind : str : shortcut for halo kind to plot:
        'halo', 'galaxy', 'cluster' and/or 'satellite', 'isolated'
    hal_indicess : array or list of arrays : indices of halos to plot
    gal_indices : array : indices of galaxies to plot
    gal_host_names : list : names of hosts for observed galaxy catalog
    func_kind : str : number kind to plot:
        'sum', 'sum.cum', 'fraction', 'fraction.cum', 'density'
    axis_y_limits : list : min and max limits to impose on y-axis
    axis_y_scaling : str : scaling along y-axis: 'log', 'linear'
    write_plot : bool : whether to write figure to file
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    dimension_number = 3

    if hals is None:
        hals = []
    if isinstance(hals, dict):
        hals = [hals]

    if mass_limitss is not None:
        mass_limitss = np.array(mass_limitss)
        if np.ndim(mass_limitss) == 1:
            mass_limitss = np.array([mass_limitss])
        mass_number = mass_limitss.shape[0]

    if not isinstance(hal_indicess, list):
        hal_indicess = [hal_indicess for _ in hals]

    DistanceBin = ut.binning.DistanceBinClass(
        distance_scaling, distance_limits, distance_width, None, dimension_number, include_max=True)

    hal_number = {}

    # get numbers for halos
    if hals is not None and len(hals):
        for hal_i, hal in enumerate(hals):
            if hal_indicess[hal_i] is None or not len(hal_indicess[hal_i]):
                hal_indices = ut.array.get_arange(hal.prop(mass_kind))
            else:
                hal_indices = hal_indicess[hal_i]

            hal_number_h = {}

            for _m_i, mass_limits in enumerate(mass_limitss):
                his_m = ut.array.get_indices(hal.prop(mass_kind), mass_limits, hal_indices)
                his_m = hal.get_indices(halo_kind=halo_kind, hal_indices=his_m)

                hal_number_m = DistanceBin.get_sum_profile(
                    hal.prop('host.distance.total', his_m), get_fraction=True)
                ut.array.append_dictionary(hal_number_h, hal_number_m)
            ut.array.append_dictionary(hal_number, hal_number_h)
        ut.array.arrayize_dictionary(hal_number)

    # get numbers for observed galaxies
    if gal is not None and mass_kind in gal:
        gal_number = {}
        for gal_host_name in gal_host_names:
            #gis_h = gis[gal['host.name'][gis] == host_name.encode()]
            gis_h = ut.array.get_indices(gal['host.name'], gal_host_name.encode(), gal_indices)
            gal_number_h = {}
            for m_i, mass_limits in enumerate(mass_limitss):
                gis_m = ut.array.get_indices(gal[mass_kind], mass_limits, gis_h)
                gal_number_m = DistanceBin.get_sum_profile(
                    gal['host.distance.total'][gis_m], get_fraction=True)
                ut.array.append_dictionary(gal_number_h, gal_number_m)
            ut.array.append_dictionary(gal_number, gal_number_h)
        ut.array.arrayize_dictionary(gal_number)

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    if hals is not None and len(hals):
        numbers_all = hal_number[func_kind]
    elif gal is not None:
        numbers_all = gal_number[func_kind]

    ut.plot.set_axes_scaling_limits(
        subplot, distance_scaling, distance_limits, None,
        axis_y_scaling, axis_y_limits, numbers_all)

    #if 'log' in distance_scaling:
    #    subplot.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))

    subplot.set_xlabel('distance $\\left[ {\\rm kpc} \\right]$')

    if '.cum' in func_kind:
        axis_y_label = '$N_{{\\rm satellite}}(< d)$'
    else:
        if distance_scaling == 'log':
            axis_y_label = '${\\rm d}n/{\\rm d}log(d) \, \\left[ {\\rm kpc^{-3}} \\right]$'
        else:
            axis_y_label = '${\\rm d}n/{\\rm d}d \, \\left[ {\\rm kpc^{-2}} \\right]$'
    subplot.set_ylabel(axis_y_label)

    colors = ut.plot.get_colors(len(hals))
    line_styles = ut.plot.get_line_styles(mass_number)

    distance_kind = 'distance.mid'
    if '.cum' in func_kind:
        distance_kind = 'distance.cum'

    # plot observed galaxies
    host_label_dict = {
        'MW': {'color': 'black', 'linestyle': '--'},
        'M31': {'color': 'black', 'linestyle': ':'}
    }
    if gal is not None and mass_kind in gal:
        for m_i, mass_limits in enumerate(mass_limitss):
            for host_i, host_name in enumerate(gal_host_names):
                label = host_name  # .replace('MW', 'Milky Way').replace('M31', 'M31 (Andromeda)')

                y_values = gal_number[func_kind][host_i, m_i]
                # ensure n = 1 is clear on log plot
                if 'sum' in func_kind and 'log' in axis_y_scaling:
                    y_values = np.clip(y_values, 0.5, Inf)
                masks = np.where(y_values > -1)[0]

                subplot.plot(
                    gal_number[distance_kind][host_i, m_i][masks], y_values[masks],
                    color=host_label_dict[host_name]['color'],
                    linestyle=host_label_dict[host_name]['linestyle'],
                    linewidth=3.0, alpha=0.8, label=label
                )

    # plot halos
    if hals is not None and len(hals):
        for hal_i, hal in enumerate(hals):
            for m_i, _mass_limits in enumerate(mass_limitss):
                linewidth = 3.0
                alpha = 0.9
                color = colors[hal_i]

                label = None
                if m_i == 0:
                    label = hal.info['simulation.name']
                    #label = 'Latte simulation'

                y_values = hal_number[func_kind][hal_i, m_i]
                # ensure n = 1 is clear on log plot
                if 'sum' in func_kind and 'log' in axis_y_scaling:
                    y_values = np.clip(y_values, 0.5, Inf)

                """
                if '57000' in hal.info['simulation.name']:
                    #label = 'Latte low-res'
                    label = None
                    linewidth = 1.5
                    #if 'star.mass' in mass_kind_default:
                    #    y_values[x_values < 3e7] = np.nan
                    #elif 'star.vel.std' in mass_kind_default:
                    #    y_values[x_values < 9] = np.nan
                    #color = colors[0]
                    color = ut.plot.get_color('blue.lite')
                """

                masks = np.where(y_values > -1)[0]
                subplot.plot(
                    hal_number[distance_kind][hal_i, m_i][masks], y_values[masks],
                    color=color, linestyle=line_styles[m_i], linewidth=linewidth, alpha=alpha,
                    label=label
                )

    if len(hals) > 1 or gal is not None:
        legend = subplot.legend(loc='best')
        legend.get_frame()

    galaxy_label = ''
    if gal is not None:
        galaxy_label = '_lg'
    redshift_label = ''
    if len(hals):
        redshift_label = ut.plot.get_time_name('redshift', hals[0].snapshot)
    plot_name = 'number.{}_v_distance{}{}'.format(func_kind, galaxy_label, redshift_label)
    ut.plot.parse_output(write_plot, plot_name, plot_directory)


#===================================================================================================
# properties
#===================================================================================================
def plot_property_v_property(
    hals=None, gal=None,
    x_property_name='mass.bound', x_property_limits=[], x_property_scaling='log',
    y_property_name='star.mass', y_property_limits=[], y_property_scaling='log',
    host_distance_limitss=None, near_halo_distance_limits=None, hal_indicess=None,
    plot_histogram=False, property_bin_number=200,
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot property v property.

    Parameters
    ----------
    hals : dict : catalog[s] of halos at snapshot
    gal : dict : catalog of galaxies
    x_property_name : str : name of property for x-axis
    x_property_limits : list : min and max limits to impose on x_property_name
    x_property_scaling : str : scaling for x_property_name: 'log' or 'linear'
    y_property_name : str :  name of property for y-axis
    y_property_limits : list : min and max limits to impose on y_property_name
    y_property_scaling : str : scaling for y_property_name: 'log' or 'linear'
    host_distance_limitss : list : min and max limits for distance from galaxy
    near_halo_distance_limits : list : distance to nearest halo [d / R_neig]
    hal_indicess : array or list of arrays :
    plot_histogram : bool : whether to plot 2-D histogram instead of individual points
    property_bin_number : int : number of bins along each axis (if histogram)
    write_plot : bool : whether to write figure to file
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''

    def get_label_distance(cat, distance_limits):
        '''
        .
        '''
        if 'halo' in cat.info['catalog.kind']:
            label = 'simulated'
        elif 'galaxy' in cat.info['catalog.kind']:
            label = 'observed'

        if np.max(distance_limits) < 400:
            label += ' satellite'
        elif np.min(distance_limits) > 100:
            label += ' isolated'

        return label

    Say = ut.io.SayClass(plot_property_v_property)

    if hals is None:
        hals = []
    elif isinstance(hals, dict):
        hals = [hals]

    if host_distance_limitss is not None:
        host_distance_limitss = np.array(host_distance_limitss)
        if np.ndim(host_distance_limitss) == 1:
            host_distance_limitss = np.array([host_distance_limitss])
        host_distance_bin_number = host_distance_limitss.shape[0]
    else:
        host_distance_bin_number = 1
        host_distance_limitss = [None]

    if not isinstance(hal_indicess, list):
        hal_indicess = [hal_indicess]

    x_property_values = []
    y_property_values = []

    for hal_i, hal in enumerate(hals):
        his = hal_indicess[hal_i]
        if his is None:
            his = ut.array.get_arange(hal['mass'])

        if near_halo_distance_limits is not None:
            his = ut.array.get_indices(
                hal['nearest.distance/Rneig'], near_halo_distance_limits, his)

        x_prop_vals_h = []
        y_prop_vals_h = []

        for host_distance_limits in host_distance_limitss:
            if host_distance_limits is not None and len(host_distance_limits):
                his_d = ut.array.get_indices(
                    hal.prop('host.distance.total'), host_distance_limits, his)
            else:
                his_d = his

            x_prop_vals_d = hal.prop(x_property_name, his_d)
            y_prop_vals_d = hal.prop(y_property_name, his_d)
            #if 'metallicity' in y_property_name:
            #    y_prop_vals_d = ut.math.get_log(y_prop_vals_d)

            Say.say('{} range = [{:.3e}, {:.3e}], med = {:.3e}'.format(
                    x_property_name, x_prop_vals_d.min(), x_prop_vals_d.max(),
                    np.median(x_prop_vals_d)))
            Say.say('{} range = [{:.3e}, {:.3e}], med = {:.3e}'.format(
                    y_property_name, y_prop_vals_d.min(), y_prop_vals_d.max(),
                    np.median(y_prop_vals_d)))

            #if ('gas.mass' in y_property_name and 'star.mass' in y_property_name and
            #'/' in y_property_name):
            #    y_prop_vals_d = y_prop_vals_d.clip(1.2e-4, Inf)

            if x_property_limits:
                indices = ut.array.get_indices(x_prop_vals_d, x_property_limits)
                x_prop_vals_d = x_prop_vals_d[indices]
                y_prop_vals_d = y_prop_vals_d[indices]

            if y_property_limits:
                indices = ut.array.get_indices(y_prop_vals_d, y_property_limits)
                x_prop_vals_d = x_prop_vals_d[indices]
                y_prop_vals_d = y_prop_vals_d[indices]

            if not len(x_prop_vals_d) or not len(y_prop_vals_d):
                Say.say('! no halos in bin')
                return

            #print(his_d[indices])

            x_prop_vals_h.append(x_prop_vals_d)
            y_prop_vals_h.append(y_prop_vals_d)

        x_property_values.append(x_prop_vals_h)
        y_property_values.append(y_prop_vals_h)

    x_property_values = np.array(x_property_values)
    y_property_values = np.array(y_property_values)

    gal_x_property_values = []
    gal_y_property_values = []

    if gal is not None:
        # compile observed galaxies
        gal_x_property_name = x_property_name.replace('.part', '')
        gal_y_property_name = y_property_name.replace('.part', '')
        gis_m = ut.array.get_indices(gal[gal_x_property_name], x_property_limits)
        for dist_i, host_distance_limits in enumerate(host_distance_limitss):
            gis_d = ut.array.get_indices(gal['host.distance.total'], host_distance_limits, gis_m)
            #gis_d = gis[gal['host.name'][gis] == b'MW']

            gal_x_property_values.append(gal.prop(gal_x_property_name, gis_d))
            gal_y_property_values.append(gal.prop(gal_y_property_name, gis_d))

        gal_x_property_values = np.array(gal_x_property_values)
        gal_y_property_values = np.array(gal_y_property_values)

    if len(hals) > 1:
        colors = ut.plot.get_colors(len(hals))
    else:
        colors = ut.plot.get_colors(max(host_distance_bin_number, 2))

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    axis_x_limits, axis_y_limits = ut.plot.set_axes_scaling_limits(
        subplot, x_property_scaling, x_property_limits, [x_property_values, gal_x_property_values],
        y_property_scaling, y_property_limits, [y_property_values, gal_y_property_values])

    axis_x_label = ut.plot.Label.get_label(x_property_name)
    subplot.set_xlabel(axis_x_label)
    axis_y_label = ut.plot.Label.get_label(y_property_name)
    subplot.set_ylabel(axis_y_label, fontsize=30)

    #if 'linear' in y_property_scaling:
    #    subplot.yaxis.set_minor_locator(AutoMinorLocator(5))

    label = None

    if plot_histogram:
        # plot histogram
        for hal_i, hal in enumerate(hals):
            for dist_i, host_distance_limits in enumerate(host_distance_limitss):
                if 'log' in x_property_scaling:
                    x_property_values[hal_i, dist_i] = ut.math.get_log(
                        x_property_values[hal_i, dist_i])

                if 'log' in y_property_scaling:
                    y_property_values[hal_i, dist_i] = ut.math.get_log(
                        y_property_values[hal_i, dist_i])

                if host_distance_limits is not None and len(host_distance_limits):
                    label = ut.plot.Label.get_label('host.distance.total', host_distance_limits)

            valuess, _xs, _ys = np.histogram2d(
                x_property_values[hal_i, dist_i], y_property_values[hal_i, dist_i],
                property_bin_number)
            # norm=LogNorm()

            subplot.imshow(
                valuess.transpose(),
                #norm=LogNorm(),
                cmap=plt.cm.YlOrBr,  # @UndefinedVariable
                aspect='auto',
                #interpolation='nearest',
                interpolation='none',
                extent=(axis_x_limits[0], axis_x_limits[1], axis_y_limits[0], axis_y_limits[1]),
                vmin=np.min(valuess), vmax=np.max(valuess),
                label=label,
            )

        #plt.colorbar()

    else:
        # plot individual points

        # draw observed galaxies
        if gal is not None:
            alpha = 0.5
            if hals is None or not len(hals):
                alpha = 0.7
            for dist_i, host_distance_limits in enumerate(host_distance_limitss):
                if host_distance_limits is not None and len(host_distance_limits):
                    #label = ut.plot.get_label_distance('host.distance.total', host_distance_limits)
                    label = get_label_distance(gal, host_distance_limits)

                subplot.plot(gal_x_property_values[dist_i], gal_y_property_values[dist_i], '*',
                             color=colors[dist_i], markersize=12, alpha=alpha, label=label)

        if ('mass' in x_property_name and 'star.mass' in y_property_name and
                '/' not in y_property_name):
            #subplot.plot([1e1, 1e14], [1e1, 1e14], ':', color='black', linewidth=2, alpha=0.3)
            #subplot.plot([1e1, 1e14], [1e-1, 1e12], '--', color='black', linewidth=2, alpha=0.2)
            mass_peaks = 10 ** np.arange(1, 12, 0.1)
            mstars_from_mpeaks = 3e6 * (mass_peaks / 1e10) ** 1.92
            subplot.plot(mass_peaks, mstars_from_mpeaks, '--', color='black', linewidth=2,
                         alpha=0.3)
        #"""
        if 'star.mass' in x_property_name and 'metallicity' in y_property_name:
            #subplot.plot(metal_fire['star.mass'], metal_fire['star.metallicity'], 'o',
            #             color='gray', markersize=8, alpha=0.5, label='FIRE isolated')

            for k in ['MW', 'M31', 'isolated']:
                if k == 'MW':
                    color = colors[0]
                    label = 'observed satellite'
                elif k == 'M31':
                    color = colors[0]
                    label = None
                else:
                    color = colors[1]
                    label = 'observed isolated'

                subplot.plot(metal_kirby['star.mass'][k], metal_kirby['star.metallicity'][k], '*',
                             color=color, markersize=12, alpha=0.5, label=label)

        #"""
        # draw simulated galaxies
        markers = ['.', '.']
        #marker_sizes = [22, 7]
        marker_sizes = [3, 3]
        for hal_i, hal in enumerate(hals):
            for dist_i, host_distance_limits in enumerate(host_distance_limitss):
                if host_distance_limits is not None and len(host_distance_limits):
                    #label = ut.plot.get_label_distance('host.distance.total', host_distance_limits)
                    label = get_label_distance(hal, host_distance_limits)
                    if hal_i > 0:
                        label = None

                subplot.plot(
                    x_property_values[hal_i, dist_i], y_property_values[hal_i, dist_i],
                    markers[hal_i], color=colors[dist_i], markersize=marker_sizes[hal_i], alpha=0.8,
                    label=label)

    if label is not None:
        legend = subplot.legend(loc='best')
        legend.get_frame()

    plot_name = y_property_name + '_v_' + x_property_name
    if hals is None and gal is not None:
        plot_name += '_lg'
    if hals is not None and len(hals):
        plot_name += ut.plot.get_time_name('redshift', hals[0].snapshot)
    else:
        plot_name = plot_name.replace('.part', '')
    ut.plot.parse_output(write_plot, plot_name, plot_directory)


def plot_property_v_distance(
    hals=None, mass_kind='mass', mass_limitss=[[]],
    distance_limits=[0, 300], distance_width=1, distance_scaling='linear',
    property_name='host.velocity.tan', statistic='median',
    axis_y_limits=None, axis_y_scaling='linear',
    halo_kind='halo', hal_indicess=None,
    write_plot=False, plot_directory='.', figure_index=1):
    '''
    Plot property v distance, in bins of mass_kind.

    Parameters
    ----------
    hals : dict or list : catalog[s] of halos at snapshot
    mass_kind : str : halo mass kind to plot
    mass_limitss : list or list of lists : min and max limits of halo mass
    distance_limits : list : min and max distance from host [kpc physical]
    distance_width : float : width of distance bin
    distance_scaling : str : 'log' or 'linear'
    property : str :
    statistic : str :
    axis_y_limits : list : min and max limits to impose on y-axis
    axis_y_scaling : str : scaling along y-axis: 'log', 'linear'
    halo_kind : str : shortcut for halo kind to plot:
        'halo', 'galaxy', 'cluster' and/or 'satellite', 'isolated'
    hal_indicess : array or list of arrays : indices of halos to plot
    write_plot : bool : whether to write figure to file
    plot_directory : str : directory to write figure file
    figure_index : int : index of figure for matplotlib
    '''
    dimension_number = 3

    if hals is None:
        hals = []
    if isinstance(hals, dict):
        hals = [hals]

    if mass_limitss is not None:
        mass_limitss = np.array(mass_limitss)
        if np.ndim(mass_limitss) == 1:
            mass_limitss = np.array([mass_limitss])
        mass_number = mass_limitss.shape[0]

    if not isinstance(hal_indicess, list):
        hal_indicess = [hal_indicess for _ in hals]

    DistanceBin = ut.binning.DistanceBinClass(
        distance_scaling, distance_limits, distance_width, None, dimension_number, include_max=True)

    hal_stat = {}

    # get statistics for halos
    if hals is not None and len(hals):
        for hal_i, hal in enumerate(hals):
            if hal_indicess[hal_i] is None or not len(hal_indicess[hal_i]):
                hal_indices = ut.array.get_arange(hal.prop(mass_kind))
            else:
                hal_indices = hal_indicess[hal_i]

            hal_stat_h = {}

            for _m_i, mass_limits in enumerate(mass_limitss):
                his_m = ut.array.get_indices(hal.prop(mass_kind), mass_limits, hal_indices)
                his_m = hal.get_indices(halo_kind=halo_kind, hal_indices=his_m)

                hal_stat_m = DistanceBin.get_statistics_profile(
                    hal.prop('host.distance.total', his_m), hal.prop(property_name, his_m))
                ut.array.append_dictionary(hal_stat_h, hal_stat_m)
            ut.array.append_dictionary(hal_stat, hal_stat_h)
        #ut.array.arrayize_dictionary(hal_stat)

    # plot ----------
    _fig, subplot = ut.plot.make_figure(figure_index)

    ut.plot.set_axes_scaling_limits(
        subplot, distance_scaling, distance_limits, None,
        axis_y_scaling, axis_y_limits)

    subplot.set_xlabel('distance $\\left[ {\\rm kpc} \\right]$')
    subplot.set_ylabel(property_name)
    #subplot.set_ylabel('$V_{tan} / ( \sqrt{2} V_{rad} )$')

    colors = ut.plot.get_colors(len(hals))
    line_styles = ut.plot.get_line_styles(mass_number)

    # plot halos
    if hals is not None and len(hals):
        for hal_i, hal in enumerate(hals):
            for m_i, _mass_limits in enumerate(mass_limitss):
                linewidth = 3.0
                alpha = 0.9
                color = colors[hal_i]

                label = hal.info['simulation.name']
                print(label)

                subplot.plot(
                    hal_stat['distance.mid'][hal_i][m_i], hal_stat[statistic][hal_i][m_i],
                    color=color, linestyle=line_styles[m_i], linewidth=linewidth, alpha=alpha,
                    label=label)

    if len(hals) > 1:
        legend = subplot.legend(loc='best')
        legend.get_frame()

    redshift_label = ''
    if len(hals):
        redshift_label = ut.plot.get_time_name('redshift', hals[0].snapshot)
    plot_name = '{}.{}_v_distance{}'.format(property_name, statistic, redshift_label)
    ut.plot.parse_output(write_plot, plot_name, plot_directory)


#===================================================================================================
# observations
#===================================================================================================
metal_kirby = {
    # stellar metallicity [Fe/H] from Kirby et al 2013
    'star.mass': {
        'MW': 10 ** np.array([7.39, 6.69, 6.59, 6.07, 5.84, 5.73, 5.51, 5.48,
                              4.57, 4.28, 3.93, 3.90, 3.73, 3.68, 3.14]),
        'isolated': 10 ** np.array([7.92, 8.01, 6.92, 6.82, 6.47, 6.15, 5.13]),
        'M31': 10 ** np.array([8.67, 7.83, 8.00, 7.17, 6.96, 6.88, 6.26, 5.79, 5.90, 5.89, 5.58,
                               5.38, 5.15])},
    'star.mass.err.hi': [.14, .13, .21, .13, .2, .2, .1, .09, .14, .13, .15, .2,
                         .23, .22, .13, .09, .06, .08, .08, .09, .05, .2, .05, .05,
                         .05, .13, .08, .05, .12, .09, .3, .16, .2, .44, .4],
    'star.mass.err.lo': [.14, .13, .21, .13, .2, .2, .1, .09, .14, .13, .11, .2,
                         .23, .22, .13, .09, .06, .08, .08, .09, .05, .2, .05, .05,
                         .05, .13, .08, .05, .12, .09, .3, .13, .3, .44, .4],
    'star.metallicity': {
        'MW': np.array([-1.04, -1.45, -1.68, -1.63, -1.94, -2.13, -1.98, -1.91, -2.39, -2.1, -2.45,
                        -2.12, -2.18, -2.25, -2.14]),
        'isolated': np.array([-1.05, -1.19, -1.43, -1.39, -1.58, -1.44, -1.74]),
        'M31': np.array([-0.92, -1.12, -0.83, -1.62, -1.47, -1.33, -1.84, -1.94, -1.35, -1.70,
                         -2.21, -1.93, -2.46])},
    'star.metallicity.err': [.01, .01, .01, .01, .01, .01, .01, .01, .04, .03, .07, .05,
                             .05, .04, .05, .01, .01, .02, .01, .02, .03, .04, .13, .36,
                             .25, .21, .37, .17, .05, .18, .2, .2, .01, .2, .2]
}

metal_fire = {
    # stellar and gas metallicity [total/Solar] from Ma et al 2015
    # for stars, they find [Fe/H] = log(Z_tot/Z_sun) - 0.2. these already include this conversion
    'star.mass': 10 ** np.array([
        10.446, 4.615, 6.359, 7.075, 6.119, 9.374, 8.174, 8.401, 8.337, 9.611, 10.3178, 10.779,
        9.274, 9.367, 11.135]),
    'star.metallicity': 10 ** np.array([
        0.219, -2.859, -1.886, -1.384, -1.829, -0.731, -1.309, -1.025, -1.163, -0.686, -0.157,
        0.114, -0.614, -0.604, 0.133]),
    'gas.metallicity': 10 ** np.array([
        0.250, -3.299, -1.157, -0.969, -1.580, -0.370, -1.092, -0.611, -0.800, -0.415, -0.183,
        0.137, -0.183, -0.347, 0.371])
}
