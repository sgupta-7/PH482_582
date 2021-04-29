'''
Helper functions for plotting with matplotlib.

@author: Andrew Wetzel <arwetzel@gmail.com>
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import numpy as np
from matplotlib import pyplot as plt
# local ----
from . import basic as ut


#===================================================================================================
# create plot
#===================================================================================================
def make_figure(
    figure_index=1, panel_numbers=[1, 1], axis_secondary='',
    left=0.19, right=0.955, top=0.965, bottom=0.17, hspace=0.03, wspace=0.03,
    sharex=True, sharey=True, background_color='white'):
    '''
    Create matplotlib figure, return figure and plot objects.

    Parameters
    ----------
    figure_index : int : index number of figure
    panel_numbers : list : number of panels along x- and y-axis
    axis_secondary : str : which axes will have secondary label: 'x', 'y', 'xy'
    left : float
    right : float
    top : float
    bottom : float
    hspace : float
    wspace : float
    sharex : bool : whether to share x-axes across panels/subplots
    sharey : bool : whether to share y-axes across panels/subplots
    background_color : str : 'white', 'black', etc

    Returns
    -------
    fig : figure object
    subplot : plot object, or list thereof (list is of dimension panel_numbers)
    '''
    fig = plt.figure(figure_index)
    fig.clf()

    if panel_numbers == [1, 1]:
        subplots = fig.add_subplot(111, facecolor=background_color)
    else:
        fig, subplots = plt.subplots(
            panel_numbers[0], panel_numbers[1], num=figure_index, sharex=sharex, sharey=sharey)
        if panel_numbers[0] > 1 and panel_numbers[1] > 1:
            for subplot_i in range(panel_numbers[0]):
                for subplot_j in range(panel_numbers[1]):
                    subplots[subplot_i, subplot_j].set_facecolor(background_color)
        else:
            panel_number = max(panel_numbers)
            for subplot_i in range(panel_number):
                subplots[subplot_i].set_facecolor(background_color)

    if 'y' in axis_secondary:
        right = 0.9
    if 'x' in axis_secondary:
        top = 0.85

    fig.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)

    return fig, subplots


#===================================================================================================
# axis scaling and limits
#===================================================================================================
def get_axis_limits(
    limits=None, values=None, scaling='linear', other_values=None, other_limits=None,
    exclude_zero=False, padding=0.04):
    '''
    Get axis limits for a plot.
    If input limits, use them instead to get limits (where not None).
    If input other_values and other_limits limits, will get values limits using only values
    corresponding to other_values within other_limits.

    Parameters
    ----------
    limits : list : min and max limits to impose (instead of using values)
    values : array : values to plot (use maxima to define limits)
    scaling : str : scaling of values: 'log', 'linear'
    other_values : array : another set of values of same length as values
    other_limits : list : min and max limits to impose on other_values
    exclude_zero : bool : whether to force limits to be non-zero
    padding : float : padding (fraction) to add to limits beyond input values

    Returns
    -------
    limits : array : x- and y-axis limits
    '''

    def get_limits(values, scaling, other_values, other_limits, exclude_zero, padding):
        '''
        Get axis limits for a plot.
        If provide other_values and other_limits limits, will get values limits using only values
        corresponding to other_values within other_limits.

        Parameters
        ----------
        values : array
        scaling : str : scaling of values: 'log', 'linear'
        other_values : array : another set of values of same length as values
        other_limits : list : limits to impose on other_values
        exclude_zero : bool : whether to force limits to be non-zero
        padding : float : padding (fraction) to add to limits beyond values
        '''
        Say = ut.io.SayClass(get_limits)

        values = ut.array.get_array_1d(values)

        if other_values is not None:
            other_values = np.array(other_values)

        if (other_values is not None and len(other_values) and
                other_limits is not None and len(other_limits)):
            ixs = ut.array.get_arange(other_values)
            masks = (other_values >= other_limits[0])
            masks *= (other_values <= other_limits[1])
            ixs = ixs[masks]
            if len(values.shape) == 1:
                if ixs.size == values.size:
                    values = values[ixs]
                else:
                    Say.say('! size mismatch between x-axis and y-axis arrays')
                    values[-1e20]
            elif len(values.shape) == 2:
                if ixs.size == values.shape[1]:
                    values = values[:, ixs]
                else:
                    Say.say('! size mismatch between x-axis and y-axis arrays')
                    values[-1e20]
            elif len(values.shape) == 3:
                if ixs.size == values.shape[2]:
                    values = values[:, :, ixs]
                else:
                    Say.say('! size mismatch between x-axis and y-axis arrays')
                    values[-1e20]
            else:
                raise ValueError('not recognize shape of y-array')

        values = values[np.isfinite(values)]

        if 'log' in scaling or exclude_zero:
            values = values[values != 0]

        limits = [values.min(), values.max()]

        if 'log' in scaling:
            limits = ut.math.get_log(limits)

        if padding:
            value_range = limits[1] - limits[0]
            if np.isfinite(value_range):
                if limits[0] != 0:
                    limits[0] -= padding * value_range
                if limits[1] != 0:
                    limits[1] += padding * value_range

        if 'log' in scaling:
            limits = 10 ** limits

        return limits

    if limits is None or not len(limits):
        return get_limits(values, scaling, other_values, other_limits, exclude_zero, padding)

    assert len(limits) == 2

    limits = np.array(limits)

    if limits[0] is None or limits[1] is None or -np.Inf in limits or np.Inf in limits:
        if values is not None and len(values):
            value_limits = get_limits(
                values, scaling, other_values, other_limits, exclude_zero, padding)
            if limits[0] is None or np.abs(limits[0]) == np.Inf:
                limits[0] = value_limits[0]
            if limits[1] is None or np.abs(limits[1]) == np.Inf:
                limits[1] = value_limits[1]

    return limits


def set_axes_scaling_limits(
    subplot,
    x_scaling='linear', x_limits=None, x_values=None,
    y_scaling='linear', y_limits=None, y_values=None):
    '''
    Set scaling along x- and y-axis for matplotlib.

    Parameters
    ----------
    subplot : plot object
    x_scaling : str : x-axis scaling: 'log', 'linear'
    x_limits : list : min and max limits to impose on x-axis
    x_values : array : values along x-axis (if not impose x_limits)
    y_scaling : str : y-axis scaling: 'log', 'linear'
    y_limits : list : min and max limits to impose on y-axis]
    y_values : array : values along y-axis (if not impose y_limits)

    Returns
    -------
    x_limits_use : array : x-axis limits
    y_limits_use : array : y-axis limits
    '''
    if x_scaling:
        subplot.set_xscale(x_scaling)

    if y_scaling:
        subplot.set_yscale(y_scaling)

    subplot.minorticks_on()

    x_limits_use = None
    if x_limits is not None or x_values is not None:
        x_limits_use = get_axis_limits(x_limits, x_values, x_scaling)

        x_limits_range = max(x_limits_use) - min(x_limits_use)

        subplot.set_xlim(x_limits_use)

        if 'log' in x_scaling and min(x_limits_use) > 1e-4 and max(x_limits_use) < 1e4:
            # re-format tick labels for log axis to be float if near 1
            tick_values = subplot.get_xticks()
            tick_labels = []
            for tick_value in tick_values:
                tick_label = '{:.4f}'.format(tick_value)
                while tick_label[-1] in ['0', '.'] and '.' in tick_label:
                    tick_label = tick_label[:-1]
                tick_labels.append(tick_label)
            subplot.set_xticklabels(tick_labels)
            #if min(x_limits_use) < 1:
            #    subplot.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            #else:
            #    subplot.xaxis.set_major_formatter(FormatStrFormatter('%d'))

            subplot.set_xlim(x_limits_use)

        elif 'linear' in x_scaling and x_limits_range > 4:
            # increase number of major tick labels for linear axis
            if x_limits_range < 10:
                tick_values = np.arange(-100, 100, 1)
                subplot.xaxis.set_ticks(tick_values)
                subplot.xaxis.set_ticks(tick_values + 0.5, minor=True)
            elif x_limits_range < 20:
                tick_values = np.arange(-100, 100, 2)
                subplot.xaxis.set_ticks(tick_values)
                subplot.xaxis.set_ticks(tick_values + 1, minor=True)
            subplot.set_xlim(x_limits_use)

    y_limits_use = None
    if y_limits is not None or y_values is not None:
        y_limits_use = get_axis_limits(y_limits, y_values, y_scaling, x_values, x_limits_use)

        subplot.set_ylim(y_limits_use)

        if 'log' in y_scaling:
            log_y_limits_range = np.log10(max(y_limits_use)) - np.log10(min(y_limits_use))
            if log_y_limits_range < 12:
                # ensure that log plot has major ticks at each factor of 10 on y-axis
                tick_values = 10 ** np.arange(-30., 30., 1)
                subplot.yaxis.set_ticks(tick_values)

                minor_tick_values = []
                if log_y_limits_range < 8:
                    for tick_value in tick_values:
                        minor_tick_values.extend(tick_value * np.arange(2, 10, 1))
                subplot.yaxis.set_ticks(minor_tick_values, minor=True)

                subplot.set_ylim(y_limits_use)

    return x_limits_use, y_limits_use


#===================================================================================================
# axis creation
#===================================================================================================
def make_axis_secondary(
    subplot, axis_name, axis_limits, tick_locations, tick_labels, axis_label):
    '''
    Make secondary axis label.

    Parameters
    ----------
    subplot : plot object
    axis_name : str : 'x', 'y'
    axis_limits : list : min and max limits for axis (this should be same as for normal axis)
    tick_locations : array-like : locations of axis ticks
    tick_labels : str : labels for ticks
    axis_label : label for axis
    '''
    if axis_name == 'x':
        subplot_2 = subplot.twiny()
    elif axis_name == 'y':
        subplot_2 = subplot.twinx()

    subplot_2.set_xticks(tick_locations)
    subplot_2.set_xticklabels(tick_labels)
    subplot_2.set_xlim(axis_limits)
    subplot_2.set_xlabel(axis_label, labelpad=9)
    subplot_2.tick_params(pad=3)


def make_axis_secondary_time(subplot, time_kind, time_limits, Cosmology):
    '''
    Make secondary axis for time, look-back time, redshift, or scale-factor.

    Parameters
    ----------
    subplot : plot object
    time_kind : str : 'time', 'time.lookback', 'redshift', 'scalefactor'
    time_limits : list : min and max limits for time_kind
    Cosmology : class : cosmology class
    '''
    if 'time' in time_kind:
        axis_2_label = 'redshift'
        axis_2_tick_labels = ['6', '4', '3', '2', '1', '0.5', '0.2', '0']
        axis_2_tick_values = [float(v) for v in axis_2_tick_labels]
        axis_2_tick_locations = Cosmology.convert_time(time_kind, 'redshift', axis_2_tick_values)

    elif time_kind == 'redshift':
        axis_2_label = 'lookback time $\\left[ {\\rm Gyr} \\right]$'
        axis_2_tick_labels = ['0', '4', '8', '10', '11', '12', '12.5', '13']
        axis_2_tick_values = [float(v) for v in axis_2_tick_labels]
        axis_2_tick_locations = Cosmology.convert_time(
            time_kind, 'time.lookback', axis_2_tick_values)

    make_axis_secondary(
        subplot, 'x', time_limits, axis_2_tick_locations, axis_2_tick_labels, axis_2_label)


#===================================================================================================
# labeling
#===================================================================================================
color_map_dict = {
    'BlackBlueWhite': {
        'red': (
            (0, 0, 0),
            (0.5, 0, 0),
            (1, 1, 1),
        ),
        'green': (
            (0, 0, 0),
            (0.5, 0, 0),
            (1, 1, 1),
        ),
        'blue': (
            (0, 0, 0),
            (0.5, 1, 1),
            (1, 1, 1),

        )
    },

    'BlackYellowWhite': {
        'red': (
            (0, 0, 0),
            (0.5, 1, 1),
            (1, 1, 1),
        ),
        'green': (
            (0, 0, 0),
            (0.5, 200 / 255, 200 / 255),
            (1, 1, 1),
        ),
        'blue': (
            (0, 0, 0),
            (0.5, 0, 0),
            (1, 1, 1),
        )
    }
}


def get_color(color_name):
    '''
    .
    '''
    color_dict = {
        'black': (0, 0, 0),
        'grey.mid': (90, 90, 90),
        'grey.dark': (50, 50, 50),
        'grey.lite': (180, 180, 180),
        'grey.lite2': (230, 230, 230),
        'blue.mid': (10, 10, 255),
        'blue.dark': (0, 0, 135),
        'blue.lite': (40, 150, 255),
        'blue.lite2': (170, 225, 255),
        'red.mid': (225, 0, 0),
        'red.dark': (150, 0, 0),
        'red.lite': (255, 105, 105),
        'red.lite2': (255, 160, 160),
        'green.mid': (0, 155, 0),
        'green.dark': (0, 100, 0),
        'green.lite': (0, 225, 0),
        'green.lite2': (0, 255, 0),
        'orange.mid': (235, 120, 0),
        'orange.dark': (160, 80, 0),
        'orange.lite': (255, 175, 0),
        'orange.lite2': (255, 205, 0),
        'violet.mid': (148, 0, 211),
        'violet.dark': (128, 0, 128),
        'violet.lite': (238, 130, 238),
        'violet.lite2': (245, 160, 245),
        'yellow.mid': (255, 185, 15),
        'yellow.dark': (160, 120, 8),
        'yellow.lite': (255, 255, 0),
        'brown.mid': (210, 105, 30),
        'brown.dark': (190, 85, 10),
    }

    return tuple(np.array(color_dict[color_name]) / 255)


def get_colors(color_number=0, reverse=False, use_black=False):
    '''
    Get list of colors.

    Parameters
    ----------
    color_number : int : number of colors to use
    reverse : bool : whether to reverse color order
    use_black : bool : whether to use black color

    Returns
    -------
    colors : list : tuples of RGB values
    '''
    if color_number is None:
        color_number = 10
    if use_black:
        color_number -= 1

    if color_number == 0:
        colors = []
    elif color_number == 1:
        colors = ['blue.mid']
    elif color_number == 2:
        #colors = ['blue.mid', 'red.mid']
        colors = ['blue.mid', 'orange.mid']
    elif color_number == 3:
        #colors = ['blue.mid', 'green.mid', 'red.mid']
        colors = ['blue.mid', 'orange.mid', 'red.mid']
    elif color_number == 4:
        colors = ['blue.mid', 'green.mid', 'orange.mid', 'red.mid']
    elif color_number == 5:
        colors = ['violet.mid', 'blue.mid', 'green.mid', 'orange.mid', 'red.mid']
    elif color_number == 6:
        colors = ['violet.mid', 'blue.mid', 'green.mid', 'yellow.mid', 'orange.mid', 'red.mid']
    elif color_number == 7:
        colors = ['violet.mid', 'blue.dark', 'blue.lite', 'green.mid', 'orange.mid', 'red.mid',
                  'red.dark']
    elif color_number == 8:
        colors = ['violet.mid', 'blue.dark', 'blue.mid', 'green.mid', 'yellow.mid',
                  'orange.mid', 'red.mid', 'red.dark']
    elif color_number == 9:
        colors = ['violet.lite', 'violet.dark', 'blue.lite', 'blue.dark', 'green.mid', 'green.dark',
                  'yellow.mid', 'orange.mid', 'red.mid']
    elif color_number == 10:
        colors = ['violet.dark', 'violet.mid', 'blue.dark', 'blue.mid', 'green.dark', 'green.mid',
                  'yellow.mid', 'orange.mid', 'red.mid', 'red.dark']
    elif color_number == 11:
        colors = ['violet.dark', 'violet.mid', 'blue.dark', 'blue.mid', 'green.dark', 'green.mid',
                  'yellow.mid', 'orange.mid', 'orange.dark', 'red.mid', 'red.dark']
    elif color_number == 12:
        colors = ['violet.dark', 'violet.mid', 'blue.dark', 'blue.mid', 'green.dark', 'green.mid',
                  'yellow.mid', 'yellow.dark', 'orange.mid', 'orange.dark', 'red.mid', 'red.dark']
    elif color_number == 13:
        colors = ['violet.lite', 'violet.dark', 'blue.lite', 'blue.dark',
                  'green.lite', 'green.mid', 'green.dark',
                  'yellow.mid', 'orange.mid', 'orange.dark',
                  'red.lite', 'red.mid', 'red.dark']
    elif color_number == 14:
        colors = ['violet.lite', 'violet.dark', 'blue.lite', 'blue.dark',
                  'green.lite', 'green.mid', 'green.dark',
                  'yellow.mid', 'orange.mid', 'orange.dark',
                  'red.lite', 'red.mid', 'red.dark', 'brown.dark']
    elif color_number == 15:
        colors = ['violet.lite', 'violet.dark', 'blue.lite', 'blue.dark',
                  'green.lite', 'green.mid', 'green.dark',
                  'yellow.mid', 'orange.mid', 'orange.dark',
                  'red.lite', 'red.mid', 'red.dark', 'brown.mid', 'brown.dark']
    else:
        colors = np.r_[color_number * ['black']]
    """
    if not color_number:
        color_number = 7
    if use_black:
        color_number -= 1
    if color_number == 0 and use_black:
        colors = []
    elif color_number == 1:
        colors = ['blue']
    elif color_number == 2:
        colors = ['blue', 'red']
    elif color_number == 3:
        colors = ['blue', 'green', 'red']
    elif color_number == 4:
        colors = ['blue', 'green', 'orange', 'red']
    elif color_number == 5:
        colors = ['magenta', 'blue', 'green', 'orange', 'red']
    elif color_number == 6:
        colors = ['magenta', 'cyan', 'blue', 'green', 'orange', 'red']
    elif color_number == 7:
        #colors = ['magenta', 'cyan', 'blue', 'green', 'orange', 'red', 'gray']
        colors = ['violet', 'blue', 'green', 'yellow', 'orange', 'red', 'red']
        #colors = ['348ABD', 'A60628', '7A68A6', '467821', 'D55E00', 'CC79A7', '56B4E9',
        '009E73', 'F0E442', '0072B2']
    else:
        print('! not support color number = {}'.format(color_number))
        colors = np.r_[color_number * 'black']
    """
    if use_black:
        colors = ['black'] + colors
        if color_number == 1:
            # switch from black and blue to black and red
            colors[1] = 'red.mid'

    if reverse:
        colors = colors[::-1]

    return [get_color(color_name) for color_name in colors]


def get_line_styles(line_number=0, reverse=False):
    '''
    Get list of line styles.

    Parameters
    ----------
    line_number : int : number of line types to use
    reverse : bool : whether to reverse line type order

    Returns
    -------
    line_styles : list
    '''
    if not line_number:
        line_number = 7
    elif line_number == 1:
        line_styles = ['-']
    elif line_number == 2:
        line_styles = ['-', '--']
    elif line_number == 3:
        line_styles = ['-', '--', ':']
    elif line_number == 4:
        line_styles = ['-', '--', '-.', ':']
    else:
        line_styles = np.r_[line_number * '-']

    if reverse:
        line_styles = line_styles[::-1]

    return line_styles


class LabelClass:
    '''
    Class for creating string label for plot axes and legends.
    '''
    mass_units = '{\\rm M}_\odot'
    time_units = '{\\rm Gyr}'
    distance_units = '{\\rm kpc}'
    velocity_units = '{\\rm km \, s^{-1}}'

    label_dict = {
        'redshift': {
            'words': 'redshift', 'symbol': 'z', 'units': ''},

        'time': {
            'words': 'time', 'symbol': 't', 'units': time_units},
        'time.lookback': {
            'words': 'lookback time', 'symbol': 't_{\\rm lookback}', 'units': time_units},
        'age': {
            'words': 'age', 'symbol': 't_{\\rm age}', 'units': time_units},
        'form.time': {
            'words': 'formation time', 'symbol': 't_{\\rm form}', 'units': time_units},
        'consume.time': {
            'words': 'consumption time', 'symbol': 't_{\\rm consume}', 'units': time_units},

        'smooth.length': {
            'words': 'smoothing length', 'symbol': 'h', 'units': '{\\rm pc}'},

        'radius': {
            'words': 'radius', 'symbol': 'r', 'units': distance_units},
        'radius.50': {
            'words': 'radius', 'symbol': 'R_{50}', 'units': distance_units},
        'distance': {
            'words': 'distance', 'symbol': 'd', 'units': distance_units},
        'host.distance': {
            'words': 'distance', 'symbol': 'd_{\\rm host}', 'units': distance_units},
        'central.distance': {
            'words': 'distance', 'symbol': 'd_{\\rm cen}', 'units': distance_units},
        'distance/R200m': {
            'words': 'distance/R200m', 'symbol': 'd / R_{\\rm 200m}', 'units': ''},
        'distance/Rvir': {
            'words': 'distance/Rvir', 'symbol': 'd / R_{\\rm vir}', 'units': ''},
        'distance.peri': {
            'words': 'pericenter distance', 'symbol': 'd_{\\rm peri}', 'units': distance_units},
        'distance.apo': {
            'words': 'apocenter distance', 'symbol': 'd_{\\rm apo}', 'units': distance_units},
        'height': {
            'words': 'height', 'symbol': 'Z', 'units': distance_units},

        'temperature': {
            'words': 'temperature', 'symbol': 'T', 'units': '{\\rm K}'},

        'pressure': {
            'words': 'pressure', 'symbol': 'P', 'units': ''},

        'entropy': {
            'words': 'entropy', 'symbol': 'S', 'units': '{\\rm kev \, cm^{-2}}'},

        'density': {
            'words': 'density', 'symbol': '\\rho',
            'units': '{{\\rm M}}_\\odot \, {{\\rm kpc}} ^ {{-{}}}'},
        'mass.density': {
            'words': 'density', 'symbol': '\\rho',
            'units': '{{\\rm M}}_\\odot \, {{\\rm kpc}} ^ {{-{}}}'},
        'number.density': {
            'words': 'number density', 'symbol': 'n', 'units': '{{\\rm cm}} ^ {{-{}}}'},
        'velocity.rad': {
            'words': 'radial velocity', 'symbol': 'v_{\\rm rad}', 'units': velocity_units},
        'velocity.tan': {
            'words': 'tangential velocity', 'symbol': 'v_{\\rm tan}', 'units': velocity_units},
        'velocity.total': {
            'words': 'velocity', 'symbol': 'v_{\\rm tot}', 'units': velocity_units},
        'velocity.beta': {
            'words': 'velocity', 'symbol': '\\beta', 'units': ''},

        'vel.circ': {
            'words': 'circular velocity', 'symbol': 'v_{\\rm circ}', 'units': velocity_units},
        'vel.circ.peak': {
            'words': 'peak circular velocity', 'symbol': 'V_{\\rm circ,peak}',
            'units': velocity_units},
        'vel.circ.max': {
            'words': 'max circular velocity', 'symbol': 'V_{\\rm circ,max}',
            'units': velocity_units},

        'vel.disp': {
            'words': 'velocity dispersion', 'symbol': '\\sigma_{\\rm v,total}',
            'units': velocity_units},
        'vel.std': {'words': 'velocity dispersion', 'symbol': '\\sigma_{\\rm velocity}',
                    'units': velocity_units},
        'vel.std.50': {
            'words': 'velocity dispersion', 'symbol': '\\sigma_{\\rm velocity,1D}',
            'units': velocity_units},
        'star.vel.disp': {
            'words': 'stellar velocity dispersion', 'symbol': '\\sigma_{\\rm v,star}',
            'units': velocity_units},
        'star.vel.std': {
            'words': 'velocity dispersion', 'symbol': '\\sigma_{\\rm velocity,star}',
            'units': velocity_units},
        'star.vel.std.50': {
            'words': 'velocity dispersion',
            'symbol': '\\sigma_{\\rm velocity,star}', 'units': velocity_units},
        'star.vel.circ.50': {
            'words': 'velocity dispersion', 'symbol': 'v_{\\rm circ,1D}',
            'units': velocity_units},

        'mass': {
            'words': 'mass', 'symbol': 'M', 'units': mass_units},
        'mass.bound': {
            'words': 'bound mass', 'symbol': 'M_{\\rm bound}', 'units': mass_units},
        'mass.peak': {
            'words': 'peak mass', 'symbol': 'M_{\\rm peak}', 'units': mass_units},
        'mass.200m': {
            'words': 'virial mass', 'symbol': 'M_{\\rm 200m}', 'units': mass_units},
        'total.mass': {
            'words': 'mass', 'symbol': 'M_{\\rm total}', 'units': mass_units},
        'subhalo.mass': {
            'words': 'subhalo mass', 'symbol': 'M_{\\rm subhalo}', 'units': mass_units},
        'halo.mass': {
            'words': 'halo mass', 'symbol': 'M_{\\rm halo}', 'units': mass_units},
        'star.mass': {
            'words': 'stellar mass', 'symbol': 'M_{\\rm star}', 'units': mass_units},
        'gas.mass': {
            'words': 'gas mass', 'symbol': 'M_{\\rm gas}', 'units': mass_units},

        'mag.r': {'words': 'r-band magnitude', 'symbol': 'M_r', 'units': ''},

        'baryon.fraction': {
            'words': 'baryon fraction $\\rho_{\\rm b} / \\rho_{\\rm tot}$',
            'symbol': '\\frac{\\rho_{\\rm baryon}}{\\rho_{\\rm total}}',
            'units': '\Omega_{\\rm b} / \Omega_{\\rm m}'},

        'sat.first.t': {
            'words': 'time since first infall', 'symbol': '\Delta\,t_{first\,infall}',
            'units': time_units},
        'sat.host.t': {
            'words': 'time since host infall', 'symbol': '\Delta\,t_{host\,infall}',
            'units': time_units},
        'sat.last.t': {
            'words': 'time since last infall', 'symbol': '\Delta\,t_{last\,infall}',
            'units': time_units},

        'star.form.rate': {
            'words': 'star formation rate', 'symbol': 'SFR',
            'units': '{\\rm M_{\odot} \, yr^{-1}}'},
        'form.rate': {
            'words': 'star formation rate', 'symbol': 'SFR',
            'units': '{\\rm M_{\odot} \, yr^{-1}}'},
        'star.form.rate.specific': {
            'words': 'specific star formation rate', 'symbol': 'sSFR', 'units': '{yr^{-1}}'},
        'form.rate.specific': {
            'words': 'specific star formation rate', 'symbol': 'sSFR', 'units': '{yr^{-1}}'},

        'metallicity.total': {'words': '', 'symbol': '{\\rm Z} / {\\rm H}', 'units': ''},
        'metallicity.metals': {'words': '', 'symbol': '{\\rm Z} / {\\rm H}', 'units': ''},
        'metallicity.he': {'words': '', 'symbol': '{\\rm He} / {\\rm H}', 'units': ''},
        'metallicity.c': {'words': '', 'symbol': '{\\rm C} / {\\rm H}', 'units': ''},
        'metallicity.n': {'words': '', 'symbol': '{\\rm N} / {\\rm H}', 'units': ''},
        'metallicity.o': {'words': '', 'symbol': '{\\rm O} / {\\rm H}', 'units': ''},
        'metallicity.ne': {'words': '', 'symbol': '{\\rm Ne} / {\\rm H}', 'units': ''},
        'metallicity.mg': {'words': '', 'symbol': '{\\rm Mg} / {\\rm H}', 'units': ''},
        'metallicity.si': {'words': '', 'symbol': '{\\rm Si} / {\\rm H}', 'units': ''},
        'metallicity.s': {'words': '', 'symbol': '{\\rm S} / {\\rm H}', 'units': ''},
        'metallicity.ca': {'words': '', 'symbol': '{\\rm Ca} / {\\rm H}', 'units': ''},
        'metallicity.fe': {'words': '', 'symbol': '{\\rm Fe} / {\\rm H}', 'units': ''},
        'metallicity.o-metallicity.fe': {
            'words': '', 'symbol': '{\\rm O} / {\\rm Fe}', 'units': ''},
        'metallicity.mg-metallicity.fe': {
            'words': '', 'symbol': '{\\rm Mg} / {\\rm Fe}', 'units': ''},
        'metallicity.si-metallicity.fe': {
            'words': '', 'symbol': '{\\rm Si} / {\\rm Fe}', 'units': ''},
        'metallicity.ca-metallicity.fe': {
            'words': '', 'symbol': '{\\rm Ca} / {\\rm Fe}', 'units': ''},
        'metallicity.alpha-metallicity.fe': {
            'words': '', 'symbol': '{\\rm \\alpha} / {\\rm Fe}', 'units': ''},
    }

    def get_label(
        self, property_name, property_statistic='', species_name='', property_limits=[],
        get_words=False, get_units=True, redshift=None, dimension_number=3):
        '''
        Get label for property.

        Parameters
        ----------
        property_name : str : name of property to get label of
        property_statistic : str : statistic of property:
            'prob', 'prob.cum', 'sum.cum', 'med', 'ave'
        species_name : str : name of particle species to add to label
        property_limits : list : min and max limits of property to add to label
        get_words : bool : whether to get words instead of symbol
        get_units : bool : whether to include units in label
        redshift : float : redshift to add - example: (z=0)
        dimension_number : int : number of spatial dimensions (for units)

        Returns
        -------
        label : str
        '''
        property_name_in = property_name  # keep copy of input

        property_name = property_name.replace('log ', '')
        #property_name = property_name.replace('.std', '')
        if 'metallicity' in property_name:
            property_name = property_name.replace(' ', '')

        if 'density' in property_statistic:
            property_name = property_statistic

        # convert element names to symbols
        for element in ut.constant.element_symbol_from_name:
            if '.' + element in property_name:
                property_name = property_name.replace(
                    element, ut.constant.element_symbol_from_name[element])

        units_label = ''

        if 'metallicity' in property_name:
            get_words = False

        if get_words:
            label_kind = 'words'
        else:
            label_kind = 'symbol'

        if 'star.' in property_name and not species_name:
            property_name = property_name.replace('star.', '')
            species_name = 'star'
        elif 'gas.' in property_name and not species_name:
            property_name = property_name.replace('gas.', '')
            species_name = 'gas'

        if not np.isscalar(species_name):
            if len(species_name) == 1:
                species_name = species_name[0]
            elif len(species_name) == 2 and 'star' in species_name and 'gas' in species_name:
                species_name = 'baryon'
            else:
                species_name = 'total'

        if property_name in self.label_dict:
            # easy case, use directly from dictionary
            label = self.label_dict[property_name][label_kind]
            units_label = self.label_dict[property_name]['units']
        else:
            # have to determine dictionary key for property
            if ('mass' in property_name and 'fraction' in property_statistic and
                    species_name == 'baryon'):
                property_name = 'baryon.fraction'

            elif 'velocity' in property_name:
                units_label = self.velocity_units

                if 'velocity.tan / velocity.rad' in property_name:
                    label = 'v_{\\rm tan} / v_{\\rm rad}'
                    units_label = ''

                elif 'velocity.rad / velocity.tan' in property_name:
                    label = 'v_{\\rm rad} / v_{\\rm tan}'
                    units_label = ''
                else:
                    if get_words:
                        label = 'velocity'
                    else:
                        label = 'v'

            elif property_name == 'sfr':
                property_name = 'star.form.rate'
            elif property_name == 'ssfr':
                property_name = 'star.form.rate.specific'

            # now try to get label
            try:
                label = self.label_dict[property_name][label_kind]
                units_label = self.label_dict[property_name]['units']
            except (KeyError, ValueError):
                print('! not recognize property = ' + property_name)
                label = property_name

        # adjustments
        if 'metallicity' in property_name:
            label = '\\left[' + label + '\\right]'

        if 'density' in property_name:
            if dimension_number in [2, 1] and get_words:
                label = 'surface ' + label

            if '*r^2' in property_name or '*r^2' in property_statistic:
                label += ' x $r^2$'
            elif '*r' in property_name or '*r' in property_statistic:
                label += ' x $r$'

        if species_name and get_words:
            species_label = species_name.replace('star', 'stellar').replace('dark', 'dark matter')
            label = species_label + ' ' + label

        if (property_name is 'baryon.fraction' and 'cum' in property_statistic and
                'prob' not in property_statistic):
            label = label.replace('\\rho', 'm')

        if 'log ' in property_name_in:
            label = '\log \, ' + label

        if not get_words:
            #if species_name:
            #    if 'smooth.length' in property_name:
            #        label = label.replace('h', 'h_{{\\rm {}}}'.format(species_name))
            #    elif 'number.density' in property_name:
            #        label = label.replace('n', 'n_{{\\rm {}}}'.format(species_name))
            #    elif 'density' in property_name:
            #        label = label.replace('\\rho', '\\rho_{{\\rm {}}}'.format(species_name))

            if species_name and 'metallicity' not in property_name and '_' not in label:
                label += '_{{\\rm {}}}'.format(species_name)

            if 'density' in property_name:
                if dimension_number in [2, 1]:
                    label = label.replace('\\rho', '\\Sigma')

                if '*r^2' in property_name or '*r^2' in property_statistic:
                    label += 'r^2'
                elif '*r' in property_name or '*r' in property_statistic:
                    label += 'r'

                units_label = units_label.format(dimension_number)

            if 'distance' in property_name:
                if dimension_number == 2:
                    label = label.replace('d', 'd_{{\\rm proj}')

            if '.cum' in property_statistic:
                label += '(< r)'

            if redshift:
                label += '(z={:.1f})'.format(redshift)

            if ('.std' in property_name_in or 'std' in property_statistic or
                    'disp' in property_statistic):
                label = '\\sigma_{{}}'.format(label)

            if 'prob' in property_statistic:
                if 'cum' in property_statistic:
                    label = 'p(< {})'.format(label)
                else:
                    label = '{{\\rm d}}p / {{\\rm d}}{}'.format(label)
            elif 'histogram' in property_statistic:
                if 'cum' in property_statistic:
                    label = 'N(< {})'.format(label)
                else:
                    label = '{{\\rm d}}N / {{\\rm d}}{}'.format(label)

            if property_limits is not None and len(property_limits):
                label += self.get_label_property_limits(property_name, property_limits)

            label = '$' + label + '$'

        if get_units and units_label:
            if property_limits is None or not len(property_limits):
                units_label = '\\left[' + units_label + '\\right]'
            label += ' $' + units_label + '$'

        return label

    def get_label_property_limits(self, property_name='', limits=[]):
        '''
        Get label property limits.

        Parameters
        ----------
        limits : list : min and max limits for mass_kind

        Returns
        -------
        label : str
        '''
        digits = 1

        if 'mag.' in property_name:
            limits = -np.array(limits)

        lower = True
        upper = True
        if limits[0] is None or limits[0] == -np.Inf:
            lower = False
        if limits[1] is None or limits[1] == np.Inf:
            upper = False

        if lower and upper:
            if np.round(limits[0]) == limits[0] and np.round(limits[1]) == limits[1]:
                digits = 0
            label = ' = \\left[ {}, {} \\right]'.format(
                ut.io.get_string_from_numbers(limits[0], digits),
                ut.io.get_string_from_numbers(limits[1], digits))
        elif lower:
            label = ' > {}'.format(ut.io.get_string_from_numbers(limits[0], digits, strip=True))
        elif upper:
            label = ' < {}'.format(ut.io.get_string_from_numbers(limits[1], digits, strip=True))
        else:
            label = ''

        return label


Label = LabelClass()


#===================================================================================================
# legends
#===================================================================================================
def make_label_legend(subplot, label='', location='best', font_size='x-small'):
    '''
    Make label legend (with no line markers).

    Parameters
    ----------
    subplot : plot object
    location : str : location for label
    size : str or float : font size for label
    '''
    subplot.legend(
        [plt.Line2D((0, 0), (0, 0), linestyle='')], [label],
        #bbox_to_anchor=(-0.05, -0.06),
        loc=location,
        fontsize=font_size, handlelength=0
    )


def make_legends(
    subplot, location='best', font_size='x-small',
    time_value=None, time_kind='redshift', time_location='lower left', time_size='x-small'):
    '''
    Make legend[s] for:
      properties (associated with lines), if more than one;
      time/redshift (if input)

    Parameters
    ----------
    subplot : plot object
    location : str : location for primary property label[s]
    font_size : str or float : font size for primary property labels
    time_value : float : value for time
    time_kind : str : kind of time: 'redshift', 'scalefactor', 'time'
    time_location : str : location for time label
    time_size : str or float : font size for time label
    '''
    if time_value is not None:
        # make legend for time
        if time_kind == 'redshift':
            time_label = '$z={:.1f}$'.format(time_value)
        elif time_kind == 'scalefactor':
            time_label = '$a={:.1f}$'.format(time_value)
        elif time_kind == 'time':
            time_label = '$t={:.1f}$'.format(time_value)

        time_label = time_label.replace('.0', '')

        time_legend = subplot.legend(
            [plt.Line2D((0, 0), (0, 0), linestyle='')], [time_label],
            bbox_to_anchor=(-0.21, -0.24),
            #bbox_to_anchor=(-0.05, -0.06),
            loc=time_location,
            fontsize=time_size, handlelength=0)

        #time_legend.get_frame().set_alpha(0.5)  # affects only border

        subplot.add_artist(time_legend)

    # make legend for properties, if more than one
    prop_labels = subplot.get_legend_handles_labels()[1]

    if len(prop_labels) > 1:
        # make legend for properties
        _property_legend = subplot.legend(loc=location, fontsize=font_size)


#===================================================================================================
# output
#===================================================================================================
def get_time_name(time_kind='redshift', snapshot_dict=None, time_value=None):
    '''
    Get string to add to file name to label time/reshift.

    Parameters
    ----------
    time_kind : str : 'time', 'redshift', 'scalefactor'
    snapshot_dict : dict : dictionary that contains time information for single snapshot
    time_value : value for time_kind (if not snapshot_dict)

    Returns
    -------
    time_name : str
    '''

    if time_kind == 'redshift':
        time_name = '_z'
    elif time_kind == 'scalefactor':
        time_name = '_a'
    elif time_kind == 'time':
        time_name = '_t'

    if snapshot_dict:
        time_value = snapshot_dict[time_kind]
    elif not time_value:
        raise ValueError('need to input snapshot dictionary or time value')

    time_name += '.' + ut.io.get_string_from_numbers(time_value, 1, strip=True)

    return time_name


def get_file_name(
    y_property_name, x_property_name, species_name='',
    time_kind='redshift', snapshot_dict=None, time_value=None,
    host_distance_limits=None, prefix=''):
    '''
    Get file name for plot.

    Parameters
    ----------
    y_property_name : str : name of property on y-axis
    x_property_name : str : name of proerty on x-axis
    species_name : str : name of particle species
    time_kind : str : 'time', 'redshift', 'scalefactor'
    snapshot_dict : dict : dictionary that contains time information for single snapshot
    time_value : value for time_kind (if not snapshot_dict)
    host_distance_limits : list : min and max host distances used
    prefix : str : str to add to beginning of file name

    Returns
    -------
    plot_name : str
    '''
    plot_name = y_property_name

    if 'position' in x_property_name or 'distr' in x_property_name:
        plot_name += '_' + x_property_name
    else:
        plot_name += '_v_' + x_property_name

    if species_name:
        plot_name = species_name + '.' + plot_name

    # adjust names
    plot_name = plot_name.replace('log ', '')
    plot_name = plot_name.replace('.average', '.ave')
    plot_name = plot_name.replace('.median', '.med')
    plot_name = plot_name.replace('.total', '.tot')
    for name in ut.constant.element_symbol_from_name:
        if name in plot_name:
            plot_name = plot_name.replace(name, ut.constant.element_symbol_from_name[name])
    for element in ['alpha', 'mg', 'si', 'ca', 'o']:
        plot_name = plot_name.replace(
            'metallicity.{} - metallicity.fe'.format(element), 'metallicity.{}-fe'.format(element))
        plot_name = plot_name.replace(
            'metallicity.{}-metallicity.fe'.format(element), 'metallicity.{}-fe'.format(element))

    # add distance limits
    if host_distance_limits is not None and len(host_distance_limits):
        plot_name += '_d.{:.0f}-{:.0f}'.format(host_distance_limits[0], host_distance_limits[1])

    # add time/redshift
    if time_kind:
        if time_kind == 'redshift':
            plot_name += '_z'
        elif time_kind == 'scalefactor':
            plot_name += '_a'
        elif time_kind == 'time':
            plot_name += '_t'
        else:
            raise ValueError('! not recognize time_kind = {}'.format(time_kind))

        if snapshot_dict:
            time_value = snapshot_dict[time_kind]
        elif not time_value:
            raise ValueError('need to input snapshot dictionary or time_value')
        value = ut.io.get_string_from_numbers(time_value, 1, strip=True, exponential=False)
        plot_name += '.{}'.format(value)

    # add prefix
    if prefix:
        plot_name = prefix + '_' + plot_name

    # clean problematic characters
    plot_name = plot_name.replace(' ', '.')
    plot_name = plot_name.replace('*', '.')
    plot_name = plot_name.replace('^', '')

    return plot_name


def parse_output(write_to_file=False, file_name='', directory='.', file_format='pdf'):
    '''
    Parse whether to display plot or write to file.

    Parameters
    ----------
    write_to_file : bool : whether to write plot to file
    file_name : str : file name for plot
    directory : str : directory to write plot file
    file_format : str : file format ('pdf' is default)
    '''
    if write_to_file:
        if file_name[-4:] == '.pdf':
            file_format = 'pdf'

        file_format = file_format.replace('.', '')

        file_extension = '.' + file_format
        if file_extension not in file_name:
            file_name += file_extension

        plot_directory = ut.io.get_path(directory, create_path=True)

        plt.savefig(plot_directory + file_name, format=file_format)
        print('* wrote ' + plot_directory + file_name)
    else:
        plt.show(block=False)
