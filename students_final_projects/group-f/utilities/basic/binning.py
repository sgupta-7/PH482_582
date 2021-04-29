'''
Utility functions for binning arrays.

@author: Andrew Wetzel <arwetzel@gmail.com>
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import numpy as np
# local ----
from . import array, io, math


#===================================================================================================
# binning utility
#===================================================================================================
def get_bin_limits(values, bin_kind='limits'):
    '''
    Get bin limits (list) according to input value array and bin kind.

    Parameters
    ----------
    values : list : two values
    bin_kind : str :
        'limits' = return input values
        'error' = return bin defined by values[0] +/- values[1]
        'error.X' = return bin defined by values[0] +/- X * values[1]

    Returns
    -------
    bin_limits : list : min and max values (edges) of bin
    '''
    if bin_kind == 'limits':
        bin_limits = [np.min(values), np.max(values)]
    elif 'error' in bin_kind:
        sigma = 1.0
        if 'error.' in bin_kind:
            sigma = float(bin_kind.replace('error.', ''))
        bin_limits = [values[0] - sigma * values[1], values[0] + sigma * values[1]]
    else:
        raise ValueError('not recnognize bin_kind = ' + bin_kind)

    return bin_limits


def get_bin_indices(
    values, bin_mins, bin_max=None, round_kind='down', clip_to_bin_limits=False, warn_outlier=True):
    '''
    Get bin indices for each input value, using given rounding kind.

    Add extra bin to get outliers beyond bin_max and normalize so bin count starts at 0.
    If clip_to_bin_limits is false: if below min bin value, assign -1, if above max bin value,
    assign len(bin_mins).
    Uses bin_mins as is, regardless if sorted.

    Parameters
    ----------
    values : array : value[s]
    bin_mins : array : lower limits of bins
    bin_max : float upper limit for last bin
    round_kind : str : direction to round: 'up', 'down', 'near'
    clip_to_bin_limits : bool : whether to clip input values to be within bin range (count all)
    warn_outlier : bool : whether to warn if there are values beyond bin range

    Returns
    -------
    bin_indices : array : bin index for each input value
    '''
    Say = io.SayClass(get_bin_indices)

    scalarize = False
    if np.isscalar(values):
        values = [values]
        scalarize = True
    values = np.array(values)

    bin_mins = array.arrayize(bin_mins)
    if bin_mins.size == 1:
        bin_max = np.Inf
    elif bin_max is None:
        bin_max = 2 * bin_mins[-1] - bin_mins[-2]

    assert round_kind in ['up', 'down', 'near']

    # add bin max to catch outliers and round properly
    bin_edges = np.append(bin_mins, bin_max)
    bin_indices = np.digitize(values, bin_edges) - 1

    bin_indices = bin_indices.astype(array.parse_data_type(bin_mins.size))

    if warn_outlier:
        if bin_indices.min() < 0 or bin_indices.max() >= bin_mins.size:
            Say.say('! input value limits = {} exceed bin limits = {}'.format(
                    array.get_limits(values, digit_number=2),
                    array.get_limits(bin_edges, digit_number=2)))

    if round_kind == 'up':
        bin_indices[bin_indices < bin_mins.size - 1] += 1
    elif round_kind == 'near':
        biis = array.get_indices(bin_indices, [0, bin_mins.size - 1.9])  # safely in bin limits
        biis_shift = biis[abs(bin_mins[bin_indices[biis] + 1] - values[biis]) <
                          abs(bin_mins[bin_indices[biis]] - values[biis])]
        bin_indices[biis_shift] += 1  # shift up if that is closer

    if clip_to_bin_limits:
        # clip values to be within input bin limits
        bin_indices = bin_indices.clip(0, bin_mins.size - 1)

    if scalarize:
        bin_indices = bin_indices[0]

    return bin_indices


def get_indices_equal_number_in_bin(values, limits=[], bin_width=0.5, number_in_bin=5):
    '''
    Get indices that randomly sample value array with equal number in each bin.

    Parameters
    ----------
    values : array
    limits : list : min and max limits to impose
    bin_width : float : width of each bin
    number_in_bin : int : number of values to sample in each bin

    Returns
    -------
    val_indices : array : indices corresponding to input values that sample equally
    '''
    Bin = BinClass(limits, bin_width)

    val_indices = []
    for bin_i in range(Bin.number):
        val_indices_bin = array.get_indices(values, Bin._get_bin_limits(bin_i))
        if len(val_indices_bin) < number_in_bin:
            number_in_bin_use = len(val_indices_bin)
        else:
            number_in_bin_use = number_in_bin
        val_indices.extend(array.sample_array(val_indices_bin, number_in_bin_use))

    return np.array(val_indices)


def get_indices_to_match_distribution(
    values_ref, values_select, limits, bin_width=None, bin_number=None):
    '''
    Get indices that sample from from values_select to yield same relative distribution
    (based on binning) as in values_ref.

    Parameters
    ----------
    values_ref : array : reference values
    values_select : array : values to sample from
    values_select : list : min and max limits to impose on values
    bin_width: float : width of each bin
    bin_number : int : number of bins

    Returns
    -------
    array : indices of values_select that sample input distribution as given by values_ref
    '''
    Bin = BinClass(limits, bin_width, bin_number)

    bin_is_ref = Bin.get_bin_indices(values_ref)
    bin_is_select = Bin.get_bin_indices(values_select)

    num_in_bins_ref = np.zeros(Bin.number)
    num_in_bins_select = np.zeros(Bin.number)
    for bin_i in range(Bin.number):
        num_in_bins_ref[bin_i] = np.sum(bin_is_ref == bin_i)
        num_in_bins_select[bin_i] = np.sum(bin_is_select == bin_i)

    frac_in_bins_ref = num_in_bins_ref / values_ref.size
    ibin_mode = np.nanargmax(num_in_bins_ref)
    frac_in_bins_keep = frac_in_bins_ref / frac_in_bins_ref[ibin_mode]
    num_in_bins_keep = np.round(frac_in_bins_keep * num_in_bins_select[ibin_mode])
    vis_select = array.get_arange(values_select)

    vis_keep = []
    for bin_i in range(Bin.number):
        vis_bin = vis_select[bin_i == bin_is_select]
        if bin_i == ibin_mode:
            vis_keep.extend(vis_bin)
        else:
            vis_keep.extend(array.sample_array(vis_bin, num_in_bins_keep[bin_i]))

    return np.array(vis_keep, vis_select.dtype)


#===================================================================================================
# main binning classes
#===================================================================================================
class BinClass(io.SayClass):
    '''
    Get binning information for array of values.
    '''

    def __init__(
        self, limits, width=None, number=None, include_max=False, scaling='', vary_bin_width=False,
        values=None):
        '''
        Assign bin information.

        Parameters
        ----------
        limits : list : min and max limits to impose on all values
        width : float : width of each bin
        number : int : number of bins within limits
        include_max : bool : whether to inclue last bin that starts at max(limits)
        scaling : str : 'log' if want to re-scale limits to log units and use log binning
        vary_bin_width : bool : whether to vary bin width
        values : array : values, if vary_bin_width is True
        '''
        self.number = None
        self.limits = None
        self.limits_input = None
        self.width = None
        self.widths = None
        self.mins = self.mids = self.maxs = None

        self.scaling = str(scaling)

        if vary_bin_width:
            self._assign_bins_variable_width(values, limits, number, include_max)
        else:
            self._assign_bins_fixed_width(limits, width, number, include_max)

    def _assign_bins_fixed_width(self, limits, width, number, include_max=False):
        '''
        Assign bin widths to self, assuming fixed scaling within limits.

        If limit is infinite, set width to infinity.
        If number defined, set width to give that number of bins within limits.

        Parameters
        ----------
        limits : list : min and max *linear* limits to impose
        width : float : bin width
        number : int : number of bins within limits
        include_max : bool : whether to inclue last bin that starts at max(limits)
        '''
        # store input limits, but may have to adjust limits_max if also input bin width
        self.limits_input = np.array([min(limits), max(limits)])

        if 'log' in self.scaling:
            self.log_limits_input = math.get_log(limits)
            self.limits_use = self.log_limits_input
        else:
            self.limits_use = limits

        self.width = self._get_bin_width(self.limits_use, width, number)
        self.mins = array.get_arange_safe(self.limits_use, self.width, include_max)

        if include_max:
            self.mids = self.mins + 0.5 * self.width
            self.maxs = self.mins + self.width
        else:
            if self.mins.size == 1 and np.isinf(self.mins):
                self.mids = np.abs(self.mins)
                self.maxs = np.max(self.limits)
            else:
                self.mids = self.mins + 0.5 * self.width
                self.maxs = self.mins + self.width

        self.number = self.mins.size
        self.widths = np.zeros(self.number) + self.width

        if 'log' in self.scaling:
            self.log_mins = self.mins
            self.mins = 10 ** self.log_mins
            self.log_mids = self.mids
            self.mids = 10 ** self.log_mids
            self.log_maxs = self.maxs
            self.maxs = 10 ** self.log_maxs
            self.log_widths = self.widths
            self.widths = 10 ** self.log_widths
            # store actual limits used, given bin widths
            self.log_limits = np.array([self.log_mins[0], self.log_maxs[-1]])
        self.limits = np.array([self.mins[0], self.maxs[-1]])

    def _assign_bins_variable_width(self, values, limits=[], number=30, include_max=False):
        '''
        Assign bin widths to self, using variable widths to give same number of points in each bin.

        If limit is infinite, set width to infinity.
        If number defined, set width to give that number of bins.

        Parameters
        ----------
        values : array-like : value[s]
        limits : list : min and max linear limits to impose
        number : int : number of bins within limits
        include_max : bool : whether to inclue last bin that starts at max(limits)
        '''
        self.number = number
        values = np.array(values)

        if limits:
            values = values[values >= min(limits)]
            if include_max:
                values = values[values <= max(limits)]
            else:
                values = values[values < max(limits)]
        else:
            # make sure bin goes just beyond value maximum
            limit_max = values.max() * (1 + np.sign(values.max()) * 1e-6)
            if limit_max == 0:
                limit_max += 1e-6
            limits = np.array([values.min(), limit_max])

        self.limits = limits
        if 'log' in self.scaling:
            self.log_limits = math.get_log(limits)
            self.limits_use = self.log_limits
        else:
            self.limits_use = limits

        number_per_bin = int(values.size / number)
        values = np.sort(values)
        self.mins, self.widths = np.zeros((2, number))

        # assign bin minima
        self.mins[0] = limits[0]
        for bin_i in range(1, number):
            # place bin minimum value half-way between adjacent points
            # unless equal, then place just above
            if values[bin_i * number_per_bin - 1] == values[bin_i * number_per_bin]:
                self.mins[bin_i] = values[bin_i * number_per_bin] * (1 + 1e-5)
            else:
                self.mins[bin_i] = 0.5 * (values[bin_i * number_per_bin - 1] +
                                          values[bin_i * number_per_bin])

        # assign bin widths
        bin_indices = np.arange(number - 1)
        self.widths[bin_indices] = self.mins[bin_indices + 1] - self.mins[bin_indices]
        self.widths[-1] = max(self.limits) - self.mins[-1]

        # assign bin centers
        if 'log' in self.scaling:
            self.log_widths = self.widths
            self.widths = 10 ** self.log_widths
            self.log_mins = self.mins
            self.mins = 10 ** self.log_mins
            self.log_mids = self.log_mins + 0.5 * self.log_widths
            self.mids = 10 ** self.log_mids
            self.log_maxs = self.log_mins + self.log_widths
            self.maxs = 10 ** self.log_maxs
        else:
            self.mids = self.mins + 0.5 * self.widths
            self.maxs = self.mins + self.widths

    def _get_bin_width(self, limits, width, number):
        '''
        Get width of single bin.

        If input width <= 0, use single bin across entire range.
        If input limit is infinite, set bin width to infinity.
        If bin number defined, set width to give that number of bins within limits.

        Parameters
        ----------
        limits : list : min and max limits to impose
        width : float : width of bin width
        number : int : number of bins within limits

        Returns
        -------
        bin_width : float : width of single bin
        '''
        if number is None:
            if width is None or width <= 0:
                bin_width = limits[1] - limits[0]
            elif limits[0] == -np.Inf or limits[1] == np.Inf:
                bin_width = np.Inf
            else:
                bin_width = width

        elif width is None and number > 0:
            bin_width = (limits[1] - limits[0]) / number

        else:
            raise ValueError(
                'input bin width = {} and bin number = {}, not sure how to interpret'.format(
                    width, number))

        return bin_width

    def get_bin_limits(self, bin_index):
        '''
        Get linear limits of single bin.

        Parameters
        ----------
        bin_index : int

        Returns
        -------
        list : min and max values (edges) of bin
        '''
        return [self.mins[bin_index], self.mins[bin_index] + self.widths[bin_index]]

    def get_bin_values(self, func_kind):
        '''
        Get values of bin appropriate for function kind, specifically, cumulative versus
        differential binning.

        Parameters
        ----------
        func_kind : str : function kind, to determine appropriate bin values

        Returns
        -------
        values : array : default values of each bin
        '''
        if 'cum' in func_kind:
            values = self.mins
        else:
            values = self.mids

        return values

    def get_bin_indices(
        self, values, round_kind='down', clip_to_bin_limits=False, warn_outlier=True):
        '''
        Get bin indices at each input value, using given rounding kind.
        If self.scaling is 'log', round to nearest bin in log space.

        Parameters
        ----------
        values : array-like : value[s]
        round_kind : str : direction to round: 'up', 'down', 'near'
        clip_to_bin_limits : bool : whether to clip bin values to input range (so count all)
        warn_outlier : bool : whether to print warning if input values beyond bin limits

        Returns
        -------
        bin_indices : array : index of corresponding bin for each input value
        '''
        if 'log' in self.scaling:
            use_values = math.get_log(values)
            use_mins = self.log_mins
            use_max = np.max(self.log_limits)
        else:
            use_values = values
            use_mins = self.mins
            use_max = np.max(self.limits)

        return get_bin_indices(
            use_values, use_mins, use_max, round_kind, clip_to_bin_limits, warn_outlier)

    def get_histogram(self, values, normed=False, weights=None, density=None):
        '''
        Wrapper for np.histogram, using self's bins.

        Parameters
        ----------
        same as np.histogram

        Returns
        -------
        array : numbers or values in each bin
        '''
        if 'log' in self.scaling:
            values_use = math.get_log(values)
            limits_use = self.log_limits + 1e-7  # minor tweak for stability
        else:
            values_use = values
            limits_use = self.limits

        return np.histogram(values_use, self.number, limits_use, normed, weights, density)[0]

    def get_distribution(self, values, normed=False, include_above_limits=True):
        '''
        Get number/probability distribution, differential and cumulative, of input values,
        including Poisson errors, using self's bins.

        Parameters
        ----------
        values : array
        normed : bool : whether to normalize distribution to 1
        include_above_limits : bool : whether to include values above limits for cumulative

        Returns
        -------
        distr : dict : dictionary of histogram of number or probability
        '''
        numbers = self.get_histogram(values, normed=False)

        distr = {}

        # histogram
        distr['sum'] = numbers
        distr['sum.err'] = np.sqrt(numbers)

        # differential distribution
        distr['sum.dif'] = numbers / self.width
        distr['sum.dif.err'] = np.sqrt(numbers) / self.width

        # cumulative distribution
        distr['sum.cum'] = numbers[::-1].cumsum()[::-1]
        if include_above_limits:
            masks = np.isfinite(values)
            number_above_limit = np.sum(values[masks] >= max(self.limits))
            distr['sum.cum'] += number_above_limit
        distr['sum.cum.err'] = np.sqrt(distr['sum.cum'])

        if normed:
            for k in ['sum.dif', 'sum.dif.err']:
                distr[k] /= numbers.size
            for k in ['sum.cum', 'sum.cum.err']:
                distr[k] /= distr['sum.cum'].max()

        # assign redundant key, clearer key for some circumstances
        for k in list(distr):
            k_mod = k.replace('sum', 'number')
            distr[k_mod] = distr[k]

        distr['bin'] = self.mids
        distr['bin.cum'] = self.mins
        if 'log' in self.scaling:
            distr['log bin'] = self.log_mids
            distr['log bin.cum'] = self.log_mins

        return distr

    def get_statistics_of_array(self, values_x, values_y):
        '''
        Get values and statistics of y-values binned by corresponding x-values.
        Ignore values outside of self bin limits.

        Parameters
        ----------
        values_x : array : values along some axis
        values_y : array : corresponding to values along another axis

        Returns
        -------
        stat : dict : dictionary of statistics, with a value for each bin
        '''
        from . import statistic

        stat = {'bin.mid': [], 'y.values': [], 'indices': [], 'number': []}

        values_y, values_x = np.array(values_y), np.array(values_x)
        assert values_y.size == values_x.size

        Statistic = statistic.StatisticClass()

        val_is = array.get_arange(values_x)
        bin_is = self.get_bin_indices(values_x)
        stat['bin.min'] = self.mins
        stat['bin.mid'] = self.mids
        stat['bin.max'] = self.maxs

        for bi in range(self.number):
            vals_y_in_bin = values_y[bin_is == bi]
            stat['y.values'].append(vals_y_in_bin)
            stat['indices'].append(val_is[bin_is == bi])
            stat['number'].append(vals_y_in_bin.size)
            Statistic.append_to_dictionary(vals_y_in_bin)
        stat.update(Statistic.stat)

        for k in stat:
            if k != 'y.values':
                stat[k] = np.array(stat[k])

        return stat


class DistanceBinClass(BinClass, io.SayClass):
    '''
    Create and store distance / radius bin information.
    '''

    def __init__(
        self, scaling='', limits=[], width=None, number=None, dimension_number=3,
        include_max=False):
        '''
        Assign distance / radius bins, of fixed width, in units defined by scaling.

        Parameters
        ----------
        scaling : str : bin scaling: 'log' or 'linear'
        limits : list: *linear* min and max limits to impose
        width : float : width of each bin, in scaling units (input this or number, but not both)
        number : int : number of bins (input this or width, but not both)
        dimension_number : int : number of spatial dimensions
        include_max : bool : whether to inclue last bin, that starts at limits[1]
        '''
        self.scaling = str(scaling)
        self.dimension_number = dimension_number
        # store input limits, but may have to adjust limits_max if also input bin width
        self.limits_input = np.array([min(limits), max(limits)])

        if limits[0] == 0:
            self.log_limits_input = np.array([-np.Inf, math.get_log(max(self.limits_input))])
        else:
            self.log_limits_input = math.get_log(self.limits_input)

        if 'log' in scaling:
            self.log_width = self._get_bin_width(self.log_limits_input, width, number)
            self.log_mins = array.get_arange_safe(
                self.log_limits_input, self.log_width, include_max)
            self.log_mids = self.log_mins + 0.5 * self.log_width
            self.log_maxs = self.log_mins + self.log_width
            self.log_widths = np.zeros(self.log_mins.size) + self.log_width
            self.mids = 10 ** self.log_mids
            self.mins = 10 ** self.log_mins
            self.maxs = 10 ** self.log_maxs
            self.widths = self.mins * (10 ** self.log_width - 1)
            # store actual limits used, given bin widths
            self.log_limits = np.array([self.log_mins[0], self.log_maxs[-1]])
            self.limits = np.array([self.mins[0], self.maxs[-1]])
        else:
            self.width = self._get_bin_width(self.limits_input, width, number)
            self.mins = array.get_arange_safe(self.limits_input, self.width, include_max)
            self.mids = self.mins + 0.5 * self.width
            self.maxs = self.mins + self.width
            self.widths = np.zeros(self.mins.size) + self.width
            self.log_mins = math.get_log(self.mins)
            self.log_mids = math.get_log(self.mids)
            self.log_maxs = math.get_log(self.maxs)
            # store actual limits used, given bin widths
            self.limits = np.array([self.mins[0], self.maxs[-1]])
            self.log_limits = np.array([self.log_mins[0], self.log_maxs[-1]])
            # deal with possibility of 0 in mins
            self.log_widths = np.zeros(self.mins.size)
            masks = (self.mins > 0)
            self.log_widths[masks] = math.get_log(self.width / self.mins[masks] + 1)

        if dimension_number > 0:
            if dimension_number == 1:
                self.volume_normalization = 1
            elif dimension_number == 2:
                self.volume_normalization = np.pi
            elif dimension_number == 3:
                self.volume_normalization = 4 / 3 * np.pi
            self.volume_in_limit = self.volume_normalization * (
                max(self.limits) ** dimension_number - min(self.limits) ** dimension_number)
            self.volumes = self.volume_normalization * (
                (self.mins + self.widths) ** dimension_number - self.mins ** dimension_number)
            self.volume_fracs = self.volumes / self.volume_in_limit
            self.volumes_cum = ((self.volume_normalization * min(self.limits) ** dimension_number) +
                                np.cumsum(self.volumes))

        self.number = self.mins.size

    def print_diagnostics(self, distances):
        '''
        Print diagnostic information for input distances wrt bin limits.

        Parameters
        ----------
        distances : array
        '''
        self.say(
            'input {:8d} distances - {:8d} ({:.1f}%) are within limits = [{:.3f}, {:.3f}]'.format(
                len(distances), array.get_indices(distances, self.limits).size,
                100 * array.get_indices(distances, self.limits).size / len(distances),
                self.limits[0], self.limits[1]))

    def get_bin_limits(self, scaling, bin_i):
        '''
        Get distance limits of single bin.

        Parameters
        ----------
        scaling : str : distance scaling: 'log' or 'linear'
        bin_i : int : infex of bin

        Returns
        -------
        bin_limits : list : min and max values corresponding to bin edges
        '''
        if 'log' in scaling:
            bin_limits = [self.log_mins[bin_i], (self.log_mins[bin_i] + self.log_widths[bin_i])]
        else:
            bin_limits = [self.mins[bin_i], self.mins[bin_i] + self.widths[bin_i]]

        return bin_limits

    def get_sum_profile(
        self, distances, properties=None, get_fraction=False, get_spline=False):
        '''
        Get dictionary of summed properties (such as mass or density) in bins of distance.
        If properties is None, instead compute number of distance values in bin (histogram).

        Parameters
        ----------
        distances : array : distances (linear)
        properties : array : property value for each distance value
        get_fraction : bool : whether to compute fraction in bin wrt total within limtis
        get_spline : bool : whether to compute and return spline of property v distance

        Returns
        -------
        pro : dict : dictionary of summed quantities, with a value for each bin
        '''
        pro = {}
        if properties is not None:
            assert distances.size == properties.size

        self.print_diagnostics(distances)

        # get properties in bins
        pro['sum'] = self.get_histogram(distances, False, properties)

        # mass within distance minimum, for computing cumulative values
        dis = np.where(distances < np.min(self.limits))[0]
        if properties is not None:
            pro['sum.cum'] = np.sum(properties[dis])
        else:
            pro['sum.cum'] = dis.size
        del(dis)
        pro['sum.cum'] += np.cumsum(pro['sum'])

        pro['log sum'] = math.get_log(pro['sum'])
        pro['log sum.cum'] = math.get_log(pro['sum.cum'])

        # mass densities in bins
        pro['density'] = pro['sum'] / self.volumes
        pro['log density'] = math.get_log(pro['density'])
        pro['density.cum'] = pro['sum.cum'] / self.volumes_cum
        pro['log density.cum'] = math.get_log(pro['density.cum'])

        pro['density*r'] = pro['density'] * self.mids
        pro['log density*r'] = pro['log density'] + self.log_mids
        pro['density*r^2'] = pro['density'] * self.mids ** 2
        pro['log density*r^2'] = pro['log density'] + 2 * self.log_mids

        if get_fraction:
            # fraction in bin (normalized to total within distance limits)
            pro['fraction'] = pro['sum'] / np.sum(pro['sum'])
            pro['log fraction'] = math.get_log(pro['fraction'])
            pro['fraction.cum'] = pro['sum.cum'] / pro['sum.cum'].max()
            pro['log fraction.cum'] = math.get_log(pro['fraction.cum'])

        pro['distance.mid'] = self.mids
        pro['distance.cum'] = self.maxs
        pro['log distance.mid'] = self.log_mids
        pro['log distance.cum'] = self.log_maxs

        if get_spline:
            spline = {}
            for prop in ['sum', 'density']:
                for bin_kind in ['.mid', '.cum']:
                    distances = pro['distance' + bin_kind]
                    spline[prop + bin_kind] = math.SplinePointClass(
                        distances, pro[prop + bin_kind], make_inverse=False)
            return pro, spline
        else:
            return pro

    def get_statistics_profile(self, distances, properties, weights=None):
        '''
        Get dictionary of statistics (such as median, average) of input property in bins of
        distance.

        Parameters
        ----------
        distances : array : distances (linear)
        properties : array : property value at each distance value
        weights : array : if defined, use to weight each property

        Returns
        -------
        pro : dict : dictionary of statistics arrays, with a value for each bin
        '''
        from . import statistic

        distances, properties = np.array(distances), np.array(properties)
        assert distances.size == properties.size

        self.print_diagnostics(distances)

        pro = {}

        Statistic = statistic.StatisticClass()

        if 'log' in self.scaling:
            dists, bins, lim = math.get_log(distances), self.log_mins, np.max(self.log_limits)
        else:
            dists, bins, lim = distances, self.mins, np.max(self.limits)
        bin_indices = get_bin_indices(dists, bins, lim, 'down', False, False)

        for bin_i in range(self.number):
            props_bin = properties[bin_indices == bin_i]
            weights_bin = None
            if weights is not None:
                weights_bin = weights[bin_indices == bin_i]
            Statistic.append_to_dictionary(props_bin, weights=weights_bin)
        pro.update(Statistic.stat)

        for k in pro:
            pro[k] = np.array(pro[k])

        pro['distance.mid'] = self.mids
        pro['distance.cum'] = self.maxs
        pro['log distance.mid'] = self.log_mids
        pro['log distance.cum'] = self.log_maxs

        return pro

    def get_velocity_profile(self, distance_vectors, velocity_vectors, masses=None):
        '''
        Get dictionary of velocity statistics in bins of distance.

        Parameters
        ----------
        distance_vectors : array : distance vectors wrt center (object number x dimension number)
        velocity_vectors : array : velocity vectors wrt center (object number x dimension number)
        masses : array : if defined, use to weight property by mass

        Returns
        -------
        pro : dict : dictionary of property arrays, with a value for each bin
        '''
        from .. import orbit

        assert distance_vectors.shape == velocity_vectors.shape
        if masses is not None:
            assert masses.size == distance_vectors.shape[0]

        orb = orbit.get_orbit_dictionary(distance_vectors, velocity_vectors, get_integrals=False)

        self.print_diagnostics(orb['distance.total'])

        prop_names = [prop for prop in orb if 'velocity' in prop and
                      ('.total' in prop or '.rad' in prop or '.tan' in prop)]

        pro = {}
        for prop in prop_names:
            pro[prop] = {}
            for stat in ['average', 'average.cum', 'median', 'median.cum', 'std', 'std.cum',
                         'disp.med', 'disp.med.cum']:
                pro[prop][stat] = np.zeros(self.number) + np.nan  # default to nan

        if 'log' in self.scaling:
            dists, bins, lim = (
                math.get_log(orb['distance.total']), self.log_mins, np.max(self.log_limits))
        else:
            dists, bins, lim = orb['distance.total'], self.mins, np.max(self.limits)

        bin_indices = get_bin_indices(dists, bins, lim, 'down', False, False)

        indices = array.get_arange(distance_vectors.shape[0])
        for bin_i in range(self.number):
            indices_bin = indices[bin_indices == bin_i]
            indices_cum = indices[orb['distance.total'] < self.maxs[bin_i]]
            masses_bin = None
            masses_cum = None
            if masses is not None:
                # renormalize masses for numerical stability
                masses_bin = masses[indices_bin] / np.median(masses[indices_bin])
                masses_cum = masses[indices_cum] / np.median(masses[indices_cum])

            for prop in prop_names:
                if indices_bin.size:
                    # (weighted) average within distance bin
                    pro[prop]['average'][bin_i] = np.average(
                        orb[prop][indices_bin], weights=masses_bin)

                    # (weighted) average within distance bin
                    pro[prop]['median'][bin_i] = math.percentile_weighted(
                        orb[prop][indices_bin], 50, weights=masses_bin)

                    # (weighted) standard deviation within distance bin
                    pro[prop]['std'][bin_i] = np.sqrt(
                        np.average(orb[prop][indices_bin] ** 2, weights=masses_bin))

                    # (weighted) median of dispersion within distance bin
                    pro[prop]['disp.med'][bin_i] = np.sqrt(
                        math.percentile_weighted(orb[prop][indices_bin] ** 2, 50, masses_bin))

                if indices_cum.size:
                    # cumulative (weighted) average inside distance
                    pro[prop]['median.cum'][bin_i] = math.percentile_weighted(
                        orb[prop][indices_cum], 50, weights=masses_cum)

                    # cumulative (weighted) average inside distance
                    pro[prop]['average.cum'][bin_i] = np.average(
                        orb[prop][indices_cum], weights=masses_cum)

                    # cumulative (weighted) standard deviation inside distance
                    pro[prop]['std.cum'][bin_i] = np.sqrt(
                        np.average(orb[prop][indices_cum] ** 2, weights=masses_cum))

                    # cumulative (weighted) median of dispersion inside distance
                    pro[prop]['disp.med.cum'][bin_i] = np.sqrt(
                        math.percentile_weighted(orb[prop][indices_cum] ** 2, 50, masses_cum))

        pro['distance.mid'] = self.mids
        pro['distance.cum'] = self.maxs
        pro['log distance.mid'] = self.log_mids
        pro['log distance.cum'] = self.log_maxs

        return pro


class GalaxyHaloMassBinClass(BinClass, io.SayClass):
    '''
    Make and retrieve bin information for both galaxy/subhalo mass and host halo mass.
    '''

    def __init__(
        self, gal_kind, gal_limits, gal_width, hal_kind, hal_limits, hal_width, vary_kind,
        include_max=False):
        '''
        Assign galaxy and halo mass bin information.

        Parameters
        ----------
        gal_kind : str : subhalo/galaxy [mass] kind
        gal_limits : list : min and max limits of galaxy mass
        gal_width : float : width of bin of galaxy mass
        hal_kind : str : halo mass kind
        hal_limits : list : min and max limits of halo mass
        hal_width : float : width of bin of halo mass
        vary_kind : str : which to mass vary: halo, galaxy
        include_max : bool : whether to inclue last bin that starts at max(limits)
        '''
        self.gal = BinClass(gal_limits, gal_width, include_max=include_max)
        self.hal = BinClass(hal_limits, hal_width, include_max=include_max)
        self.gal.kind = gal_kind
        self.hal.kind = hal_kind

        if vary_kind in [hal_kind, 'halo']:
            vary_mass_kind = hal_kind
            vary_kind = 'halo'
            fix_mass_kind = gal_kind
            fix_kind = 'galaxy'
        elif vary_kind in [gal_kind, 'galaxy']:
            vary_mass_kind = gal_kind
            vary_kind = 'galaxy'
            fix_mass_kind = hal_kind
            fix_kind = 'halo'
        else:
            raise ValueError('not recognize vary_kind = ' + vary_kind)

        if vary_kind == 'halo':
            self.vary = self.hal
            self.fix = self.gal
        elif vary_kind == 'galaxy':
            self.vary = self.gal
            self.fix = self.hal

        self.vary.mass_kind = vary_mass_kind
        self.vary.kind = vary_kind
        self.fix.mass_kind = fix_mass_kind
        self.fix.kind = fix_kind

        # copy varying parameters to self for shortcut
        self.limits = self.vary.limits
        self.width = self.vary.width
        self.widths = self.vary.widths
        self.mins = self.vary.mins
        self.mids = self.vary.mids
        self.maxs = self.vary.maxs
        self.number = self.vary.number

    def get_bins_limits(self, vary_i, fix_i=None, print_bin=False):
        '''
        Get limits of single bin for both vary and fix masses.

        Parameters
        ----------
        vary_i : int : index of vary bin
        fix_i : int : index of fixed bin
        print_bin : bool : whether to print mass bin

        Returns
        -------
        gal_limits : list : min and max limits of galaxy mass
        hal_limits : list : min and max limits of host halo mass
        '''
        vary_limits = self.vary._get_bin_limits(vary_i)

        if fix_i is not None:
            fix_limits = self.fix._get_bin_limits(fix_i)
        else:
            fix_limits = self.fix.limits

        if print_bin:
            self.Say('{} [{:.2f}, {:.2f}]'.format(
                     self.vary.kind, min(vary_limits), max(vary_limits)))
            if fix_i is not None:
                self.Say('{} [{:.2f}, {:.2f}]'.format(
                         self.fix.kind, min(vary_limits), max(vary_limits)))

        if self.vary.kind == self.hal.kind:
            gal_limits = fix_limits
            hal_limits = vary_limits
        elif self.vary.kind == self.gal.kind:
            gal_limits = vary_limits
            hal_limits = fix_limits

        return gal_limits, hal_limits
