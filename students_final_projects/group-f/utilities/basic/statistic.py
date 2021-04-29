'''
Utility functions for statistics.

@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import numpy as np
from scipy import stats
# local ----
from . import array, binning, io, math


#===================================================================================================
# statistics
#===================================================================================================
class StatisticClass(io.SayClass):
    '''
    Store statistics and probability distribution of input array.
    '''

    def __init__(
        self, values=None, limits=[], bin_width=None, bin_number=10, scaling='linear',
        weights=None, vary_bin_width=False, values_possible=None):
        '''
        Parameters
        ----------
        values : array
        limits : list : min and max limits to impose
        bin_width : float : width of each bin
        bin_number : int : number of bins
        scaling : str : scaling for binning: 'log', 'linear'
        weights : array : weight for each value
        vary_bin_width : bool : whether to vary bin width
        values_possible : array : all possible input values
            use to map input values to int value, to then bin by intrinsic scaling in values
            if defined, this overrides bin_number and vary_bin_width
            for example, if values correspond to redshifts of simulation snapshots,
            input every possible snapshot redshift to bin according to that width
        '''
        self.stat = {}
        self.distr = {}

        if values is not None and len(values):
            self.stat = self.get_statistic_dict(values, limits, weights)
            self.distr = self.get_distribution_dict(
                values, limits, bin_width, bin_number, scaling, weights, vary_bin_width,
                values_possible)

    def parse_limits(self, values, limits=None):
        '''
        Get limits, either as input or from input values.
        Impose sanity checks.

        Parameters
        ----------
        values : array : value[s]
        limits : list : *linear* min and max limits to impose
        '''
        limit_buffer = 1e-6

        if limits is None or not len(limits) or limits[0] is None or limits[1] is None:
            limits_values = array.get_limits(values)
            if limits is None or not len(limits):
                limits = [None, None]
            if limits[0] is None:
                limits[0] = limits_values[0]
            if limits[1] is None:
                limits[1] = limits_values[1]
                limits[1] *= 1 + limit_buffer  # make sure single value remains valid
                limits[1] += limit_buffer

        if limits[0] == limits[1] or isinstance(limits[1], int):
            limits[1] *= 1 + limit_buffer  # make sure single value remains valid
            limits[1] += limit_buffer

        return limits

    def get_statistic_dict(self, values, limits=[-np.Inf, np.Inf], weights=None):
        '''
        Get dicionary of statistics within limits.

        Parameters
        ----------
        values : array
        limits : list : min and max limits to impose
        weights : array : weight for each value
        '''
        stat = {
            'limits': [],  # impose limits or limits of input values
            'number': 0,  # number of values
            # values at confidence
            'median': 0,
            'percent.16': 0, 'percent.84': 0, 'percent.2': 0, 'percent.98': 0,
            'percent.0.1': 0, 'percent.99.9': 0,
            'percent.25': 0, 'percent.75': 0,
            'percents.68': [0, 0], 'percents.95': [0, 0],
            'median.dif.2': 0, 'median.dif.16': 0, 'median.dif.84': 0, 'median.dif.98': 0,
            'median.difs.68': [0, 0], 'median.difs.95': [0, 0],
            'average': 0, 'std': 0, 'sem': 0,  # average, std dev, std dev of mean
            'std.lo': 0, 'std.hi': 0,  # values of std limits
            'min': 0, 'max': 0,  # minimum and maximum
        }

        if values is None or not len(values):
            return stat

        values = np.array(values)

        limits = self.parse_limits(values, limits)

        masks = array.get_indices(values, limits)
        if masks.size < values.size:
            values = values[masks]
            if weights is not None:
                weights = weights[masks]

        if not values.size:
            self.say('! no values are within bin limit: [{:.3f}, {:.3f}]'.format(
                     min(limits), max(limits)))
            return stat

        # scalar statistics
        stat['limits'] = limits
        stat['number'] = values.size
        if weights is None or not len(weights):
            stat['median'] = np.median(values)
            stat['percent.50'] = np.median(values)
            stat['percent.16'] = np.percentile(values, 16)
            stat['percent.84'] = np.percentile(values, 84)
            stat['percent.2'] = np.percentile(values, 2.275)
            stat['percent.98'] = np.percentile(values, 97.725)
            stat['percent.0.1'] = np.percentile(values, 0.135)
            stat['percent.99.9'] = np.percentile(values, 99.865)
            stat['percent.25'] = np.percentile(values, 25)
            stat['percent.75'] = np.percentile(values, 75)
        else:
            stat['median'] = math.percentile_weighted(values, 50, weights)
            stat['percent.50'] = math.percentile_weighted(values, 50, weights)
            stat['percent.16'] = math.percentile_weighted(values, 16, weights)
            stat['percent.84'] = math.percentile_weighted(values, 84, weights)
            stat['percent.2'] = math.percentile_weighted(values, 2.275, weights)
            stat['percent.98'] = math.percentile_weighted(values, 97.725, weights)
            stat['percent.0.1'] = math.percentile_weighted(values, 0.135, weights)
            stat['percent.99.9'] = math.percentile_weighted(values, 99.865, weights)
            stat['percent.25'] = math.percentile_weighted(values, 25, weights)
            stat['percent.75'] = math.percentile_weighted(values, 75, weights)

        stat['percents.68'] = [stat['percent.16'], stat['percent.84']]
        stat['percents.95'] = [stat['percent.2'], stat['percent.98']]

        stat['median.dif.2'] = stat['median'] - stat['percent.2']
        stat['median.dif.16'] = stat['median'] - stat['percent.16']
        stat['median.dif.84'] = stat['percent.84'] - stat['median']
        stat['median.dif.98'] = stat['percent.98'] - stat['median']
        stat['median.difs.68'] = [stat['median.dif.16'], stat['median.dif.84']]
        stat['median.difs.95'] = [stat['median.dif.2'], stat['median.dif.98']]

        if weights is None or not len(weights):
            stat['average'] = np.mean(values)
            stat['average.50'] = np.mean(
                values[(values > stat['percent.25']) * (values < stat['percent.75'])])
            stat['std'] = np.std(values, ddof=1)
            stat['sem'] = stats.sem(values)
        else:
            stat['average'] = np.sum(values * weights) / np.sum(weights)
            masks = (values > stat['percent.25']) * (values < stat['percent.75'])
            stat['average.50'] = np.sum(values[masks] * weights[masks]) / np.sum(weights[masks])
            stat['std'] = np.sqrt(np.sum(weights / np.sum(weights) *
                                         (values - stat['average']) ** 2))
            stat['sem'] = stat['std'] / np.sqrt(values.size)
        stat['std.lo'] = stat['average'] - stat['std']
        stat['std.hi'] = stat['average'] + stat['std']
        stat['sem.lo'] = stat['average'] - stat['sem']
        stat['sem.hi'] = stat['average'] + stat['sem']
        stat['min'] = values.min()
        stat['max'] = values.max()
        """
        # make sure array has more than one value
        if stat['max'] != stat['min']:
            vals_sort = np.unique(values)
            if vals_sort.size > 2:
                stat['min.2'] = vals_sort[1]
                stat['max.2'] = vals_sort[-2]
            if vals_sort.size > 4:
                stat['min.3'] = vals_sort[2]
                stat['max.3'] = vals_sort[-3]
        """

        return stat

    def get_distribution_dict(
        self, values, limits=None, bin_width=None, bin_number=0, scaling='linear', weights=None,
        vary_bin_width=False, values_possible=None):
        '''
        Get dicionary for histogram/probability distribution.

        Parameters
        ----------
        values : array : value[s]
        limits : list : *linear* max and min limits to impose
        bin_width : float : width of each bin
        bin_number : int : number of bins
        scaling : str : scaling for binning: 'log' or 'linear'
        weights : array : weight for each value
        vary_bin_width : bool : whether adaptively to vary bin width
        values_possible : array : all possible input values
            use to map input values to int value, to then bin by intrinsic scaling in values
            if defined, this overrides bin_number and vary_bin_width
            for example, if values correspond to redshifts of simulation snapshots,
            input every possible snapshot redshift to bin according to that width
        '''
        distr = {
            'limits': np.array([]),
            'bin.min': np.array([]),
            'bin.mid': np.array([]),
            'bin.max': np.array([]),
            'bin.width': np.array([]),
            'probability': np.array([]),
            'probability.err': np.array([]),
            'probability.cum': np.array([]),
            'histogram': np.array([]),
            'histogram.err': np.array([]),
        }

        if values is None or not len(values):
            return distr

        values = np.array(values)

        limits = self.parse_limits(values, limits)

        distr['limits'] = limits

        if 'log' in scaling:
            limits = math.get_log(limits)
            values = math.get_log(values)
            if values_possible is not None:
                values_possible = math.get_log(values_possible)

        val_indices = array.get_indices(values, limits)
        if not val_indices.size:
            self.say('! no values within bin limits = {}'.format(distr['limits']))
            return distr
        values_in_limit = values[val_indices]

        if weights is not None:
            weights_in_limit = weights[val_indices]
        else:
            weights_in_limit = None

        if (values_possible is not None or (bin_number is not None and bin_number > 0) or
                (bin_width is not None and bin_width > 0)):
            if values_possible is not None:
                self.say('use spacing of input values_possible to set bin widths')
                values_possible = np.unique(values_possible)
                vals_possible_width = np.abs(values_possible[:-1] - values_possible[1:])
                value_bin_indices = binning.get_bin_indices(
                    values_in_limit, values_possible, values_possible.max(), 'down')
                val_bin_limits = [0, value_bin_indices.max() + 2]
                val_bin_range = np.arange(val_bin_limits[0], val_bin_limits[1], dtype=np.int64)
                values_possible = values_possible[val_bin_range]
                vals_possible_width = vals_possible_width[val_bin_range]

                if 'log' in scaling:
                    distr['bin.min'] = 10 ** values_possible
                    distr['bin.mid'] = 10 ** (values_possible + 0.5 * vals_possible_width)
                    distr['bin.max'] = 10 ** (values_possible + vals_possible_width)
                else:
                    distr['bin.min'] = values_possible
                    distr['bin.mid'] = values_possible + 0.5 * vals_possible_width
                    distr['bin.max'] = values_possible + vals_possible_width

                distr['histogram'] = np.histogram(
                    value_bin_indices, values_possible.size, val_bin_limits, False,
                    weights_in_limit)[0]

                if weights is not None:
                    distr['probability'] = (
                        distr['histogram'] / np.sum(weights_in_limit) / vals_possible_width)
                else:
                    distr['probability'] = (distr['histogram'] / values_in_limit.size /
                                            vals_possible_width)

            elif ((bin_number is not None and bin_number > 0) or
                    (bin_width is not None and bin_width > 0)):
                Bin = binning.BinClass(limits, bin_width, bin_number, vary_bin_width, values)

                if 'log' in scaling:
                    distr['bin.min'] = 10 ** Bin.mins
                    distr['bin.mid'] = 10 ** Bin.mids
                    distr['bin.max'] = 10 ** Bin.maxs
                else:
                    distr['bin.min'] = Bin.mins
                    distr['bin.mid'] = Bin.mids
                    distr['bin.max'] = Bin.maxs

                if vary_bin_width:
                    bin_mins = np.append(Bin.mins, Bin.limits[1])
                    distr['histogram'] = np.histogram(
                        values_in_limit, bin_mins, limits, False, weights_in_limit)[0]
                    if weights is not None:
                        distr['probability'] = (
                            distr['histogram'] / np.sum(weights_in_limit) / Bin.wids)
                    else:
                        distr['probability'] = distr['histogram'] / values_in_limit.size / Bin.wids
                else:
                    distr['histogram'] = np.histogram(
                        values_in_limit, Bin.number, limits, False, weights_in_limit)[0]
                    distr['probability'] = np.histogram(
                        values_in_limit, Bin.number, limits, True, weights_in_limit)[0]

            distr['bin.width'] = distr['bin.max'] - distr['bin.min']

            distr['histogram.err'] = distr['histogram'] ** 0.5
            value_low_number = np.sum(values < limits[0])
            if weights is not None:
                distr['probability.cum'] = (
                    (np.cumsum(distr['histogram']) + value_low_number) / np.sum(weights))
            else:
                distr['probability.cum'] = (
                    (np.cumsum(distr['histogram']) + value_low_number) / values.size)

            distr['probability.err'] = math.Fraction.get_fraction(
                distr['probability'], distr['histogram.err'])

            for prop in list(distr):
                if '.err' not in prop and np.min(distr[prop]) > 0:
                    distr['log ' + prop] = math.get_log(distr[prop])

        return distr

    def append_to_dictionary(
        self, values, limits=None, bin_width=None, bin_number=0, scaling='linear', weights=None,
        vary_bin_width=False, values_possible=None):
        '''
        Make dictionaries for statistics and histogram/probability distribution, append to self.

        Parameters
        ----------
        values : array : value[s]
        limits : list : *linear* min and max limits to impose
        bin_width : float : width of each bin
        bin_number : int : number of bins
        scaling : str : scaling for binning: 'log', 'linear'
        weights : array : weights for each value
        vary_bin_width : bool : whether adaptively to vary bin width
        values_possible : array : all possible input values
            use to map input values to int value, to then bin by intrinsic scaling in values
            if defined, this overrides bin_number and vary_bin_width
            for example, if values correspond to redshifts of simulation snapshots,
            input every possible snapshot redshift to bin according to that width
        '''
        # check if need to arrayize dictionaries
        if (self.distr and self.distr['probability'] and len(self.distr['probability']) and
                np.isscalar(self.distr['probability'][0])):
            for k in self.stat:
                self.stat[k] = [self.stat[k]]
            for k in self.distr:
                self.distr[k] = [self.distr[k]]

        array.append_dictionary(self.stat, self.get_statistic_dict(values, limits, weights))

        array.append_dictionary(
            self.distr,
            self.get_distribution_dict(
                values, limits, bin_width, bin_number, scaling, weights, vary_bin_width,
                values_possible)
        )

    def append_class_to_dictionary(self, StatIn):
        '''
        Append statistics class dictionaries to self.

        Parameters
        ----------
        StatIn : another statistic/distribution class
        '''
        array.append_dictionary(self.stat, StatIn.stat)
        array.append_dictionary(self.distr, StatIn.distr)

    def arrayize(self):
        '''
        Convert dicionary lists to arrays.
        '''
        self.stat = array.arrayize(self.stat)
        self.distr = array.arrayize(self.distr)

    def print_statistics(self, bin_index=None):
        '''
        Print statistics in self.

        Parameters
        ----------
        bin_index : int : bin index to print statistic of
        '''
        stat_list = [
            'min', 'max',
            'median',
            'average', 'std',
            'percent.0.1', 'percent.2', 'percent.16', 'percent.50',
            'percent.84', 'percent.98', 'percent.99.9'
        ]
        #, 'min.2', 'min.3', 'max.2', 'max.3']

        if bin_index is None and not np.isscalar(self.stat['median']):
            raise ValueError('no input index, but stat is multi-dimensional')

        if bin_index is not None:
            value = self.stat['number'][bin_index]
        else:
            value = self.stat['number']

        self.say('number = {}\n'.format(value))

        for k in stat_list:
            if bin_index is not None:
                value = self.stat[k][bin_index]
            else:
                value = self.stat[k]
            self.say('{} = {}'.format(k, io.get_string_from_numbers(value, 3)))
            if k in ['max', 'average', 'std']:
                self.say('')


def print_statistics(values, plot=False):
    '''
    For input array, print statistics (and plot histogram).

    Parameters
    ----------
    values : array : value[s]
    plot : bool : whether to plot histogram
    '''
    values = np.array(values)
    if np.ndim(values) > 1:
        values = np.concatenate(values)

    Stat = StatisticClass(values)
    Stat.print_statistics()

    if 0 in values:
        print('  contains 0')
        print('  minimum value > 0 = {:.4f}'.format(np.min(values[values > 0])))

    if -np.Inf in values:
        print('  contains -Inf')
        print('  minimum value > -Inf = {:.4f}'.format(np.min(values[values > -np.Inf])))

    if np.Inf in values:
        print('  contains Inf')
        print('  maximum value < Inf = {:.4f}'.format(np.min(values[values < np.Inf])))

    if plot:
        from matplotlib import pyplot as plt
        bin_number = np.int(np.clip(values.size / 10, 0, 1000))
        plt.hist(values, bins=bin_number)
