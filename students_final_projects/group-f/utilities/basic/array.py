'''
Utility functions for arrays: creation, manipulation, diagnostic

@author: Andrew Wetzel
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import numpy as np
from scipy import ndimage, signal
# local ----
from . import io as io


#===================================================================================================
# useful classes
#===================================================================================================
class DictClass(dict):
    pass


class ListClass(list):
    pass


#===================================================================================================
# initialize array
#===================================================================================================
def parse_data_type(array_length, dtype=None):
    '''
    Parse input data type, if None, use default int, according to array size.

    Parameters
    ----------
    array_length : int : length of array
    dtype : data type to override default selection

    Returns
    -------
    dtype : data type
    '''
    if dtype is None:
        if array_length > 2147483647:
            dtype = np.int64
        else:
            dtype = np.int32

    return dtype


def get_array_null(array_shape, dtype=None):
    '''
    Make array of given data type with initial values that are safely negative and out of bounds.

    Parameters
    ----------
    array_shape : int or list : size/shape for array
    dtype : data type for array

    Returns
    -------
    array : array initialized with null values
    '''
    if np.isscalar(array_shape):
        size = array_shape
    else:
        size = array_shape[0]

    dtype = parse_data_type(size, dtype)

    return np.zeros(array_shape, dtype) - size - 1


def get_arange(array_or_length_or_imin=None, i_max=None, dtype=None):
    '''
    Get np.arange corresponding to input limits or input array size.

    Parameters
    ----------
    array_or_length_or_imin : array or int : array or array length or starting value
    i_max : int : ending value (if array_or_length_or_imin is starting value)
    dtype : data type (if None, use size to determine 32 or 64-bit int)

    Returns
    -------
    array : array with values 0, 1, 2, 3, etc
    '''
    if i_max is None:
        i_min = 0
        if np.isscalar(array_or_length_or_imin):
            i_max = array_or_length_or_imin
        else:
            i_max = len(array_or_length_or_imin)
    else:
        i_min = array_or_length_or_imin

    dtype = parse_data_type(i_max, dtype)

    return np.arange(i_min, i_max, dtype=dtype)


def get_arange_safe(limits, width=None, include_max=False, dtype=np.float64):
    '''
    Get arange that safely does [not] reach limit maximum.
    If width < 0, return input limit.

    Parameters
    ----------
    limits : list : min and max limits for array
    width : float : width of each bin
    include_max : bool : whether to include maximum(limits) as a bin
    dtype : data type for array

    Returns
    -------
    array : array with values 0, 1, 2, 3, etc that does [not] include max
    '''
    if limits is None:
        return limits

    lim = get_limits(limits)

    # ensure upper limit does not reach input limit
    lim_max = lim[1] * (1 - np.sign(lim[1]) * 1e-6)
    if lim_max == 0:
        lim_max -= 1e-6

    if include_max:
        if lim[0] == -np.Inf or lim[1] == np.Inf:
            return np.array(lim, dtype)
        elif width == np.Inf:
            return np.array([lim[0], np.Inf], dtype)
        elif width <= 0:
            return np.array(lim, dtype)
        else:
            return np.arange(lim[0], lim_max + width, width, dtype)
    else:
        if lim[0] == -np.Inf or lim_max == np.Inf:
            return np.array([lim[0]], dtype)
        elif width == np.Inf:
            return np.array([lim[0]], dtype)
        elif width <= 0:
            return np.array([lim[0]], dtype)
        else:
            return np.arange(lim[0], lim_max, width, dtype)


#===================================================================================================
# limits of array
#===================================================================================================
def get_limits(
    values, cut_number=0, cut_percent=0, use_unique=False, ignore_nan=False, ignore_inf=False,
    digit_number=None):
    '''
    Get tuple of minimum and maximum values, applying all cuts, keeping given number of digits.

    Parameters
    ----------
    values : array
    cut_number : int : n'th unique value above/below minimum/maximum to keep
    cut_percent : float : cut percent from mininum/maximum to keep
    use_unique : bool : use only unique values in array when cutting by percent/value
    ignore_nan : bool : whether to ignore NaN values in array
    ignore_inf : bool : whether to ignore np.Inf values in array
    digit_number : bool : number of digits to keep (for printing)

    Returns
    -------
    value_limits : array : limits of input values, after applying cuts
    '''
    values = np.array(values)

    if np.ndim(values) > 1:
        values = np.concatenate(values)

    if ignore_nan:
        values = values[np.invert(np.isnan(values))]

    if ignore_inf:
        values = values[np.isfinite(values)]

    value_limits = [np.min(values), np.max(values)]

    if use_unique:
        values = np.unique(values)  # returns sorted values

    if cut_number > 0:
        value_limits = [values[cut_number], values[-cut_number]]

    if cut_percent > 0:
        value_limits_temp = [np.percentile(values, cut_percent),
                             np.percentile(values, 100 - cut_percent)]
        value_limits = [max(value_limits[0], value_limits_temp[0]),
                        min(value_limits[1], value_limits_temp[1])]

    if digit_number is not None:
        value_limits = [round(value_limits[0], digit_number), round(value_limits[1], digit_number)]

    return value_limits


def get_limits_string(values, digit_number=3, exponential=False, strip=False):
    '''
    Get string of minimum and maximum values, in nice format.

    Parameters
    ----------
    value[s] : int/float or list thereof
    digits : int : number of digits after period
    exponential : bool : whether to use exponential (instead of float) format
      if None, chose automatically
    strip : bool : whether to strip trailing 0s (and .)

    Returns
    -------
    value_limits : str : limits of input values, in nice format
    '''
    value_limits = io.get_string_from_numbers(
        get_limits(values), digit_number, exponential, strip)

    return '[{}]'.format(value_limits)


def get_limits_expanded(values, expand_value, expand_kind='add'):
    '''
    Get limits of input values, expanding both ends by given amount.

    Parameters
    ----------
    values : array
    expand_value : float :  amount to expand limits in both directions
    expand_kind : str : method to expand: add, multiply

    Returns
    -------
    value_limits : str : limits of input values, expanded
    '''
    if values is None or expand_value is None:
        value_limits = None
    else:
        if expand_kind == 'add':
            value_limits = [np.min(values) - expand_value, np.max(values) + expand_value]
        elif expand_kind == 'multiply':
            value_limits = [np.min(values) * (1 - expand_value),
                            np.max(values) * (1 + expand_value)]
        else:
            raise ValueError('not recognize expand_kind = {}'.format(expand_kind))

    return value_limits


#===================================================================================================
# indices in array
#===================================================================================================
def get_indices(values, limits=[-np.Inf, np.Inf], prior_indices=None, get_masks=False, dtype=None):
    '''
    Get the indices of the input values array that are within the input limits
    [>= min(limits), < max(limits)),
    that also are in the input prior_indices array (if defined).

    Either entry in input limits can be of same size as input values array if want to select each
    value with a different limit.

    Parameters
    ----------
    values : array
    limits : list : limits to use to select values, within range [min(limits), max(limits))
    prior_indices : array : indices of input value array to select on first (before imposing limits)
    get_masks : bool : whether to return selection indices of input indices array
    dtype : data type for array of indices

    Returns
    -------
    indices : array : indices of values array that are within limits, after applying prior_indices
    '''
    if not isinstance(values, np.ndarray):
        values = np.array(values)

    dtype = parse_data_type(values.size, dtype)

    # check if input indices
    if prior_indices is None:
        prior_indices = np.arange(values.size, dtype=dtype)
    elif not len(prior_indices):
        # input prior indices is [], so select no values/indices
        values = np.array([])
    else:
        values = values[prior_indices]

    indices_keep = prior_indices

    if limits is None:
        masks = np.r_[[True] * indices_keep.size]

    elif np.isscalar(limits):
        # input limits is just one value
        masks = (values == limits)

    elif len(limits) == 2:
        # input limits is multiple values
        limit_min, limit_max = limits

        # treat None as np.Inf
        if limit_min is None:
            limit_min = -np.Inf
        if limit_max is None:
            limit_max = np.Inf

        if np.isscalar(limit_min) and np.isscalar(limit_max):
            if limit_min > limit_max:
                limit_min, limit_max = limit_max, limit_min

            if isinstance(limit_min, int) and isinstance(limit_max, int):
                if limit_min == limit_max:
                    raise ValueError('input limit = {}, has same value'.format(limits))
                if limit_min != limit_max and 'int' in values.dtype.name:
                    print('! ut.get_indices discards values at max(limits) = {}'.format(limit_max))

        masks = np.invert(np.isnan(values))

        if not np.isscalar(limit_min) or limit_min > -np.Inf:
                masks[masks] = (values[masks] >= limit_min)

        if not np.isscalar(limit_max) or limit_max < np.Inf:
            masks[masks] = (values[masks] < limit_max)

    else:
        raise ValueError('! not sure how to interpret input limits = {}'.format(limits))

    if get_masks:
        return indices_keep[masks], np.arange(prior_indices.size, dtype=dtype)[masks]
    else:
        return indices_keep[masks]


#===================================================================================================
# sub-sample array
#===================================================================================================
def sample_array(values, number):
    '''
    Get randomly sampled version of array.

    If number > values.size, randomly sample *with* repeat.
    if number <= values.size, randomly sample *without* repeat.

    Parameters
    ----------
    values: array
    number : int : number of elements to sample

    Returns
    -------
    sampled_array : array : array sampled
    '''
    if not values.size or number == 0:
        sampled_array = np.array([])

    if number > values.size:
        sampled_array = values[np.random.randint(0, values.size, number)]
    else:
        vals_rand = values.copy()
        np.random.shuffle(vals_rand)
        sampled_array = vals_rand[:number]

    return sampled_array


#===================================================================================================
# convert to/from array
#===================================================================================================
def get_array_1d(values):
    '''
    Convert input list/array of values to a 1-D array.

    If values is list, treat as independent arrays, else treat as one whole array.

    Parameters
    ----------
    values : array-like : [list of] value[s] / list[s] / array[s]

    Returns
    -------
    array : 1-D array of values
    '''
    if np.isscalar(values):
        return values

    try:
        if np.ndim(values) == 1:
            if isinstance(values, list):
                dtype = np.float64  # default data type if input is list
            else:
                dtype = None
            return np.asarray(values, dtype=dtype)
    except Exception:
        pass

    try:
        return np.squeeze(np.hstack(values))
    except Exception:
        values_1d = []
        for vals in values:
            if np.isscalar(vals):
                values_1d.append(vals)
            else:
                vals = get_array_1d(vals)
                values_1d.extend(vals)

        return np.array(values_1d)


def arrayize(values, bit_number=64, repeat_number=1):
    '''
    Convert input list of values to array of given bit size.

    If values is tuple, treat as independent arrays, else treat as one whole array.

    Parameters
    ----------
    values : array-like : [tuple of] value[s] / list[s] / array[s]
    bit_number : int : precision in bits for array
    repeat_number : int : factor by which to repeat values periodically in array

    Returns
    -------
    array : 1-D array of values
    '''

    def get_array(value, repeat_number, bit_number):
        if np.isscalar(value):
            if repeat_number == 1:
                value = [value]
            elif repeat_number > 1:
                value = np.r_[repeat_number * [value]]

        value = np.array(value)

        if bit_number == 32:
            if value.dtype == 'float64':
                value = value.astype('float32')
            elif value.dtype == 'int64':
                value = value.astype('int32')

        return value

    if np.isscalar(values):
        return get_array(values, repeat_number, bit_number)
    elif len(values) == 1:
        return get_array(values, repeat_number, bit_number)
    elif isinstance(values, tuple):
        arrays = []
        for value in values:
            arrays.append(get_array(value, repeat_number, bit_number))
        return arrays
    elif isinstance(values, dict):
        for k in values:
            values[k] = get_array(values[k], repeat_number, bit_number)
        return values
    else:
        return get_array(values, repeat_number, bit_number)


def scalarize(values):
    '''
    If input list has length of 1, return as scalar.
    Else, return as is.

    Parameters
    ----------
    values : array-like : [tuple of] value[s]

    Returns
    -------
    float or array : scalar (if single value) or array
    '''
    if values is None or np.isscalar(values):
        return values
    elif len(values) == 1:
        return values[0]
    else:
        return values


#===================================================================================================
# get list
#===================================================================================================
def get_list_combined(list_or_dict_1, value_or_list_or_dict_2, combine_kind='combine'):
    '''
    Get list of values, either that overlap or combined.

    Parameters
    ----------
    list_or_dict_1 : list or dict
    value_or_list_or_dict_2 : value or list or dict : values to compute intersection or to exclude
    combine_kind : str : action to combine lists: 'combine', 'intersect', or 'exclude'

    Returns
    -------
    list_1 : list : values
    '''
    list_1 = list(list_or_dict_1)
    if np.isscalar(value_or_list_or_dict_2):
        value_or_list_or_dict_2 = [value_or_list_or_dict_2]
    list_2 = list(value_or_list_or_dict_2)

    assert combine_kind in ['combine', 'intersect', 'exclude']

    # get list of values in either array
    if combine_kind == 'combine':
        for value in list_2:
            if value not in list_1:
                list_1.append(value)

    # get list of values in both arrays
    if combine_kind == 'intersect':
        for value in list_1:
            if value not in list_2:
                list_1.remove(value)
                print(value)

    # get list with values in list 2 removed from list 1
    elif combine_kind == 'exclude':
        for value in list_1:
            if value in list_2:
                list_1.remove(value)

    return list_1


#===================================================================================================
# print information about array
#===================================================================================================
def print_list(values, digit_number=3, delimeter=' ', print_vertical=False):
    '''
    Print list in nice format.

    Parameters
    ----------
    values : array
    digit_number : int : number of digits to print
    delimeter : str : what to print between values
    print_vertical : bool : whether to print values vertically (horizontally is default)
    '''
    string = '{:.' + '{}'.format(digit_number) + 'f}'

    if print_vertical:
        for value in values:
            print(string.format(value))
    else:
        for value in values[:-1]:
            print(string.format(value) + delimeter, end='')
        print(string.format(values[-1]))


def print_extrema(values, number=5, digit_number=3, delimeter=' ', print_vertical=False):
    '''
    Print number unique extrema values in array.

    Parameters
    ----------
    values : array
    number : int : number of minimum / maximum values to print
    digit_number : int : number of digits to print
    print_vertical : bool : whether to print values vertically (horizontally is default)
    print_comma : bool : whether to print comma between values
    '''
    vals_unique = np.unique(values)  # returns sorted values
    print('# minima: ', end='')
    print_list(vals_unique[:number], digit_number, delimeter, print_vertical)
    print('# maxima: ', end='')
    print_list(vals_unique[-number:], digit_number, delimeter, print_vertical)


#===================================================================================================
# compare arrays
#===================================================================================================
def compare_arrays(array_1, array_2, print_mismatch=True, tolerance=0.01):
    '''
    Check if values in arrays are the same (within tolerance percent if float).

    Parameters
    ----------
    array_1, array_2 : two arrays
    print_mismatch : bool : whether to print values of mismatches
    tolerance : float : fractional difference tolerance
    '''
    bad_number = 0

    if len(array_1) != len(array_2):
        print('! array_1 len = {}, array_2 len = {}'.format(len(array_1), len(array_2)))
        return

    if np.shape(array_1) != np.shape(array_2):
        print('! array_1 shape =', np.shape(array_1), 'array_2 shape =', np.shape(array_2))
        return

    if 'int' in array_1.dtype.name:
        for a1_i in range(len(array_1)):
            if np.isscalar(array_1[a1_i]):
                if array_1[a1_i] != array_2[a1_i]:
                    if print_mismatch:
                        print('!', a1_i, array_1[a1_i], array_2[a1_i])
                    bad_number += 1
            else:
                for a1_ii in range(len(array_1[a1_i])):
                    if array_1[a1_i][a1_ii] != array_2[a1_i][a1_ii]:
                        if print_mismatch:
                            print('!', a1_i, a1_ii, array_1[a1_i][a1_ii], array_2[a1_i][a1_ii])
                        bad_number += 1

    elif 'float' in array_1.dtype.name:
        for a1_i in range(len(array_1)):
            if np.isscalar(array_1[a1_i]):
                if array_1[a1_i] == array_2[a1_i]:
                    continue
                elif (abs(np.max((array_1[a1_i] - array_2[a1_i]) / (array_1[a1_i] + 1e-10))) >
                      tolerance):
                    if print_mismatch:
                        print('!', a1_i, array_1[a1_i], array_2[a1_i])
                    bad_number += 1
            else:
                for a1_ii in range(len(array_1[a1_i])):
                    if array_1[a1_i][a1_ii] == array_2[a1_i][a1_ii]:
                        continue
                    elif (abs(array_1[a1_i][a1_ii] - array_2[a1_i][a1_ii]) /
                          abs(array_1[a1_i][a1_ii]) > tolerance):
                        if print_mismatch:
                            print('!', a1_i, a1_ii, array_1[a1_i][a1_ii], array_2[a1_i][a1_ii],
                                  abs(array_1[a1_i][a1_ii] - array_2[a1_i][a1_ii]) /
                                  abs(array_1[a1_i][a1_ii]))
                        bad_number += 1

    else:
        print('! dtype = {}, not examined'.format(array_1.dtype))

    print('bad count = {}'.format(bad_number))


#===================================================================================================
# filter array
#===================================================================================================
def filter_array(values, filter_kind='triang', filter_size=3):
    '''
    Get array with smoothing filer applied.

    Parameters
    ----------
    values : array
    filter_kind: str : 'triang', 'boxcar'
    filter_size : int : number of array get_indices

    Returns
    -------
    array : values filtered by input filter_kind
    '''
    window = signal.get_window(filter_kind, filter_size)
    window /= window.sum()

    return ndimage.convolve(values, window)


#===================================================================================================
# dictionary of arrays
#===================================================================================================
def arrayize_dictionary(dic):
    '''
    Convert list entries to numpy arrays.

    Parameters
    ----------
    dic : dictionary of lists
    '''
    for k in dic:
        if isinstance(dic[k], dict):
            dic[k] = arrayize_dictionary(dic[k])
        elif isinstance(dic[k], list) and len(dic[k]):
            dic[k] = np.array(dic[k])


def append_dictionary(dict_1, dict_2):
    '''
    Append elements of dict_2 that are in dict_1 to dict_1.
    If dict_1 is empty, append all elements of dict_2 to it.
    If dict_2 dictionary contains lists/arrays, create/append list of lists/arrays to dict_1.

    Parameters
    ----------
    dict_1, dict_2 : two dictionaries (dict_1 can be empty)
    '''
    # initialize dict_1, if necessary
    if not dict_1:
        for k in dict_2:
            if isinstance(dict_2[k], dict):
                dict_1[k] = {}
                for kk in dict_2[k]:
                    dict_1[k][kk] = []
            else:
                dict_1[k] = []

    # append values to dict_1
    for k in dict_1:
        if k in dict_2:
            if isinstance(dict_1[k], dict):
                for kk in dict_1[k]:
                    if kk in dict_2[k]:
                        if np.isscalar(dict_1[k][kk]):
                            dict_1[k][kk] = [dict_1[k][kk]]
                        dict_1[k][kk].append(dict_2[k][kk])
            else:
                if np.isscalar(dict_1[k]):
                    dict_1[k] = [dict_1[k]]
                dict_1[k].append(dict_2[k])


def compare_dictionaries(dict_1, dict_2, print_mismatch=True, tolerance=0.01):
    '''
    Check if values in dictionaries are the same (within tolerance percent if float).

    Parameters
    ----------
    dict_1, dict_2 : two dictionaries
    print_mismatch : bool : whether to print mismatches
    tolerance : float : fractional difference tolerance
    '''
    bad_number = 0

    if len(dict_1) != len(dict_2):
        print('! dict_1 has {} keys but dict_2 has {}'.format(len(dict_1), len(dict_2)))

    keys = np.intersect1d(list(dict_1), list(dict_2))

    for k in dict_1:
        if k not in dict_2:
            print('! {} is not in dict_2'.format(k))
    for k in dict_2:
        if k not in dict_1:
            print('! {} is not in dict_1'.format(k))

    for k in keys:
        if np.isscalar(dict_1[k]):
            if dict_1[k] != dict_2[k]:
                if print_mismatch:
                    print('!', k, dict_1[k], dict_2[k])
                bad_number += 1
        elif len(dict_1[k]):
            print('{:20s} '.format(k), end='')
            compare_arrays(dict_1[k], dict_2[k], print_mismatch, tolerance)
