'''
Utility functions for math and function fitting.

@author: Andrew Wetzel <arwetzel@gmail.com>
'''

# system ----
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import integrate, interpolate, ndimage, special, stats
# local ----
from . import array, constant, io


#===================================================================================================
# math utility
#===================================================================================================
def get_log(values):
    '''
    Safely get log of values.
    Values = 0 is ok, but print warning if values < 0.

    Parameters
    ----------
    values : array

    Returns
    -------
    values : array
    '''
    if np.isscalar(values):
        if values <= 0:
            values = -np.Inf
        else:
            values = np.log10(values)
    else:
        if not isinstance(values, np.ndarray):
            values = np.array(values, np.float32)
        else:
            values = 1.0 * np.array(values)

        if np.min(values) < 0:
            print('! input value minimum = {:.3f} < 0, cannot take log, setting to -Inf!'.format(
                  np.min(values)))

        masks_negative = (values <= 0)

        if np.sum(masks_negative):
            values[masks_negative] = -np.Inf

        masks_positive = (values > 0)

        values[masks_positive] = np.log10(values[masks_positive])

    return values


def percentile_weighted(values, percentiles, weights=None):
    '''
    Compute weighted percentiles.
    If weights are equal, this is the same as normal percentiles.
    Elements of the C{data} and C{weights} arrays correspond to each other and must have
    equal length (unless C{weights} is C{None}).

    Parameters
    ----------
    values : array (1-D)
    percentiles : float or array : percentiles to get corresponding values of [0, 100]
    weights : array : weight to give to each value. if none, weight all equally

    Returns
    -------
    values_at_percents : array : value[s] at each input percentile[s]
    '''
    if weights is None:
        return np.percentile(values, percentiles)
        #weights = np.ones(len(values), dtype=np.float)

    values = np.array(values)
    assert len(values.shape) == 1

    value_number = values.shape[0]
    assert value_number > 0

    if value_number == 1:
        return values[0]

    assert np.min(percentiles) >= 0, 'input percentiles < 0'
    assert np.max(percentiles) <= 100, 'input percentiles > 100'
    if np.isscalar(percentiles):
        percentiles = [percentiles]
    percentiles = np.asarray(percentiles)
    fractions = 0.01 * percentiles

    weights = np.asarray(weights, dtype=np.float)

    assert np.min(weights) >= 0, 'input weights < 0'
    assert weights.shape == values.shape

    indices = np.argsort(values)
    values_sorted = np.take(values, indices, axis=0)
    weights_sorted = np.take(weights, indices, axis=0)
    weights_cumsum = np.cumsum(weights_sorted)
    if not weights_cumsum[-1] > 0:
        raise ValueError('nonpositive weight sum')

    # normalize like np.percentile
    weight_fractions = (weights_cumsum - weights_sorted) / (weights_cumsum[-1] - weights_cumsum[0])
    indices = np.digitize(fractions, weight_fractions) - 1

    values_at_percents = []
    for (ii, frac) in zip(indices, fractions):
        if ii == value_number - 1:
            values_at_percents.append(values_sorted[-1])
        else:
            weight_dif = weight_fractions[ii + 1] - weight_fractions[ii]
            f1 = (frac - weight_fractions[ii]) / weight_dif
            f2 = (weight_fractions[ii + 1] - frac) / weight_dif
            assert f1 >= 0 and f2 >= 0 and f1 <= 1 and f2 <= 1
            assert abs(f1 + f2 - 1.0) < 1e-6
            values_at_percents.append(values_sorted[ii + 1] * f1 + values_sorted[ii] * f2)

    if len(values_at_percents) == 1:
        values_at_percents = values_at_percents[0]

    return values_at_percents


def sample_random_reject(func, params, x_limits, y_max, size):
    '''
    Use rejection method to sample distribution and return as array, assuming minimum of func is 0.

    Parameters
    ----------
    func : function
    params : list : function's parameters
    x_limits : list : min and max limits for function's x-range
    y_max : list : min and max limits for function's y-range
    size : int : number of values to sample

    Returns
    -------
    xs_rand : array : sampled distribution
    '''
    xs_rand = np.random.uniform(x_limits[0], x_limits[1], size)
    ys_rand = np.random.uniform(0, y_max, size)
    ys = func(xs_rand, params)
    xs_rand = xs_rand[ys_rand < ys]
    x_number = xs_rand.size
    if x_number < size:
        xs_rand = np.append(
            xs_rand, sample_random_reject(func, params, x_limits, y_max, size - x_number))

    return xs_rand


def deconvolve(ys_conv, scatter, x_width, iter_number=10):
    '''
    Get Gaussian-deconvolved values via Lucy routine.

    Parameters
    ----------
    ys_conv : array : y-values that already are convolved with gaussian
    scatter : float : gaussian scatter
    x_width : float : bin width
    iter_number : int : number of iterations to do

    Returns
    -------
    y_it : array : deconvolved y values
    '''
    y_it = ys_conv
    for _ in range(iter_number):
        ratio = ys_conv / ndimage.filters.gaussian_filter1d(y_it, scatter / x_width)
        y_it = y_it * ndimage.filters.gaussian_filter1d(ratio, scatter / x_width)
        # this is part of lucy's routine, but seems less stable
        #y_it = y_it * ratio

    return y_it


def convert_luminosity(kind, luminosities):
    '''
    Convert luminosity to magnitude, or vice-versa.

    Parameters
    ----------
    kind : str : luminosity kind: 'luminosity', 'mag.r'
    luminosities : array : value[s] (if luminosity, in Solar luminosity)

    Returns
    -------
    values : array : luminosity[s] or magnitude[s]
    '''
    assert kind in ['luminosity', 'mag.r']

    if kind == 'luminosity':
        values = constant.sun_magnitude - 2.5 * np.log10(luminosities)
    elif kind == 'mag.r':
        if np.min(luminosities) > 0:
            luminosities = -luminosities
        values = 10 ** ((constant.sun_magnitude - luminosities) / 2.5)

    return values


#===================================================================================================
# fraction
#===================================================================================================
class FractionClass(dict, io.SayClass):
    '''
    Compute fraction safely, convert from fraction to ratio, store fractions in self dictionary.
    '''

    def __init__(self, array_shape=None, error_kind=''):
        '''
        Initialize dictionary to store fraction values and uncertainties.

        Parameters
        ----------
        array_shape : int or list : shape of array to store fractions
        error_kind : str : uncertainty kind for fraction: normal, beta
        '''
        self.error_kind = error_kind

        if array_shape is not None:
            array_shape = array.arrayize(array_shape)
            if error_kind:
                if error_kind == 'normal':
                    self['error'] = np.zeros(array_shape)
                elif error_kind == 'beta':
                    self['error'] = np.zeros(np.append(array_shape, 2))
                else:
                    raise ValueError('not recognize uncertainty kind: ' + error_kind)
            else:
                self.say('! not calculating uncertainty for fraction')

            self['value'] = np.zeros(array_shape)
            self['numer'] = np.zeros(array_shape)
            self['denom'] = np.zeros(array_shape)

    def get_fraction(self, numers, denoms, error_kind=''):
        '''
        Get numers/denoms [and uncertainty if uncertainty kind defined].
        Assume numers < denoms, and that numers = 0 if denoms = 0.

        Parameters
        ----------
        numers : float or array : subset count[s]
        denoms : float or array : total count[s]
        error_kind : str : uncertainty kind: '' (= none), 'normal', 'beta'

        Returns
        -------
        frac_values : float or array
        [frac_errors : float or array]
        '''
        if not error_kind:
            error_kind = self.error_kind

        if np.isscalar(numers):
            if numers == 0 and denoms == 0:
                # avoid dividing by 0
                if not error_kind:
                    return 0.0
                elif error_kind == 'normal':
                    return 0.0, 0.0
                elif error_kind == 'beta':
                    return 0.0, np.array([0.0, 0.0])
                else:
                    raise ValueError('not recognize uncertainty kind = ' + error_kind)
            elif denoms == 0:
                raise ValueError('numers != 0, but denoms = 0')
        else:
            numers = np.array(numers)
            denoms = np.array(denoms).clip(1e-20)

        frac_values = numers / denoms

        if error_kind:
            if error_kind == 'normal':
                frac_errors = ((numers / denoms * (1 - numers / denoms)) / denoms) ** 0.5
            elif error_kind == 'beta':
                # Cameron 2011
                conf_inter = 0.683  # 1 - sigma
                p_lo = (numers / denoms - stats.distributions.beta.ppf(
                    0.5 * (1 - conf_inter), numers + 1, denoms - numers + 1))
                p_hi = stats.distributions.beta.ppf(
                    1 - 0.5 * (1 - conf_inter), numers + 1, denoms - numers + 1) - numers / denoms
                frac_errors = np.array([p_lo, p_hi]).clip(0)
            else:
                raise ValueError('not recognize error_kind = {}'.format(error_kind))

            return frac_values, frac_errors
        else:
            return frac_values

    def get_fraction_from_ratio(self, ratio):
        '''
        Get fraction relative to total: x / (x + y).

        Parameters
        ----------
        ratio : float or array : x / y

        Returns
        -------
        float or array : fraction
        '''
        return 1 / (1 + 1 / ratio)

    def get_ratio_from_fraction(self, frac):
        '''
        Get ratio: x / y.

        Parameters
        ----------
        frac : float or array : fraction of total x / (x + y)

        Returns
        -------
        float or array : ratio
        '''
        return frac / (1 - frac)

    def assign_to_dict(self, indices, numer, denom):
        '''
        Assign fraction to self dictionary.

        Parameters
        ----------
        indices : array : index[s] to assign to in self dictionary
        numer : int : subset count[s]
        denom : int : total count[s]
        '''
        if np.ndim(indices):
            indices = tuple(indices)
        self['value'][indices], self['error'][indices] = self.get_fraction(
            numer, denom, self.error_kind)
        self['numer'][indices] = numer
        self['denom'][indices] = denom


Fraction = FractionClass()


#===================================================================================================
# spline fitting
#===================================================================================================
class SplineFunctionClass(io.SayClass):
    '''
    Fit spline [and its inverse] to input function.
    '''

    def __init__(
        self, func, x_limits=[0, 1], number=100, dtype=np.float64, make_inverse=True, **kwargs):
        '''
        Fit f(x) to spline, and fit x(f) if f is monotonic.

        Parameters
        ----------
        func : function f(x)
        x_limits : list : min and max limits on x
        number : int : number of spline points
        dtype : data type to store
        make_inverse : bool : whether to make inverse spline, x(f)
        kwargs : keyword arguments for func
        '''
        self.dtype = dtype
        self.xs = np.linspace(min(x_limits), max(x_limits), number).astype(dtype)

        self.fs = np.zeros(number, dtype)
        for x_i, x in enumerate(self.xs):
            self.fs[x_i] = func(x, **kwargs)

        self.spline_f_from_x = interpolate.splrep(self.xs, self.fs)

        if make_inverse:
            self.make_spline_inverse()

    def make_spline_inverse(self):
        '''
        Make inverse spline, x(f).
        '''
        xs_temp = self.xs
        fs_temp = self.fs
        if fs_temp[1] < fs_temp[0]:
            fs_temp = fs_temp[::-1]
            xs_temp = xs_temp[::-1]
        fis = array.get_arange(fs_temp.size - 1)

        if (fs_temp[fis] < fs_temp[fis + 1]).min():
            self.spline_x_from_f = interpolate.splrep(fs_temp, xs_temp)
        else:
            self.say('! unable to make inverse spline: function values not monotonic')

    # wrappers for spline evaluation, ext=2 raises ValueError if input x in outside of limits
    def value(self, x, ext=2):
        return interpolate.splev(x, self.spline_f_from_x, ext=ext).astype(self.dtype)

    def derivative(self, x, ext=2):
        return interpolate.splev(x, self.spline_f_from_x, der=1, ext=ext).astype(self.dtype)

    def value_inverse(self, f, ext=2):
        return interpolate.splev(f, self.spline_x_from_f, ext=ext).astype(self.dtype)

    def derivative_inverse(self, f, ext=2):
        return interpolate.splev(f, self.spline_x_from_f, der=1, ext=ext).astype(self.dtype)


class SplinePointClass(SplineFunctionClass):
    '''
    Fit spline [and its inverse] to input points.
    '''

    def __init__(self, x_values, f_values, dtype=np.float64, make_inverse=True):
        '''
        Fit f(x) to spline, and fit x(f) if f is monotonic.
        Store to self.

        Parameters
        ----------
        x_values : array : x values
        f_values : array : f(x) values
        dtype : data type to store
        make_inverse : bool : whether to make inverse spline
        '''
        self.Say = io.SayClass(SplineFunctionClass)
        self.dtype = dtype
        self.xs = np.array(x_values)
        self.fs = np.array(f_values)
        self.spline_f_from_x = interpolate.splrep(self.xs, self.fs)
        if make_inverse:
            self.make_spline_inverse()


#===================================================================================================
# general functions
#===================================================================================================
class FunctionClass:
    '''
    Collection of functions, for fitting.
    '''

    def get_ave(self, func, params, x_limits=[0, 1]):

        def integrand_func_ave(x, func, params):
            return x * func(x, params)

        return integrate.quad(integrand_func_ave, x_limits[0], x_limits[1], (func, params))[0]

    def gaussian(self, x, params):
        return (1 / ((2 * np.pi) ** 0.5 * params[1]) *
                np.exp(-0.5 * ((x - params[0]) / params[1]) ** 2))

    def gaussian_normalized(self, x):
        return 1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * x ** 2)

    def gaussian_double(self, x, params):
        return (params[2] * np.exp(-0.5 * ((x - params[0]) / params[1]) ** 2) + (1 - params[2]) *
                np.exp(-0.5 * ((x - params[3]) / params[4]) ** 2))

    def gaussian_double_skew(self, x, params):
        return (params[3] * self.skew(x, params[0], params[1], params[2]) + (1 - params[3]) *
                self.skew(x, params[4], params[5], params[6]))

    def skew(self, x, e=0, width=1, skew=0):
        t = (x - e) / width
        return 2 * stats.norm.pdf(t) * stats.norm.cdf(skew * t) / width

    def erf_0to1(self, x, params):
        '''
        Varies from 0 to 1.
        '''
        return 0.5 * (1 + special.erf((x - params[0]) / (np.sqrt(2) * params[1])))

    def erf_AtoB(self, x, params):
        '''
        Varies from params[2] to params[3].
        '''
        return (params[2] * 0.5 * (1 + special.erf((x - params[0]) / ((np.sqrt(2) * params[1])))) +
                params[3])

    def line(self, x, params):
        return params[0] + x * params[1]

    def power_law(self, x, params):
        return params[0] + params[1] * x ** params[2]

    def line_exp(self, x, params):
        return params[0] + params[1] * x * np.exp(-x ** params[2])

    def m_function_schechter(self, m, params):
        '''
        Compute d(num-den) / d(log m) = ln(10) * amplitude * (10 ^ (m - m_char)) ^ slope *
        exp(-10**(m - m_char)).

        Parameters
        ----------
        m : float : (stellar) mass
        params : list : parameters (0 = amplitude, 1 = m_char, 2 = slope)
        '''
        m_ratios = 10 ** (m - params[1])

        return np.log(10) * params[0] * m_ratios ** params[2] * np.exp(-m_ratios)

    def numden_schechter(self, m, params, m_max=20):
        '''
        Get cumulative number density above m.

        Parameters
        ----------
        m : float : (stellar) mass
        params : list : parameters (0 = amplitude, 1 = m_char, 2 = slope)
        m_max : float : maximum mass for integration
        '''
        return integrate.quad(self.m_function_schechter, m, m_max, (params))[0]


Function = FunctionClass()


#===================================================================================================
# function fitting
#===================================================================================================
def get_chisq_reduced(values_test, values_ref, values_ref_err, param_number=1):
    '''
    Get reduced chi ^ 2, excising reference values with 0 uncertainty.

    Parameters
    ----------
    values_test : array : value[s] to test
    values_ref : array : reference value[s]
    values_ref_err : array : reference uncertainties (can be asymmetric)
    param_number : int : number of free parameters in getting test values

    Returns
    -------
    array : chi^2 / dof
    '''
    Say = io.SayClass(get_chisq_reduced)

    values_test = array.arrayize(values_test)
    values_ref = array.arrayize(values_ref)
    values_ref_err = array.arrayize(values_ref_err)

    if np.ndim(values_ref_err) > 1:
        # get uncertainty on correct side of reference values
        if values_ref_err.shape[0] != values_ref.size:
            values_ref_err = values_ref_err.transpose()
        values_ref_err_sided = np.zeros(values_ref_err.shape[0])
        values_ref_err_sided[values_test <= values_ref] = \
            values_ref_err[:, 0][values_test <= values_ref]
        values_ref_err_sided[values_test > values_ref] = \
            values_ref_err[:, 1][values_test > values_ref]
    else:
        values_ref_err_sided = values_ref_err

    val_indices = array.get_arange(values_ref)[values_ref_err_sided > 0]
    if val_indices.size != values_ref.size:
        Say.say('excise {} reference values with uncertainty = 0'.format(
                values_ref.size - val_indices.size))

    chi2 = np.sum(((values_test[val_indices] - values_ref[val_indices]) /
                   values_ref_err_sided[val_indices]) ** 2)
    dof = val_indices.size - 1 - param_number

    return chi2 / dof

"""
NEED TO UPDATE FOR np fit

def fit(func, params, x_values, y_values, y_errors=None):
    '''
    Fit function via mpfit and return as Fit.params.

    Parameters
    ----------
    func : function to fit to
    params : list of lists : inital parameter values and ranges [[value, lo, hi], etc]
    x_values : array : x values
    y_values : array : y values
    y_errors : array : uncertainties
    '''

    def test_fit(params, func, x_values, y_values, y_errors, fjac=None):  # @UnusedVariable
        '''
        Parameters
        ----------
        params : list of lists : parameter values
        func : function to fit
        x_values : array
        y_values : array
        y_errors : array : uncertainties in y_values
        [fjac = if want partial derivs, not used here, but need to keep]
        '''
        model = func(x_values, params)
        status = 0  # non-negative status value means MPFIT should continue

        return [status, (y_values - model) / y_errors]

    Say = io.SayClass(fit)

    if y_errors is None or not len(y_errors):
        y_errors = np.zeros(y_values.size) + 0.1 * y_values.mean()
    else:
        y_errors = np.array(y_errors, np.float64)

    if np.NaN in y_values:
        Say.say('! Nan values in y_values')
    if np.NaN in y_errors:
        Say.say('! Nan values in y_errors')

    if np.min(y_errors) <= 0:
        Say.say('! 0 (or negative) values in y_errors, excising these from fit')
        yis = array.get_arange(y_errors)
        yis = yis[y_errors > 0]
        x_values = x_values[yis]
        y_values = y_values[yis]
        y_errors = y_errors[yis]

    pinfo = []
    for p in params:
        # if parameters limit are the same, interpret as fixed parameter
        if p[1] == p[2]:
            pinfo.append({'value': p[0], 'fixed': 1})
        else:
            pinfo.append({'value': p[0], 'fixed': 0, 'limited': [1, 1], 'limit': [p[1], p[2]]})

    fa = {'func': func, 'x': x_values, 'y': y_values, 'y_err': y_errors}

    Fit = nmpfit.MpFit(test_fit, parinfo=pinfo, functkw=fa, quiet=1)

    if Fit.status >= 5:
        Say.say('! mpfit status {}, tolerances too small'.format(Fit.status))
    elif Fit.status <= 0:
        Say.say('! mpfit error: ' + Fit.errmsg)

    for pi in range(Fit.params.size):
        if Fit.params[pi] == params[pi][1] or Fit.params[pi] == params[pi][2]:
            Say.say('! fit parameter {} = {:.3f} is at its input limit'.format(pi, Fit.params[pi]))

    return Fit
"""

