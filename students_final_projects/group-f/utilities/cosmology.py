'''
Cosmological parameters and cosmology functions.

@author: Andrew Wetzel

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
from scipy import integrate
# local ----
from . import basic as ut


class CosmologyClass(dict):
    '''
    Class to store cosmological parameters and cosmology functions.

    Parameters
    ----------
    dictionary class (to allow this class to store as if a dictionary)
    '''

    def __init__(
        self, omega_lambda=0.73, omega_matter=0.27, omega_baryon=0.047, hubble=0.7, sigma_8=0.82,
        n_s=0.95, w=-1.0, source=''):
        '''
        Store cosmology parameters.

        Parameters
        ----------
        omega_lambda : float : Omega_lambda(z = 0)
        omega_matter : float : Omega_matter(z = 0)
        omega_baryon : float : Omega_baryon(z = 0)
        hubble : float : dimensionless hubble constant (at z = 0)
        sigma_8 : float : sigma_8(z = 0)
        n_s : float : index (slope) of primordial power spectrum
        w : float : dark energy equation of state
        source : str : use published cosmology: 'planck2015', 'wmap9','agora', 'bolshoi'
        '''
        if source:
            if source == 'planck2015':
                hubble = 0.68
                omega_matter = 0.31
                omega_lambda = 0.69
                omega_baryon = 0.048
                sigma_8 = 0.82
                n_s = 0.97
                w = -1.0
            elif source == 'wmap9':
                hubble = 0.69
                omega_matter = 0.29
                omega_lambda = 0.71
                omega_baryon = 0.046
                sigma_8 = 0.82
                n_s = 0.97
                w = -1.0
            elif source == 'agora':
                hubble = 0.702
                omega_matter = 0.272
                omega_lambda = 0.728
                omega_baryon = 0.0455
                sigma_8 = 0.807
                n_s = 0.961
                w = -1.0
            elif source == 'bolshoi':
                hubble = 0.7
                omega_matter = 0.27
                omega_lambda = 0.73
                omega_baryon = 0.047
                sigma_8 = 0.82
                n_s = 0.95
                w = -1.0
            else:
                raise ValueError('not recognize cosmology source = ' + source)
            self.source = source
        else:
            self.source = None

        # defaults if did not input some cosmology parameters
        if omega_lambda is None and omega_matter is None:
            raise ValueError('! must supply omega_matter or omega_lambda')
        elif omega_lambda is None:
            omega_lambda = 1 - omega_matter
            self.say('assuming flatness, setting omega_lambda = {:.3f}'.format(omega_lambda))
        elif omega_matter is None:
            omega_matter = 1 - omega_lambda
            self.say('assuming flatness, setting omega_matter = {:.3f}'.format(omega_matter))

        # sanity checks
        assert 0 <= omega_lambda <= 1
        assert 0 <= omega_matter <= 1
        assert 0 <= omega_baryon <= 1
        assert 0.5 < hubble <= 1
        assert 0.5 < sigma_8 < 1
        assert 0.9 <= n_s <= 1.1
        assert -1.5 < w < -0.5

        # store as dictionary
        self['omega_lambda'] = omega_lambda
        self['omega_matter'] = omega_matter
        self['omega_baryon'] = omega_baryon
        self['omega_curvature'] = 1 - self['omega_matter'] - self['omega_lambda']
        self['omega_dm'] = self['omega_matter'] - self['omega_baryon']
        self['baryon.fraction'] = self['omega_baryon'] / self['omega_matter']
        self['hubble'] = hubble
        self['sigma_8'] = sigma_8
        self['n_s'] = n_s
        self['w'] = w
        self.DistanceRedshiftSpline = None
        self.TimeScalefactorSpline = None

    # parameters ----------
    def get_omega(self, species_kind='matter', redshifts=0):
        '''
        Get Omega[s] of species at redshift[s].

        Parameters
        ----------
        species_kind : str : species to get density of: 'matter', 'lambda'
        redshifts : float or array : redshift[s]
        '''
        assert species_kind in ['matter', 'lambda']

        if species_kind == 'matter':
            omegas = (self['omega_matter'] /
                      (self['omega_matter'] + self['omega_lambda'] / (1 + redshifts) ** 3 +
                       self['omega_curvature'] / (1 + redshifts) ** 3))
        elif species_kind == 'lambda':
            omegas = (self['omega_lambda'] /
                      (self['omega_matter'] * (1 + redshifts) ** 3 + self['omega_lambda'] +
                       self['omega_curvature'] * (1 + redshifts) ** 2))

        return omegas

    def get_density(self, density_kind='critical', redshifts=0, units='kpc comoving'):
        '''
        Get critical density[s] [M_sun / kpc[/h]^3 comoving or physical] at redshift[s].

        Parameters
        ----------
        density_kind : str : thing to get density of: 'critical', 'matter', 'lambda'
        redshifts : float or array : redshift[s]
        units : str : distance units: 'kpc', 'kpc/h', 'Mpc', 'Mpc/h' + 'comoving' or 'physical'
        '''
        assert density_kind in ['critical', 'matter', 'lambda']

        # [M_sun/h / (kpc/h)^3 comoving]
        densities = (ut.constant.density_critical_0 *
                     (self['omega_lambda'] / (1 + redshifts) ** 3 + self['omega_matter'] +
                      self['omega_curvature'] / (1 + redshifts)))

        densities /= self['hubble']  # [M_sun / (kpc/h)^3 comoving]

        if '/h' not in units:
            densities *= self['hubble'] ** 3  # [M_sun / kpc^3 comoving]

        if 'physical' in units:
            densities *= (1 + redshifts) ** 3  # [M_sun / kpc[/h]^3 physical]
        elif 'comoving' in units:
            pass
        else:
            raise ValueError('need to specify comoving or physical in units = ' + units)

        if density_kind in ['matter', 'lambda']:
            densities *= self.get_omega(density_kind, redshifts)

        return densities

    def get_hubble_parameter(self, redshifts):
        '''
        Get Hubble parameter[s] [sec ^ -1] at redshift[s].

        Parameters
        ----------
        redshifts : float or array : redshift[s]
        '''
        return (ut.constant.hubble_parameter_0 * self['hubble'] *
                (self['omega_matter'] * (1 + redshifts) ** 3 + self['omega_lambda'] +
                 self['omega_curvature'] * (1 + redshifts) ** 2) ** 0.5)

    # time and distance ----------
    def make_time_v_scalefactor_spline(self, scalefactor_limits=[0.01, 1.01], number=500):
        '''
        Make and store spline to get time [Gyr] from scale-factor.
        Use scale-factor (as opposed to redshift) because it is more stable:
        time maps ~linearly onto scale-factor.

        Parameters
        ----------
        scalefactor_limits : list : min and max limits of scale-factor
        number : int : number of spline points within limits
        '''
        self.TimeScalefactorSpline = ut.math.SplineFunctionClass(
            self.get_time, scalefactor_limits, number, value_kind='scalefactor')

    def get_time(self, values, value_kind='redshift'):
        '''
        Get age[s] of the Universe [Gyr] at redshift[s] or scale-factor[s].

        Parameters
        ----------
        values : float or array : redshift[s] or scale-factor[s]
        value_kind : str : 'redshift' or 'scalefactor'

        Returns
        -------
        times : float or array : age[s] of Universe [Gyr]
        '''

        def get_dt(scalefactor, self):
            return (self['omega_matter'] / scalefactor + self['omega_lambda'] * scalefactor ** 2 +
                    self['omega_curvature']) ** -0.5

        if not np.isscalar(values):
            values = np.asarray(values)

        if value_kind == 'scalefactor':
            scalefactors = values
        elif value_kind == 'redshift':
            scalefactors = 1 / (1 + values)
        else:
            raise ValueError('! not recognize value_kind = {}'.format(value_kind))

        if np.isscalar(scalefactors):
            times = ut.constant.hubble_time / self['hubble'] * integrate.quad(
                get_dt, 1e-10, scalefactors, (self))[0]
        else:
            if self.TimeScalefactorSpline is None:
                self.make_time_v_scalefactor_spline()
            times = self.TimeScalefactorSpline.value(scalefactors)

        return times

    def convert_time(self, time_kind_get, time_kind_input, values):
        '''
        Get to_kind value[s] at from_kind value[s].

        Parameters
        ----------
        time_kind_get : str : time kind to get:
            'time', 'time.lookback', 'redshift', 'scalefactor'
        time_kind_input : str : time kind to input:
            'time', 'time.lookback', 'redshift', 'scalefactor'
        values : float or array : input values to convert from
            time[s] [Gyr], lookback-time[s] [Gyr], redshift[s], scale-factor[s]

        Returns
        -------
        age[s] of the Universe [Gyr], lookback-time[s] [Gyr], redshift[s], or scale-factor[s]:
            float or array
        '''

        def get_dt(scalefactor, self):
            return (self['omega_matter'] / scalefactor + self['omega_lambda'] * scalefactor ** 2 +
                    self['omega_curvature']) ** -0.5

        spline_scalefactor_limits = [0.01, 1.01]

        assert time_kind_get != time_kind_input

        if 'time' in time_kind_get:
            if time_kind_input == 'scalefactor':
                scalefactors = np.array(values)
                redshifts = 1 / scalefactors - 1
            elif time_kind_input == 'redshift':
                redshifts = np.array(values)
                scalefactors = 1 / (1 + redshifts)

            if np.isscalar(scalefactors):
                # if scalar, do direct integration
                values_get = ut.constant.hubble_time / self['hubble'] * integrate.quad(
                    get_dt, 1e-10, scalefactors, (self))[0]
            else:
                # if multiple values, use spline fit
                if self.TimeScalefactorSpline is None:
                    if np.min(scalefactors) < spline_scalefactor_limits[0]:
                        spline_scalefactor_limits[0] = np.min(scalefactors)
                    if np.max(scalefactors) > spline_scalefactor_limits[1]:
                        spline_scalefactor_limits[1] = np.max(scalefactors)
                    self.make_time_v_scalefactor_spline(spline_scalefactor_limits)
                values_get = self.TimeScalefactorSpline.value(scalefactors)

            if 'lookback' in time_kind_get:
                # convert to lookback-time
                values_get = self.get_time(0) - values_get

        elif 'time' in time_kind_input:
            if self.TimeScalefactorSpline is None:
                if np.min(scalefactors) < spline_scalefactor_limits[0]:
                    spline_scalefactor_limits[0] = np.min(redshifts)
                if np.max(scalefactors) > spline_scalefactor_limits[1]:
                    spline_scalefactor_limits[1] = np.max(scalefactors)
                self.make_time_v_scalefactor_spline(spline_scalefactor_limits)

            times = np.array(values)
            if 'lookback' in time_kind_input:
                times = self.get_time(0) - times

            values_get = self.TimeScalefactorSpline.value_inverse(times)

            if values_get.size == 1:
                values_get = float(values_get)

            if 'redshift' in time_kind_get:
                values_get = 1 / values_get - 1

        return values_get

    def make_distance_v_redshift_spline(
        self, redshift_limits=[0, 0.2], redshift_number=100, units='kpc comoving'):
        '''
        Make and store spline to get distances from redshifts.

        Parameters
        ----------
        redshift_limits : list : min and max limits of redshift
        redshift_number : int : number of spline points within limits
        units : str : distance units: 'kpc', 'kpc/h', 'Mpc', 'Mpc/h' + 'comoving' or 'physical'
        '''
        self.DistanceRedshiftSpline = ut.math.SplineFunctionClass(
            self.get_distance, redshift_limits, redshift_number, units=units)

    def get_distance(
        self, redshifts, compute_kind='integrate', redshift_limits=[0, 0.2], units='kpc comoving'):
        '''
        Get distance[s] [in input units] from z = 0 to redshift[s].

        Parameters
        ----------
        redshifts : float or array : redshift[s]
        compute_kind : str : integrate, spline
        redshift_limits : list : min and max limits of redshift (if spline)
        units : str : distance units: 'kpc', 'kpc/h', 'Mpc', 'Mpc/h' + 'comoving' or 'physical'

        Returns
        -------
        distances : float or array : distance[s]
        '''

        def get_ddist(redshifts, self):
            return (self['omega_matter'] * (1 + redshifts) ** 3 + self['omega_lambda'] +
                    self['omega_curvature'] * (1 + redshifts) ** 2) ** -0.5

        if compute_kind == 'spline' or not np.isscalar(redshifts):
            if self.DistanceRedshiftSpline is None:
                self.make_distance_v_redshift_spline(redshift_limits, units=units)
            distances = self.DistanceRedshiftSpline.value(redshifts)

        elif compute_kind == 'integrate':
            distances_int = integrate.quad(get_ddist, 1e-10, redshifts, (self))[0]
            distances = ut.constant.hubble_distance * distances_int  # [kpc/h comoving]

            if '/h' in units:
                distances /= self['hubble']

            if 'Mpc' in units:
                distances *= ut.constant.mega_per_kilo
            elif 'kpc' in units:
                pass
            else:
                raise ValueError('need to specify Mpc or kpc in units = {}'.format(units))

            if 'physical' in units:
                distances /= (1 + redshifts)
            elif 'comoving' in units:
                pass
            else:
                raise ValueError('need to specify comoving or physical in units = {}'.format(units))

        return distances

    def get_distance_angular_diameter(
        self, redshifts, compute_kind='integrate', units='kpc comoving'):
        '''
        Get angular diameter distance[s] [in input units] from z = 0 to redshift[s].
        By convention, returns as *physical* (proper) distance.

        Parameters
        ----------
        redshifts : float or array : redshift[s]
        compute_kind : str : integrate, spline
        units : str : distance units: 'kpc', 'kpc/h', 'Mpc', 'Mpc/h' + 'comoving' or 'physical'

        Returns
        -------
        angular diameter distance[s] : float or array : angular diameter distance[s]
        '''
        # ensure scale relative to comoving distance (by convention)
        if 'comoving' in units:
            pass
        elif 'physical' in units:
            units.replace('physical', 'comoving')
        else:
            units += ' comoving'

        # returns physical (proper) distance
        return self.get_distance(redshifts, compute_kind, units=units) / (1 + redshifts)

    def get_distance_luminosity(self, redshifts, compute_kind='integrate', units='kpc comoving'):
        '''
        Get luminosity distance[s] [in input units] from z = 0 to redshift[s].

        Parameters
        ----------
        redshifts : float or arr : redshift[s]
        compute_kind : str : integrate, spline
        units : str : distance units: 'kpc', 'kpc/h', 'Mpc', 'Mpc/h' + 'comoving' or 'physical'

        Returns
        -------
        luminosity distance[s] : float or array : luminosity distance[s]
        '''
        return self.get_distance(redshifts, compute_kind, units=units) * (1 + redshifts)

    def get_size_per_angle(
        self, redshifts, units='kpc physical', angle_kind='arcsec', compute_kind='integrate'):
        '''
        Get size per angle at redshift[s].

        Parameters
        ----------
        redshifts : float or arr : redshift[s]
        units : str : size units: 'kpc', 'kpc/h', 'Mpc', 'Mpc/h' + 'comoving' or 'physical'
        angle_kind : str : radian, degree, arcsec
        compute_kind : str : integrate, spline

        Returns
        -------
        angles : float or array : angle[s] [in input units]
        '''
        angles = self.get_distance(redshifts, compute_kind, units=units)

        if angle_kind == 'degree':
            angles *= ut.constant.radian_per_degree
        elif angle_kind == 'arcsec':
            angles *= ut.constant.radian_per_arcsec
        elif angle_kind == 'arcmin':
            angles *= ut.constant.radian_per_arcmin
        else:
            raise ValueError('not recognize angle kind = {}'.format(angle_kind))

        return angles

    def get_volume(
        self, redshifts=[0, 0.1], area=7966, volume_kind='regular', units='kpc comoving'):
        '''
        Get volume [in input units] in input redshift interval across input area.

        Parameters
        ----------
        redshifts : list : min and max limits for redshift
        area : float : observed area [degree ^ 2]
            note: null = full sky, default is SDSS 'legacy' DR7/DR8 spectroscopic sample
        volume_kind : str : regular, luminosity
        units : str : distance units: 'kpc', 'kpc/h', 'Mpc', 'Mpc/h' + 'comoving' or 'physical'

        Returns
        -------
        volume : float : volume [in input units]
        '''
        area_frac = 1

        if area:
            area_frac = area / ut.constant.deg2_per_sky
        distance_min = self.get_distance(redshifts[0], units=units)
        distance_max = self.get_distance(redshifts[1], units=units)
        if volume_kind == 'luminosity':
            distance_min *= redshifts[0]
            distance_max *= redshifts[1]

        return 4 / 3 * np.pi * area_frac * (distance_max ** 3 - distance_min ** 3)

    # density ----------
    def get_growth_factor(self, redshift):
        '''
        Get growth factor by which density perturbations were smaller at redshift,
        normalized so growth(z = 0) = 1.

        Parameters
        ----------
        redshift: float
        '''

        def get_dgrowth(scalefactor, omega_matter, omega_lambda, omega_curvature):
            return (omega_matter / scalefactor + omega_lambda * scalefactor ** 2 +
                    omega_curvature) ** -1.5

        scalefactor = 1 / (1 + redshift)
        scalefactor_min = 1e-3
        scalefactor_max = 1.0
        omega_matter = self.get_omega('matter', redshift)
        omega_lambda = self.get_omega('lambda', redshift)
        omega_curvature = 1 - omega_matter - omega_lambda
        g_0 = self['omega_matter'] * integrate.quad(
            get_dgrowth, scalefactor_min, scalefactor_max,
            (self['omega_matter'], self['omega_lambda'], self['omega_curvature']))[0]
        g_a = omega_matter * integrate.quad(get_dgrowth, scalefactor_min, scalefactor_max,
                                            (omega_matter, omega_lambda, omega_curvature))[0]

        return g_a / g_0 * scalefactor

    def get_transfer_function(self, k, source='e&h'):
        '''
        Get transfer function at input wave-number k.

        Parameters
        ----------
        k : float : wave number
        source: str : published source: 'e&h', 'e&h-paper', 'ebw'
        '''
        if 'e&h' in source:
            # Eisenstein & Hu 1999
            # CMB temperature conversion from 2.7 K
            theta = 2.728 / 2.7
            # comoving distance that sound wave can propagate
            s = (44.5 * np.log(9.83 / (self['omega_matter'] * self['hubble'] ** 2)) /
                 (1 + 10 * (self['omega_baryon'] * self['hubble'] ** 2) ** 0.75) ** 0.5 *
                 self['hubble'])
            alpha = (1 - 0.328 * np.log(431 * (self['omega_matter'] * self['hubble'] ** 2)) *
                     self['omega_baryon'] / self['omega_matter'] +
                     0.380 * np.log(22.3 * (self['omega_matter'] * self['hubble'] ** 2)) *
                     (self['omega_baryon'] / self['omega_matter']) ** 2)
            gamma = (self['omega_matter'] * self['hubble'] ** 2) * (alpha + (1 - alpha) /
                                                                    (1 + (0.43 * k * s) ** 4))
            # convert q to h/kpc
            q = k * theta ** 2 / gamma * self['hubble']
            if source == 'e&h':
                # modified version
                L = np.log(2 * np.e + 1.8 * q)
                C = 14.2 + 731 / (1 + 62.5 * q)
            elif source == 'e&h-paper':
                # original paper version
                beta = 1 / (1 - 0.949 * self['omega_baryon'] / self['omega_matter'])
                L = np.log(np.e + 1.84 * beta * alpha * q)
                C = 14.4 + 325 / (1 + 60.5 * q ** 1.11)
            return L / (L + C * q ** 2)

        elif source == 'ebw':
            # Efstathiou, Bond & White 1992
            shape = self['omega_matter'] * self['hubble']
            a = 6.4 / shape
            b = 3.0 / shape
            c = 1.7 / shape
            nu = 1.13
            return (1 + (a * k + (b * k) ** (3 / 2) + (c * k) ** 2) ** nu) ** (-1 / nu)

        else:
            raise ValueError('not recognize transfer function source = ' + source)

    def get_delta_2(self, k, source='e&h'):
        '''
        Get *non-normalized* Delta ^ 2(k).
        Need to scale to sigma_8 at z = 0, then need to scale by growth function at redshift.

        Parameters
        ----------
        k : float : wave number
        source : str : published source: 'e&h', 'e&h-paper', 'ebw'
        '''
        return ((ut.constant.hubble_distance * k) ** (3 + self['n_s']) *
                self.get_transfer_function(k, source) ** 2)

    # galaxies ----------
    def convert_magnitude(self, magnitudes, redshifts, kcorrects=0):
        '''
        Get converted magnitude (either absolute from apparent or vice versa).
        Assume absolute magnitudes represented as positive.

        Parameters
        ----------
        magnitudes : float or array : magnitude[s]
        redshifts : float or array : redshift[s]
        kcorrects : float or array : k-correction[s]
        '''
        if np.isscalar(redshifts):
            distances_lum = self.get_distance(redshifts, units='kpc/h comoving') * (1 + redshifts)
        else:
            redshifts = ut.array.arrayize(redshifts)
            self.make_distance_v_redshift_spline(ut.array.get_limits(redshifts), 100)
            distances_lum = self.get_distance_luminosity(
                redshifts, 'spline', units='kpc/h comoving')

        return -magnitudes + 5 * np.log10(distances_lum * ut.constant.kilo / 10) + kcorrects

    def assign_magnitude_v_redshift_spline(
        self, mag_app_max=17.72, redshift_limits=[0, 0.5], redshift_width=0.001):
        '''
        Get spline of absolute magnitude limit v redshift.
        Neglects evolution of k-correct.

        Parameters
        ----------
        mag_app_max : float : apparent magnitude limit
        redshift_limits : list : min and max limits for redshift
        redshift_width : float : redshift bin width
        '''
        redshifts = ut.array.get_arange_safe(
            np.array(redshift_limits) + redshift_width, redshift_width)
        mags_abs = self.convert_magnitude(mag_app_max, redshifts)

        self.MagnitudeRedshiftSpline = ut.math.SplinePointClass(redshifts, mags_abs)
