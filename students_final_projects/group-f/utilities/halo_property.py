'''
Calculate and convert halo properties, including mass, density, radius.

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


def get_circular_velocity(masses, distances):
    '''
    Get circular velocity[s] [km / s] at distance[s]: vel_circ = sqrt(G M(< r) / r)

    Parameters
    ----------
    masses : float or array : mass[es] within distance[s] [M_sun]
    distances : float or array : distance[s] [kpc physical]

    Returns
    -------
    circular velocity[s] at distance[s] [km / s] : float or array
    '''
    return np.sqrt(ut.constant.grav_kpc_msun_sec * masses / distances) * ut.constant.km_per_kpc


class HaloPropertyClass():
    '''
    Class to calculate or convert halo properties, including mass, density, radius.
    '''

    def __init__(self, Cosmology, redshift=None):
        '''
        Store variables from cosmology class and spline for converting between virial density
        definitions.

        Parameters
        ----------
        Cosmology : cosmology class
        redshift : float
        '''
        self.Cosmology = Cosmology
        if redshift is not None:
            self.redshift = redshift
        self.DensityRadiusSpline = None

    def get_nfw_integral_factor(self, radiuss_wrt_scale):
        '''
        Get factor from integrating NFW density profile over volume, such that:
            M(< r) = 4 / 3 * pi * nfw_normalize * background_density * r ^ 3 * nfw_integral_factor

        Parameters
        ----------
        radiuss_wrt_scale : float or array : r / R_scale

        Returns
        -------
        nfw_integral_factor : float or array
        '''
        return (3 / radiuss_wrt_scale ** 3 * (np.log(1 + radiuss_wrt_scale) - 1 /
                                              (1 + 1 / radiuss_wrt_scale)))

    def get_nfw_integral_factor_inv(self, radiuss_wrt_scale):
        '''
        Get inverse of factor from integrating NFW density profile over volume.

        Parameters
        ----------
        radiuss_wrt_scale : float or array :  r / R_scale
        '''
        return 1 / self.get_nfw_integral_factor(radiuss_wrt_scale)

    def get_nfw_normalization(self, concentrations, overdensity=1):
        '''
        Get NFW density amplitude factor, such that:
            density(r) = nfw_normalization * background_density / (r / r_s * (1 + r / r_s) ** 2)
        This is a *fixed number* that does not change with virial definition for a given halo.

        Parameters
        ----------
        concentrations : float or array : concentration[s] = r_vir / r_scale
        overdensity : float : corresponding virial overdensity (with respect to given background)

        Returns
        -------
        nfw_normalization : float or array
        '''
        return overdensity * self.get_nfw_integral_factor_inv(concentrations)

    def get_overdensity_linking_length(self, ll=0.168, reference_kind='matter', redshift=None):
        '''
        Get edge iso-overdensity corresponding to linking length.

        Parameters
        ----------
        ll : float : FoF linking length
        reference_kind : str : reference density: critical, matter
        redshift : float

        Returns
        -------
        overdensity : float
        '''
        if redshift is None:
            redshift = self.redshift

        overdensity = 3 / (2 * np.pi * ll ** 3)  # wrt matter
        if reference_kind[0] == 'c':
            overdensity *= self.Cosmology.get_omega('matter', redshift)

        return overdensity

    def get_overdensity(
        self, virial_kind, reference_kind='critical', redshift=None, units='kpc comoving'):
        '''
        Get virial overdensity wrt reference density and reference density itself [in input units].

        Parameters
        ----------
        virial_kind : str : virial overdensity definition:
          '180m' -> average density is 180 x matter
          '200c' -> average density is 200 x critical
          'vir' -> average density is Bryan & Norman
          'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
          'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
        reference_kind : str : density reference kind: 'critical', 'matter'
        redshift : float

        Returns
        -------
        virial overdensity wrt reference density : float
        reference density [input units] : float
        '''
        if redshift is None:
            redshift = self.redshift

        reference_density = self.Cosmology.get_density(reference_kind, redshift, units)

        if 'vir' in virial_kind:
            # use virial definition according to Bryan & Norman 1998
            if self.Cosmology['omega_curvature'] != 0:
                raise ValueError('cannot use Bryan & Norman fit for Omega_k = {}'.format(
                                 self.Cosmology['omega_curvature']))
            x = self.Cosmology.get_omega('matter', redshift) - 1
            overdensity_value = 18 * np.pi ** 2 + 82 * x - 39 * x ** 2  # wrt critical
            if reference_kind[0] == 'm':
                overdensity_value /= self.Cosmology.get_omega('matter', redshift)
        else:
            overdensity_kind = virial_kind[-1]
            overdensity_value = np.float32(ut.io.get_numbers_in_string(virial_kind, scalarize=True))
            if reference_kind[0] == 'c' and overdensity_kind == 'm':
                overdensity_value *= self.Cosmology.get_omega('matter', redshift)
            elif reference_kind[0] == 'm' and overdensity_kind == 'c':
                overdensity_value /= self.Cosmology.get_omega('matter', redshift)

        return overdensity_value, reference_density

    def convert_concentration(
        self, virial_kind_to, virial_kind_from, concentrations_from, redshift=None,
        solve_kind='fit'):
        '''
        Get halo concentration[s] = R_virial / R_scale for virial kind to get.

        Parameters
        ----------
        virial_kind_to : str : virial overdensity definition to get
          '180m' -> average density is 180 x matter
          '200c' -> average density is 200 x critical
          'vir' -> average density is Bryan & Norman
          'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
          'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
        virial_kind_from : str : virial overdensity definition from which to calculate
        concentrations_from : float : concentration[s] corresponding to virial_kind_from
        redshift : float
        solve_kind : str : options: 'spline', 'fit'

        Returns
        -------
        concentrations_to : float or array : halo concentration[s]
        '''
        if redshift is None:
            redshift = self.redshift

        # get reference overdensities wrt critial
        overdensity_to = self.get_overdensity(virial_kind_to, 'critical', redshift)[0]
        overdensity_from = self.get_overdensity(virial_kind_from, 'critical', redshift)[0]
        if 'fof' in virial_kind_to or 'fof' in virial_kind_from:
            # one of overdensities is defined via halo edge
            if 'fof' in virial_kind_to and 'fof' not in virial_kind_from:
                density_ratio = (self.get_nfw_normalization(concentrations_from, overdensity_from) /
                                 overdensity_to)
                # concens_to_all = np.roots([1, 2, 1, -density_ratio])
                # concentrations_to = float(concens_to_all[np.isreal(concens_to_all)][0])
                # real solution of cubic
                c0 = (10 + 27 * density_ratio) / 54
                c1 = np.sqrt((-1 / 9) ** 3 + c0 ** 2)
                concentrations_to = (c0 + c1) ** (1 / 3) + (c0 - c1) ** (1 / 3) - 2 / 3
            else:
                raise ValueError('not support conversion from fof overdensity definition')
        else:
            # both overdensities are defined via spherical average
            factors = overdensity_from / overdensity_to * self.get_nfw_integral_factor_inv(
                concentrations_from)

            if solve_kind == 'spline':
                if self.DensityRadiusSpline is None:
                    self.DensityRadiusSpline = ut.math.SplineFunctionClass(
                        self.get_nfw_integral_factor_inv, [0.01, 100], 500)
                # ensure input concentrations are in spline range
                factors = factors.clip(self.DensityRadiusSpline.ys.min(),
                                       self.DensityRadiusSpline.ys.max())
                concentrations_to = self.DensityRadiusSpline.value_inverse(factors)
            elif solve_kind == 'fit':
                # fit from Hu & Kravtsov 2002
                factors = 1 / 3 / factors
                p = -0.4283 + -3.13e-3 * np.log(factors) - 3.52e-5 * np.log(factors) ** 2
                concentrations_to = (1 / ((0.5116 * factors ** (2 * p) + 0.5625) ** -0.5 +
                                          2 * factors))

        return concentrations_to

    def get_virial_properties(
        self, virial_kind_to, virial_kind_from, masses=None, halo_radiuss=None,
        concentrations=None, scale_radiuss=None, redshift=None):
        '''
        Get dictionary of virial properties.
        To get ratio[s] of mass and/or radius, set masses = 0 and/or halo_radiuss = 1.

        Parameters
        ----------
        virial_kind_to : str : virial overdensity definition to get:
          '180m' -> average density is 180 x matter
          '200c' -> average density is 200 x critical
          'vir' -> average density is Bryan & Norman
          'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
          'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
        virial_kind_from : str : virial kind from which to calculate
        masses : float or array : halo virial mass[es] [M_sun]
        halo_radiuss : float or array : halo virial radius[es] [kpc physical]
        concentrations : float or array : halo virial concentration[s]
        scale_radiuss : float or array : scale radius[s] [kpc physical]
        redshift: float

        Returns
        -------
        density[s] [M_sun / kpc^3 physical] : float or array
        '''
        if redshift is None:
            redshift = self.redshift

        if 'fof' in virial_kind_from:
            raise ValueError('not support converting from virial edge definition')

        overdensity_from, density_critical = self.get_overdensity(
            virial_kind_from, 'critical', redshift)
        overdensity_to = self.get_overdensity(virial_kind_to, 'critical', redshift)[0]

        if (concentrations is None and
                ((halo_radiuss is None and masses is None) or scale_radiuss is None)):
            raise ValueError(
                'need to input either concentration or virial mass / radius + scale radius')

        if masses is None and halo_radiuss is not None:
            masses = 4 / 3 * np.pi * overdensity_from * density_critical * halo_radiuss ** 3

        if halo_radiuss is None and masses is not None:
            halo_radiuss = (3 / 4 / np.pi * masses / overdensity_from /
                            density_critical) ** (1. / 3)

        if scale_radiuss is not None:
            concentrations = halo_radiuss / scale_radiuss

        virdic = {}
        virdic['concentration'] = self.convert_concentration(
            virial_kind_to, virial_kind_from, concentrations, redshift)

        virdic['mass'] = (masses * (overdensity_to / overdensity_from) ** 3 *
                          (virdic['concentration'] / concentrations) ** 3)

        virdic['radius'] = (
            (virdic['mass'] / masses * overdensity_from / overdensity_to) ** (1 / 3) *
            halo_radiuss)

        virdic['scale.radius'] = virdic['radius'] / virdic['concentration']

        return virdic

    def get_radius_virial(self, virial_kind, masses, redshift=None):
        '''
        Get virial radius[s] [kpc physical].

        Parameters
        ----------
        virial_kind : str : virial overdensity definition
          '180m' -> average density is 180 x matter
          '200c' -> average density is 200 x critical
          'vir' -> average density is Bryan & Norman
          'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
          'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
        masses : float or array : virial mass[s] [M_sun]
        redshift : float

        Returns
        -------
        virial radius[s] [kpc physical] : float or array
        '''
        if redshift is None:
            redshift = self.redshift

        if 'fof' in virial_kind:
            raise ValueError('not support derivation from edge overdensity definition')

        overden, density_critical = self.get_overdensity(virial_kind, 'critical', redshift)

        return (3 / 4 / np.pi * masses / (overden * density_critical)) ** (1 / 3)

    def get_radius_of_zero_acceleration(
        self, mass, redshift, scale_to_halo_radius=False, virial_kind='200m', concentration=10):
        '''
        Get radius around a halo at which inward acceleration from halo matches outward acceleration
        from dark energy.
        Can ignore mass beyond virial radius.

        Parameters
        ----------
        mass : float : virial mass [M_sun]
        redshift : float
        scale_to_halo_radius : bool : whether to scale to halo virial radius
        virial_kind : str : virial overdensity definition
        concentration : float : halo concentration

        Returns
        -------
        radius at which acceleration is 0 [kpc physical or R_halo] : float
        '''
        # density of dark energy [M_sun / kpc^3 physical]
        lambda_density = self.Cosmology.get_density('lambda', redshift, 'kpc physical')

        radius_0 = (3 / (8 * np.pi) * mass / lambda_density) ** (1 / 3)  # [kpc physical]

        # iterate to get mass beyond R_halo
        #"""
        radius_0_old = 0
        mass_radius_0 = self.get_mass_within_radius(
            virial_kind, mass, concentration, radius_0, redshift)

        while np.abs(radius_0 - radius_0_old) / radius_0 > 0.01:
            radius_0_old = radius_0
            radius_0 = (3 / (8 * np.pi) * mass_radius_0 / lambda_density) ** (1 / 3)
            mass_radius_0 = self.get_mass_within_radius(
                virial_kind, mass, concentration, radius_0, redshift)
        #"""

        if scale_to_halo_radius:
            halo_radius = self.get_radius_virial(virial_kind, mass, redshift)
            radius_0 /= halo_radius

        return radius_0

    def get_density_at_radius(
        self, virial_kind, masses, concentrations, radiuss, dimension_number=3, radius_end=None,
        redshift=None):
        '''
        Get density[s] [M_sun / kpc^3 physical] at radius[s].
        Cannot do parallel for 2-D.

        Parameters
        ----------
        virial_kind : str : virial overdensity definition
          '180m' -> average density is 180 x matter
          '200c' -> average density is 200 x critical
          'vir' -> average density is Bryan & Norman
          'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
          'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
        masses : float or array : virial mass[s] [M_sun]
        concentrations : float or array : concentration[s]
        radiuss : float or array : radius[s] [kpc physical]
        dimension_number : int : number of spatial dimensions
        radius_end : float : ending radius (for 2-D integral)
        redshift : float

        Returns
        -------
        density[s] [M_sun / kpc^3 physical] : float or array
        '''
        assert dimension_number in [2, 3]

        if redshift is None:
            redshift = self.redshift

        overdensity, density_critical = self.get_overdensity(
            virial_kind, 'critical', redshift, 'kpc physical')
        normalization = self.get_nfw_normalization(concentrations, overdensity)
        scale_radiuss = self.get_radius_virial(virial_kind, masses, redshift) / concentrations

        if dimension_number == 3:
            return (normalization * density_critical /
                    (radiuss / scale_radiuss * (1 + radiuss / scale_radiuss) ** 2))

        elif dimension_number == 2:
            if not radius_end:
                raise ValueError('need to define ending radius (R_halo) for 2-d projected density')
            elif radiuss > radius_end:
                return 0

            def kernel(r3d, rs, r2d):
                return np.log10(r3d / ((r3d ** 2 - r2d ** 2) ** 0.5 * r3d / rs *
                                       (1 + r3d / rs) ** 2))

            return 10 ** (2 * normalization * integrate.quad(
                kernel, radiuss * 1.001, radius_end, (scale_radiuss, radiuss))[0])

    def get_density_within_radius(
        self, virial_kind, masses, concentrations, radiuss, redshift=None):
        '''
        Get average density[s] [M_sun / kpc^3 physical] within 3-D radius[s].

        Parameters
        ----------
        virial_kind : str : virial overdensity definition
          '180m' -> average density is 180 x matter
          '200c' -> average density is 200 x critical
          'vir' -> average density is Bryan & Norman
          'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
          'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
        masses : float or array : virial mass[s] [M_sun]
        concentrations : float or array
        radiuss : float or array : radius[s] [kpc physical]
        redshift : float

        Returns
        -------
        density[s] [M_sun / kpc^3 physical] : float or array
        '''
        if redshift is None:
            redshift = self.redshift

        overdensity, density_critical = self.get_overdensity(
            virial_kind, 'critical', redshift, 'kpc physical')
        xs = radiuss / (self.get_radius_virial(virial_kind, masses, redshift) / concentrations)

        return (self.get_nfw_normalization(concentrations, overdensity) * density_critical *
                self.get_nfw_integral_factor(xs))

    def get_mass_within_radius(self, virial_kind, masses, concentrations, radiuss, redshift=None):
        '''
        Get mass[s] [M_sun] within 3-D radius[s].

        Parameters
        ----------
        virial_kind : str : virial overdensity definition
          '180m' -> average density is 180 x matter
          '200c' -> average density is 200 x critical
          'vir' -> average density is Bryan & Norman
          'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
          'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
        masses : float or array : virial mass[s] [M_sun]
        concentrations : float or array : concentration[s]
        radiuss : float or array : radius[s] [kpc physical]
        redshift : float

        Returns
        -------
        mass[es] [M_sun] : float or array
        '''
        if redshift is None:
            redshift = self.redshift

        xs = radiuss / (self.get_radius_virial(virial_kind, masses, redshift) / concentrations)

        return (masses * ((np.log(1 + xs) - 1 / (1 + 1 / xs)) /
                          (np.log(1 + concentrations) - 1 / (1 + 1 / concentrations))))

    def get_density_v_radius(
        self, virial_kind, concentrations, DistanceBin, radius_limits_normalize=[], redshift=None):
        '''
        Get density[s] [M_sun / kpc^3 physical] v radii,
        normalized to density within DistanceBin.limits or radius_limits_normalize.

        Parameters
        ----------
        virial_kind : str : virial overdensity definition
          '180m' -> average density is 180 x matter
          '200c' -> average density is 200 x critical
          'vir' -> average density is Bryan & Norman
          'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
          'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
        concentrations : float : concentration
        DistanceBin : distance bin class
        radius_limits_normalize : list : min and max limits of radius to normalize density
        redshift : float

        Returns
        -------
        pro : dict : dictionary of radii and densities as radii
        '''
        rad_bin_values = DistanceBin.mins
        if rad_bin_values[-1] != DistanceBin.limits[1]:
            # extend radius bins to boundaries
            rad_bin_values = np.concatenate((DistanceBin.mins, [DistanceBin.limits[1]]))

        if redshift is None:
            redshift = self.redshift

        if not radius_limits_normalize:
            radius_limits_normalize = DistanceBin.limits

        halo_mass = 13  # dummy
        halo_radius = self.get_radius_virial(virial_kind, halo_mass, concentrations)

        if DistanceBin.dimension_number == 3:
            densities = self.get_density_at_radius(
                virial_kind, halo_mass, concentrations, rad_bin_values * halo_radius)
            mass_in_limit = (
                self.get_mass_within_radius(
                    virial_kind, halo_mass, concentrations,
                    radius_limits_normalize[1] * halo_radius) -
                self.get_mass_within_radius(
                    virial_kind, halo_mass, concentrations,
                    radius_limits_normalize[0] * halo_radius))
            densities *= (DistanceBin.volume_in_limit *
                          halo_radius ** DistanceBin.dimension_number / mass_in_limit)
        elif DistanceBin.dimension_number == 2:
            densities = np.zeros(DistanceBin.number)
            for ri in range(DistanceBin.number):
                densities[ri] = self.get_density_at_radius(
                    virial_kind, halo_mass, concentrations, DistanceBin.mins[ri] * halo_radius,
                    DistanceBin.dimension_number, radius_limits_normalize[1] * halo_radius)
            mass_in_limit = (
                self.get_mass_within_radius(
                    virial_kind, halo_mass, concentrations,
                    radius_limits_normalize[1] * halo_radius) -
                self.get_mass_within_radius(
                    virial_kind, halo_mass, concentrations,
                    radius_limits_normalize[0] * halo_radius))
            densities *= (DistanceBin.volume_in_limit *
                          halo_radius ** DistanceBin.dimension_number / mass_in_limit)

        pro = {'density': densities, 'log density': ut.math.get_log(densities)}

        if 'log' in DistanceBin.scaling:
            pro['radius'] = ut.math.get_log(rad_bin_values)
        else:
            pro['radius'] = rad_bin_values

        return pro

    def get_dynamical_time(self, virial_kind, redshift=None):
        '''
        Get dynamical (gravitational collapse) time of halo [Myr].

        Parameters
        ----------
        virial_kind : str : virial overdensity definition
          '180m' -> average density is 180 x matter
          '200c' -> average density is 200 x critical
          'vir' -> average density is Bryan & Norman
          'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
          'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
        redshift : float

        Returns
        -------
        dynamical (gravitational collapse) time of halo [Myr] : float
        '''
        if redshift is None:
            redshift = self.redshift

        overdensity, density_critical = self.get_overdensity(
            virial_kind, 'critical', redshift, 'kpc physical')

        return (ut.constant.grav_kpc_msun_yr / ut.constant.mega * overdensity *
                density_critical * (1 + redshift) ** 3) ** -0.5

    def get_velocity_circular(self, virial_kind, masses, concentrations, distances, redshift=None):
        '''
        Get circular velocity[s] [km / s] at distance[s].

        Parameters
        ----------
        virial_kind : str : virial overdensity definition
          '180m' -> average density is 180 x matter
          '200c' -> average density is 200 x critical
          'vir' -> average density is Bryan & Norman
          'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
          'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
        masses : float or array : virial mass[s] [M_sun]
        concentrations : float or array : concentration[s]
        distances : float or array : distance[s] [kpc physical]
        redshift : float

        Returns
        -------
        maximum circular velocity [km / s] : float or array
        '''
        if redshift is None:
            redshift = self.redshift

        masses_int = self.get_mass_within_radius(
            virial_kind, masses, concentrations, distances, redshift)

        return get_circular_velocity(masses_int, distances)

    def get_velocity_virial(self, virial_kind, masses, redshift=None):
        '''
        Get virial circular velocity[s] [km / s].

        Parameters
        ----------
        virial_kind : str : virial overdensity definition
          '180m' -> average density is 180 x matter
          '200c' -> average density is 200 x critical
          'vir' -> average density is Bryan & Norman
          'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
          'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
        masses : float or array : virial mass[s] [M_sun]
        redshift : float

        Returns
        -------
        virial velocity [km / s] : float or array
        '''
        if redshift is None:
            redshift = self.redshift

        vir_radiuss = self.get_radius_virial(virial_kind, masses, redshift)  # [kpc physical]

        return get_circular_velocity(masses, vir_radiuss)

    def get_gas_property_virial(self, property_name, virial_kind, masses, redshift=None):
        '''
        Get virial properties associated with gas:
            temperature [Kelvin]
            entropy [erg * cm^2] (assumes equation of state = 5 / 3)
            pressure [erg / cm^3]

        Parameters
        ----------
        property_name : str : halo property: 'temperature', 'entropy', 'pressure'
        virial_kind : str : virial overdensity definition
          '180m' -> average density is 180 x matter
          '200c' -> average density is 200 x critical
          'vir' -> average density is Bryan & Norman
          'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
          'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
        masses : float or array : virial mass[s] [M_sun]
        redshift : float

        Returns
        -------
        prop_values : float or array : virial properties
        '''
        molecular_weight = 0.59  # for fully ionized gas (T > 1e4 K)

        if redshift is None:
            redshift = self.redshift

        # [cm physical]
        vir_radiuss = self.get_radius_virial(virial_kind, masses, redshift) * ut.constant.cm_per_kpc
        vir_vel2s = ut.constant.grav * ut.constant.gram_per_sun * masses / vir_radiuss  # [cm / s]

        if property_name == 'temperature':
            prop_values = (molecular_weight * ut.constant.proton_mass /
                           (2 * ut.constant.boltzmann) * vir_vel2s)

        elif property_name in ['entropy', 'pressure']:
            overdensity, density_critical = self.get_overdensity(
                virial_kind, 'critical', redshift, 'kpc physical')
            vir_gas_densities = (self.Cosmology['omega_baryon'] / self.Cosmology['omega_matter'] *
                                 overdensity * density_critical * ut.constant.gram_per_sun *
                                 ut.constant.kpc_per_cm ** 3)
            if property_name == 'entropy':
                prop_values = (
                    molecular_weight * ut.constant.proton_mass * vir_vel2s / 2 /
                    (vir_gas_densities / (molecular_weight * ut.constant.proton_mass)) ** (2 / 3))
            elif property_name == 'pressure':
                prop_values = 1 / 2 * vir_vel2s * vir_gas_densities
        else:
            raise ValueError('not recognize property = {}'.format(property_name))

        return prop_values
