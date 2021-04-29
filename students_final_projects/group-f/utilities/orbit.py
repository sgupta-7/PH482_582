'''
Class to calculate orbital properties in a general potential.

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
from scipy import integrate, optimize
# local ----
from . import basic as ut
from . import catalog


#===================================================================================================
# general orbit calculation
#===================================================================================================
def get_orbit_dictionary(distance_vectors, velocity_vectors, get_integrals=False):
    '''
    Get dictionary of orbital parameters given input distances and velocities.
    Convert integrals to cgs, assuming distances [kpc physical] and velocities [km / s].

    Parameters
    ----------
    distance_vectors : array :
        distances wrt some center [physical] (object number x dimension number)
    velocity_vectors : array :
        velocities wrt some center [physical] (object number x dimension number)

    Returns
    -------
    orb : dict : dictionary or orbit properties
    '''
    orb = {}

    # distance
    orb['distance'] = distance_vectors
    orb['distance.total'] = np.sqrt(np.sum(distance_vectors ** 2, 1))  # dot product
    orb['distance.norm'] = np.zeros(distance_vectors.shape, distance_vectors.dtype)  # normalized
    masks = np.where(orb['distance.total'] > 0)[0]
    orb['distance.norm'][masks] = np.transpose(
        distance_vectors[masks].transpose() / orb['distance.total'][masks])  # need to do this way

    # velocity
    orb['velocity'] = velocity_vectors
    orb['velocity.total'] = np.sqrt(np.sum(velocity_vectors ** 2, 1))  # dot product
    orb['velocity.tan'] = np.sqrt(np.sum(np.cross(velocity_vectors, orb['distance.norm']) ** 2, 1))
    orb['velocity.rad'] = np.sum(velocity_vectors * orb['distance.norm'], 1)
    # velocity ratio
    orb['velocity.ratio'] = np.zeros(orb['velocity.tan'].shape, orb['velocity.tan'].dtype)
    masks = np.where(orb['velocity.tan'] > 0)[0]
    orb['velocity.ratio'][masks] = np.abs(
        orb['velocity.rad'][masks] / (np.sqrt(0.5) * orb['velocity.tan'][masks]))
    #orb['velocity.beta'] = np.zeros(orb['velocity.rad'].shape, orb['velocity.rad'].dtype)
    #orb['velocity.beta'] = 1 - 0.5 * (orb['velocity.tan'] / orb['velocity.rad']) ** 2

    orb['velocity.norm'] = np.zeros(velocity_vectors.shape, velocity_vectors.dtype)  # normalized
    orb['velocity.norm'] = np.transpose(
        velocity_vectors[masks].transpose() / orb['velocity.total'][masks])  # need to do this way

    if get_integrals:
        # integrals of orbit [cgs]
        # [(cm / s) ^ 2 physical]
        orb['energy.kinetic'] = 0.5 * (orb['velocity.total'] * ut.constant.centi_per_kilo ** 2)
        # [cm^2 / s physical]
        orb['momentum.angular'] = (orb['distance.total'] * ut.constant.cm_per_kpc *
                                   orb['velocity.tan'])

    return orb


class OrbitClass:
    '''
    Given input potential, calculate orbital parameters, including peri/apo-center distance,
    and numerically integrate orbit.
    '''

    def __init__(self):
        '''
        Assign gravitational constant (in simulation units: kpc, M_sun, Gyr) to self.

        This sets the units for all subsequent calculations, so change this if you want to work
        in different units.
        '''
        self.grav_sim_units = ut.constant.grav_kpc_msun_Gyr

    def get_potential(self, distances, masses_tot, halo_radiuss=None, halo_concentrations=None):
        '''
        Get gravitational potential at radius [in simulation units, physical].

        Parameters
        ----------
        distance : float or array : distance[s] [physical]
        mass_tot : float or array : total / main mass[es]
        halo_radiuss : float or array : virial radius[s] [physical]
        halo_concentrations : float : virial concentration[s] (if NFW potential)

        Returns
        -------
        potential[s] : float or array
        '''
        if halo_radiuss is None:
            return -self.grav_sim_units * masses_tot / distances
        else:
            log_concen1 = np.log(1 + halo_concentrations)
            return (self.grav_sim_units * masses_tot *
                    (-1 / distances * np.log(1 + distances / (halo_radiuss / halo_concentrations)) /
                     (log_concen1 - halo_concentrations / (1 + halo_concentrations)) +
                     1 / halo_radiuss *
                     (1 / (1 - halo_concentrations / (1 + halo_concentrations) / log_concen1) - 1)))

    def get_distance_extremum(
        self, distance_kind, distances, energys, angular_momentums, masses_tot, halo_radiuss=None,
        halo_concens=None):
        '''
        Get orbital pericenter or apocenter distance in given gravitational potential
        (NFW potential normalized to Keplerian at R_halo).

        Parameters
        ----------
        distance_kind : str : distance extremum kind: 'peri', 'apo'
        distances : float or array : distance[s] [physical]
        energys : float or array : specific energy[s]
        angular_momentums : float or array : specific angular momentum[s]
        masses_tot : floar or array : main / total mass[es]
        halo_radiuss : float or array : virial radius[s] [physical]
        halo_concens : float or array : virial concentration[s] (if NFW potential)

        Returns
        -------
        extremes : floar or array
        '''

        def root_kep(r, c2, c1, c0):
            return r ** 2 * c2 + r * c1 + c0

        def root_nfw(r, c2, c1, c0, r_s):
            return r ** 2 * c2 + r * np.log(1 + r / r_s) * c1 + c0

        distances, energys, angular_momentums, masses_tot = ut.array.arrayize(
            (distances, energys, angular_momentums, masses_tot))

        if halo_radiuss:
            halo_radiuss = ut.array.arrayize(halo_radiuss)
        if halo_concens:
            halo_concens = ut.array.arrayize(halo_concens)

        if halo_radiuss is None:
            # keplerian potential
            extremes = np.zeros((energys.size, 2)) - 1
            for ene_i in range(energys.size):
                extremes[ene_i] = np.real(
                    np.roots([energys[ene_i], self.grav_sim_units * masses_tot[ene_i],
                              -0.5 * angular_momentums[ene_i] ** 2]))
                #rad_peri = optimize.brentq(root_kep, 0, rad_vir,
                #(e, self.grav_sim_units * mass_tot, -0.5 * ang_mom**2))
            if distance_kind == 'peri':
                extremes = extremes.min(1)
            elif distance_kind == 'apo':
                extremes = extremes.max(1)
        else:
            # NFW potential
            # beware of issues in finding different sign limits for brentq
            extremes = np.zeros(energys.size) - 1
            radiuss_scale = halo_radiuss / halo_concens

            if distance_kind == 'peri':
                radius_mins, radius_maxs = np.zeros(halo_radiuss.size), distances
            elif distance_kind == 'apo':
                radius_mins, radius_maxs = distances, np.zeros(halo_radiuss.size) + 99

            for ene_i in range(energys.size):
                args = (
                    energys[ene_i] -
                    (self.grav_sim_units * masses_tot[ene_i] / halo_radiuss[ene_i] *
                     (1 / (1 - halo_concens[ene_i] / (1 + halo_concens[ene_i]) /
                           np.log(1 + halo_concens[ene_i])) - 1)),
                    self.grav_sim_units * masses_tot[ene_i] /
                    (np.log(1 + halo_concens[ene_i]) - halo_concens[ene_i] /
                     (1 + halo_concens[ene_i])),
                    -0.5 * angular_momentums[ene_i] ** 2, radiuss_scale[ene_i])
                if root_nfw(radius_mins[ene_i], *args) * root_nfw(radius_maxs[ene_i], *args) > 0:
                    extremes[ene_i] = distances[ene_i]
                else:
                    extremes[ene_i] = optimize.brentq(
                        root_nfw, radius_mins[ene_i], radius_maxs[ene_i], args)

            if extremes.size == 1:
                extremes = extremes[0]

            return extremes

    def get_distance_integrate(
        self, time_steps, distances, energys, angular_momentums, masses_tot, halo_radiuss=None,
        halo_concens=None):
        '''
        Integrate each orbit to smaller distance to each of its time step[s].

        Parameters
        ----------
        time_steps : float or array : integration time interval[s] at which to get distances
        distances : float or array : current distance[s] [physical]
        energys : float or array : specific energy[s]
        angular_momentums : float or array : specific angular momentum[s]
        masses_tot : float or array : total / main mass[es]
        halo_radiuss : float or array : virial radius[s] [physical]
        halo_concens : float or array : virial concentration[s] (if NFW potential)

        Returns
        -------
        distances_int : float or array
        '''
        if np.ndim(distances) > 0:
            time_steps = ut.array.arrayize(time_steps, distances.size)
            time_limits = np.array([np.zeros(len(time_steps)), time_steps]).transpose()
            distances_int = np.zeros(distances.size, dtype=distances.dtype)
        else:
            time_limits = [0, time_steps]

        # if want to plot trajectory
        #step_number = 10
        #time_limits = np.arange(0, dt, dt/nstep)
        if halo_radiuss is None:
            # keplerian potential
            if np.ndim(distances) > 0:
                for dist_i in range(distances.size):
                    # get each one's distance at its ending integration time
                    distances_int[dist_i] = integrate.odeint(
                        self.drdt_kep, distances[dist_i], time_limits[dist_i],
                        (energys[dist_i], angular_momentums[dist_i],
                         self.grav_sim_units * masses_tot[dist_i]))[-1, 0]
            else:
                # get its distances at all integration times
                distances_int = integrate.odeint(
                    self.drdt_kep, distances, time_limits,
                    (energys, angular_momentums, self.grav_sim_units * masses_tot))[:, 0]
        else:
            # NFW potential
            potential_amplitudes, potential_constants, radiuss_scale = self.get_nfw_potential_terms(
                masses_tot, halo_radiuss, halo_concens)

            if np.ndim(distances) > 0:
                for dist_i in range(distances.size):
                    # get each one's radius at its ending integration time
                    distances_int[dist_i] = integrate.odeint(
                        self.drdt_nfw, distances[dist_i], time_limits[dist_i],
                        (energys[dist_i], angular_momentums[dist_i], potential_amplitudes[dist_i],
                         potential_constants[dist_i], radiuss_scale[dist_i]))[-1, 0]
            else:
                # get its distances at all integration times
                distances_int = integrate.odeint(
                    self.drdt_nfw, distances, time_limits,
                    (energys, angular_momentums, potential_amplitudes, potential_constants,
                     radiuss_scale))[:, 0]

        if np.ndim(distances) > 0:
            # get each one's radius at its ending integration time
            return distances_int
        else:
            # return just distance at last integration time
            return distances_int[-1]
            # return all distances at all integration times
            #return distances_int.transpose()

    def get_time_integrate(
        self, distances_start, distances_end, energys, angular_momentums, masses_tot,
        halo_radiuss=None, halo_concens=None):
        '''
        Integrate each orbit from starting to ending distance,
        get orbital time across distance range.

        Parameters
        ----------
        distances_start, distances_end : float or array :
            starting and ending orbital distance[s] [physical]
        energys : float or array : specific energy[s]
        angular_momentums : float or array : specific angular momentum[s]
        masses_tot : float or array : total / main mass[es]
        halo_radiuss : float or array : virial radius[s] [physical]
        halo_concens : float or array : virial concentration[s] (if NFW potential)

        Returns
        -------
        orb_times : float or array
        '''
        orb_times = np.zeros(distances_start.size)

        if halo_radiuss is None:
            # keplerian potential
            if np.ndim(distances_start) > 0:
                for dist_i in range(distances_start.size):
                    orb_times[dist_i] = integrate.odeint(
                        self.dtdr_kep, 0, [distances_start[dist_i], distances_end[dist_i]],
                        (energys[dist_i], angular_momentums[dist_i],
                         self.grav_sim_units * masses_tot[dist_i]))[-1, 0]
            else:
                orb_times = integrate.odeint(
                    self.dtdr_kep, 0, [distances_start, distances_end],
                    (energys, angular_momentums, self.grav_sim_units * masses_tot))[-1, 0]
        else:
            # NFW potential
            potential_amplitudes, potential_constants, radiuss_scale = self.get_nfw_potential_terms(
                masses_tot, halo_radiuss, halo_concens)

            if np.ndim(distances_start) > 0:
                for dist_i in range(distances_start.size):
                    orb_times[dist_i] = integrate.odeint(
                        self.dtdr_nfw, 0, [distances_start[dist_i], distances_end[dist_i]],
                        (energys[dist_i], angular_momentums[dist_i], potential_amplitudes[dist_i],
                         potential_constants[dist_i], radiuss_scale[dist_i]))[-1, 0]
            else:
                orb_times = integrate.odeint(
                    self.dtdr_nfw, 0, [distances_start, distances_end],
                    (energys, angular_momentums, potential_amplitudes, potential_constants,
                     radiuss_scale))[-1, 0]

        return orb_times

    # orbit integration kernels ----------
    def drdt_kep(self, radius, _t, energy, momentum_angular, potential_amplitude):
        '''
        Compute dr/dt at t for keplerian potential.

        Parameters
        ----------
        radius : float : radius at time
        _t : float : time
        energy : float : specific energy
        momentum_angular : float : specific angular momentum
        potential_amplitude : float : potential amplitude (G * M_tot)
        '''
        drdt2 = 2 * (energy + potential_amplitude / radius) - (momentum_angular / radius) ** 2

        if drdt2 <= 0:
            return 0
        else:
            return -drdt2 ** 0.5

    def drdt_nfw(
        self, radius, _t, energy, momentum_angular, potential_amplitude, potential_constant,
        radius_scale):
        '''
        Compute dr/dt at t for NFW potential.

        Parameters
        ----------
        radius : float : radius at time
        _t : float : time
        energy : float : specific energy
        momentum_angular : float : specific angular momentum
        potential_amplitude : float : potential amplitude (G * M_tot)
        potential_constant : float : potential constant
        radius_scale : float : NFW scale radius
        '''
        potential = (-potential_amplitude * np.log(1 + radius / radius_scale) / radius +
                     potential_constant)
        drdt2 = 2 * (energy - potential) - (momentum_angular / radius) ** 2

        if drdt2 <= 0:
            return 0
        else:
            return -drdt2 ** 0.5

    def dtdr_kep(self, _t, radius, energy, momentum_angular, potential_amplitude):
        '''
        Compute dt/dr at t for keplerian potential.

        Parameters
        ----------
        _t : float : time
        radius : float : radius at time
        energy : float : specific energy
        momentum_angular : float : specific angular momentum
        potential_amplitude : float : potential amplitude (G * M_tot)
        '''
        drdt2 = 2 * (energy + potential_amplitude / radius) - (momentum_angular / radius) ** 2

        if drdt2 <= 0:
            return 0
        else:
            return -drdt2 ** -0.5

    def dtdr_nfw(
        self, _t, radius, energy, momentum_angular, potential_amplitude, potential_constant,
        radius_scale):
        '''
        Compute dt/dr at t for NFW potential.

        Parameters
        ----------
        _t : float : time
        radius : float : radius at time
        energy : float : specific energy
        momentum_angular : float : specific angular momentum
        potential_amplitude : float : potential amplitude (G * M_tot)
        potential_constant : float : potential constant
        radius_scale : float : NFW scale radius
        '''
        potential = (-potential_amplitude * np.log(1 + radius / radius_scale) / radius +
                     potential_constant)
        drdt2 = 2 * (energy - potential) - (momentum_angular / radius) ** 2

        if drdt2 <= 0:
            return 0
        else:
            return -drdt2 ** -0.5

    def get_nfw_potential_terms(self, masses_tot, halo_radiuss, halo_concens):
        '''
        Get tally of NFW potential terms to speed orbit integration.

        Parameters
        ----------
        masses_tot : float or array : total mass[es]
        halo_radiuss : float or array : virial radius[s] [physical]
        halo_concens : float or array : virial concentration[s]

        Returns
        -------
        potential_amplitudes : float or array
        potential_constants : float or array
        radiuss_scale : float or array
        '''
        radiuss_scale = halo_radiuss / halo_concens
        log_concen_1 = np.log(1 + halo_concens)
        potential_constants = self.grav_sim_units * masses_tot / halo_radiuss * (
            1 / (1 - halo_concens / (1 + halo_concens) / log_concen_1) - 1)
        potential_amplitudes = (self.grav_sim_units *
                                masses_tot / (log_concen_1 - halo_concens / (1 + halo_concens)))

        return potential_amplitudes, potential_constants, radiuss_scale


#===================================================================================================
# orbit for [sub]halo catalog
#===================================================================================================
def get_orbit_dictionary_catalog(
    cat, orbit_indices, center_indices, orbit_mass_kind='', center_mass_kind='',
    get_integrals=False):
    '''
    Get dictionary of orbital parameters of orbit_indices around center_indices.

    Parameters
    ----------
    cat : dict : catalog of [sub]halos at snapshot
    orbit_indices : array : index[s] of [sub]halo[s]
    center_indices : array : index[s] of [sub]halo[s]
    orbit_mass_kind, center_mass_kind : str : mass kinds of [sub]halo[s]
        (to compute mass-dependent integrals of orbit, assuming point masses)
    get_integrals : bool : whether to get orbital integrals

    Returns
    -------
    orb : dict : dictionary of orbit properties
    '''
    orbit_indices, center_indices = ut.array.arrayize(
        (orbit_indices, center_indices), bit_number=32)
    if (orbit_indices.size != center_indices.size and orbit_indices.size != 1 and
            center_indices.size != 1):
        raise ValueError('input indices are arrays but have different size')
    elif not orbit_indices.size or not center_indices.size:
        return {}

    distance_vectors = ut.coordinate.get_distances(
        cat['position'][orbit_indices], cat['position'][center_indices],
        cat.info['box.length'], cat.snapshot['scalefactor'])  # [kpc physical]

    velocity_vectors = catalog.get_velocity_differences(
        cat, cat, orbit_indices, center_indices)  # [km / s]

    orb = get_orbit_dictionary(distance_vectors, velocity_vectors, get_integrals)

    # integrals that depend on mass, assuming point masses [cgs]
    if get_integrals and orbit_mass_kind and center_mass_kind:
        masses_1 = 10 ** cat[orbit_mass_kind][orbit_indices]
        masses_2 = 10 ** cat[center_mass_kind][center_indices]
        masses_tot = masses_1 + masses_2
        #masses_red = masses_1 * masses_2 / (masses_1 + masses_2)

        orb['energy.potental'] = (-ut.constant.grav * masses_tot * ut.constant.sun_mass /
                                  (orb['distance.total'] * ut.constant.cm_per_kpc))
        orb['energy.total'] = orb['energy.potental'] + orb['energy.kinetic']
        orb['eccentricity'] = (1 + 2 * orb['energy.total'] * orb['momentum.angular'] ** 2 /
                               (ut.constant.grav * masses_tot * ut.constant.sun_mass) ** 2) ** 0.5
        if orb['eccentricity'].min() < 0:
            raise ValueError('eccentricity = {} < 0'.format(orb['eccentricity'].min()))
        distances_semi = (
            orb['momentum.angular'] ** 2 /
            ((1 - orb['eccentricity'] ** 2) * ut.constant.grav * masses_tot *
             ut.constant.sun_mass))
        distances_semi *= ut.constant.kpc_per_cm  # [kpc physical]
        orb['distance.peri'] = distances_semi * (1 - orb['eccentricity'])
        orb['distance.apo'] = distances_semi * (1 + orb['eccentricity'])

    if max(orbit_indices.size, center_indices.size) == 1:
        for k in orb:
            orb[k] = orb[k][0]

    return orb
