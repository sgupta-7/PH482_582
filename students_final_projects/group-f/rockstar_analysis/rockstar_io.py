#!/usr/bin/env python3

'''
Read rockstar halo/galaxy catalogs and ConsistentTrees halo merger trees.

@author: Andrew Wetzel <arwetzel@gmail.com>


Units: unless otherwise noted, all quantities are converted to (combinations of):
    mass [M_sun]
    position [kpc comoving]
    distance, radius [kpc physical]
    velocity [km / s]
    time [Gyr]
    elemental abundance [mass fraction]


Halo overdensity definition

By default, I run Rockstar using R_200m as the halo overdensity definition.
Thus, halos are defined by computing the radius that encloses 200 x mean matter density.


Reading a halo catalog

Within a simulation directory, read all halos in the snapshot at redshift 0 via:
    hal = rockstar.io.IO.read_catalogs('redshift', 0)
hal is a dictionary, with a key for each property. So, access via:
    hal[property_name]
For example:
    hal['mass']
returns a numpy array of masses, one for each halo, while
    hal['position']
returns a numpy array of positions (of dimension particle_number x 3).


Default/stored properties

The most common/important are:
    'id' : catalog ID, valid at given snapshot (starts at 0)
    'position' : 3-D position, along simulations's (cartesian) x,y,z grid [kpc comoving]
    'velocity' : 3-D velocity, along simulations's (cartesian) x,y,z grid [km / s]
    'mass' : default total mass - M_200m is default overdensity definition [M_sun]
    'radius' : halo radius, for 'default' overdensity definition of R_200m [kpc physical]
    'scale.radius' : NFW scale radius [kpc physical]
    'mass' : total mass defined via 200 x mean matter density [M_sun]
    'mass.vir' : total mass defined via Bryan & Norman 1998
    'mass.200c' : total mass defined via 200 x critical density [M_sun]
    'mass.bound' : total mass within R_200m that is bound to the halo [M_sun]
    'vel.circ.max' : maximum of the circular velocity profile [km / s]
    'vel.std' : standard deviation of the velocity of particles [km/s physical]
    'mass.lowres' : mass from low-resolution dark-matter particles in halo [M_sun]
    'host.index' : catalog index of the primary host (highest halo mass) in catalog
    'host.distance' : 3-D distance wrt center of primary host [kpc physical]
    'host.velocity' : 3-D velocity wrt center of primary host [km / s]
    'host.velocity.tan' : tangential velocity wrt primary host [km / s]
    'host.velocity.rad' : radial velocity wrt primary host (negative = inward) [km / s]
If you set host_number > 1, or the simulation directory name contains 'elvis', you also have:
    'host2.index' : catalog index of the secondary host in catalog
    'host2.distance' : 3-D distance wrt center of secondary host [kpc physical]
    'host2.velocity' : 3-D velocity wrt center of secondary host [km / s]
    'host2.velocity.tan' : tangential velocity wrt secondary host [km / s]
    'host2.velocity.rad' : radial velocity wrt secondary host (negative = inward) [km / s]

If you read the halo main progenitor histories (hlist*.list), you also have:
    'major.merger.snapshot' : snapshot index of last major merger
    'mass.peak' : maximum of mass throughout history [M_sun]
    'mass.peak.snapshot': snapshot index at which above occurs
    'vel.circ.peak' : maximum of vel.circ.max throughout history [km / s]
    'infall.snapshot' : snapshot index when most recently fell into a host halo
    'infall.mass' : mass when most recently fell into host halo [M_sun]
    'infall.vel.circ.max' : vel.circ.max when most recently fell into a host halo [km / s]
    'infall.first.snapshot' : snapshot index when first became a satellite
    'infall.first.mass' : mass when first fell into a host halo (became a satellite) [M_sun]
    'infall.first.vel.circ.max' : vel.circ.max when first became a satellite [km / s]
    'mass.half.snapshot' : snapshot when first had half of current mass
    'accrete.rate' : instantaneous accretion rate [M_sun / yr]
    'accrete.rate.100Myr : mass growth rate averaged over 100 Myr [M_sun / yr]
    'accrete.rate.tdyn : mass growth rate averaged over dynamical time [M_sun / yr]

If you read the halo merger trees (tree*.dat), you also have:
    'tid' : tree ID, unique across all halos across all snapshots (starts at 0)
    'snapshot' : snapshot index of halo
    'am.phantom' : whether halo is interpolated across snapshots
    'descendant.snapshot' : snapshot index of descendant
    'descendant.index' : tree index of descendant
    'am.progenitor.main' : whether am most massive progenitor of my descendant
    'progenitor.number' : number of progenitors
    'progenitor.main.index' : index of main (most massive) progenitor
    'progenitor.co.index' : index of next co-progenitor (with same descendant)
    'final.index' : tree index at final snapshot
    'dindex' : depth-first order (index) within tree
    'progenitor.co.dindex' : depth-first index of next co-progenitor
    'progenitor.last.dindex' : depth-first index of last progenitor - includes *all* progenitors
    'progenitor.main.last.dindex' : depth-first index of last progenitor - only via main progenitors
    'central.index' tree index of most massive central halo (which must be a central)
    'central.local.index' : tree index of local (lowest-mass) central (which could be a satellite)
    'host.index' : tree index of the primary host (following back main progenitor branch)
    'host.distance' : 3-D distance wrt center of primary host [kpc physical]
    'host.velocity' : 3-D velocity wrt center of primary host [km / s]
    'host.velocity.tan' : tangential velocity wrt primary host [km / s]
    'host.velocity.rad' : radial velocity wrt primary host (negative = inward) [km / s]
If you set host_number > 1, or the simulation directory name contains 'elvis', you also have:
    'host2.index' : tree index of the secondary host in catalog
    'host2.distance' : 3-D distance wrt center of secondary host [kpc physical]
    'host2.velocity' : 3-D velocity wrt center of secondary host [km / s]
    'host2.velocity.tan' : tangential velocity wrt secondary host [km / s]
    'host2.velocity.rad' : radial velocity wrt secondary host (negative = inward) [km / s]

If you read in a halo star catalog, with star particles assigned, you also have:
    'star.number' : number of star particles in halo [M_sun]
    'star.mass' : mass from all star particles in halo [M_sun]
    'star.radius.50' : radius that encloses 50% of stellar mass [kpc physical]
    'star.vel.std.50' : stellar velocity dispersion (standard deviation) at R_50 [km / s]
    'star.position' : center-of-mass position of star particles [kpc comoving]
    'star.velocity' : center-of-mass velocity of star particles [km / s]
    'star.indices' : indices of member star particles in the particle catalog at the same snapshot
        example: pis = hal['star.indices'][0] for halo 0,
        then get star particle properties via part['star'][property_name][pis]
    'star.form.time.50' : time (age of Universe) when formed 50% of current stellar mass [Gyr]
    'star.host.index' : index of primary host galaxy (highest stellar mass) in catalog
    'star.host.distance' : 3-D distance wrt center of primary host galaxy [kpc physical]
    'star.host.velocity' : 3-D velocity wrt center of primary host galaxy [km / s]
    'star.host.velocity.tan' : tangential velocity wrt primary host galaxy [km / s]
    'star.host.velocity.rad' : radial velocity wrt primary host galaxy (negative = inward) [km / s]
    'dark2.mass' : low-resolution DM mass within halo radius


Derived properties

hal is a HaloDictionaryClass that can compute derived properties on the fly.
Call derived (or stored) properties via:
    hal.prop(property_name)
For example:
    hal.prop('host.distance.total')
    hal.prop('star.density.50')
    hal.prop('star.age.50')
You also can call stored properties via hal.prop(property_name).
It will know that it is a stored property and return as is.
For example, hal.prop('position') is the same as hal['position'].

See HaloDictionaryClass.prop() for full option for parsing of derived properties.
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import os
import collections
import numpy as np
from numpy import Inf
from scipy import spatial
# local ----
import utilities as ut

#===================================================================================================
# defaults
#===================================================================================================
# subset of 64 snapshots indices on which to run halo finder, particle assignment, etc
snapshot_indices_subset = np.array([
    20, 26, 33, 41, 52,  # z = 10 - 6
    55, 57, 60, 64, 67,  # z = 5.8 - 5.0
    71, 75, 79, 83, 88,  # z = 4.8 - 4.0
    91, 93, 96, 99, 102, 105, 109, 112, 116, 120,  # z = 3.9 - 3.0
    124, 128, 133, 137, 142, 148, 153, 159, 165, 172,  # z = 2.9 - 2.0
    179, 187, 195, 204, 214, 225, 236, 248, 262, 277,  # z = 1.9 - 1.0
    294, 312, 332, 356, 382, 412, 446, 486, 534,  # z = 0.9 - 0.1
    539, 544, 550, 555, 561, 567, 573, 579, 585,  # z = 0.09 - 0.01
    600  # z = 0
])

# default rockstar sub-directory (within simulation directory)
ROCKSTAR_DIRECTORY = 'halo/rockstar_dm'

# default directory to store halo 'raw' text files
HALO_CATALOG_DIRECTORY = 'catalog'

# default directory to store halo 'processed' hdf5 files
HALO_CATALOG_HDF5_DIRECTORY = 'catalog_hdf5'

# minimum fraction of mass that is bound to consider/trust a halo
BOUND_MASS_FRAC_MIN = 0.4

# maximum contamination from low-resolution DM to consider/trust a halo
LOWRES_MASS_FRAC_MAX = 0.02


#===================================================================================================
# read rockstar halo/galaxy catalog
#===================================================================================================
class HaloDictionaryClass(dict):
    '''
    Dictionary class to store halo/galaxy properties.
    Allows production of derived quantities.
    '''

    def __init__(self):
        # use to translate between element name and index in element table
        self.element_dict = collections.OrderedDict()
        self.element_dict['metals'] = self.element_dict['total'] = 0
        self.element_dict['helium'] = self.element_dict['he'] = 1
        self.element_dict['carbon'] = self.element_dict['c'] = 2
        self.element_dict['nitrogen'] = self.element_dict['n'] = 3
        self.element_dict['oxygen'] = self.element_dict['o'] = 4
        self.element_dict['neon'] = self.element_dict['ne'] = 5
        self.element_dict['magnesium'] = self.element_dict['mg'] = 6
        self.element_dict['silicon'] = self.element_dict['si'] = 7
        self.element_dict['sulphur'] = self.element_dict['s'] = 8
        self.element_dict['calcium'] = self.element_dict['ca'] = 9
        self.element_dict['iron'] = self.element_dict['fe'] = 10

        # to use if read only subset of elements
        self.element_pointer = np.arange(len(self.element_dict) // 2)

    def prop(self, property_name='', indices=None, dict_only=False):
        '''
        Get property, either from self dictionary or derive.
        If several properties, need to provide mathematical relationship.

        Parameters
        ----------
        property_name : str : name of property
        indices : array : list of indices to select on (of arbitrary dimensions)
        dict_only : bool : require property_name to be in self's dict - avoids endless recursion

        Returns
        -------
        values : float or array : depending on dimensionality of input indices
        '''
        ## parsing general to all catalogs ----------
        property_name = property_name.strip()  # strip white space

        # if input is in self dictionary, return as is
        if property_name in self:
            if indices is not None:
                return self[property_name][indices]
            else:
                return self[property_name]
        elif dict_only:
            raise KeyError('property = {} is not in self\'s dictionary'.format(property_name))

        # math relation, combining more than one property
        if ('/' in property_name or '*' in property_name or '+' in property_name or
                '-' in property_name):
            prop_names = property_name

            for delimiter in ['/', '*', '+', '-']:
                if delimiter in property_name:
                    prop_names = prop_names.split(delimiter)
                    break

            if len(prop_names) == 1:
                raise ValueError('not sure how to parse property = {}'.format(property_name))

            # make copy so not change values in input catalog
            prop_values = np.array(self.prop(prop_names[0], indices))

            for prop_name in prop_names[1:]:
                if '/' in property_name:
                    if np.isscalar(prop_values):
                        if self.prop(prop_name, indices) == 0:
                            prop_values = np.nan
                        else:
                            prop_values = prop_values / self.prop(prop_name, indices)
                    else:
                        masks = self.prop(prop_name, indices) != 0
                        prop_values[masks] = (
                            prop_values[masks] / self.prop(prop_name, indices)[masks])
                        masks = self.prop(prop_name, indices) == 0
                        prop_values[masks] = np.nan
                if '*' in property_name:
                    prop_values = prop_values * self.prop(prop_name, indices)
                if '+' in property_name:
                    prop_values = prop_values + self.prop(prop_name, indices)
                if '-' in property_name:
                    prop_values = prop_values - self.prop(prop_name, indices)

            if prop_values.size == 1:
                prop_values = np.float(prop_values)

            return prop_values

        # math transformation of single property
        if property_name[:3] == 'log':
            return ut.math.get_log(self.prop(property_name.replace('log', ''), indices))

        if property_name[:3] == 'abs':
            return np.abs(self.prop(property_name.replace('abs', ''), indices))

        ## parsing specific to halo catalog ----------
        if '.barysim' in property_name:
            values = self.prop(property_name.replace('.barysim', ''))
            # if halos from a DM-only simulation, re-scale mass or V_circ,max by subtracting
            # baryonic mass fraction contained in DM particles
            if not self.info['baryonic']:
                dm_fraction = self.Cosmology['omega_dm'] / self.Cosmology['omega_matter']
                if 'mass' in property_name:
                    values *= dm_fraction
                elif 'vel.circ.max' in property_name:
                    values *= np.sqrt(dm_fraction)

            return values

        if 'mass.' in property_name:
            if property_name == 'mass.hires':
                # high-res mass from Rockstar
                values = self.prop('mass - mass.lowres', indices)
            elif property_name == 'lowres.mass.frac' or property_name == 'dark2.mass.frac':
                # low-res mass from Rockstar
                values = self.prop('mass.lowres / mass', indices)
                # check if catalog has direct assigment of low-res dark2 particles from snapshot
                # if so, use larger of the two low-res masses
                if 'dark2.mass' in self:
                    # low-res mass from direct assignment of particles
                    values_t = self.prop('dark2.mass / mass', indices)
                    if np.isscalar(values) and values_t > values:
                        values = values_t
                    else:
                        masks = (values_t > values)
                        values[masks] = values_t[masks]
            else:
                # mass from individual element
                values = (self.prop('mass', indices, dict_only=True) *
                          self.prop(property_name.replace('mass.', 'massfraction.'), indices))

                if property_name == 'mass.hydrogen.neutral':
                    # mass from neutral hydrogen (excluding helium, metals, and ionized hydrogen)
                    values = values * self.prop(
                        'hydrogen.neutral.fraction', indices, dict_only=True)

            return values

        if 'vel.circ.max.' in property_name:
            scale_radius_factor = 2.1626  # R(V_circ,max) = scale_radius_factor * R_scale
            scale_radius_name = 'scale.radius.klypin'

            if 'radius' in property_name:
                # radius at which V_circ,max occurs
                values = scale_radius_factor * self[scale_radius_name][indices]
            elif 'mass' in property_name:
                # mass within R(V_circ,max)
                values = (self['vel.circ.max'][indices] ** 2 *
                          scale_radius_factor * self[scale_radius_name][indices] *
                          ut.constant.km_per_kpc / ut.constant.grav_km_msun_sec)

            return values

        # element string -> index conversion
        if 'massfraction.' in property_name or 'metallicity.' in property_name:
            if 'massfraction.hydrogen' in property_name or property_name == 'massfraction.h':
                # special case: mass fraction of hydrogen (excluding helium and metals)
                values = (
                    1 - self.prop('massfraction', indices)[:, 0] -
                    self.prop('massfraction', indices)[:, 1])

                if property_name == 'massfraction.hydrogen.neutral':
                    # mass fraction of neutral hydrogen (excluding helium, metals, and ionized)
                    values = values * self.prop(
                        'hydrogen.neutral.fraction', indices, dict_only=True)

                return values

            element_index = None
            for prop_name in property_name.split('.'):
                if prop_name in self.element_dict:
                    element_index = self.element_pointer[self.element_dict[prop_name]]
                    element_name = prop_name
                    break

            if element_index is None:
                raise KeyError('not sure how to parse property = {}'.format(property_name))

            if 'star.' in property_name:
                massfraction_name = 'star.massfraction'
            elif 'gas.' in property_name:
                massfraction_name = 'gas.massfraction'

            if indices is None:
                values = self[massfraction_name][:, element_index]
            else:
                values = self[massfraction_name][indices, element_index]

            if 'metallicity.' in property_name:
                values = ut.math.get_log(
                    values / ut.constant.sun_composition[element_name]['massfraction'])

            return values

        # average stellar density
        if 'star.density' in property_name:
            if property_name == 'star.density':
                property_name += '.50'  # use R_50 as default radius to measure stellar density

            radius_percent = float(property_name.split('.')[-1])
            radius_name = 'star.radius.' + property_name.split('.')[-1]

            values = self.prop(radius_name, indices, dict_only=True)
            #masks = np.isfinite(values)
            #masks[masks] *= (values[masks] > 0)
            #values[masks] = (
            #    radius_percent / 100 * self.prop('star.mass', indices, dict_only=True)[masks] /
            #    (4 / 3 * np.pi * self.prop(radius_name, indices)[masks] ** 3))
            values = (
                radius_percent / 100 * self.prop('star.mass', indices, dict_only=True) /
                (4 / 3 * np.pi * self.prop(radius_name, indices) ** 3))
            #if values.size == 1:
            #    values = np.asscalar(values)

            return values

        # velocity (dispersion) along 1 dimension
        if 'vel.' in property_name and '.1d' in property_name:
            values = self.prop(property_name.replace('.1d', ''), indices) / np.sqrt(3)

        # distance/velocity wrt center of a primary hostal
        if 'host' in property_name:
            #if 'host.near.' in property_name:  # TODO add this capability
            #    host_name = 'host.'
            if 'host.' in property_name or 'host1.' in property_name:
                host_name = 'host.'
            elif 'host2.' in property_name:
                host_name = 'host2.'
            elif 'host3.' in property_name:
                host_name = 'host3.'
            else:
                raise ValueError('could not identify host name in {}'.format(property_name))

            if host_name + 'distance' in property_name:
                if 'star.' in property_name:
                    values = self.prop('star.' + host_name + 'distance', indices, dict_only=True)
                else:
                    values = self.prop(host_name + 'distance', indices, dict_only=True)
            elif host_name + 'velocity' in property_name:
                if 'star.' in property_name:
                    values = self.prop('star.' + host_name + 'velocity', indices, dict_only=True)
                else:
                    values = self.prop(host_name + 'velocity', indices, dict_only=True)

            if 'cylindrical' in property_name:
                # convert to cylindrical coordinates
                if 'distance' in property_name:
                    # along major axes R (positive definite), minor axis Z (signed),
                    # angle phi (0 to 2 * pi)
                    values = ut.coordinate.get_positions_in_coordinate_system(
                        values, 'cartesian', 'cylindrical')
                if 'velocity' in property_name:
                    # along major axes (v_R), minor axis (v_Z), angular (v_phi)
                    if 'principal' in property_name:
                        distance_vectors = self.prop(host_name + 'distance.principal', indices)
                    else:
                        distance_vectors = self.prop(
                            host_name + 'distance', indices, dict_only=True)
                    values = ut.coordinate.get_velocities_in_coordinate_system(
                        values, distance_vectors, 'cartesian', 'cylindrical')

            elif 'spherical' in property_name:
                # convert to spherical coordinates
                if 'distance' in property_name:
                    # along R (positive definite), theta [0, pi), phi [0, 2 * pi)
                    values = ut.coordinate.get_positions_in_coordinate_system(
                        values, 'cartesian', 'spherical')
                if 'velocity' in property_name:
                    # along v_R, v_theta, v_phi
                    if 'principal' in property_name:
                        distance_vectors = self.prop(host_name + 'distance.principal', indices)
                    else:
                        distance_vectors = self.prop(
                            host_name + 'distance', indices, dict_only=True)
                    values = ut.coordinate.get_velocities_in_coordinate_system(
                        values, distance_vectors, 'cartesian', 'spherical')

            if 'total' in property_name:
                # compute total (scalar) distance / velocity
                if len(values.shape) == 1:
                    shape_pos = 0
                else:
                    shape_pos = 1
                values = np.sqrt(np.sum(values ** 2, shape_pos))

            return values

        # walking merger tree
        if 'progenitor' in property_name or 'descendant' in property_name:
            # for now, only handle one input halo at a time
            assert indices is not None and np.isscalar(indices)

            # get main progenitor going back in time (including self)
            if property_name == 'progenitor.main.indices':
                values = []
                hindex = indices
                while hindex >= 0:
                    values.append(hindex)
                    hindex = self['progenitor.main.index'][hindex]

            # get all progenitors at previous snapshot
            if property_name == 'progenitor.indices':
                values = []
                hindex = self['progenitor.main.index'][indices]
                while hindex >= 0:
                    values.append(hindex)
                    hindex = self['progenitor.co.index'][hindex]

            # get descendants going forward in time (including self)
            if property_name == 'descendant.indices':
                values = []
                hindex = indices
                while hindex >= 0:
                    values.append(hindex)
                    hindex = self['descendant.index'][hindex]

            return np.array(values, self['descendant.index'].dtype)

        # stellar formation time / age
        if 'form.' in property_name or '.age' in property_name:
            if '.age' in property_name:
                values = (self.snapshot['time'] -
                          self.prop(property_name.replace('.age', '.form.time'), indices))
            elif 'time' in property_name and 'lookback' in property_name:
                values = (self.snapshot['time'] -
                          self.prop(property_name.replace('.lookback', ''), indices))

            return values

        # should not get this far without a return
        raise KeyError('not sure how to parse property = {}'.format(property_name))

    def get_indices(
        self, lowres_mass_frac_max=LOWRES_MASS_FRAC_MAX, bound_mass_frac_min=BOUND_MASS_FRAC_MIN,
        star_particle_number_min=10, star_mass_limits=[1, None], star_density_limits=[300, Inf],
        star_mass_fraction_limits=None, dark_star_offset_max=None,
        host_distance_limits=None, halo_kind='', hal_indices=None):
        '''
        Get indices of 'robust' halos/galaxies that satisfy input selection limits.
        This removes the most common cases of numerical artifacts.
        If input halo_kind = 'halo' or 'galaxy', will use default selection limits
        (regardless of other inputs).

        Parameters
        ----------
        lowres_mass_frac_max : float : maximum contamination mass fraction from low-res DM
        bound_mass_frac_min : float : minimum mass.bound/mass
        star_particle_number_min : int : minimum number of star particles
        star_mass_limits : list : min and max limits for stellar mass [M_sun]
        star_density_limits : list :
            min and max limits for average stellar density within R_50 [M_sun / kpc^3]
        star_mass_fraction_limits : list : min and max limits for star.mass/mass.bound
        dark_star_offset_max : float :
            max offset between position and velocity of stars and halo (dark),
            in units of R_50 and V_50
        host_distance_limits : list : min and max limits for distance to host [kpc physical]
        halo_kind : str : shortcut to select object type:
            'halo', 'galaxy' and/or 'satellite', 'isolated'
        hal_indices : array : prior halo indices to impose

        Returns
        -------
        hindices : array : indices of 'robust' halos/galaxies
        '''
        satellite_distance_limits = [5, 350]

        # default parameters for given kind
        if 'halo' in halo_kind:
            star_particle_number_min = 0
            star_mass_limits = None
            star_density_limits = None
            star_mass_fraction_limits = None
        elif 'galaxy' in halo_kind:
            star_particle_number_min = 10
            star_mass_limits = [1, None]
            star_density_limits = [300, Inf]

        if 'satellite' in halo_kind:
            host_distance_limits = satellite_distance_limits
        elif 'isolated' in halo_kind:
            host_distance_limits = [satellite_distance_limits[1], Inf]

        hindices = hal_indices
        if hindices is None or not len(hindices):
            hindices = ut.array.get_arange(self['mass'])

        # properties common to all halos
        if lowres_mass_frac_max > 0:
            hindices = ut.array.get_indices(
                self.prop('lowres.mass.frac'), [0, lowres_mass_frac_max], hindices)

        if bound_mass_frac_min > 0:
            hindices = ut.array.get_indices(
                self.prop('mass.bound/mass'), [bound_mass_frac_min, Inf], hindices)

        # require that halo exists in merger trees
        #if 'tree.index' in hal:
        #    hindices = ut.array.get_indices(hal['tree.index'], [0, Inf], hindices)

        # properties for galaxies
        if 'star.mass' in self and np.nanmax(self['star.mass']) > 0:
            if star_particle_number_min > 0:
                hindices = ut.array.get_indices(
                    self['star.number'], [star_particle_number_min, Inf], hindices)

            if star_mass_limits is not None and len(star_mass_limits):
                hindices = ut.array.get_indices(self['star.mass'], star_mass_limits, hindices)

            if star_density_limits is not None and len(star_density_limits):
                hindices = ut.array.get_indices(
                    self.prop('star.density.50'), star_density_limits, hindices)

            if star_mass_fraction_limits is not None and len(star_mass_fraction_limits):
                hindices = ut.array.get_indices(
                    self.prop('star.mass/mass.bound'), star_mass_fraction_limits, hindices)

            if dark_star_offset_max is not None and dark_star_offset_max > 0:
                position_offsets = ut.coordinate.get_distances(
                    self['position'][hindices], self['star.position'][hindices],
                    self.info['box.length'], self.snapshot['scalefactor'], total_distance=True)
                hindices = hindices[
                    position_offsets < dark_star_offset_max * self['star.radius.50'][hindices]]
                velocity_offsets = ut.coordinate.get_velocity_differences(
                    self['velocity'][hindices], self['star.velocity'][hindices],
                    self['position'][hindices], self['star.position'][hindices],
                    self.info['box.length'], self.snapshot['scalefactor'],
                    self.snapshot['time.hubble'], total_velocity=True)
                hindices = hindices[
                    velocity_offsets < dark_star_offset_max * self['star.vel.std.50'][hindices]]

        # properties for satellites of primary host
        if host_distance_limits is not None and len(host_distance_limits):
            hindices = ut.array.get_indices(
                self.prop('host.distance.total'), host_distance_limits, hindices)

        return hindices


class IOClass(ut.io.SayClass):
    '''
    Read or write halo/galaxy files from Rockstar and/or ConsistentTrees.
    '''

    def __init__(
        self, catalog_directory=HALO_CATALOG_DIRECTORY,
        catalog_hdf5_directory=HALO_CATALOG_HDF5_DIRECTORY,
        lowres_mass_frac_max=LOWRES_MASS_FRAC_MAX):
        '''
        Parameters
        ----------
        catalog_directory : str :
            directory (within rockstar base directory) where 'raw' text files are
        catalog_hdf5_directory : str :
            directory (within rockstar base directory) where 'processed' HDF5 files are
        lowres_mass_frac_max : float :
            maximum contamination from low-resolution DM to consider a halo to be the primary host
        '''
        # set directories
        self.catalog_directory = ut.io.get_path(catalog_directory)
        self.catalog_hlist_directory = self.catalog_directory + 'hlists/'
        self.catalog_tree_directory = self.catalog_directory + 'trees/'
        self.catalog_hdf5_directory = ut.io.get_path(catalog_hdf5_directory)

        # maximum contamination from low-resolution DM to consider a halo to be the primary host
        self.lowres_mass_frac_max = lowres_mass_frac_max

        # set default names for ids and indices
        self.catalog_id_name = 'id'
        self.tree_id_name = 'tid'
        self.prop_name_default = 'mass'  # default property for iterating

        # data types to store halo properties
        self.int_type = np.int32
        self.float_type = np.float32

        self.Snapshot = None

        # halo properties to ignore when reading in
        self.ignore_properties = [
            'descendant.' + self.catalog_id_name,
            'particle.number',
            'momentum.ang.x', 'momentum.ang.y', 'momentum.ang.z',
            'axis.x', 'axis.y', 'axis.z',
            'axis.b/a.500c', 'axis.c/a.500c', 'axis.x.500c', 'axis.y.500c', 'axis.z.500c',
            'kinetic/potential', 'mass.pe.behroozi', 'mass.pe.diemer', 'type',
            'star.mass.rockstar', 'gas.mass.rockstar', 'bh.mass.rockstar',
            'mass.hires',
            'core.number', 'i.dx', 'i.so', 'i.ph', 'particle.child.number', 'max.metric',
            'descendant.central.local.' + self.tree_id_name, 'breadth.index',
            'sam.mass.vir', 'snapshot.index', 'tidal.force', 'tidal.' + self.tree_id_name,
            'accrete.rate.2tdyn', 'accrete.rate.mass.peak', 'accrete.rate.vel.circ.max',
            'accrete.rate.vel.circ.max.tyn', 'mass.peak.vel.circ.max',
        ]

    def read_catalogs(
        self, snapshot_value_kind='redshift', snapshot_values=0,
        simulation_directory='.', rockstar_directory=ROCKSTAR_DIRECTORY, file_kind='hdf5',
        assign_species=True, assign_host=True, host_number=1,
        all_snapshot_list=True, simulation_name=''):
        '''
        Read catalog of halos at snapshot[s] from Rockstar and/or ConsistentTrees.
        Return as dictionary class.

        Parameters
        ----------
        snapshot_value_kind : str : snapshot value kind: 'index', 'redshift', 'scalefactor'
        snapshot_values : int or float or list thereof :
            index[s] or redshifts[s] or scale-factor[s] of snapshot file[s]
            if 'all' or None, read all snapshots with halo catalogs
        simulation_directory : str : directory of simulation
        rockstar_directory : str : sub-directory (within simulation_directory) of halo files
        file_kind : str : kind of catalog file to read: 'out', 'ascii', 'hlist', 'hdf5'
        assign_species : bool : whether to read and assign baryonic particle properties
        assign_host : bool : whether to assign primary host[s] and relative coordinates
        host_number : int : number of hosts to assign and compute coordinates relative to
        all_snapshot_list : bool :
            if reading multiple snapshots, whether to create a list of halo catalogs of length
            equal to all snapshots in simulation (so halo catalog index = snapsht index)
        simulation_name : str : name of simulation to store for future identification

        Returns
        -------
        hals : dictionary class or list thereof : catalog[s] of halos at snapshot[s]
        '''
        # parse input properties
        assert file_kind in ['out', 'ascii', 'hlist', 'hdf5', 'star', 'gas', 'dark']
        simulation_directory = ut.io.get_path(simulation_directory)
        rockstar_directory = ut.io.get_path(rockstar_directory)

        self.Snapshot = ut.simulation.read_snapshot_times(simulation_directory)

        hals = [[] for _ in self.Snapshot['index']]  # list of halo catalogs across all snapshots

        if snapshot_values is 'all' or snapshot_values is None:
            # read all snapshots
            snapshot_indices = self.Snapshot['index']
        else:
            # get snapshot index[s] corresponding to input snapshot values
            snapshot_indices = self.Snapshot.parse_snapshot_values(
                snapshot_value_kind, snapshot_values)

        # get names of all halo files to read
        path_file_names, file_values = self._get_catalog_file_names_values(
            simulation_directory + rockstar_directory, snapshot_indices, file_kind)

        if not len(path_file_names):
            raise OSError('could not find any halo catalog files of type {} in:  {}'.format(
                file_kind, simulation_directory + rockstar_directory.strip('./')))

        # get snapshot indices corresponding to existing halo files
        if 'hlist' in file_kind:
            file_snapshot_indices = self.Snapshot.parse_snapshot_values(
                'scalefactor', file_values, verbose=False)
        else:
            file_snapshot_indices = file_values

        if assign_host:
            # if 'elvis' is in simulation directory name, force 2 hosts
            host_number = ut.catalog.get_host_number_from_directory(
                host_number, simulation_directory, os)

        # initialize
        Cosmology = None

        # read halos at all input snapshots
        for path_file_name, snapshot_index in zip(path_file_names, file_snapshot_indices):
            # read halos
            if 'hdf5' in path_file_name:
                hal, header = self._io_catalog_hdf5(
                    simulation_directory + rockstar_directory, snapshot_index)
            elif 'out' in path_file_name or 'ascii' in path_file_name or 'hlist' in path_file_name:
                hal, header = self._read_catalog_text(path_file_name)

            if len(hal):
                # assign cosmological parameters via cosmology class
                if Cosmology is None:
                    Cosmology = self._get_cosmology(simulation_directory, header)
                hal.Cosmology = Cosmology

                # assign information on all snapshots
                hal.Snapshot = self.Snapshot

                self._assign_simulation_information(
                    hal, header, snapshot_index, file_kind, simulation_directory, simulation_name)

                if assign_species and 'hdf5' in file_kind:
                    # try assigning particle species properties, if file exists
                    species_name = 'star'
                    Particle.io_species_hdf5(
                        species_name, hal, None, simulation_directory + rockstar_directory)
                    if species_name + '.mass' in hal:
                        # ensure baryonic flag
                        hal.info['baryonic'] = True

                if assign_host:
                    # assign primary host[s]
                    self._assign_hosts_to_catalog(hal, 'halo', host_number)
                    hal.info['host.number'] = host_number

            # if read single snapshot, return as dictionary instead of list
            if len(file_snapshot_indices) == 1:
                hals = hal
            else:
                hals[snapshot_index] = hal
                if snapshot_index != file_snapshot_indices[-1]:
                    print()

        if len(file_snapshot_indices) > 1 and not all_snapshot_list:
            hals = [hal for hal in hals if len(hal)]

        return hals

    def read_catalogs_simulations(
        self, snapshot_value_kind='redshift', snapshot_value=0,
        simulation_directories=[], rockstar_directory=ROCKSTAR_DIRECTORY, file_kind='hdf5',
        assign_species=True, assign_host=True, host_number=1, all_snapshot_list=True):
        '''
        Read catalog of halos at single snapshot across various simulations.
        Return as list of dictionary classes.

        Parameters
        ----------
        snapshot_value_kind : str : snapshot value kind: 'index', 'redshift', 'scalefactor'
        snapshot_value : int or float or list thereof :
            index[s] or redshifts[s] or scale-factor[s] of snapshot file[s]
            if 'all' or None, read all snapshots with halo catalogs
        simulation_directories : list of strings : directories of simulations
        rockstar_directory : str : sub-directory (within simulation_directory) of halo files
        file_kind : str : kind of catalog file to read
            options: 'out', 'ascii', 'hlist', 'hdf5', 'star', 'gas', 'dark'
        assign_species : bool : whether to read and assign baryonic particle properties
        assign_host : bool : whether to assign primary host[s] and relative coordinates
        host_number : int : number of hosts to assign and compute coordinates relative to
        all_snapshot_list : bool :
            if reading multiple snapshots, whether to create a list of halo catalogs of length
            equal to all snapshots in simulation (so halo catalog index = snapsht index)

        Returns
        -------
        hals : list of dictionary classes : catalogs of halos across simulations
        '''
        # parse list of directories
        if np.ndim(simulation_directories) == 0:
            raise ValueError('input simulation_directories = {} but need to input list'.format(
                             simulation_directories))
        elif np.ndim(simulation_directories) == 1:
            # assign null names
            simulation_directories = list(
                zip(simulation_directories, ['' for _ in simulation_directories]))
        elif np.ndim(simulation_directories) == 2:
            pass
        elif np.ndim(simulation_directories) >= 3:
            raise ValueError('not sure how to parse simulation_directories = {}'.format(
                             simulation_directories))

        hals = []
        directories_read = []
        for simulation_directory, simulation_name in simulation_directories:
            try:
                hal = self.read_catalogs(
                    snapshot_value_kind, snapshot_value, simulation_directory, rockstar_directory,
                    file_kind, assign_species, assign_host, host_number, all_snapshot_list,
                    simulation_name)

                hals.append(hal)
                directories_read.append(simulation_directory)

            except Exception:
                self.say('! could not read halo catalog at {} = {} in {}'.format(
                         snapshot_value_kind, snapshot_value, simulation_directory))

        if not len(hals):
            self.say('! could not read any halo catalogs at {} = {}'.format(
                     snapshot_value_kind, snapshot_value))
            return

        return hals

    def _read_catalog_text(self, path_file_name):
        '''
        Read catalog of halos at snapshot from Rockstar text file[s] (halos_*.ascii or out_*.list)
        or from ConsistentTrees halo history text file (hlist*.list).

        Parameters
        ----------
        path_file_name : str : path + file name of halo file - if multiple blocks, input 0th one

        Returns
        -------
        hal : class : catalog of halos at snapshot
        header : dictionary : header information
        '''
        # store as dictionary class
        hal = HaloDictionaryClass()
        header = {}

        ## read header to get cosmology ----------
        with open(path_file_name, 'r') as file_in:
            if 'ascii' in path_file_name or 'out' in path_file_name:
                for line in file_in:
                    if 'a = ' in line:
                        index = line.rfind('a = ')
                        header['scalefactor'] = float(line[index + 4: index + 12])
                    if 'h = ' in line:
                        index = line.rfind('h = ')
                        header['hubble'] = float(line[index + 4: index + 12])
                    if 'Om = ' in line:
                        index = line.rfind('Om = ')
                        header['omega_matter'] = float(line[index + 5: index + 13])
                    if 'Ol = ' in line:
                        index = line.rfind('Ol = ')
                        header['omega_lambda'] = float(line[index + 5: index + 13])
                    if 'Box size: ' in line:
                        index = line.rfind('Box size: ')
                        header['box.length/h'] = float(line[index + 10: index + 19])
                        # convert to [kpc/h comoving]
                        header['box.length/h'] *= ut.constant.kilo_per_mega
                    if 'Particle mass: ' in line:
                        index = line.rfind('Particle mass: ')
                        header['dark.particle.mass'] = float(line[index + 15: index + 26])

                header['dark.particle.mass'] /= header['hubble']  # convert to [M_sun]

            elif 'hlist' in path_file_name or 'tree' in path_file_name:
                for line in file_in:
                    if 'h0 = ' in line:
                        index = line.rfind('h0 = ')
                        header['hubble'] = float(line[index + 5: index + 13])
                    if 'Omega_M = ' in line:
                        index = line.rfind('Omega_M = ')
                        header['omega_matter'] = float(line[index + 10: index + 18])
                    if 'Omega_L = ' in line:
                        index = line.rfind('Omega_L = ')
                        header['omega_lambda'] = float(line[index + 10: index + 18])
                    if 'box size = ' in line:
                        index = line.rfind('box size = ')
                        header['box.length/h'] = float(line[index + 11: index + 20])
                        # convert to [kpc/h comoving]
                        header['box.length/h'] *= ut.constant.kilo_per_mega

                header['dark.particle.mass'] = np.nan

        # initialize rest of cosmological parameters for later
        header['omega_baryon'] = None
        header['sigma_8'] = None
        header['n_s'] = None

        it = self.int_type
        ft = self.float_type

        if 'ascii' in path_file_name:
            # get all file blocks
            file_name_base = path_file_name.replace('.0.', '.*.')
            path_file_names = ut.io.get_file_names(file_name_base)
            # loop over multiple blocks per snapshot
            for file_block_index, path_file_name in enumerate(path_file_names):
                hal_in = np.loadtxt(
                    path_file_name, comments='#',
                    dtype=[
                        (self.catalog_id_name, it),
                        ('particle.number', it),  # [ignore]
                        ('mass', ft),
                        ('mass.bound', ft),
                        ('radius', ft),
                        ('vel.circ.max', ft),
                        ('vel.circ.max.radius', ft),
                        ('vel.std', ft),
                        ('position.x', ft), ('position.y', ft), ('position.z', ft),
                        ('velocity.x', ft), ('velocity.y', ft), ('velocity.z', ft),
                        ('momentum.ang.x', ft), ('momentum.ang.y', ft), ('momentum.ang.z', ft),
                        ('energy', ft),
                        ('spin.peebles', ft),
                        ('position.err', ft), ('velocity.err', ft),
                        ('bulk.velocity.x', ft), ('bulk.velocity.y', ft),
                        ('bulk.velocity.z', ft),
                        ('bulk.velocity.err', ft),
                        ('core.number', it),  # [ignore]
                        ('mass.vir', ft), ('mass.200c', ft), ('mass.500c', ft), ('mass.180m', ft),
                        ('position.offset', ft), ('velocity.offset', ft),
                        ('spin.bullock', ft),
                        ('axis.b/a', ft), ('axis.c/a', ft),
                        ('axis.x', ft), ('axis.y', ft), ('axis.z', ft),  # [ignore]
                        ('axis.b/a.500c', ft), ('axis.c/a.500c', ft),  # [ignore]
                        ('axis.x.500c', ft), ('axis.y.500c', ft), ('axis.z.500c', ft),  # [ignore]
                        ('scale.radius', ft),
                        ('scale.radius.klypin', ft),
                        ('kinetic/potential', ft),  # [ignore]
                        ('mass.pe.behroozi', ft),  # [ignore]
                        ('mass.pe.diemer', ft),  # [ignore]
                        ('type', it),  # [ignore]
                        ('star.mass.rockstar', ft),  # [ignore for now]
                        ('gas.mass.rockstar', ft),  # [ignore for now]
                        ('bh.mass.rockstar', ft),  # [ignore for now]
                        ('i.dx', it),  # [ignore]
                        ('i.so', it),  # [ignore]
                        ('i.ph', it),  # [ignore]
                        ('particle.child.number', it),  # [ignore]
                        ('max.metric', ft),  # [ignore]
                        ('mass.hires', ft),  # [ignore]
                        ('mass.lowres', ft),
                    ]
                )

                for prop_name in hal_in.dtype.names:
                    if prop_name not in self.ignore_properties:
                        if file_block_index == 0:
                            hal[prop_name] = hal_in[prop_name]
                        else:
                            hal[prop_name] = np.concatenate((hal[prop_name], hal_in[prop_name]))

            self.say('* read {} halos from:'.format(hal[self.prop_name_default].size))
            for path_file_name in path_file_names:
                self.say(path_file_name.strip('./'))
            self.say('')

        elif 'out' in path_file_name:
            hal_in = np.loadtxt(
                path_file_name, comments='#', dtype=[
                    (self.catalog_id_name, it),  # catalog ID at snapshot
                    ('descendant.' + self.catalog_id_name, it),  # catalog ID of descendant [ignore]
                    ('mass.bound', ft),  # bound mass
                    ('vel.circ.max', ft),  # maximum of circular velocity
                    ('vel.std', ft),  # velocity dispersion
                    ('radius', ft),  # halo radius
                    ('scale.radius', ft),  # NFW scale radius
                    ('particle.number', it),  # number of particles in halo [ignore]
                    ('position.x', ft), ('position.y', ft), ('position.z', ft),  # center position
                    ('velocity.x', ft), ('velocity.y', ft), ('velocity.z', ft),  # center velocity
                    ('momentum.ang.x', ft), ('momentum.ang.y', ft), ('momentum.ang.z', ft),  # [ign]
                    ('spin.peebles', ft),  # dimensionless spin parameter
                    ('scale.radius.klypin', ft),  # NFW scale radius from radius(vel.circ.max)
                    ('mass', ft),  # total mass within radius (including unbound)
                    ('mass.vir', ft), ('mass.200c', ft), ('mass.500c', ft), ('mass.180m', ft),
                    # offset of density peak from particle average position
                    ('position.offset', ft), ('velocity.offset', ft),
                    ('spin.bullock', ft),  # dimensionless spin from Bullock++ (J/(sqrt(2)*GMVR))
                    # ratio of 2nd & 3rd to 1st largest shape ellipsoid axes (Allgood et al 2006)
                    ('axis.b/a', ft), ('axis.c/a', ft),
                    ('axis.x', ft), ('axis.y', ft), ('axis.z', ft),  # [ignore]
                    ('axis.b/a.500c', ft), ('axis.c/a.500c', ft),  # [ignore]
                    ('axis.x.500c', ft), ('axis.y.500c', ft), ('axis.z.500c', ft),  # [ignore]
                    ('kinetic/potential', ft),  # ratio of kinetic to potential energies [ignore]
                    ('mass.pe.behroozi', ft),  # [ignore]
                    ('mass.pe.diemer', ft),  # [ignore]
                    ('type', ft),  # [ignore]
                    ('star.mass.rockstar', ft),  # [ignore for now]
                    ('gas.mass.rockstar', ft),  # [ignore for now]
                    ('bh.mass.rockstar', ft),  # [ignore for now]
                    ('mass.hires', ft),  # mass in high-res DM particles [ignore]
                    ('mass.lowres', ft),  # mass in low-res DM particles
                ]
            )

            for prop_name in hal_in.dtype.names:
                if prop_name not in self.ignore_properties:
                    hal[prop_name] = hal_in[prop_name]

            self.say('* read {} halos from:  {}\n'.format(
                hal[self.prop_name_default].size, path_file_name.strip('./')))

        elif 'hlist' in path_file_name:
            hal_in = np.loadtxt(
                path_file_name, comments='#',
                dtype=[
                    ## properties copied from merger tree
                    ('scalefactor', ft),  # [convert to snapshot index] of halo
                    (self.tree_id_name, it),  # tree ID (unique across all snapshots)
                    ('descendant.scalefactor', ft),  # [snapshot index] of descendant
                    ('descendant.' + self.tree_id_name, it),  # [tree index] of descendant [ignore]
                    ('progenitor.number', it),  # number of progenitors
                    # [tree index] of local (least mass) central (can be a satellite)
                    ('central.local.' + self.tree_id_name, it),  # [ignore]
                    ('central.' + self.tree_id_name, it),  # [tree index] of most massive central []
                    ('descendant.central.local.' + self.tree_id_name, it),  # [ignore]
                    ('am.phantom', it),  # whether halo is interpolated across snapshots
                    ('sam.mass.vir', ft),  # [ignore]
                    ('mass.bound', ft),  # bound mass
                    ('radius', ft),  # halo radius
                    ('scale.radius', ft),  # NFW scale radius
                    ('vel.std', ft),  # velocity dispersion
                    ('am.progenitor.main', it),  # whether am most massive progenitor of descendant
                    ('major.merger.scalefactor', ft),  # [snapshot index] of last major merger
                    ('vel.circ.max', ft),  # maximum of circular velocity
                    ('position.x', ft), ('position.y', ft), ('position.z', ft),  # center position
                    ('velocity.x', ft), ('velocity.y', ft), ('velocity.z', ft),  # center velocity
                    ('momentum.ang.x', ft), ('momentum.ang.y', ft), ('momentum.ang.z', ft),  # [ign]
                    ('spin.peebles', ft),  # dimensionless spin parameter
                    ('breadth.index', it),  # (same as tree index) [ignore]
                    ('dindex', it),  # depth-first order (index) within tree
                    ('final.' + self.tree_id_name, it),  # [tree index] at final snapshot [ignore]
                    (self.catalog_id_name, it),  # catalog ID at snapshot from rockstar catalog
                    ('snapshot.index', it),  # [ignore]
                    # depth-first index of next co-progenitor
                    ('progenitor.co.dindex', it),
                    # depth-first index of last progenitor
                    ('progenitor.last.dindex', it),
                    # depth-first index of last progenitor on main progenitor branch
                    ('progenitor.main.last.dindex', it),
                    ('tidal.force', ft),  # [ignore]
                    ('tidal.' + self.tree_id_name, it),  # [ignore]
                    ('scale.radius.klypin', ft),  # NFW scale radius from radius(vel.circ.max)
                    ('mass', ft),  # total mass within halo radius (including unbound)
                    ('mass.vir', ft), ('mass.200c', ft), ('mass.500c', ft), ('mass.180m', ft),
                    # offset of density peak from particle average position
                    ('position.offset', ft), ('velocity.offset', ft),
                    ('spin.bullock', ft),  # dimensionless spin from Bullock++ (J/(sqrt(2)*GMVR))
                    # ratio of 2nd & 3rd to 1st largest shape ellipsoid axes (Allgood et al 2006)
                    ('axis.b/a', ft), ('axis.c/a', ft),
                    ('axis.x', ft), ('axis.y', ft), ('axis.z', ft),  # [ignore]
                    ('axis.b/a.500c', ft), ('axis.c/a.500c', ft),  # [ignore]
                    ('axis.x.500c', ft), ('axis.y.500c', ft), ('axis.z.500c', ft),  # [ignore]
                    ('kinetic/potential', ft),  # ratio of kinetic to potential energies [ignore]
                    ('mass.pe.behroozi', ft),  # [ignore]
                    ('mass.pe.diemer', ft),  # [ignore]
                    ('type', it),  # [ignore]
                    ('star.mass.rockstar', ft),  # [ignore for now]
                    ('gas.mass.rockstar', ft),  # [ignore for now]
                    ('bh.mass.rockstar', ft),  # [ignore for now]
                    ('mass.hires', ft),  # mass in high-res DM particles [ignore]
                    ('mass.lowres', ft),  # mass in low-res DM particles
                    ## properties computed from main progenitor history
                    ('infall.mass', ft),  # mass before fell into host halo (becoming a satellite)
                    ('mass.peak', ft),  # peak mass throughout history
                    ('infall.vel.circ.max', ft),  # vel.cirx.max before fall into a host halo
                    ('vel.circ.peak', ft),  # peak vel.cirx.max throughout history
                    ('mass.half.scalefactor', ft),  # [snapshot] when first half current mass
                    ('accrete.rate', ft),  # mass growth rate between snapshots
                    ('accrete.rate.100Myr', ft),  # mass growth rate averaged over 100 Myr
                    ('accrete.rate.tdyn', ft),  # mass growth rate averaged over dynamical time
                    ('accrete.rate.2tdyn', ft),  # mass growth rate averaged over 2 t_dyn [ignore]
                    ('accrete.rate.mass.peak', ft),  # [ignore]
                    ('accrete.rate.vel.circ.max', ft),  # [ignore]
                    ('accrete.rate.vel.circ.max.tyn', ft),  # [ignore]
                    ('mass.peak.scalefactor', ft),  # [snapshot] when reached mass.peak
                    ('infall.scalefactor', ft),  # [snapshot] before fell into host halo
                    ('infall.first.scalefactor', ft),  # [snapshot] before first fell into host halo
                    ('infall.first.mass', ft),  # mass before first fell into host halo
                    ('infall.first.vel.circ.max', ft),  # vel.circ.max before first fell in
                    ('mass.peak.vel.circ.max', ft),  # [ignore]
                ]
            )

            header['scalefactor'] = hal_in['scalefactor'][0]

            for prop_name in hal_in.dtype.names:
                if (prop_name not in self.ignore_properties and prop_name != 'scalefactor' and
                        ('.' + self.tree_id_name) not in prop_name and 'dindex' not in prop_name):
                    hal[prop_name] = hal_in[prop_name]

            self.say('* read {} halos from:  {}\n'.format(
                hal[self.prop_name_default].size, path_file_name.strip('./')))

        del(hal_in)

        # convert properties
        for prop_name in hal:
            # if only 1 halo, make sure is array
            if hal[prop_name].size == 1:
                hal[prop_name] = np.array([hal[prop_name]], hal[prop_name].dtype)
            if 'mass' in prop_name and 'scalefactor' not in prop_name:
                hal[prop_name] *= 1 / header['hubble']  # to [M_sun]
            elif 'radius' in prop_name:
                hal[prop_name] *= header['scalefactor'] / header['hubble']  # to [kpc physical]
            elif 'position' in prop_name:
                hal[prop_name] *= ut.constant.kilo_per_mega / header['hubble']  # to [kpc comoving]
            elif 'momentum.ang' in prop_name:
                hal[prop_name] *= (header['scalefactor'] / header['hubble']) ** 2  # to [kpc phys]
            elif 'energy' in prop_name:
                hal[prop_name] *= header['scalefactor'] / header['hubble']  # to [kpc physical]
            elif 'index' in prop_name and np.min(hal[prop_name]) == -1:
                # ensure null pointer index points safely out of range
                hindices = np.where(hal[prop_name] == -1)[0]
                hal[prop_name][hindices] -= hal[prop_name].size

        # assign derived masses
        hal['mass.200m'] = hal['mass']  # pointer for clarity
        if 'star.mass.rockstar' in hal:
            hal['baryon.mass.rockstar'] = hal['gas.mass.rockstar'] + hal['star.mass.rockstar']
            hal['dark.mass'] = hal['mass'] - hal['baryon.mass.rockstar']

        # convert position and velocity to halo number x dimension number array
        for prop_name in ['position', 'velocity', 'bulk.velocity', 'momentum.ang', 'axis',
                          'axis.500c']:
            if prop_name + '.x' in hal:
                hal[prop_name] = np.transpose(
                    [hal[prop_name + '.x'], hal[prop_name + '.y'], hal[prop_name + '.z']])
                del(hal[prop_name + '.x'], hal[prop_name + '.y'], hal[prop_name + '.z'])

        # convert properties of snapshot scale-factor to snapshot index
        for prop_name in list(hal.keys()):
            if '.scalefactor' in prop_name:
                prop_name_new = prop_name.replace('.scalefactor', '.snapshot')
                hal[prop_name_new] = np.zeros(
                    hal[self.prop_name_default].size, self.int_type) - 601  # init safely
                hindices = ut.array.get_indices(hal[prop_name], [1e-10, 1.00001])
                if hindices.size:
                    hal[prop_name_new][hindices] = self.Snapshot.get_snapshot_indices(
                        'scalefactor', hal[prop_name][hindices])
                del(hal[prop_name])

        # assign conversion between halo id and index
        ut.catalog.assign_id_to_index(hal, self.catalog_id_name, 0)

        return hal, header

    def _io_catalog_hdf5(
        self, rockstar_directory=ROCKSTAR_DIRECTORY, snapshot_index=None, hal=None):
        '''
        Read/write halo catalog at snapshot to/from HDF5 file.
        If reading, return as dictionary class.

        Parameters
        ----------
        rockstar_directory : str : directory (full path) of rockstar halo files
        snapshot_index : int : index of snapshot
        hal : class : catalog of halos at snapshot, if writing

        Returns
        -------
        hal : class : catalog of halos at snapshot
        '''
        # parse inputs
        assert (snapshot_index is not None or hal is not None)
        file_path = ut.io.get_path(rockstar_directory) + self.catalog_hdf5_directory
        if not snapshot_index:
            snapshot_index = hal.snapshot['index']

        file_name = 'halo_{:03d}'.format(snapshot_index)
        path_file_name = file_path + file_name

        if hal is not None:
            # write to file
            file_path = ut.io.get_path(file_path, create_path=True)

            properties_add = []
            for prop_name in hal.info:
                if not isinstance(hal.info[prop_name], str):
                    hal['info:' + prop_name] = np.array(hal.info[prop_name])
                    properties_add.append('info:' + prop_name)

            for prop_name in hal.snapshot:
                hal['snapshot:' + prop_name] = np.array(hal.snapshot[prop_name])
                properties_add.append('snapshot:' + prop_name)

            for prop_name in hal.Cosmology:
                hal['cosmology:' + prop_name] = np.array(hal.Cosmology[prop_name])
                properties_add.append('cosmology:' + prop_name)

            ut.io.file_hdf5(path_file_name, hal)

            for prop_name in properties_add:
                del(hal[prop_name])

        else:
            # read from file

            # store as dictionary class
            hal = HaloDictionaryClass()
            header = {}

            try:
                # try to read from file
                hal_in = ut.io.file_hdf5(path_file_name, verbose=False)

                for prop_name in hal_in:
                    if 'info:' in prop_name:
                        hal_prop_name = prop_name.split(':')[-1]
                        header[hal_prop_name] = float(hal_in[prop_name])
                    elif 'snapshot:' in prop_name:
                        hal_prop_name = prop_name.split(':')[-1]
                        if hal_prop_name == 'index':
                            header[hal_prop_name] = int(hal_in[prop_name])
                        else:
                            header[hal_prop_name] = float(hal_in[prop_name])
                    elif 'cosmology:' in prop_name:
                        hal_prop_name = prop_name.split(':')[-1]
                        header[hal_prop_name] = float(hal_in[prop_name])
                    else:
                        hal[prop_name] = hal_in[prop_name]

                self.say('* read {} halos from:  {}.hdf5'.format(
                    hal[self.prop_name_default].size, path_file_name.strip('./')))

            except OSError:
                raise OSError('! cannot read halo catalog at snapshot index = {}'.format(
                              snapshot_index))

            return hal, header

    def _get_catalog_file_names_values(
        self, rockstar_directory=ROCKSTAR_DIRECTORY, snapshot_indices=None, file_kind='out'):
        '''
        Get name[s] and snapshot value[s] (index or scale-factor) of Rockstar halo catalog file[s].

        Parameters
        ----------
        rockstar_directory : str : directory (full path) of rockstar halo files
        snapshot_indices : int or array thereof : index of snapshot
        file_kind : str : kind of file: 'out', 'ascii', 'hlist', 'hdf5', 'star', 'gas', 'dark'

        Returns
        -------
        path_file_names : list : path + name[s] of halo file[s]
        file_values : list : snapshot value[s] (index or scale-factor) of halo file[s]
        '''
        assert file_kind in ['out', 'ascii', 'hlist', 'hdf5', 'star', 'gas', 'dark']

        snapshot_values = snapshot_indices

        if 'out' in file_kind:
            file_name_base = 'out_*.list'
            file_number_type = int
            directory = ut.io.get_path(rockstar_directory) + self.catalog_directory
        elif 'ascii' in file_kind:
            file_name_base = 'halos_*.ascii'
            file_number_type = float
            directory = ut.io.get_path(rockstar_directory) + self.catalog_directory
        elif 'hlist' in file_kind:
            file_name_base = 'hlist_*.list'
            file_number_type = float
            directory = ut.io.get_path(rockstar_directory) + self.catalog_hlist_directory
            snapshot_values = self.Snapshot['scalefactor'][snapshot_indices]
        elif 'hdf5' in file_kind:
            if 'star' in file_kind:
                file_name_base = 'star_*.hdf5'
            elif 'gas' in file_kind:
                file_name_base = 'gas_*.hdf5'
            elif 'dark' in file_kind:
                file_name_base = 'dark_*.hdf5'
            else:
                file_name_base = 'halo_*.hdf5'
            file_number_type = int
            directory = ut.io.get_path(rockstar_directory) + self.catalog_hdf5_directory
        elif 'star' in file_kind:
            file_name_base = 'star_*.hdf5'
            file_number_type = int
            directory = ut.io.get_path(rockstar_directory) + self.catalog_hdf5_directory
        elif 'gas' in file_kind:
            file_name_base = 'gas_*.hdf5'
            file_number_type = int
            directory = ut.io.get_path(rockstar_directory) + self.catalog_hdf5_directory
        elif 'dark' in file_kind:
            file_name_base = 'dark_*.hdf5'
            file_number_type = int
            directory = ut.io.get_path(rockstar_directory) + self.catalog_hdf5_directory

        # get names and indices/scale-factors of all files matching name base
        # this can include multiple snapshots and/or multiple blocks per snapshot
        path_file_names_all, file_values_all = ut.io.get_file_names(
            directory + file_name_base, file_number_type, verbose=False)

        if snapshot_values is not None:
            path_file_names = []
            file_values = []
            for file_i, file_value in enumerate(file_values_all):
                if 'hlist' in file_kind:
                    # hlist files are labeled via scale-factor
                    if np.min(np.abs(file_value - snapshot_values)) < 1e-5:
                        path_file_names.append(path_file_names_all[file_i])
                        file_values.append(file_value)
                else:
                    # all other files are labeled via snapshot index
                    # keep only block 0 if multiple blocks per snapshot
                    if np.max(file_value == snapshot_values):
                        path_file_names.append(path_file_names_all[file_i])
                        file_values.append(file_value)

            if np.isscalar(snapshot_values):
                snapshot_values = [snapshot_values]
            if len(snapshot_values) != len(path_file_names):
                self.say('! input {} snapshot indices but found only {} halo catalog files'.format(
                    len(snapshot_values), len(path_file_names)))
        else:
            # return all that found
            path_file_names = path_file_names_all
            file_values = file_values_all

        if not len(path_file_names):
            self.say('! cannot find halo {} files in:  {}'.format(file_kind, directory))

        return path_file_names, file_values

    def _get_cosmology(self, simulation_directory='.', cosmo={}):
        '''
        Get Cosmology class of cosmological parameters.
        If all cosmological parameters in input cosmo dictionary, use them.
        Else, try to read cosmological parameters from MUSIC initial condition config file.
        Else, assume AGORA cosmology as default.

        Parameters
        ----------
        simulation_directory : str : directory of simulation
        cosmo : dict : dictionary that includes cosmological parameters

        Returns
        -------
        Cosmology : cosmology class, which also stores cosmological parameters
        '''

        def check_value(line, value_test=None):
            frac_dif_max = 0.01
            value = float(line.split('=')[-1].strip())
            if 'h0' in line:
                value /= 100
            if value_test is not None:
                frac_dif = np.abs((value - value_test) / value)
                if frac_dif > frac_dif_max:
                    print('! read {}, but previously assigned = {}'.format(line, value_test))
            return value

        if (cosmo and cosmo['omega_lambda'] and cosmo['omega_matter'] and cosmo['omega_baryon'] and
                cosmo['hubble'] and cosmo['sigma_8'] and cosmo['n_s']):
                pass
        else:
            try:
                # try to find MUSIC file, assuming named *.conf
                file_name_find = ut.io.get_path(simulation_directory) + '*/*.conf'
                path_file_name = ut.io.get_file_names(file_name_find)[0]
                self.say('* read cosmological parameters from:  {}\n'.format(
                    path_file_name.strip('./')))
                # read cosmological parameters
                with open(path_file_name, 'r') as file_in:
                    for line in file_in:
                        line = line.lower().strip().strip('\n')  # ensure lowercase for safety
                        if 'omega_l' in line:
                            cosmo['omega_lambda'] = check_value(line, cosmo['omega_lambda'])
                        elif 'omega_m' in line:
                            cosmo['omega_matter'] = check_value(line, cosmo['omega_matter'])
                        elif 'omega_b' in line:
                            cosmo['omega_baryon'] = check_value(line, cosmo['omega_baryon'])
                        elif 'h0' in line:
                            cosmo['hubble'] = check_value(line, cosmo['hubble'])
                        elif 'sigma_8' in line:
                            cosmo['sigma_8'] = check_value(line, cosmo['sigma_8'])
                        elif 'nspec' in line:
                            cosmo['n_s'] = check_value(line, cosmo['n_s'])
            except (OSError, IndexError):
                self.say('! cannot find MUSIC config file:  {}'.format(file_name_find.strip('./')))
                self.say('! assuming missing cosmological parameters from the AGORA box')
                if cosmo['omega_baryon'] is None:
                    cosmo['omega_baryon'] = 0.0455
                    self.say('assuming omega_baryon = {}'.format(cosmo['omega_baryon']))
                if cosmo['sigma_8'] is None:
                    cosmo['sigma_8'] = 0.807
                    self.say('assuming sigma_8 = {}'.format(cosmo['sigma_8']))
                if cosmo['n_s'] is None:
                    cosmo['n_s'] = 0.961
                    self.say('assuming n_s = {}'.format(cosmo['n_s']))
                self.say('')

        Cosmology = ut.cosmology.CosmologyClass(
            cosmo['omega_lambda'], cosmo['omega_matter'], cosmo['omega_baryon'], cosmo['hubble'],
            cosmo['sigma_8'], cosmo['n_s'])

        return Cosmology

    def _assign_simulation_information(
        self, hal, header, snapshot_index, file_kind, simulation_directory='', simulation_name=''):
        '''
        Add information about snapshot to halo catalog.
        Append as dictionaries to halo dictionary class.

        Parameters
        ----------
        hal : dictionary class: catalog of halos at snapshot
        header : dictionary : header information from halo text file
        snapshot_index : int : index of snapshot
        file_kind : str : kind of catalog file to read:
            'out', 'ascii', 'hlist', 'hdf5', 'star', 'gas', 'dark'
        simulation_directory : str : directory of simulation
        simulation_name : str : name of simulation to store for future identification
        '''
        # assign information on current snapshot
        redshift = 1 / header['scalefactor'] - 1
        hal.snapshot = {
            'index': snapshot_index,
            'scalefactor': header['scalefactor'],
            'redshift': redshift,
            'time': hal.Cosmology.get_time(header['scalefactor'], 'scalefactor'),
            'time.lookback': (hal.Cosmology.get_time(0) -
                              hal.Cosmology.get_time(header['scalefactor'], 'scalefactor')),
            'time.hubble': ut.constant.Gyr_per_sec / hal.Cosmology.get_hubble_parameter(redshift),
        }

        # assign general information about simulation
        if not simulation_name and simulation_directory != './':
            simulation_name = simulation_directory.split('/')[-2]
            simulation_name = simulation_name.replace('_', ' ')
            simulation_name = simulation_name.replace('res', 'r')

        hal.info = {
            'dark.particle.mass': header['dark.particle.mass'],
            'box.length/h': header['box.length/h'],
            'box.length': header['box.length/h'] / header['hubble'],
            'catalog.kind': 'halo.catalog',
            'file.kind': file_kind,
            'baryonic': ut.catalog.is_baryonic_from_directory(simulation_directory, os),
            'host.number': 0,
            'simulation.name': simulation_name,
        }
        if hal.info['baryonic']:
            hal.info['gas.particle.mass'] = (
                header['dark.particle.mass'] *
                hal.Cosmology['omega_baryon'] / hal.Cosmology['omega_dm'])

    def _assign_hosts_to_catalog(self, hal, host_kind='halo', host_number=1):
        '''
        Assign primary host halo/galaxy[s] and coordinates relative to it/them.

        Parameters
        ----------
        hal : dictionary class : catalog of halos at snapshot
        host_kind : str : property to determine primary host: 'halo', 'star'
        host_number : int : number of hosts to assign
        '''
        if host_number < 1:
            host_number = 1

        for host_rank in range(host_number):
            host_name = ut.catalog.get_host_name(host_rank) + 'index'
            host_index_name = host_name + 'index'

            if host_index_name not in hal:
                self._assign_host_to_catalog(hal, host_kind, host_rank)

        if host_number > 1:
            # multiple hosts - assign nearest one to each halo
            self.say('* assigning nearest host')
            host_distancess = np.zeros(
                (hal['host.index'].size, host_number), dtype=hal['host.distance'].dtype)
            for host_rank in range(host_number):
                host_name = ut.catalog.get_host_name(host_rank)
                host_distancess[:, host_rank] = hal.prop(host_name + 'distance.total')

            host_nearest_indices = np.argmin(host_distancess, 1)

            # initialize all halos to the primary host
            for prop_name in list(hal.keys()):
                if 'host.' in prop_name and 'near.' not in prop_name:
                    prop_name_near = prop_name.replace('host.', 'host.near.')
                    hal[prop_name_near] = np.array(hal[prop_name])

            # assign other hosts
            for host_rank in range(1, host_number):
                hindices = np.where(host_nearest_indices == host_rank)[0]
                if hindices.size:
                    host_name = ut.catalog.get_host_name(host_rank)
                    self.say('{} halos are closest to {}'.format(
                             hindices.size, host_name.replace('.', '')))
                    for prop_name in hal:
                        if host_name in prop_name and 'near.' not in prop_name:
                            prop_name_near = prop_name.replace(host_name, 'host.near.')
                            hal[prop_name_near][hindices] = hal[prop_name][hindices]
        print()

    def _assign_host_to_catalog(self, hal, host_kind='halo', host_rank=0):
        '''
        Assign primary (or secondary etc) host halo/galaxy and position + velocity wrt it.
        Define host as being host_rank highest in host_prop_name.
        Require low contamination from low-resolution dark matter.

        If host_kind is 'halo', define primary host as most massive halo in catalog,
        and use coordinates in halo catalog.
        If host_kind is 'star', define primary host as highest stellar mass galaxy in catalog,
        and use coordinate defined via stars.

        Parameters
        ----------
        hal : dictionary class : catalog of halos at snapshot
        host_kind : str : property to determine primary host: 'halo', 'star'
        host_rank : int : which host (sorted by host_prop_name) to assign
        '''
        assert host_kind in ['halo', 'star']

        host_name = ut.catalog.get_host_name(host_rank)

        if host_kind == 'halo':
            host_prop_name = 'mass'  # property to use to determine primary host
            spec_prefix = ''
        elif host_kind == 'star':
            host_prop_name = 'star.mass'  # property to use to determine primary host
            spec_prefix = 'star.'
            host_name = spec_prefix + host_name

        self.say('* assigning primary {} and coordinates wrt it to halo catalog'.format(
                 host_name.replace('.', '')))

        # assign primary host coordinates only to halos with well defined mass
        hindices = ut.array.get_indices(hal.prop(host_prop_name), [1e-10, Inf])
        hindices_pure = ut.array.get_indices(
            hal.prop('lowres.mass.frac'), [0, self.lowres_mass_frac_max], hindices)

        host_index = hindices_pure[np.argsort(hal[host_prop_name][hindices_pure])][-host_rank - 1]

        hal[host_name + 'index'] = (
            np.zeros(hal[host_prop_name].size, dtype=self.int_type) + host_index)

        # distance to primary host
        hal[host_name + 'distance'] = np.zeros(
            hal[spec_prefix + 'position'].shape, hal[spec_prefix + 'position'].dtype) * np.nan
        hal[host_name + 'distance'][hindices] = ut.coordinate.get_distances(
            hal[spec_prefix + 'position'][hindices], hal[spec_prefix + 'position'][host_index],
            hal.info['box.length'], hal.snapshot['scalefactor'])  # [kpc physical]

        # velocity wrt primary host
        hal[host_name + 'velocity'] = np.zeros(
            hal[spec_prefix + 'velocity'].shape, hal[spec_prefix + 'velocity'].dtype) * np.nan
        hal[host_name + 'velocity'][hindices] = ut.coordinate.get_velocity_differences(
            hal[spec_prefix + 'velocity'][hindices], hal[spec_prefix + 'velocity'][host_index],
            hal[spec_prefix + 'position'][hindices], hal[spec_prefix + 'position'][host_index],
            hal.info['box.length'], hal.snapshot['scalefactor'], hal.snapshot['time.hubble'])

        # orbital velocities wrt primary host - use only halos with well defined host distance
        hindices = hindices[np.where(hal.prop(host_name + 'distance.total', hindices) > 0)[0]]

        distances_norm = np.transpose(
            hal[host_name + 'distance'][hindices].transpose() /
            hal.prop(host_name + 'distance.total', hindices))  # need to do this way

        hal[host_name + 'velocity.tan'] = np.zeros(
            hal[host_prop_name].size, hal[spec_prefix + 'velocity'].dtype) * np.nan
        hal[host_name + 'velocity.tan'][hindices] = np.sqrt(np.sum(np.cross(
            hal[host_name + 'velocity'][hindices], distances_norm) ** 2, 1))
        hal[host_name + 'velocity.tan'][host_index] = 0

        hal[host_name + 'velocity.rad'] = np.zeros(
            hal[host_prop_name].size, hal[spec_prefix + 'velocity'].dtype) * np.nan
        hal[host_name + 'velocity.rad'][hindices] = np.sum(
            hal[spec_prefix + 'velocity'][hindices] * distances_norm, 1)
        hal[host_name + 'velocity.rad'][host_index] = 0

    def _transfer_properties_catalog(self, hal_1, hal_2):
        '''
        Transfer/assign properties from hal_2 catalog to hal_1 catalog (at same snapshot).
        Primary use: transfer properties from ConsistentTrees halo history catalog (hlist) to
        Rockstar halo catalog.

        Parameters
        ----------
        hal_1 : dictionary class : catalog of halos at snapshot
        hal_2 : dictionary class : another catalog of same halos at same snapshot
        '''
        # parse input catalogs
        assert hal_1.snapshot['index'] == hal_2.snapshot['index']

        pointer_name = self.catalog_id_name + '.to.index'

        if pointer_name not in hal_1 or not len(hal_1[pointer_name]):
            ut.catalog.assign_id_to_index(hal_1, self.catalog_id_name, 0)

        hal_2_indices = ut.array.get_indices(
            hal_2[self.catalog_id_name], [0, hal_1[self.catalog_id_name].max() + 1])
        hal_1_indices = hal_1[pointer_name][hal_2[self.catalog_id_name][hal_2_indices]]
        masks = (hal_1_indices >= 0)
        hal_1_indices = hal_1_indices[masks]
        hal_2_indices = hal_2_indices[masks]

        # sanity check - compare shared properties
        self.say('\n* shared properties with offsets: min, med, max, N_offset')
        for prop_name in hal_2:
            if prop_name in hal_1 and prop_name != pointer_name:
                prop_difs = hal_1[prop_name][hal_1_indices] - hal_2[prop_name][hal_2_indices]
                if np.abs(np.min(prop_difs)) > 1e-4 and np.abs(np.max(prop_difs)) > 1e-4:
                    self.say('{}: [{}, {}, {}] {}'.format(
                        prop_name, np.min(prop_difs), np.median(prop_difs), np.max(prop_difs),
                        np.sum(np.abs(prop_difs) > 0)))

        self.say('* assigning new properties')
        for prop_name in hal_2:
            if prop_name not in hal_1 and prop_name != pointer_name:
                self.say('{}'.format(prop_name))
                dtype = hal_2[prop_name].dtype
                if dtype == np.float32 or dtype == np.float64:
                    value_null = np.nan
                else:
                    if 'snapshot' in prop_name:
                        value_null = hal_1.Snapshot['index'].size - 1
                    else:
                        value_null = -1
                hal_1[prop_name] = np.zeros(
                    hal_1[self.catalog_id_name].size, hal_2[prop_name].dtype) + value_null
                hal_1[prop_name][hal_1_indices] = hal_2[prop_name][hal_2_indices]

    ## halo merger trees ----------
    def read_tree(
        self, simulation_directory='.', rockstar_directory=ROCKSTAR_DIRECTORY, file_kind='hdf5',
        assign_species=True, species_snapshot_indices=None, assign_host=True, host_number=1,
        simulation_name=''):
        '''
        Read catalog of halo merger trees from ConsistentTrees (tree_*.dat or tree.hdf5).
        Return as dictionary class.

        Parameters
        ----------
        simulation_directory : str : directory of simulation
        rockstar_directory : str : sub-directory (within simulation_directory) of halo files
        file_kind : str : kind of halo tree file to read: 'text', 'hdf5'
        assign_species : bool : whether to read and assign baryonic particle properties
        species_snapshot_indices : array :
            list of snapshot indices at which to assign particle species to tree
            if None, assign at all snapshots with particle species data
        assign_host : bool : whether to assign primary host[s] and relative coordinates
        host_number : int : number of primary hosts to assign
        simulation_name : str : name of simulation to store for future identification

        Returns
        -------
        halt : dictionary class or list thereof : catalog of halo merger trees across all snapshots
        '''
        # parse input properties
        assert file_kind in ['text', 'hdf5']
        simulation_directory = ut.io.get_path(simulation_directory)
        rockstar_directory = ut.io.get_path(rockstar_directory)

        # assign information about all snapshot times
        self.Snapshot = ut.simulation.read_snapshot_times(simulation_directory)

        if file_kind == 'text':
            halt, header = self._read_tree_text(simulation_directory + rockstar_directory)
        elif file_kind == 'hdf5':
            halt, header = self._io_tree_hdf5(simulation_directory + rockstar_directory)

        ## assign auxilliary information
        # assign cosmological parameters via cosmology class
        halt.Cosmology = self._get_cosmology(simulation_directory, header)

        # assign information about all snapshot times
        halt.Snapshot = self.Snapshot

        # assign general information about simulation
        if not simulation_name and simulation_directory != './':
            simulation_name = simulation_directory.split('/')[-2]
            simulation_name = simulation_name.replace('_', ' ')
            simulation_name = simulation_name.replace('res', 'r')

        halt.info = {
            'box.length/h': header['box.length/h'],
            'box.length': header['box.length/h'] / header['hubble'],
            'catalog.kind': 'halo.tree',
            'file.kind': file_kind,
            'baryonic': ut.catalog.is_baryonic_from_directory(simulation_directory, os),
            'host.number': 0,
            'simulation.name': simulation_name,
        }

        if assign_species and 'hdf5' in file_kind:
            # try assigning particle species properties, if file exists
            species_name = 'star'
            self._assign_species_to_tree(
                halt, species_name, species_snapshot_indices, simulation_directory,
                rockstar_directory)
            if species_name + 'mass' in halt:
                # ensure baryonic flag
                halt.info['baryonic'] = True

        if assign_host:
            # assign one or multiple hosts
            # if 'elvis' is in simulation directory name, force 2 hosts
            host_number = ut.catalog.get_host_number_from_directory(
                host_number, simulation_directory, os)
            self._assign_hosts_to_tree(halt, 'halo', host_number)
            halt.info['host.number'] = host_number

        return halt

    def read_trees_simulations(
        self, simulation_directories=[], rockstar_directory=ROCKSTAR_DIRECTORY, file_kind='hdf5',
        assign_species=True, species_snapshot_indices=None, assign_host=True, host_number=1):
        '''
        Read catalog of halo merger trees across different simulations.
        Return as list of dictionary classes.

        Parameters
        ----------
        simulation_directories : list of strings : directories of simulations
        rockstar_directory : str : sub-directory (within simulation_directory) of halo files
        file_kind : str : kind of halo tree file to read: 'text', 'hdf5'
        assign_species : bool : whether to read and assign baryonic particle properties
        species_snapshot_indices : array :
            list of snapshot indices at which to assign particle species to tree
            if None, assign at all snapshots with particle species data
        assign_host : bool : whether to assign primary host[s] and relative coordinates
        host_number : int : number of primary hosts to assign

        Returns
        -------
        halts : list of dictionary classes : catalogs of halo merger trees across simulations
        '''
        # parse list of directories
        if np.ndim(simulation_directories) == 0:
            raise ValueError('input simulation_directories = {} but need to input list'.format(
                             simulation_directories))
        elif np.ndim(simulation_directories) == 1:
            # assign null names
            simulation_directories = list(
                zip(simulation_directories, ['' for _ in simulation_directories]))
        elif np.ndim(simulation_directories) == 2:
            pass
        elif np.ndim(simulation_directories) >= 3:
            raise ValueError('not sure how to parse simulation_directories = {}'.format(
                             simulation_directories))

        halts = []
        directories_read = []
        for simulation_directory, simulation_name in simulation_directories:
            try:
                halt = self.read_tree(
                    simulation_directory, rockstar_directory, file_kind, assign_species,
                    species_snapshot_indices, assign_host, host_number, simulation_name)

                halts.append(halt)
                directories_read.append(simulation_directory)

            except Exception:
                self.say('! could not read halo merger trees in {}'.format(simulation_directory))

        if not len(halts):
            self.say('! could not read any halo merger trees')
            return

        return halts

    def _read_tree_text(self, rockstar_directory=ROCKSTAR_DIRECTORY):
        '''
        Read catalog of halo merger trees (text file) from ConsistentTrees (tree_*.dat).
        Return as dictionary class.

        Parameters
        ----------
        rockstar_directory : str : directory (full path) of rockstar halo files

        Returns
        -------
        halt : dictionary class : catalog of halo merger trees across all snapshots
        '''
        file_name = 'tree_0_0_0.dat'
        path_file_name = (ut.io.get_path(rockstar_directory) + self.catalog_tree_directory +
                          file_name)

        ## store as dictionary class ----------
        halt = HaloDictionaryClass()
        header = {}

        ## read header to get cosmology ----------
        with open(path_file_name, 'r') as file_in:
            for line in file_in:
                if 'h0 = ' in line:
                    index = line.rfind('h0 = ')
                    header['hubble'] = float(line[index + 5: index + 13])
                if 'Omega_M = ' in line:
                    index = line.rfind('Omega_M = ')
                    header['omega_matter'] = float(line[index + 10: index + 18])
                if 'Omega_L = ' in line:
                    index = line.rfind('Omega_L = ')
                    header['omega_lambda'] = float(line[index + 10: index + 18])
                if 'box size = ' in line:
                    index = line.rfind('box size = ')
                    header['box.length/h'] = float(line[index + 11: index + 20])
                    # convert to [kpc/h comoving]
                    header['box.length/h'] *= ut.constant.kilo_per_mega

            header['dark.particle.mass'] = np.nan

        # initialize rest of cosmological parameters for later
        header['omega_baryon'] = None
        header['sigma_8'] = None
        header['n_s'] = None

        it = self.int_type
        ft = self.float_type

        halt_in = np.loadtxt(
            path_file_name, comments='#',
            skiprows=49,  # because ConsistentTrees writes total number of halos here
            dtype=[
                ('scalefactor', ft),  # [convert to snapshot index] of halo
                (self.tree_id_name, it),  # tree ID (unique across all snapshots)
                ('descendant.scalefactor', ft),  # [convert to snapshot index] of descendant
                ('descendant.' + self.tree_id_name, it),  # [convert to tree index] of descendant
                ('progenitor.number', it),  # number of progenitors
                # [convert to tree index] of local (least mass) central (can be a satellite)
                ('central.local.' + self.tree_id_name, it),
                # [convert to tree index] of most massive central
                ('central.' + self.tree_id_name, it),
                ('descendant.central.local.' + self.tree_id_name, it),  # [ignore]
                ('am.phantom', it),  # whether halo is interpolated across snapshots
                ('sam.mass.vir', ft),  # [ignore]
                ('mass.bound', ft),  # bound mass
                ('radius', ft),  # halo radius
                ('scale.radius', ft),  # NFW scale radius
                ('vel.std', ft),  # velocity dispersion
                ('am.progenitor.main', it),  # whether am most massive progenitor of my descendant
                ('major.merger.scalefactor', ft),  # [convert to snapshot index] of last maj merger
                ('vel.circ.max', ft),  # maximum of circular velocity
                ('position.x', ft), ('position.y', ft), ('position.z', ft),  # center position
                ('velocity.x', ft), ('velocity.y', ft), ('velocity.z', ft),  # center velocity
                ('momentum.ang.x', ft), ('momentum.ang.y', ft), ('momentum.ang.z', ft),  # [ignore]
                ('spin.peebles', ft),  # dimensionless spin parameter
                ('breadth.index', it),  # (same as tree index) [ignore]
                ('dindex', it),  # depth-first order (index) within tree
                ('final.' + self.tree_id_name, it),  # [convert to tree index] at final snapshot
                (self.catalog_id_name, it),  # catalog ID from rockstar
                ('snapshot.index', it),  # [ignore]
                # depth-first index of next co-progenitor
                ('progenitor.co.dindex', it),
                # depth-first index of last progenitor (earliest time), including *all* progenitors
                ('progenitor.last.dindex', it),
                # depth-first index of last progenitor (earliest time), only along main prog branch
                ('progenitor.main.last.dindex', it),
                ('tidal.force', ft),  # [ignore]
                ('tidal.' + self.tree_id_name, it),  # [ignore]
                ('scale.radius.klypin', ft),  # NFW scale radius from radius(vel.circ.max)
                ('mass', ft),  # total mass within halo radius (including unbound)
                ('mass.vir', ft), ('mass.200c', ft), ('mass.500c', ft), ('mass.180m', ft),
                # offset of density peak from particle average position
                ('position.offset', ft), ('velocity.offset', ft),
                ('spin.bullock', ft),  # dimensionless spin from Bullock et al (J/(sqrt(2)*GMVR))
                # ratio of 2nd & 3rd to 1st largest shape ellipsoid axes (Allgood et al 2006)
                ('axis.b/a', ft), ('axis.c/a', ft),
                ('axis.x', ft), ('axis.y', ft), ('axis.z', ft),  # [ignore]
                ('axis.b/a.500c', ft), ('axis.c/a.500c', ft),  # [ignore]
                ('axis.x.500c', ft), ('axis.y.500c', ft), ('axis.z.500c', ft),  # [ignore]
                ('kinetic/potential', ft),  # ratio of kinetic to potential energy [ignore]
                ('mass.pe.behroozi', ft),  # [ignore]
                ('mass.pe.diemer', ft),  # [ignore]
                ('type', it),  # [ignore]
                ('star.mass.rockstar', ft),  # [ignore for now]
                ('gas.mass.rockstar', ft),  # [ignore for now]
                ('bh.mass.rockstar', ft),  # [ignore for now]
                ('mass.hires', ft),  # mass in high-res DM particles [ignore]
                ('mass.lowres', ft),  # mass in low-res DM particles
            ]
        )

        for prop_name in halt_in.dtype.names:
            if prop_name not in self.ignore_properties:
                halt[prop_name] = halt_in[prop_name]

        self.say('* read {} halos from:  {}\n'.format(
            halt[self.prop_name_default].size, path_file_name.strip('./')))

        del(halt_in)

        # convert properties
        for prop_name in halt:
            if 'mass' in prop_name and 'scalefactor' not in prop_name:
                halt[prop_name] *= 1 / header['hubble']  # [M_sun]
            elif 'radius' in prop_name:
                halt[prop_name] *= halt['scalefactor'] / header['hubble']  # [kpc physical]
            elif 'position' in prop_name:
                halt[prop_name] *= ut.constant.kilo_per_mega / header['hubble']  # [kpc comoving]
            elif 'momentum.ang' in prop_name:
                halt[prop_name] *= (halt['scalefactor'] / header['hubble']) ** 2  # [kpc physical]
            elif 'energy' in prop_name:
                halt[prop_name] *= halt['scalefactor'] / header['hubble']  # [kpc physical]
            elif 'index' in prop_name and np.min(halt[prop_name]) == -1:
                # ensure null pointer index points safely out of range
                hindices = np.where(halt[prop_name] == -1)[0]
                halt[prop_name][hindices] -= halt[prop_name].size

        # assign derived masses
        halt['mass.200m'] = halt['mass']  # pointer for clarity/convenience
        if 'star.mass.rockstar' in halt:
            halt['baryon.mass.rockstar'] = halt['gas.mass.rockstar'] + halt['star.mass.rockstar']
            halt['dark.mass'] = halt['mass'] - halt['baryon.mass.rockstar']

        # convert position and velocity to halo number x dimension number array
        for prop_name in ['position', 'velocity', 'bulk.velocity', 'momentum.ang', 'axis',
                          'axis.500c']:
            if prop_name + '.x' in halt:
                halt[prop_name] = np.transpose(
                    [halt[prop_name + '.x'], halt[prop_name + '.y'], halt[prop_name + '.z']])
                del(halt[prop_name + '.x'], halt[prop_name + '.y'], halt[prop_name + '.z'])

        # convert properties of snapshot scale-factor to snapshot index
        for prop_name in list(halt.keys()):
            if 'scalefactor' in prop_name:
                prop_name_new = prop_name.replace('scalefactor', 'snapshot')
                # initialize safely out of bounds
                halt[prop_name_new] = (
                    np.zeros(halt[prop_name].size, np.int32) - self.Snapshot['index'].size - 1)
                hindices = ut.array.get_indices(halt[prop_name], [1e-10, 1.0001])
                if hindices.size:
                    halt[prop_name_new][hindices] = self.Snapshot.get_snapshot_indices(
                        'scalefactor', halt[prop_name][hindices])
                del(halt[prop_name])

        # convert halo tree id pointer to index pointer
        ut.catalog.assign_id_to_index(halt, self.tree_id_name, 0)
        for prop_name in list(halt.keys()):
            if ('.' + self.tree_id_name) in prop_name:
                prop_name_new = prop_name.replace(self.tree_id_name, 'index')
                halt[prop_name_new] = ut.array.get_array_null(halt[prop_name].size)
                hindices = ut.array.get_indices(halt[prop_name], [0, Inf])
                halt[prop_name_new][hindices] = halt[self.tree_id_name + '.to.index'][
                    halt[prop_name][hindices]]
                del(halt[prop_name])

        # assign progenitor information from descendant information
        # first assign main (most massive) progenitor
        am_prog_indices = np.where(
            (halt['am.progenitor.main'] > 0) * (halt['snapshot'] < halt['snapshot'].max()))[0]
        desc_hindices = halt['descendant.index'][am_prog_indices]
        assert np.min(desc_hindices) >= 0
        halt['progenitor.main.index'] = ut.array.get_array_null(
            halt['descendant.index'].size, halt['descendant.index'].dtype)
        halt['progenitor.main.index'][desc_hindices] = am_prog_indices
        # assign co-progenitors if multiple progenitors
        halt['progenitor.co.index'] = ut.array.get_array_null(
            halt['progenitor.main.index'].size, halt['progenitor.main.index'].dtype)
        has_mult_prog_hindices = np.where(halt['progenitor.number'] > 1)[0]
        for has_mult_prog_hindex in has_mult_prog_hindices:
            prog_indices = np.where(
                halt['descendant.index'] == has_mult_prog_hindex)[0]
            assert halt['am.progenitor.main'][prog_indices[0]]  # sanity check
            for prog_i, prog_hindex in enumerate(prog_indices[:-1]):
                halt['progenitor.co.index'][prog_hindex] = prog_indices[prog_i + 1]

        return halt, header

    def _io_tree_hdf5(self, rockstar_directory=ROCKSTAR_DIRECTORY, halt=None):
        '''
        Read/write catalog of halo merger trees across snapshots to/from HDF5 file.
        If reading, return as dictionary class.

        Parameters
        ----------
        rockstar_directory : str : directory (full path) of rockstar halo files
        halt : dictionary class : catalog of halo merger trees, if writing

        Returns
        -------
        halt : dictionary class : catalog of halo merger trees across all snapshots
        '''
        file_name = 'tree.hdf5'

        file_path = ut.io.get_path(rockstar_directory) + self.catalog_hdf5_directory
        path_file_name = file_path + file_name

        if halt is not None:
            # write to file
            assert halt.info['catalog.kind'] == 'halo.tree'
            file_path = ut.io.get_path(file_path, create_path=True)

            properties_add = []
            for prop_name in halt.info:
                if not isinstance(halt.info[prop_name], str):
                    halt['info:' + prop_name] = np.array(halt.info[prop_name])
                    properties_add.append('info:' + prop_name)

            for prop_name in halt.Cosmology:
                halt['cosmology:' + prop_name] = np.array(halt.Cosmology[prop_name])
                properties_add.append('cosmology:' + prop_name)

            ut.io.file_hdf5(path_file_name, halt)

            for prop_name in properties_add:
                del(halt[prop_name])

        else:
            # store as dictionary class
            halt = HaloDictionaryClass()
            header = {}

            try:
                # try to read from file
                halt_in = ut.io.file_hdf5(path_file_name, verbose=False)

                for prop_name in halt_in:
                    if 'info:' in prop_name:
                        hal_prop_name = prop_name.split(':')[-1]
                        header[hal_prop_name] = float(halt_in[prop_name])
                    elif 'cosmology:' in prop_name:
                        hal_prop_name = prop_name.split(':')[-1]
                        header[hal_prop_name] = float(halt_in[prop_name])
                    else:
                        halt[prop_name] = halt_in[prop_name]

                self.say('* read {} halos from:  {}'.format(
                    halt[self.prop_name_default].size, path_file_name.strip('./')))

            except OSError:
                raise OSError('! no halo merger tree file = {}'.format(path_file_name.strip('./')))

            return halt, header

    def _assign_hosts_to_tree(self, halt, host_kind='halo', host_number=1):
        '''
        Assign one or multiple primary host halo/galaxy and relative coordinates.

        Parameters
        ----------
        hal : dictionary class : catalog of halos at snapshot
        host_kind : str : property to determine primary host: 'halo', 'star'
        host_number : int : number of hosts to assign
        '''
        if host_number < 1:
            host_number = 1

        for host_rank in range(host_number):
            host_index_name = ut.catalog.get_host_name(host_rank) + 'index'

            if host_index_name not in halt:
                self._assign_host_to_tree(halt, host_kind, host_rank)

        if host_number > 1:
            # multiple hosts - assign nearest one to each halo

            # initialize all halos relative to the primary host
            for prop_name in list(halt.keys()):
                if 'host.' in prop_name and 'near.' not in prop_name:
                    prop_name_near = prop_name.replace('host.', 'host.near.')
                    halt[prop_name_near] = np.array(halt[prop_name])

            snapshot_indices = np.arange(halt['snapshot'].min(), halt['snapshot'].max() + 1)

            for snapshot_index in snapshot_indices:
                hindices = np.where(halt['snapshot'] == snapshot_index)[0]
                if hindices.size:
                    host_distancess = np.zeros(
                        (hindices.size, host_number), dtype=halt['host.distance'].dtype)

                    for host_rank in range(host_number):
                        host_name = ut.catalog.get_host_name(host_rank)
                        host_distancess[:, host_rank] = halt.prop(
                            host_name + 'distance.total', hindices)

                    host_nearest_indices = np.argmin(host_distancess, 1)

                    # assign halos whose nearest is not the primary
                    for host_rank in range(1, host_number):
                        hindices_h = hindices[np.where(host_nearest_indices == host_rank)[0]]
                        if hindices_h.size:
                            host_name = ut.catalog.get_host_name(host_rank)
                            for prop_name in halt:
                                if host_name in prop_name and 'near.' not in prop_name:
                                    prop_name_near = prop_name.replace(host_name, 'host.near.')
                                    halt[prop_name_near][hindices_h] = halt[prop_name][hindices_h]
        print()

    def _assign_host_to_tree(self, halt, host_kind='halo', host_rank=0):
        '''
        Assign primary (secondary, etc) host halo/galaxy and position + velocity wrt it.
        Determine host as being host_rank order sorted by host_prop_name.
        Require host to have low contamination from low-resolution dark matter at final snapshot.
        Determine host at final snapshot and follow back its main progenitor via tree.

        If host_kind is 'halo', define primary host as most massive halo in catalog,
        and use coordinates in halo catalog.
        If host_kind is 'star', define primary host as highest stellar mass galaxy in catalog,
        and use coordinate defined via stars.

        Parameters
        ----------
        halt : dictionary class : catalog of halo merger trees across all snapshots
        host_kind : str : property to determine primary host: 'halo', 'star'
        host_rank : int : rank of host halo (sorted by host_prop_name) to assign
        '''
        assert host_kind in ['halo', 'star']

        host_name = ut.catalog.get_host_name(host_rank)

        if host_kind == 'halo':
            host_prop_name = 'mass'  # property to use to determine primary host
            spec_prefix = ''
        elif host_kind == 'star':
            host_prop_name = 'star.mass'  # property to use to determine primary host
            spec_prefix = 'star.'
            host_name = spec_prefix + host_name

        self.say('* assigning primary {} and coordinates wrt it to merger trees'.format(
                 host_name.replace('.', '')))

        # initialize arrays
        halt[host_name + 'index'] = ut.array.get_array_null(
            halt[self.prop_name_default].size)
        halt[host_name + 'distance'] = np.zeros(
            halt[spec_prefix + 'position'].shape, halt[spec_prefix + 'position'].dtype) * np.nan
        halt[host_name + 'velocity'] = np.zeros(
            halt[spec_prefix + 'velocity'].shape, halt[spec_prefix + 'velocity'].dtype) * np.nan
        halt[host_name + 'velocity.tan'] = np.zeros(
            halt[host_prop_name].size, halt[spec_prefix + 'velocity'].dtype) * np.nan
        halt[host_name + 'velocity.rad'] = np.zeros(
            halt[host_prop_name].size, halt[spec_prefix + 'velocity'].dtype) * np.nan

        # get host at final snapshot
        snapshot_index = halt['snapshot'].max()
        pure_hindices = ut.array.get_indices(
            halt.prop('lowres.mass.frac'), [0, self.lowres_mass_frac_max])
        hindices = ut.array.get_indices(halt['snapshot'], snapshot_index, pure_hindices)

        # get host_rank'th halo
        host_index = hindices[np.argsort(halt[host_prop_name][hindices])][-host_rank - 1]

        # follow back main progenitor
        while host_index >= 0:
            snapshot_index = halt['snapshot'][host_index]

            hindices = ut.array.get_indices(halt['snapshot'], snapshot_index)

            halt[host_name + 'index'][hindices] = host_index

            # assign host coordinates only to halos with well defined mass
            hindices = ut.array.get_indices(halt.prop(host_prop_name), [1e-10, Inf], hindices)

            # distance to primary host
            halt[host_name + 'distance'][hindices] = ut.coordinate.get_distances(
                halt[spec_prefix + 'position'][hindices],
                halt[spec_prefix + 'position'][host_index],
                halt.info['box.length'], halt.Snapshot['scalefactor'][snapshot_index])

            # velocity wrt primary host
            halt[host_name + 'velocity'][hindices] = ut.coordinate.get_velocity_differences(
                halt[spec_prefix + 'velocity'][hindices],
                halt[spec_prefix + 'velocity'][host_index],
                halt[spec_prefix + 'position'][hindices],
                halt[spec_prefix + 'position'][host_index],
                halt.info['box.length'], halt.Snapshot['scalefactor'][snapshot_index],
                ut.constant.Gyr_per_sec / halt.Cosmology.get_hubble_parameter(
                    halt.Snapshot['redshift'][snapshot_index]))

            # orbital velocities wrt primary host - only those with well defined host distance
            hindices = ut.array.get_indices(
                halt.prop(host_name + 'distance.total'), [1e-10, Inf], hindices)

            distances_norm = np.transpose(
                halt[host_name + 'distance'][hindices].transpose() /
                halt.prop(host_name + 'distance.total', hindices))  # need to do this way

            halt[host_name + 'velocity.tan'][hindices] = np.sqrt(np.sum(np.cross(
                halt[host_name + 'velocity'][hindices], distances_norm) ** 2, 1))
            halt[host_name + 'velocity.tan'][host_index] = 0

            halt[host_name + 'velocity.rad'][hindices] = np.sum(
                halt[host_name + 'velocity'][hindices] * distances_norm, 1)
            halt[host_name + 'velocity.rad'][host_index] = 0

            # get host's main progenitor
            host_index = halt['progenitor.main.index'][host_index]
            if (host_index >= 0 and halt['snapshot'][host_index] > 0 and
                    halt['snapshot'][host_index] != snapshot_index - 1):
                self.say('! {} main progenitor skips snapshot {}'.format(
                         host_name.replace('.', ''), snapshot_index - 1))

    def _convert_tree(self, halt):
        '''
        Experimental.
        '''
        snapshot_index_max = halt['snapshot'].max()
        snapshot_index_min = halt['snapshot'].min()

        halo_number_max = 0
        halo_number_max_snapshot = None
        snapshot_indices = np.arange(snapshot_index_min, snapshot_index_max + 1)
        for snapshot_index in snapshot_indices:
            hindices = np.where(halt['snapshot'] == snapshot_index)[0]
            if hindices.size > halo_number_max:
                halo_number_max = hindices.size
                halo_number_max_snapshot = snapshot_index

        # start at final snapshot, work back to assign main progenitor indices
        hindices_final = np.where(halt['snapshot'] == halt['snapshot'].max())[0]

        self.say('number of halos = {} at snapshot index = {}'.format(
                 hindices_final.size, snapshot_index_max))
        self.say('number of halos maximum = {} at snapshot index = {}'.format(
                 halo_number_max, halo_number_max_snapshot))

        # make halo merger tree index pointer
        catalog_shape = (halo_number_max, snapshot_index_max + 1)
        dtype = ut.array.parse_data_type(halt['snapshot'].size)
        hindicess = np.zeros(catalog_shape, dtype=dtype) - halt['snapshot'].size - 1

        # halos sorted by tree depth
        halt_indices_depth = np.argsort(halt['dindex'])

        for hii, hindex in enumerate(hindices_final):
            hindices_final = halt_indices_depth[
                hindex: halt['progenitor.main.last.dindex'][hindex] + 1]
            halt_snapshot_indices = halt['snapshot'][hindices_final]
            hindicess[hii][halt_snapshot_indices] = hindices_final

        for prop_name in halt:
            props = halt[prop_name]
            if np.ndim(props) == 1:
                halt[prop_name] = np.zeros(catalog_shape, props.dtype) - 1
                if props.dtype in [np.int32, np.int64]:
                    halt[prop_name] -= halt['snapshot'].size

                masks = (hindicess >= 0)
                halt[prop_name][masks] = props[hindicess[masks]]

    def get_catalog_from_tree(self, halt, snapshot_index):
        '''
        Parameters
        ----------
        halt : dictionary class : catalog of halo merger trees across all snapshots
        snapshot_index : int : index of snapshot at which to get halo catalog

        Returns
        -------
        hal : dictionary class : catalog of halos at snapshot_index
        '''
        hal = HaloDictionaryClass()

        hindices_at_snapshot = np.where(halt['snapshot'] == snapshot_index)[0]
        for prop_name in halt:
            if prop_name != 'snapshot':
                if isinstance(halt[prop_name], list):
                    hal[prop_name] = [[] for _ in hindices_at_snapshot]
                    for hii, hi in enumerate(hindices_at_snapshot):
                        hal[prop_name][hii] = halt[prop_name][hi]
                else:
                    hal[prop_name] = halt[prop_name][hindices_at_snapshot]

        hal.info = halt.info
        hal.Cosmology = halt.Cosmology
        hal.Snapshot = halt.Snapshot
        hal.snapshot = {
            'index': snapshot_index,
            'scalefactor': halt.Snapshot['scalefactor'][snapshot_index],
            'redshift': halt.Snapshot['redshift'][snapshot_index],
            'time': halt.Snapshot['time'][snapshot_index],
            'time.hubble': ut.constant.Gyr_per_sec / hal.Cosmology.get_hubble_parameter(
                halt.Snapshot['redshift'][snapshot_index])
        }

        return hal

    ## both halo catalog at snapshot and merger trees across snapshots ----------
    def _convert_id_to_index_catalogs_tree(self, hals, halt):
        '''
        Convert ids to indices for pointers between halo catalogs and halo merger tree.

        Parameters
        ----------
        hals : list of dictionary classes : catalog of halos at each snapshot
        halt : dictionary class : catalog of halo merger trees across all snapshots
        '''
        # parse input catalogs
        assert len(hals) and hals[-1].info['catalog.kind'] == 'halo.catalog'
        assert halt.info['catalog.kind'] == 'halo.tree'

        # set pointer names
        catalog_pointer_name = self.catalog_id_name + '.to.index'
        tree_pointer_name = self.tree_id_name + '.to.index'

        self.say('\n* converting pointer id to index between halo catalogs and merger tree')

        halt['catalog.index'] = ut.array.get_array_null(halt[self.tree_id_name].size)
        if tree_pointer_name not in halt or not len(halt[tree_pointer_name]):
            ut.catalog.assign_id_to_index(halt, self.tree_id_name, 0)

        for hal in hals:
            if len(hal) and len(hal[self.catalog_id_name]):
                # get real (non-phantom) halos at this snapshot in trees
                halt_indices = np.where(
                    (halt['am.phantom'] == 0) * (halt['snapshot'] == hal.snapshot['index']))[0]
                if halt_indices.size:
                    if catalog_pointer_name not in hal or not len(hal[catalog_pointer_name]):
                        ut.catalog.assign_id_to_index(hal, self.catalog_id_name, 0)

                    # assign halo catalog index to tree - all halos in trees should be in catalog
                    hal_ids = halt[self.catalog_id_name][halt_indices]
                    assert hal_ids.min() >= 0
                    hal_indices = hal[catalog_pointer_name][hal_ids]
                    assert hal_indices.min() >= 0
                    halt['catalog.index'][halt_indices] = hal_indices

                    # assign halo tree indices to halo catalog - note: not all halos are in trees
                    for prop_name in list(hal.keys()):
                        if self.tree_id_name in prop_name:
                            prop_name_new = 'tree.' + prop_name.replace(self.tree_id_name, 'index')
                            hal[prop_name_new] = (np.zeros(
                                hal[self.catalog_id_name].size, halt[self.tree_id_name].dtype) -
                                halt[self.tree_id_name].size)
                            hal_indices = np.where(hal[prop_name] >= 0)[0]
                            if hal_indices.size:
                                halt_ids = hal[prop_name][hal_indices]
                                assert halt_ids.min() >= 0
                                halt_indices = halt[tree_pointer_name][halt_ids]
                                assert halt_indices.min() >= 0
                                hal[prop_name_new][hal_indices] = halt_indices
                            del(hal[prop_name])

        del(halt[self.catalog_id_name])

        del(hal[catalog_pointer_name])
        del(halt[tree_pointer_name])

    def _assign_species_to_tree(
        self, halt, species='star', species_snapshot_indices=None,
        simulation_directory='.', rockstar_directory=ROCKSTAR_DIRECTORY, verbose=False):
        '''
        Read halo catalogs with particle species properties and assign to halo merger trees.

        Parameters
        ----------
        halt : dictionary class : catalog of halo merger trees across all snapshots
        species : str : particle species to assign to halos: 'star', 'gas', 'dark'
        species_snapshot_indices : array :
            list of snapshot indices at which to assign particle species to tree
            if None, assign at all snapshots with particle species data
        simulation_directory : str : directory of simulation
        rockstar_directory : str : sub-directory (within simulation_directory) of halo files
        verbose : bool : whether to print diagnostics
        '''
        # parse input parameters
        assert halt.info['catalog.kind'] == 'halo.tree'
        assert species in ['star', 'gas', 'dark']
        simulation_directory = ut.io.get_path(simulation_directory)

        # get all halo species file names to read
        path_file_names, file_snapshot_indices = self._get_catalog_file_names_values(
            simulation_directory + rockstar_directory, species_snapshot_indices,
            file_kind=species)

        if not len(path_file_names):
            self.say('! found no halo {} files in:  {}'.format(
                species, simulation_directory + rockstar_directory), verbose)
            return
        else:
            self.say('\n* assigning {} properties to halo merger trees'.format(species))

        # check if input subset list of snapshot indices at which to assign particles
        if species_snapshot_indices is not None:
            snapshot_indices = np.intersect1d(species_snapshot_indices, file_snapshot_indices)
        else:
            snapshot_indices = file_snapshot_indices

        #snapshot_indices = snapshot_indices[::-1]  # reverse order to start closest to z = 0

        for snapshot_index in snapshot_indices:
            hal = Particle.io_species_hdf5(
                species, None, snapshot_index, simulation_directory + rockstar_directory)

            if species + '.mass' not in hal:
                # skip this snapshot if no particle species assigned to halos
                continue
            elif species + '.mass' not in halt:
                # initialize arrays for halo merger trees
                for prop_name in hal:
                    if self.catalog_id_name not in prop_name:
                        self.say('{}'.format(prop_name), verbose)
                        if prop_name == species + '.indices':
                            halt[prop_name] = [[] for _ in halt[self.prop_name_default]]
                        else:
                            value_min = np.min(hal[prop_name].min(), -1)
                            shape = list(hal[prop_name].shape)
                            shape[0] = halt[self.prop_name_default].size
                            halt[prop_name] = np.zeros(shape, hal[prop_name].dtype) + value_min

            # get real (non-phantom) halos in tree at this snapshot
            halt_indices = np.where(
                (halt['am.phantom'] == 0) * (halt['snapshot'] == hal.snapshot['index']))[0]
            if halt_indices.size:
                # assign halo catalog index to trees - all halos in trees should be in catalog
                hal_indices = halt['catalog.index'][halt_indices]
                assert hal_indices.min() >= 0

                if verbose:
                    # check if any halos with particle species are not in tree
                    hal_indices_no_tree = np.setdiff1d(
                        np.arange(hal[self.prop_name_default].size), hal_indices)
                    hal_indices_no_tree_has_species = ut.array.get_indices(
                        hal[species + '.number'], [1, Inf], hal_indices_no_tree)
                    if hal_indices_no_tree_has_species.size:
                        self.say('! snapshot {}: {} halos have {} but not in tree'.format(
                            hal.snapshot['index'], hal_indices_no_tree_has_species.size,
                            species))
                        self.say('M_{} = {:.1e}, M_halo = {:.1e} V_max = {:.1f}'.format(
                            species,
                            hal[species + '.mass'][hal_indices_no_tree_has_species].max(),
                            hal['mass'][hal_indices_no_tree_has_species].max(),
                            hal['vel.circ.max'][hal_indices_no_tree_has_species].max()))

                # transfer particle species properties from catalog to trees
                for prop_name in hal:
                    if self.catalog_id_name not in prop_name:
                        if prop_name == species + '.indices':
                            for halt_index, hal_index in zip(halt_indices, hal_indices):
                                if len(hal[prop_name][hal_index]):
                                    halt[prop_name][halt_index] = hal[prop_name][hal_index]
                        else:
                            halt[prop_name][halt_indices] = hal[prop_name][hal_indices]

    def _assign_species_to_tree_via_halo(
        self, halt, species='star', simulation_directory='.', rockstar_directory=ROCKSTAR_DIRECTORY,
        verbose=False):
        '''
        ARCHIVAL
        Read catalog of particle species properties and assign to halo merger trees.

        Parameters
        ----------
        halt : dictionary class : catalog of halo merger trees across all snapshots
        species : str : particle species to assign to halos
        simulation_directory : str : directory of simulation
        rockstar_directory : str : sub-directory (within simulation_directory) of halo files
        verbose : bool : whether to print diagnostics
        '''
        # parse input parameters
        assert halt.info['catalog.kind'] == 'halo.tree'
        assert species in ['star', 'gas', 'dark']
        simulation_directory = ut.io.get_path(simulation_directory)

        # get all halo species file names to read
        path_file_names, file_values = self._get_catalog_file_names_values(
            simulation_directory + rockstar_directory, file_kind=species)

        if not len(path_file_names):
            self.say('! found no halo {} files in:  {}'.format(
                species, simulation_directory + rockstar_directory))
            return
        else:
            self.say('\n* assigning {} properties to halo merger trees'.format(species))

        hals = self.read_catalogs(
            'index', file_values, simulation_directory, rockstar_directory, file_kind='hdf5',
            assign_host=False, assign_species=True, all_snapshot_list=True)

        # initialize arrays for halo merger trees
        hal = hals[-1]
        for prop_name in hal:
            if (species + '.') in prop_name or 'lowres.mass.frac' in prop_name:
                self.say('{}'.format(prop_name), verbose)

                if prop_name == species + '.indices':
                    halt[prop_name] = [[] for _ in halt[self.prop_name_default]]
                else:
                    value_min = np.min(hal[prop_name].min(), -1)
                    shape = list(hal[prop_name].shape)
                    shape[0] = halt[self.prop_name_default].size
                    halt[prop_name] = np.zeros(shape, hal[prop_name].dtype) + value_min

        for hal in hals:
            if len(hal) and len(hal[self.prop_name_default]):
                # get real (non-phantom) halos at this snapshot in trees
                halt_indices = np.where(
                    (halt['am.phantom'] == 0) * (halt['snapshot'] == hal.snapshot['index']))[0]
                if halt_indices.size:
                    # assign halo catalog index to trees - all halos in trees should be in catalog
                    hal_indices = halt['catalog.index'][halt_indices]
                    assert hal_indices.min() >= 0

                    if verbose:
                        # check if any halos with particle species are not in tree
                        hal_indices_no_tree = np.setdiff1d(
                            np.arange(hal[self.prop_name_default].size), hal_indices)
                        hal_indices_no_tree_has_species = ut.array.get_indices(
                            hal[species + '.number'], [1, Inf], hal_indices_no_tree)
                        if hal_indices_no_tree_has_species.size:
                            self.say('! snapshot {}: {} halos have {} but not in tree'.format(
                                hal.snapshot['index'], hal_indices_no_tree_has_species.size,
                                species))
                            self.say('M_{} = {:.1e}, M_halo = {:.1e} V_max = {:.1f}'.format(
                                species,
                                hal[species + '.mass'][hal_indices_no_tree_has_species].max(),
                                hal['mass'][hal_indices_no_tree_has_species].max(),
                                hal['vel.circ.max'][hal_indices_no_tree_has_species].max()))

                    # transfer particle species properties from catalog to trees
                    for prop_name in hal:
                        if (species + '.') in prop_name or 'lowres.mass.frac' in prop_name:
                            if prop_name == species + '.indices':
                                for halt_index, hal_index in zip(halt_indices, hal_indices):
                                    if len(hal[prop_name][hal_index]):
                                        halt[prop_name][halt_index] = hal[prop_name][hal_index]
                            else:
                                halt[prop_name][halt_indices] = hal[prop_name][hal_indices]

    def _transfer_properties_catalogs_tree(self, hals, halt, verbose=False):
        '''
        Transfer properties between hals (list of Rockstar halo catalogs at each snapshot) and
        halt (ConsistentTrees halo merger trees across all snaphsots).

        Parameters
        ----------
        hals : list of dictionary classes : catalog of halos at each snapshot
        halt : dictionary class : catalog of halo merger trees across all snapshots
        '''
        # parse input catalogs
        assert len(hals) and hals[-1].info['catalog.kind'] == 'halo.catalog'
        assert halt.info['catalog.kind'] == 'halo.tree'

        # initialize arrays for halo merger trees
        self.say('* assigning properties to halo merger tree catalog:')
        hal = hals[-1]
        for prop_name in hal:
            if (prop_name not in halt and 'id' not in prop_name and prop_name != 'tree.index' and
                    'host' not in prop_name and 'star.' not in prop_name and
                    'gas.' not in prop_name):
                self.say('{}'.format(prop_name))
                value_min = np.min(hal[prop_name].min(), -1)
                halt[prop_name] = np.zeros(
                    halt[self.prop_name_default].size, hal[prop_name].dtype) + value_min

        #first_print = True

        for hal in hals:
            if len(hal) and len(hal[self.prop_name_default]):
                # get real (non-phantom) halos at this snapshot in trees
                halt_indices = np.where(
                    (halt['am.phantom'] == 0) * (halt['snapshot'] == hal.snapshot['index']))[0]
                if halt_indices.size:
                    # assign halo catalog index to trees - all halos in trees should be in catalog
                    hal_indices = halt['catalog.index'][halt_indices]
                    assert hal_indices.min() >= 0

                    if verbose:
                        # sanity check - compare shared properties
                        self.say('* shared properties with offsets: [min, med, max], N_offset')
                        for prop_name in halt:
                            if (prop_name in hal and hal[prop_name][hal_indices].min() > 0 and
                                    halt[prop_name][halt_indices].min() > 0):
                                prop_difs = (halt[prop_name][halt_indices] -
                                             hal[prop_name][hal_indices])
                                if (np.abs(np.min(prop_difs)) > 1e-4 and
                                        np.abs(np.max(prop_difs)) > 1e-4):
                                    self.say('{}: [{}, {}, {}] {}'.format(
                                        prop_name, np.min(prop_difs), np.median(prop_difs),
                                        np.max(prop_difs), np.sum(np.abs(prop_difs) > 0)))

                    # transfer properties from catalog to trees
                    for prop_name in hal:
                        if (prop_name not in halt and 'id' not in prop_name and
                                prop_name != 'tree.index' and 'host' not in prop_name and
                                'star.' not in prop_name and 'gas.' not in prop_name):
                            halt[prop_name][halt_indices] = hal[prop_name][hal_indices]

    def rewrite_as_hdf5(self, simulation_directory='.', rockstar_directory=ROCKSTAR_DIRECTORY):
        '''
        Read halo catalogs at all snapshots from Rockstar text files.
        Read ConsistentTrees history text files and merger tree text file, if they exist.
        Re-write as HDF5 files.

        Parameters
        ----------
        simulation_directory : str : directory of simulation
        rockstar_directory : str : sub-directory (within simulation_directory) of halo files
        '''
        simulation_directory = ut.io.get_path(simulation_directory)
        rockstar_directory = ut.io.get_path(rockstar_directory)

        # read halo catalogs
        hals = self.read_catalogs(
            'index', 'all', simulation_directory, rockstar_directory, file_kind='out')
        if isinstance(hals, dict):
            hals = [hals]  # ensure list if catalog only at single snapshot

        try:
            # try to read halo history catalogs
            halhs = self.read_catalogs(
                'index', 'all', simulation_directory, rockstar_directory, file_kind='hlist',
                assign_host=False)
            # transfer history properties to halo catalog
            for hal, halh in zip(hals, halhs):
                if len(hal) and len(halh):
                    self._transfer_properties_catalog(hal, halh)
            del(halhs)
        except OSError:
            self.say('! could not read halo history catalogs (hlist)')

        try:
            # try to read halo merger trees
            halt = self.read_tree(simulation_directory, rockstar_directory, 'text')
            self._convert_id_to_index_catalogs_tree(hals, halt)
            self._io_tree_hdf5(simulation_directory + rockstar_directory, halt)
        except OSError:
            self.say('! could not read halo merger trees')

        for hal in hals:
            if len(hal):
                # write as HDF5 files
                self._io_catalog_hdf5(
                    simulation_directory + rockstar_directory, hal.snapshot['index'], hal)

    ## assign additional properties ----------
    def assign_nearest_neighbor(
        self, hal, mass_kind='mass', mass_limits=[1e7, Inf],
        neig_distance_max=3800, neig_distance_scaling='Rneig', neig_number_max=5000):
        '''
        Assign information about nearest neighbor halo
        (nearest := minimum in terms of physical distance or d/R_halo)
        to each halo in mass range in catalog.

        Parameters
        ----------
        hal : list : catalog of halos at snapshot
        mass_kind : str : mass kind
        mass_limits : list : min and max limits for mass_kind
        neig_distance_max : int : maximum search distance [kpc physical]
        neig_distance_scaling : str : distance kind to compute minimum of:
            'physical' or '' = use physical distance
            'Rneig' = scale to distance/R_halo(neig)
            'Rself' = scale to distance/R_halo(self)
        neig_number_max : int : maximum number of neighbors to search for within maximum distance
        '''
        NearestNeighbor = ut.catalog.NearestNeighborClass()

        NearestNeighbor.assign_to_self(
            hal, mass_kind, mass_limits, [1.0, Inf], [min(mass_limits), Inf],
            neig_distance_max, neig_distance_scaling, neig_number_max)

        NearestNeighbor.assign_to_catalog(hal)

    def sort_by_property(self, hal, property_name='mass'):
        '''
        Sort halos (in descending order) by property_name.

        Parameters
        ----------
        hal : dictionary class : catalog of halos at snapshot
        property_name : str : name of property to sort by
        '''
        hindices = ut.array.get_arange(hal[property_name])

        # put halos with significant contamination from low-resolution DM at end of list
        pure_hindices = hindices[hal.prop('lowres.mass.frac') < self.lowres_mass_frac_max]
        contam_hindices = hindices[hal.prop('lowres.mass.frac') >= self.lowres_mass_frac_max]
        pure_hindices = pure_hindices[np.argsort(hal[property_name][pure_hindices])[::-1]]
        contam_hindices = contam_hindices[
            np.argsort(hal[property_name][contam_hindices])[::-1]]
        hindices = np.concatenate([pure_hindices, contam_hindices])

        for prop_name in hal:
            hal[prop_name] = hal[prop_name][hindices]

    """
    def assign_host_orbits(self, hal, host_rank=0):
        '''
        Assign orbital properties wrt primary host.

        Parameters
        ----------
        hal : dictionary class : catalog of halos at snapshot
        host_rank : int :
        '''
        self.say('* assigning orbital properties wrt {}'.format(host_rank))

        host_position_kind = 'position'  # code does not assign star particles to primary host
        host_velocity_kind = 'velocity'  # so it does not have a star/gas position or velocity

        position_velocity_kinds = [
            ['position', 'velocity']
            #['star.position', 'star.velocity'],
        ]

        host_index_name = ut.catalog.get_host_name(host_rank) + 'index'

        # sanity check
        for position_kind, velocity_kind in position_velocity_kinds:
            if position_kind not in hal or velocity_kind not in hal:
                position_velocity_kinds.remove([position_kind, velocity_kind])

        for position_kind, velocity_kind in position_velocity_kinds:
            distance_vectors = ut.coordinate.get_distances(
                hal[position_kind], hal[host_position_kind][hal[host_index_name]],
                hal.info['box.length'], hal.snapshot['scalefactor'])  # [kpc physical]

            velocity_vectors = ut.coordinate.get_velocity_differences(
                hal[velocity_kind], hal[host_velocity_kind][hal[host_index_name]],
                hal[position_kind], hal[host_position_kind][hal[host_index_name]],
                hal.info['box.length'], hal.snapshot['scalefactor'], hal.snapshot['time.hubble'])

            orb = ut.orbit.get_orbit_dictionary(distance_vectors, velocity_vectors)

            for prop_name in orb:
                hal[host_index_name.replace('index', '') + prop_name] = orb[prop_name]
    """

    # utility for running rockstar
    def write_snapshot_indices(
        self, snapshot_selection='all', simulation_directory='../../',
        rockstar_directory=ROCKSTAR_DIRECTORY, out_file_name='snapshot_indices.txt'):
        '''
        Read all snapshot indices of the simulation, read indices that already have a halo catalog,
        print to file a list of snapshot indices that halo finder needs to run on.
        By default, set to run from within the Rockstar sub-directory.

        Parameters
        ----------
        snapshot_selection : str : 'all', 'subset'
        simulation_directory : str : directory of simulation
        rockstar_directory : str : sub-directory (within simulation_directory) of halo files
        out_file_name : str : name of output file to list snapshot indices to run on
        '''
        snapshot_index_min = 3  # exclude snapshots before this - unlikely to have any halos

        # parse inputs
        simulation_directory = ut.io.get_path(simulation_directory)
        rockstar_directory = ut.io.get_path(rockstar_directory)
        assert snapshot_selection in ['all', 'subset']
        if snapshot_selection == 'all':
            Snapshot = ut.simulation.read_snapshot_times(simulation_directory)
            snapshot_indices = Snapshot['index']
        elif snapshot_selection == 'subset':
            snapshot_indices = snapshot_indices_subset

        try:
            _file_names, file_indices = self._get_catalog_file_names_values(
                simulation_directory + rockstar_directory, file_kind='out')

            # keep only indices that do not have existing halo catalog file
            snapshot_indices = np.setdiff1d(snapshot_indices, file_indices)

            # ensure one overlapping snapshot - creates descendant index bug?!
            #snapshot_indices = np.sort(np.append(file_indices.max(), snapshot_indices))
        except OSError:
            self.say('! could not read any halo catalog files, so writing all snapshot indices')

        # exclude eary snashots
        snapshot_indices = snapshot_indices[snapshot_indices >= snapshot_index_min]

        with open(out_file_name, 'w') as file_out:
            for snapshot_index in snapshot_indices:
                file_out.write('{:03d}\n'.format(snapshot_index))

        self.say('snapshot indices: number = {}, min = {}, max = {}'.format(
                 snapshot_indices.size, snapshot_indices.min(), snapshot_indices.max()))
        self.say('wrote to file:  {}'.format(out_file_name))


IO = IOClass()


class ParticleClass(ut.io.SayClass):
    '''
    Assign indices and properties of particles to halos.
    '''

    def __init__(self, catalog_hdf5_directory=HALO_CATALOG_HDF5_DIRECTORY):
        '''
        Initialize variables.
        '''
        self.catalog_hdf5_directory = ut.io.get_path(catalog_hdf5_directory)
        self.catalog_id_name = 'id'
        self.prop_name_default = 'mass'  # default property for iterating
        self.Snapshot = None

    def assign_particle_indices(
        self, hal, part, species=['star'],
        mass_limits=[3e6, Inf], vel_circ_max_limits=[5, Inf],
        bound_mass_frac_min=BOUND_MASS_FRAC_MIN, lowres_mass_frac_max=LOWRES_MASS_FRAC_MAX,
        halo_radius_frac_max=0.8, radius_max=30, radius_mass_fraction=90, radius_factor=1.5,
        halo_velocity_frac_max=2.0, particle_number_min=6,
        require_rockstar_species_mass=False):
        '''
        Identify particles of input species that are members of a halo
        (using cuts in position, velocity, and velocity dispersion).
        Assign to each halo the total number of particles and their indices in the particle catalog.

        Work down in halo sort_prop_name to prioritize particle assignment.
        Once assigned, exclude particles from future halo assignment, so each particle assigned
        to only one halo.

        Parameters
        ----------
        hal : dict : catalog of halos at snapshot
        part : dict : catalog of particles at snapshot
        species : str or list thereof : name[s] of particle species to assign indices of
        mass_limits : list : min and max limits of total mass to keep halo [M_sun]
        vel_circ_max_limits : list :  min and max limits of vel.circ.max to keep halo [km / s]
        bound_mass_frac_min : float : minimum mass.bound/mass to keep halo
        lowres_mass_frac_max : float :
            maximum fraction of total mass contaminated by low-resolution DM to keep halo
        halo_radius_frac_max : float : max radius wrt halo (in units of halo radius)
            to consider particle
        radius_max : list : max radius wrt galaxy center to consider particle [kpc physical]
        radius_mass_fraction : float : mass fraction to define galaxy edge
        radius_factor : float : multiplier for R_{radius_mass_fraction} to keep particle
        halo_velocity_frac_max : float : maximum velocity wrt halo and galaxy
            (in units of halo and galaxy velocity dispersion) to keep particle
        particle_number_min : int : minimum number of species particles within halo to consider it
        require_rockstar_species_mass : bool :
            whether to require rockstar species mass > 0 to consider halo
        '''
        # property to sort halos by to prioritize particle assignment
        sort_prop_name = 'vel.circ.max'

        # fractional change in particle number to stop iterating
        particle_number_fraction_converge = 0.01

        species = ut.particle.parse_species(part, species)

        prop_limits = {
            'mass.bound/mass': [bound_mass_frac_min, Inf],
            'lowres.mass.frac': [0, lowres_mass_frac_max],
            'mass': mass_limits,
            'vel.circ.max': vel_circ_max_limits,
        }

        for spec in species:
            hal[spec + '.indices'] = [[] for _ in hal[self.prop_name_default]]
            hal[spec + '.number'] = np.zeros(hal[self.prop_name_default].size, dtype=np.int32)

        hal_indices = ut.catalog.get_indices_catalog(hal, prop_limits)

        self.say('* assigning {} particle indices to {} halos within property limits'.format(
            ut.array.scalarize(species), hal_indices.size))

        # sort inversely by mass/velocity (to limit particle overlap)
        hal_indices = hal_indices[np.argsort(hal.prop(sort_prop_name, hal_indices))[::-1]]

        # store particles already assigned to a halo
        part_indices_used = np.array([], dtype=np.int32)

        for spec in species:
            if require_rockstar_species_mass:
                hal_indices = ut.array.get_indices(
                    hal.prop(spec + '.mass.rockstar'), [1, Inf], hal_indices)

                self.say('{} halos have {} mass: max = {:.2e} M_sun'.format(
                    hal_indices.size, spec, hal.prop(spec + '.mass.rockstar', hal_indices).max()))

            assign_number = 0

            for _hal_ii, hal_i in enumerate(hal_indices):
                # get limits in distance and velocity to select particles
                distance_max = halo_radius_frac_max * hal.prop('radius', hal_i)
                if radius_max < distance_max:
                    distance_max = radius_max
                distance_limits = [0, distance_max]

                # keep particles within:
                #   velocity_dif_max x halo internal velocity
                #   halo_radius_frac_max x halo radius
                halo_vel_max = max(hal['vel.std'][hal_i], hal['vel.circ.max'][hal_i])
                velocity_limits = [0, halo_velocity_frac_max * halo_vel_max]
                part_indices = ut.particle.get_indices_within_coordinates(
                    part, spec, distance_limits, hal.prop('position', hal_i),
                    velocity_limits, hal.prop('velocity', hal_i))

                # skip particles already assigned to a larger halo
                part_indices = np.setdiff1d(part_indices, part_indices_used)

                if spec == 'dark':
                    # not need to use mass weights for dark matter
                    weights = None
                else:
                    # normalize mass weights by median for numerical stability
                    weights = (part[spec]['mass'][part_indices] /
                               np.median(part[spec]['mass'][part_indices]))

                # iterate to remove particles with outlier positions and velocities
                part_number_frac_dif = 1
                while (part_indices.size >= particle_number_min and
                       part_number_frac_dif > particle_number_fraction_converge):
                    part_number_prev = part_indices.size

                    # select particles via position -----
                    part_center_position = ut.coordinate.get_center_position_zoom(
                        part[spec]['position'][part_indices], weights, part.info['box.length'],
                        center_position=hal.prop('position', hal_i))

                    part_distances = ut.coordinate.get_distances(
                        part[spec]['position'][part_indices], part_center_position,
                        part.info['box.length'], part.snapshot['scalefactor'],
                        total_distance=True)
                    #part_radius = ut.math.percentile_weighted(
                    #    part_distances, radius_mass_fraction, weights)
                    # skip weights for speed
                    part_radius = np.percentile(part_distances, radius_mass_fraction)

                    # keep particles within radius_factor x R_{radius_mass_fraction}
                    # from center of *galaxy*
                    masks = (part_distances < radius_factor * part_radius)
                    part_indices = part_indices[masks]
                    weights = weights[masks]

                    if part_indices.size < particle_number_min:
                        break

                    # keep particles within radius_factor x R_{radius_mass_fraction}
                    # from center of *halo*
                    part_halo_distances = ut.coordinate.get_distances(
                        part[spec]['position'][part_indices], hal.prop('position', hal_i),
                        part.info['box.length'], part.snapshot['scalefactor'], total_distance=True)
                    masks = (part_halo_distances < radius_factor * part_radius)
                    part_indices = part_indices[masks]
                    weights = weights[masks]

                    if part_indices.size < particle_number_min:
                        break

                    # select particles via velocity -----
                    # get COM velocity of particles
                    part_center_velocity = ut.coordinate.get_center_velocity(
                        part[spec]['velocity'][part_indices], weights)

                    # total velocity of each particle wrt center velocity
                    part_vels = (part[spec]['velocity'][part_indices] - part_center_velocity) ** 2
                    part_vels = np.sqrt(np.sum(part_vels, 1))

                    # velocity dispersion of particles
                    part_vel_std = np.median(part_vels)  # skip weights for speed
                    #part_vel_std = ut.math.percentile_weighted(part_vels, 50, weights)
                    #part_vel_std = np.average(part_vels, weights=weights)
                    # cap velocity dispersion at halo value (sanity check)
                    part_vel_std = min(part_vel_std, halo_vel_max)

                    # keep only particles with velocity near center velocity
                    masks = (part_vels < halo_velocity_frac_max * part_vel_std)
                    part_indices = part_indices[masks]
                    weights = weights[masks]

                    if part_indices.size < particle_number_min:
                        break

                    part_number_frac_dif = np.abs(
                        (part_indices.size - part_number_prev) / part_number_prev)

                if part_indices.size >= particle_number_min:
                    assign_number += 1
                    hal[spec + '.indices'][hal_i] = part_indices
                    hal[spec + '.number'][hal_i] = len(part_indices)
                    part_indices_used = np.append(part_indices_used, part_indices)

            self.say('assigned {} indices to {} halos with >= {} particles'.format(
                spec, assign_number, particle_number_min))
        print()
        hal[spec + '.indices'] = np.array(hal[spec + '.indices'])

    def assign_particle_properties(
        self, hal, part, species=['star'],
        properties=[
            'position', 'velocity', 'mass',
            'radius.50', 'radius.90',
            'vel.std', 'vel.std.50', 'vel.circ.50',
            'massfraction',
            'form.time.50', 'form.time.90', 'form.time.95', 'form.time.100', 'form.time.dif.68',
            'mass.neutral']):
        '''
        Assign properties of computed from particles of input species that are within the halo.

        Parameters
        ----------
        hal : dict : catalog of halos at snapshot
        part : dict : catalog of particles at snapshot
        species : str or list : name[s] of particle species to assign  properties of
        properties : str or list : properties to assign to halo
        '''
        species = ut.particle.parse_species(part, species)

        for spec in species:
            if spec + '.indices' not in hal:
                self.say('! halo catalog does not have {}.indices'.format(spec))

            for prop_name in properties:
                hal_prop_name = spec + '.' + prop_name

                if 'position' in prop_name and 'position' in hal:
                    hal[hal_prop_name] = np.array(hal['position']) * np.nan
                elif 'velocity' in prop_name and 'velocity' in hal:
                    hal[hal_prop_name] = np.array(hal['velocity']) * np.nan
                elif 'massfraction' in prop_name and 'massfraction' in part[spec]:
                    hal[hal_prop_name] = np.zeros(
                        (hal.prop(self.prop_name_default).size,
                         part[spec]['massfraction'].shape[1]))
                    hal.element_pointer = np.array(part[spec].element_pointer)
                else:
                    hal[hal_prop_name] = np.array(hal.prop('mass')) * np.nan

        for spec in species:
            hal_indices = ut.array.get_indices(hal.prop(spec + '.number'), [1, Inf])

            self.say('* assigning {} properties to {} halos'.format(spec, hal_indices.size))

            for _hal_ii, hal_i in enumerate(hal_indices):
                pis = hal[spec + '.indices'][hal_i]

                if 'mass' in part[spec]:
                    mass_weights = part[spec]['mass'][pis] / np.median(part[spec]['mass'][pis])
                else:
                    mass_weights = None

                for prop_name in properties:
                    hal_prop_name = spec + '.' + prop_name

                    if 'position' in prop_name and 'position' in part[spec]:
                        hal[hal_prop_name][hal_i] = ut.coordinate.get_center_position_zoom(
                            part[spec]['position'][pis], mass_weights, part.info['box.length'])

                    elif 'velocity' in prop_name and 'velocity' in part[spec]:
                        hal[hal_prop_name][hal_i] = ut.coordinate.get_center_velocity(
                            part[spec]['velocity'][pis], mass_weights)

                    elif 'massfraction' in prop_name and 'massfraction' in part[spec]:
                        for element_i in range(part[spec]['massfraction'].shape[1]):
                            hal[hal_prop_name][hal_i, element_i] = (
                                np.sum(part[spec]['massfraction'][pis, element_i] * mass_weights) /
                                np.sum(mass_weights))

                    elif ('mass' in prop_name and 'mass' in part[spec] and
                          'neutral' not in prop_name):
                        hal[hal_prop_name][hal_i] = part[spec]['mass'][pis].sum()

                    elif 'vel.std' in prop_name or 'vel.circ' in prop_name:
                        distance_max = None
                        if '.50' in prop_name or '.90' in prop_name:
                            # impose maximum distance on particles
                            mass_percent = prop_name.split('.')[-1]
                            if spec == 'dark':
                                distance_max = 0.6  # radius to measure dark matter [kpc]
                            else:
                                distance_max = hal.prop('star.radius.' + mass_percent, hal_i)

                            distances = ut.coordinate.get_distances(
                                part[spec]['position'][pis],
                                hal.prop(spec + '.position', hal_i), part.info['box.length'],
                                part.snapshot['scalefactor'], total_distance=True)  # [kpc physical]

                            distance_masks = (distances < distance_max)
                            if np.sum(distance_masks) < 2:
                                continue

                            if 'vel.circ' in prop_name:
                                mass = np.sum(part[spec]['mass'][pis[distance_masks]])
                                hal[hal_prop_name][hal_i] = ut.halo_property.get_circular_velocity(
                                    mass, distance_max)

                        if 'vel.std' in prop_name:
                            weights = np.array(mass_weights)
                            if distance_max:
                                weights = mass_weights[distance_masks]

                            velocity2s = np.sum(
                                (part[spec]['velocity'][pis] - hal[spec + '.velocity'][hal_i]) ** 2,
                                1)
                            if distance_max:
                                velocity2s = velocity2s[distance_masks]

                            # compute average of velocity ** 2 (std)
                            #vel_std2 = np.average(velocity2s, weights=weights)
                            # compute median of velocity ** 2 (more stable to velocity_dif_max)
                            vel_std2 = ut.math.percentile_weighted(velocity2s, 50, weights)

                            hal[hal_prop_name][hal_i] = np.sqrt(vel_std2)

                    if spec == 'star':
                        if 'radius' in prop_name:
                            mass_percent = float(prop_name.split('.')[-1])

                            gal_prop = ut.particle.get_galaxy_properties(
                                part, spec, 'mass.percent', mass_percent,
                                hal.prop('radius', hal_i), 0.01, 'log',
                                hal.prop(spec + '.position', hal_i), part_indices=pis,
                                print_results=False)

                            hal[hal_prop_name][hal_i] = gal_prop['radius']

                        if 'form.time' in prop_name:
                            if ('.50' in prop_name or '.90' in prop_name or '.95' in prop_name or
                                    '.100' in prop_name):
                                percent = float(prop_name.split('.')[-1])
                                hal[hal_prop_name][hal_i] = ut.math.percentile_weighted(
                                    part[spec].prop('form.time', pis), percent, mass_weights)
                            elif '.dif.68' in prop_name:
                                val_16, val_84 = ut.math.percentile_weighted(
                                    part[spec].prop('form.time', pis), [16, 84], mass_weights)
                                hal[hal_prop_name][hal_i] = val_84 - val_16

                    if spec == 'gas':
                        if 'mass.neutral' in prop_name:
                            hal[hal_prop_name][hal_i] = part[spec].prop('mass.neutral', pis).sum()

            print()

        # assign 'star' properties to halos in dark-matter only simulation to compare
        if 'dark' in species and hal['star.mass.rockstar'].max() == 0:
            for prop_name in properties:
                hal['star.' + prop_name] = hal['dark.' + prop_name]

    def assign_lowres_mass(self, hal, part):
        '''
        Assign low-resolution dark matter (dark2) mass within R_halo.

        Parameters
        ----------
        hal : dict : catalog of halos at snapshot
        part : dict : catalog of particles at snapshot
        '''
        spec_name = 'dark2'
        mass_name = 'dark2.mass'

        # initialize halos to 100% low-res mass
        hal[mass_name] = np.zeros(hal.prop('mass').size, hal.prop('mass').dtype) + 1

        # some halos are completely low-res mass, yet do not have low-res particles near them (?!)
        # keep them as 100% low-res DM and skip henceforth
        hal_indices = ut.array.get_indices(hal.prop('mass.hires') > 0.1 * hal.prop('mass').min())

        hal[mass_name][hal_indices] = 0

        self.say('* assigning low-resolution {} mass to {} halos'.format(
                 spec_name, hal_indices.size))

        # cKDTree leads to seg fault, need to use KDTree
        # KDTree does not handle periodic boundaries, but should be ok for zoom-in
        KDTree = spatial.KDTree(part[spec_name]['position'])
        # , boxsize=part.info['box.length'])

        lowres_spec_mass_max = np.max(part[spec_name]['mass'])

        for hi in hal_indices:
            # convert to [kpc comoving]
            hal_radius = hal['radius'][hi] / hal.snapshot['scalefactor']

            # set maximum number of particles expected, via halo mass and particle mass
            particle_number = int(np.clip(hal.prop('mass', hi) / lowres_spec_mass_max, 1e4, 1e7))

            distances, indices = KDTree.query(
                hal['position'][hi], particle_number, distance_upper_bound=hal_radius)

            masks = (distances < hal_radius)
            if True in masks:
                hal[mass_name][hi] += np.sum(part[spec_name]['mass'][indices[masks]])

        print()

    def write_catalogs_with_species(
        self, species=['star'], snapshot_value_kind='redshift', snapshot_values=0,
        simulation_directory='../../', mass_limits=[1e6, Inf], vel_circ_max_limits=[4, Inf],
        thread_number=1):
        '''
        Read halo catalog and particles from snapshot, assign given particle species to halos,
        write to HDF5 file in halo catalog directory.
        By default, set up to run from within halo finder (rockstar) sub-directory of simulation.

        Parameters
        ----------
        species : str or list : name[s] of particle species to assign
        snapshot_value_kind : str : snapshot number kind: 'index', 'redshift', 'scalefactor'
        snapshot_values : int or float or list thereof :
            index[s] or redshifts[s] or scale-factor[s] of snapshot file[s]
        simulation_directory : str : directory of simulation
        mass_limits : list : min and max halo mass for assigning species particles
        vel_circ_max_limits : list : min and max halo vel.circ.max for assigning species particles
        thread_number : int : number of threads for parallelization
        '''
        if np.isscalar(species):
            species = [species]  # ensure is list

        if np.isscalar(snapshot_values):
            snapshot_values = [snapshot_values]  # ensure is list

        # read list of all snapshots
        self.Snapshot = ut.simulation.read_snapshot_times(simulation_directory)

        args_list = [
            (species, snapshot_value_kind, snapshot_value, simulation_directory, mass_limits,
             vel_circ_max_limits) for snapshot_value in snapshot_values]

        ut.io.run_in_parallel(
            self._write_catalog_with_species, args_list, thread_number=thread_number, verbose=True)

    def _write_catalog_with_species(
        self, species=['star'], snapshot_value_kind='redshift', snapshot_value=0,
        simulation_directory='../../', mass_limits=[1e6, Inf], vel_circ_max_limits=[4, Inf]):
        '''
        Read halo catalog and particles from snapshot, assign given particle species to halos,
        write species properties of those halos to HDF5 file.
        By default, set up to run from within rockstar halo directory of simulation.

        Parameters
        ----------
        species : str or list : name[s] of particle species to assign
        snapshot_value_kind : str : snapshot number kind: 'index', 'redshift', 'scalefactor'
        snapshot_value : int or float or list thereof :
            index[s] or redshifts[s] or scale-factor[s] of snapshot file[s]
        simulation_directory : str : directory of simulation
        mass_limits : list : min and max halo mass for assigning species particles
        vel_circ_max_limits : list : min and max halo vel.circ.max for assigning species particles
        '''
        from gizmo_analysis import gizmo_io

        if np.isscalar(species):
            species = [species]  # ensure is list

        if self.Snapshot is None:
            # read list of all snapshots
            self.Snapshot = ut.simulation.read_snapshot_times(simulation_directory)

        snapshot_index = self.Snapshot.parse_snapshot_values(snapshot_value_kind, snapshot_value)

        # assign as current directory (assume am in within halo sub-directory)
        current_directory = os.getcwd().split('/')
        rockstar_directory = current_directory[-2] + '/' + current_directory[-1]

        part = gizmo_io.Read.read_snapshots(
            species + ['dark2'], 'index', snapshot_index, simulation_directory)

        # read halo catalog
        hal = IO.read_catalogs(
            'index', snapshot_index, simulation_directory, rockstar_directory, file_kind='hdf5')

        # assign nearest neighboring halos
        #IO.assign_nearest_neighbor(hal, mass_limits=mass_limits)

        # assign contamination mass from low-resolution dark matter
        self.assign_lowres_mass(hal, part)

        # assign indices of particles
        self.assign_particle_indices(hal, part, species, mass_limits, vel_circ_max_limits)

        # assign galaxy-wide properties
        self.assign_particle_properties(hal, part, species)

        # write to HDF5 file
        self.io_species_hdf5(
            species, hal, None, simulation_directory + rockstar_directory, write=True)

    def fix_lowres_mass_catalogs(
        self, snapshot_indices='all', simulation_directory='../../', thread_number=1):
        '''
        Read halo catalog and particles from snapshot, re-assign low-res mass.

        Parameters
        ----------
        snapshot_indices : int or list thereof : snapshot index[s] or 'all'
        simulation_directory : str : directory of simulation
        thread_number : int : number of threads for parallelization
        '''
        IO = IOClass()

        # read list of all snapshots
        self.Snapshot = ut.simulation.read_snapshot_times(simulation_directory)

        if snapshot_indices is 'all' or snapshot_indices is None:
            # read all snapshots
            snapshot_indices = self.Snapshot['index']

        # assign as current directory (assume am in within halo sub-directory)
        current_directory = os.getcwd().split('/')
        rockstar_directory = current_directory[-2] + '/' + current_directory[-1]

        # get names of all halo species files to read
        _path_file_names, snapshot_indices = IO._get_catalog_file_names_values(
            simulation_directory + rockstar_directory, snapshot_indices, 'star')

        args_list = [(snapshot_index, simulation_directory) for snapshot_index in snapshot_indices]

        ut.io.run_in_parallel(
            self._fix_lowres_mass_catalog, args_list, thread_number=thread_number, verbose=True)

    def _fix_lowres_mass_catalog(self, snapshot_index=0, simulation_directory='../../'):
        '''
        Read halo catalog and particles from snapshot, re-assign low-res mass.

        Parameters
        ----------
        snapshot_index : int : snapshot index
        simulation_directory : str : directory of simulation
        '''
        from gizmo_analysis import gizmo_io

        # assign as current directory (assume am in within halo sub-directory)
        current_directory = os.getcwd().split('/')
        rockstar_directory = current_directory[-2] + '/' + current_directory[-1]

        # read halo catalog
        hal = IO.read_catalogs(
            'index', snapshot_index, simulation_directory, rockstar_directory, file_kind='hdf5')

        # read particles
        part = gizmo_io.Read.read_snapshots(
            'dark2', 'index', snapshot_index, simulation_directory, assign_host_coordinates=False,
            check_properties=False)

        # re-assign contamination mass from low-resolution dark matter
        self.assign_lowres_mass(hal, part)

        # write to HDF5 file
        self.io_species_hdf5(
            'star', hal, None, simulation_directory + rockstar_directory, write=True)

    def io_species_hdf5(
        self, species='star', hal=None, snapshot_index=None,
        rockstar_directory=ROCKSTAR_DIRECTORY, write=False, verbose=False):
        '''
        Read/write halo catalog with particle species properties to/from HDF5 file.
        If writing, write only species properties (not all halo properties).
        If reading, either assign species properties to input halo catalog or return new halo
        catalog with just particle species properties.

        Parameters
        ----------
        species : str or list thereof : particle species to read/write: 'star', 'gas', 'dark'
        hal : class : catalog of halos at snapshot
        snapshot_index : int : index of snapshot
        rockstar_directory : str :directory (full path) of rockstar halo files
        write : bool : whether to write file (instead of read)
        verbose : bool : whether to print diagnostics

        Returns
        -------
        hal : dictionary class : halo catalog with particle species properties
        '''
        # parse inputs
        file_path = ut.io.get_path(rockstar_directory) + self.catalog_hdf5_directory

        assert hal is not None or snapshot_index is not None
        if snapshot_index is None:
            snapshot_index = hal.snapshot['index']

        file_name = ''
        if 'star' in species:
            file_name += 'star_'
        if 'gas' in species:
            file_name += 'gas_'
        if 'dark' in species:
            file_name += 'dark_'
        file_name += '{:03d}'.format(snapshot_index)

        path_file_name = file_path + file_name

        if write:
            # write to file
            file_path = ut.io.get_path(file_path, create_path=True)

            # create temporary catalog to store species properties
            hal_spec = HaloDictionaryClass()
            # add species properties
            for prop_name in hal:
                if 'star.' in prop_name:
                    hal_spec[prop_name] = hal[prop_name]
            # add mass fraction from low-resolution DM
            hal_spec['dark2.mass'] = hal['dark2.mass']
            # add halo catalog id
            hal_spec[self.catalog_id_name] = hal[self.catalog_id_name]

            properties_add = []
            for prop_name in hal.info:
                if not isinstance(hal.info[prop_name], str):
                    hal_spec['info:' + prop_name] = np.array(hal.info[prop_name])
                    properties_add.append('info:' + prop_name)

            for prop_name in hal.snapshot:
                hal_spec['snapshot:' + prop_name] = np.array(hal.snapshot[prop_name])
                properties_add.append('snapshot:' + prop_name)

            for prop_name in hal.Cosmology:
                hal_spec['cosmology:' + prop_name] = np.array(hal.Cosmology[prop_name])
                properties_add.append('cosmology:' + prop_name)

            ut.io.file_hdf5(path_file_name, hal_spec)

        else:
            if hal is None:
                create_catalog = True
                # create new dictionary class to store halo catalog
                hal = HaloDictionaryClass()
            else:
                create_catalog = False
            header = {}

            try:
                # try to read from file
                hal_in = ut.io.file_hdf5(path_file_name, verbose=False)

                for prop_name in hal_in:
                    if 'info:' in prop_name:
                        hal_prop = prop_name.split(':')[-1]
                        header[hal_prop] = float(hal_in[prop_name])
                    elif 'snapshot:' in prop_name:
                        hal_prop = prop_name.split(':')[-1]
                        if hal_prop == 'index':
                            header[hal_prop] = int(hal_in[prop_name])
                        else:
                            header[hal_prop] = float(hal_in[prop_name])
                    elif 'cosmology:' in prop_name:
                        hal_prop = prop_name.split(':')[-1]
                        header[hal_prop] = float(hal_in[prop_name])
                    else:
                        if prop_name == self.catalog_id_name and prop_name in hal:
                            # sanity check - make sure halo ids match
                            if hal[prop_name].size != hal_in[prop_name].size:
                                words = '{} has catalog of different size than input halo catalog'
                                raise ValueError(words.format(path_file_name))
                            if np.max(hal[prop_name] != hal_in[prop_name]):
                                words = '{} has mis-matched ids versus input halo catalog'
                                raise ValueError(words.format(path_file_name))
                        else:
                            hal[prop_name] = hal_in[prop_name]

                            # backward compatability with old star particle files
                            if prop_name == 'lowres.mass.frac':
                                if 'mass' in hal:
                                    # convert low-res mass fraction to low-res mass
                                    hal['dark2.mass'] = 0 * hal['mass']
                                    masks = np.isfinite(hal['lowres.mass.frac'])
                                    hal['dark2.mass'][masks] *= hal['lowres.mass.frac'][masks]
                                del(hal['lowres.mass.frac'])

                self.say('* read {} halos with {} particles from:  {}.hdf5'.format(
                         hal[self.catalog_id_name].size, ut.array.scalarize(species),
                         path_file_name.strip('./')))

                if create_catalog:
                    # add simulation information
                    hal.snapshot = {
                        'index': header['index'],
                        'scalefactor': header['scalefactor'],
                        'redshift': header['redshift'],
                        'time': header['time'],
                        'time.lookback': header['time.lookback'],
                        'time.hubble': header['time.hubble'],
                    }

                    hal.info = {
                        'dark.particle.mass': header['dark.particle.mass'],
                        'gas.particle.mass': header['gas.particle.mass'],
                        'box.length/h': header['box.length/h'],
                        'box.length': header['box.length'],
                        'catalog.kind': 'halo.catalog',
                        'file.kind': 'hdf5',
                        'baryonic': False,
                        'simulation.name': '',
                    }
                    if species in ['star', 'gas']:
                        hal.info['baryonic'] = True

            except OSError:
                self.say('! no halo species file = {}'.format(path_file_name.strip('./')), verbose)

            return hal


Particle = ParticleClass()


#===================================================================================================
# output
#===================================================================================================
def write_halos_text(hal, hal_indices=None, part=None, distance_max=None, directory='.'):
    '''
    Write properties of input list of galaxies.

    Parameters
    ----------
    hal : dict : catalog of halos at snapshot
    hal_indices : array-like : indices of halos to write
    part : dict : catalog of particles at snapshot
    distance_max : float : max distance (radius) to select particles
        if none, use only particles associated with halo
    '''
    species_name = 'star'  # write galaxy properties of this particle species

    if np.isscalar(hal_indices):
        hal_indices = [hal_indices]

    directory = ut.io.get_path(directory)

    for hi in hal_indices:
        file_name = 'halo_{}.txt'.format(hal['id'][hi])

        path_file_name = ut.io.get_path(directory) + file_name

        with open(path_file_name, 'w') as file_out:
            Write = ut.io.WriteClass(file_out)

            Write.write('# halo id = {}'.format(hal['id'][hi]), print_stdout=True)
            Write.write('# star mass = {:.3e}'.format(hal.prop('star.mass', hi)),
                        print_stdout=True)
            Write.write('# star particle number = {:d}'.format(hal.prop('star.number', hi)),
                        print_stdout=True)
            Write.write('# velocity dispersion: star = {:.1f}, halo = {:.1f} km/s'.format(
                        hal.prop('star.vel.std')[hi], hal.prop('vel.std', hi)), print_stdout=True)
            #Write.write('# star radius.50 = {:.2f} kpc'.format(hal.prop('star.radius.50', hi)))
            Write.write(
                '# formation lookback-time: 50% = {:.3f}, 95% = {:.3f}, 100% = {:.3f} Gyr'.format(
                    hal.prop('star.form.time.50.lookback', hi),
                    hal.prop('star.form.time.95.lookback', hi),
                    hal.prop('star.form.time.100.lookback', hi)), print_stdout=True
            )
            #Write.write('# star metallicity: total = {:.3f}, [Fe/H] = {:.3f}'.format(
            #            hal.prop('star.metallicity.metals', hi),
            #            hal.prop('star.metallicity.iron', hi)))
            Write.write('# distance from nearest host = {:.1f} kpc'.format(
                hal.prop('host.near.distance.total', hi)), print_stdout=True)
            Write.write('# current age of Universe = {:.3f} Gyr'.format(hal.snapshot['time']),
                        print_stdout=True)

            #Write.write('position = {:.2f}, {:.2f}, {:.2f} kpc'.format(
            #            hal.prop('star.position')[hi, 0], hal.prop('star.position')[hi, 1],
            #            hal.prop('star.position')[hi, 2]))

            if distance_max:
                part_indices = ut.particle.get_indices_within_coordinates(
                    part, species_name, [0, distance_max], hal[species_name + '.position'][hi])
            else:
                part_indices = hal[species_name + '.indices'][hi]

            orb = ut.particle.get_orbit_dictionary(
                part, species_name, part_indices, part.host_positions[0], part.host_velocities[0],
                return_single=False)

            Write.write('# columns:')
            Write.write('#  id mass[M_sun] formation-lookback-time[Gyr] ' +
                        'mass-fraction(He, C, N, O, Ne, Mg, Si, S, Ca, Fe) ' +
                        'distance(x, y, z, total)[kpc] velocity-radial[km/s]')

            for pii, pi in enumerate(part_indices):
                if species_name == 'star':
                    string = ('{} {:.3e} {:.3f} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} ' +
                              '{:.3e} {:.3e} {:.3f} {:.3f} {:.3f} {:.3f} {:.1f}')
                    Write.write(
                        string.format(
                            part[species_name].prop('id', pi),
                            part[species_name].prop('mass', pi),
                            part[species_name].prop('age', pi),
                            part[species_name].prop('massfraction')[pi, 1],
                            part[species_name].prop('massfraction')[pi, 2],
                            part[species_name].prop('massfraction')[pi, 3],
                            part[species_name].prop('massfraction')[pi, 4],
                            part[species_name].prop('massfraction')[pi, 5],
                            part[species_name].prop('massfraction')[pi, 6],
                            part[species_name].prop('massfraction')[pi, 7],
                            part[species_name].prop('massfraction')[pi, 8],
                            part[species_name].prop('massfraction')[pi, 9],
                            orb['distance'][pii, 0], orb['distance'][pii, 1],
                            orb['distance'][pii, 2],
                            orb['distance.total'][pii], orb['velocity.rad'][pii],
                        )
                    )


#===================================================================================================
# run from command line
#===================================================================================================
if __name__ == '__main__':
    if len(os.sys.argv) <= 1:
        raise OSError('specify function: snapshots, hdf5')

    function_kind = str(os.sys.argv[1])
    assert ('snapshots' in function_kind or 'hdf5' in function_kind)

    # assume am in rockstar sub-directory
    current_directory = os.getcwd().split('/')
    rockstar_directory = current_directory[-2] + '/' + current_directory[-1]

    if 'snapshots' in function_kind:
        snapshot_selection = 'all'
        if len(os.sys.argv) == 3:
            snapshot_selection = str(os.sys.argv[2])
        IO.write_snapshot_indices(snapshot_selection, '../../', rockstar_directory)

    elif 'hdf5' in function_kind:
        IO.rewrite_as_hdf5('../../', rockstar_directory)
