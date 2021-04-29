'''
Select halos from Rockstar catalog (for subsequent zoom-in).

@author: Andrew Wetzel <arwetzel@gmail.com>

Units: unless otherwise noted, all quantities are in (combinations of):
    mass [M_sun]
    position [kpc comoving]
    distance, radius [kpc physical]
    velocity [km / s]
    time [Gyr]
'''

# system ----
from __future__ import absolute_import, division, print_function
import copy
import numpy as np
from numpy import Inf
# local ----
import utilities as ut


#===================================================================================================
# selection
#===================================================================================================
class SelectClass(ut.io.SayClass):
    '''
    Select halos (for zoom-in).
    '''

    def __init__(self):
        '''
        Set default parameters for halo selection.
        '''
        self.isolate_param = {
            'distance/Rneig.limits': [4.0, Inf],
            'distance/Rself.limits': [5.0, Inf],
            'distance.limits': [],
            'neig.mass.frac.limits': [0.33, Inf],
            'neig.mass.limits': [],
        }

        self.satellite_param = {
            'bin.kind': 'error',
            'mass.kind': 'vel.circ.max',
            'number': 1,

            #'mass.limits': [92, 10],  # observed
            #'distance.limits': [51, 2],  # observed
            #'velocity.total.limits': [321, 25],  # observed
            #'velocity.rad.limits': [64, 7],  # observed
            #'velocity.tan.limits': [314, 24],  # observed

            'mass.limits': [92, 5 * 2.5],
            'distance.limits': [51, 2 * 20],
            'velocity.rad.limits': [64, 7 * 2.5],
            'velocity.tan.limits': [314, 24 * 2.5],
            'velocity.total.limits': [321, 25 * 2.5],  # observed
        }

    def select(
        self, hal, hal_mass_kind='mass', hal_mass_limits=[1e12, 2e12],
        contaminate_mass_frac_max=0.01, hal_indices=None,
        isolate_param=None, satellite_param=None,
        print_properties=True):
        '''
        Impose various cuts on halo catalog to select halos.

        Parameters
        ----------
        hal : dictionary class : halo catalog
        hal_mass_kind : str : mass kind to select halos
        hal_mass_limits : list : min and max limits for hal_mass_kind
        contaminate_mass_frac_max : float : maximim low-res dark-matter mass fraction (if zoom-in)
        hal_indices : array : prior selection indices of halos
        isolate_param : dict : parameters for selecting isolated halos
        satellite_param : dict : parameters for selecting halos with satellites
        print_properties : bool : whether to print properties of selected halos

        Returns
        -------
        hindices : array : indices of selected halos
        '''
        neighbor_distance_max = 20000  # [kpc]
        neighbor_number_max = 1000

        hindices = hal_indices

        # select via halo mass
        if hal_mass_limits and len(hal_mass_limits):
            hindices = ut.array.get_indices(hal[hal_mass_kind], hal_mass_limits, hindices)
            self.say('* {} halos are within {} limits = [{}]'.format(
                hindices.size, hal_mass_kind, ut.io.get_string_from_numbers(hal_mass_limits, 2)))

        # select via purity
        if (contaminate_mass_frac_max and 'mass.lowres' in hal and
                hal.prop('lowres.mass.frac').max() > 0):
            hindices = ut.array.get_indices(
                hal.prop('lowres.mass.frac'), [0, contaminate_mass_frac_max], hindices)
            self.say('* {} have mass contamination < {:.2f}%'.format(
                     hindices.size, contaminate_mass_frac_max))

        NearestNeighbor = ut.catalog.NearestNeighborClass()

        # select neighbors above self mass
        NearestNeighbor.assign_to_self(
            hal, hal_mass_kind, hal_mass_limits, [1, Inf], None, neighbor_distance_max, 'Rneig',
            neighbor_number_max, print_diagnostics=False)
        nearest = NearestNeighbor.nearest

        # select central halos
        masks = (nearest['distance/Rneig'] > 1)
        hindices = hindices[masks]
        self.say('* {} are a central'.format(hindices.size))

        # select via isolation
        if isolate_param:
            if isolate_param == 'default':
                isolate_param = copy.copy(self.isolate_param)  # use stored defaults

            # select isolated, defined wrt neighbor's R_halo
            if isolate_param['distance/Rneig.limits']:
                if (min(isolate_param['distance/Rneig.limits']) > 1 or
                        max(isolate_param['distance/Rneig.limits']) < Inf):
                    nearest['distance/Rneig'] = nearest['distance/Rneig'][masks]
                    his = ut.array.get_indices(
                        nearest['distance/Rneig'], isolate_param['distance/Rneig.limits'])
                    self.say('* {} ({:.1f}%)  have nearest more-massive at d/Rneig = {}'.format(
                             his.size, 100 * his.size / hindices.size,
                             isolate_param['distance/Rneig.limits']))
                    hindices = hindices[his]

            # select isolated, defined wrt self's R_halo
            if isolate_param['distance/Rself.limits']:
                # get neighbors above self mass * neig.mass.frac.limits
                NearestNeighbor.assign_to_self(
                    hal, hal_mass_kind, hal_mass_limits, isolate_param['neig.mass.frac.limits'],
                    None, neighbor_distance_max, 'Rself', neighbor_number_max, hindices,
                    print_diagnostics=False)
                nearest = NearestNeighbor.nearest
                his = ut.array.get_indices(
                    nearest['distance/Rself'], isolate_param['distance/Rself.limits'])
                self.say('* {} ({:.1f}%) have nearest with M/Mself = {} at d/Rself = {}'.format(
                         his.size, 100 * his.size / hindices.size,
                         isolate_param['neig.mass.frac.limits'],
                         isolate_param['distance/Rself.limits']))
                hindices = hindices[his]

        # select via having satellite[s]
        if satellite_param:
            if satellite_param == 'default':
                satellite_param = copy.copy(self.satellite_param)  # use stored defaults

            # convert value + uncertainty into min + max limits
            for prop in satellite_param:
                if 'limits' in prop:
                    satellite_param[prop] = np.array(ut.binning.get_bin_limits(
                        satellite_param[prop], satellite_param['bin.kind']))

            # select satellites within mass limits
            if ('mass.kind' in satellite_param and len(satellite_param['mass.kind']) and
                    'mass.limits' in satellite_param and len(satellite_param['mass.limits'])):
                sat_hindices = ut.array.get_indices(
                    hal[satellite_param['mass.kind']], satellite_param['mass.limits'])

            # select satellites := within host halo radius
            sat_distance_limits = [1, np.max(hal['radius'][hindices])]
            self.say('selecting satellites within distance limits = [{}]'.format(
                ut.io.get_string_from_numbers(sat_distance_limits, 1)))

            Neighbor = ut.neighbor.NeighborClass()
            sat_distancess, sat_indicess = Neighbor.get_neighbors(
                hal['position'][hindices], hal['position'][sat_hindices], 100,
                sat_distance_limits, hal.info['box.length'], hal.snapshot['scalefactor'],
                neig_ids=sat_hindices, print_diagnostics=False)

            # select halos with input number of satellites
            his_has_sat = []
            for hi, sat_distances in enumerate(sat_distancess):
                if len(sat_distances) == satellite_param['number']:
                    his_has_sat.append(hi)
            his_has_sat = np.array(his_has_sat)
            hindices_has_sat = hindices[his_has_sat]
            self.say('* {} ({:.1f}%) have {} satellite[s] with {} = {}'.format(
                     hindices_has_sat.size, 100 * hindices_has_sat.size / hindices.size,
                     satellite_param['number'], satellite_param['mass.kind'],
                     satellite_param['mass.limits']))

            # convert to 1-D arrays, one entry per satellite (hosts may be repeated)
            hindices_has_sat = []
            sat_hindices = []
            sat_distances = []
            for hi in his_has_sat:
                for si, sat_hindex in enumerate(sat_indicess[hi]):
                    hindices_has_sat.append(hindices[hi])
                    sat_hindices.append(sat_hindex)
                    sat_distances.append(sat_distancess[hi][si])
            hindices_has_sat = np.array(hindices_has_sat)
            sat_hindices = np.array(sat_hindices)
            sat_distances = np.array(sat_distances)

            # select via satellite distance
            if 'distance.limits' in satellite_param and len(satellite_param['distance.limits']):
                sis = ut.array.get_indices(sat_distances, satellite_param['distance.limits'])
                self.say('{} ({:.1f}%) satellite[s] at d = {}'.format(
                    sis.size, 100 * sis.size / sat_hindices.size,
                    satellite_param['distance.limits']))
                hindices_has_sat = hindices_has_sat[sis]
                sat_hindices = sat_hindices[sis]

            # select via satellite total velocity
            if ('velocity.total.limits' in satellite_param and
                    len(satellite_param['velocity.total.limits'])):
                sat_velocities_tot = ut.coordinate.get_velocity_differences(
                    hal['velocity'][hindices_has_sat], hal['velocity'][sat_hindices],
                    hal['position'][hindices_has_sat], hal['position'][sat_hindices],
                    hal.info['box.length'], hal.snapshot['scalefactor'],
                    hal.snapshot['time.hubble'], total_velocity=True)
                sis = ut.array.get_indices(
                    sat_velocities_tot, satellite_param['velocity.total.limits'])
                self.say('{} ({:.1f}%) satellites with velocity.total = {}'.format(
                    sis.size, 100 * sis.size / sat_hindices.size,
                    satellite_param['velocity.total.limits']))
                hindices_has_sat = hindices_has_sat[sis]
                sat_hindices = sat_hindices[sis]

            # select via satellite radial and/or tangential velocity
            if 'velocity.rad.limits' in satellite_param or 'velocity.tan.limits' in satellite_param:
                distance_vectors = ut.coordinate.get_distances(
                    hal['position'][hindices_has_sat], hal['position'][sat_hindices],
                    hal.info['box.length'])
                velocity_vectors = ut.coordinate.get_velocity_differences(
                    hal['velocity'][hindices_has_sat], hal['velocity'][sat_hindices],
                    hal['position'][hindices_has_sat], hal['position'][sat_hindices],
                    hal.info['box.length'], hal.snapshot['scalefactor'],
                    hal.snapshot['time.hubble'])

                orb = ut.orbit.get_orbit_dictionary(distance_vectors, velocity_vectors)

                if ('velocity.rad.limits' in satellite_param and
                        len(satellite_param['velocity.rad.limits'])):
                    sis = ut.array.get_indices(
                        orb['velocity.rad'], satellite_param['velocity.rad.limits'])
                    self.say('{} ({:.1f}%) satellites with velocity.rad = {}'.format(
                        sis.size, 100 * sis.size / sat_hindices.size,
                        satellite_param['velocity.rad.limits']))
                    hindices_has_sat = hindices_has_sat[sis]
                    sat_hindices = sat_hindices[sis]
                    for prop in orb:
                        orb[prop] = orb[prop][sis]

                if ('velocity.tan.limits' in satellite_param and
                        len(satellite_param['velocity.tan.limits'])):
                    sis = ut.array.get_indices(
                        orb['velocity.tan'], satellite_param['velocity.tan.limits'])
                    self.say('{} ({:.1f}%) satellites with velocity.tan = {}'.format(
                        sis.size, 100 * sis.size / sat_hindices.size,
                        satellite_param['velocity.tan.limits']))
                    hindices_has_sat = hindices_has_sat[sis]
                    sat_hindices = sat_hindices[sis]
                    for prop in orb:
                        orb[prop] = orb[prop][sis]

            hindices = hindices_has_sat

        if print_properties:
            self.say(' ')
            for hi, hindex in enumerate(hindices):
                self.say(
                    'halo: index = {}, id = {}, M = {:.2e}, R = {:.1f}, Vmax = {:.1f}'.format(
                        hindex, hal['id'][hindex], hal[hal_mass_kind][hindex],
                        hal['radius'][hindex], hal['vel.circ.max'][hindex]))
                self.say('      position = [{:.3f}, {:.3f}, {:.3f}]'.format(
                         hal['position'][hindex, 0], hal['position'][hindex, 1],
                         hal['position'][hindex, 2]))

                if satellite_param:
                    sat_hindex = sat_hindices[hi]
                    words = 'sat: index = {}, id = {}, M = {:.2e}, Mbound = {:.2e}, Vmax = {:.1f}'
                    self.say(words.format(
                        sat_hindex, hal['id'][sat_hindex], hal['mass'][sat_hindex],
                        hal['mass.bound'][sat_hindex], hal['vel.circ.max'][sat_hindex]))
                    self.say(
                        '     d = {:.1f}, vel tot = {:.1f}, rad = {:.1f}, tan = {:.1f}\n'.format(
                            orb['distance.total'][hi], orb['velocity.total'][hi],
                            orb['velocity.rad'][hi], orb['velocity.tan'][hi]))

        return hindices


Select = SelectClass()
