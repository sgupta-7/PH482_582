'''
Utility functions to analyze particle data.

@author: Andrew Wetzel <arwetzel@gmail.com>

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
from numpy import Inf
# local ----
from . import basic as ut
from . import halo_property
from . import orbit
from . import catalog


#===================================================================================================
# utilities - parsing input arguments
#===================================================================================================
def parse_species(part, species):
    '''
    Parse input list of species to ensure all are in catalog.

    Parameters
    ----------
    part : dict : catalog of particles
    species : str or list : name[s] of particle species to analyze

    Returns
    -------
    species : list : name[s] of particle species
    '''
    Say = ut.io.SayClass(parse_species)

    if np.isscalar(species):
        species = [species]
    if species == ['all'] or species == ['total']:
        species = list(part.keys())
    elif species == ['baryon']:
        species = ['gas', 'star']

    for spec in list(species):
        if spec not in part:
            species.remove(spec)
            Say.say('! {} not in particle catalog'.format(spec))

    return species


def parse_indices(part_spec, part_indices):
    '''
    Parse input list of particle indices.
    If none, generate via arange.

    Parameters
    ----------
    part_spec : dict : catalog of particles of given species
    part_indices : array-like : indices of particles

    Returns
    -------
    part_indices : array : indices of particles
    '''
    if part_indices is None or not len(part_indices):
        if 'position' in part_spec:
            part_indices = ut.array.get_arange(part_spec['position'].shape[0])
        elif 'id' in part_spec:
            part_indices = ut.array.get_arange(part_spec['id'].size)
        elif 'mass' in part_spec:
            part_indices = ut.array.get_arange(part_spec['mass'].size)

    return part_indices


def parse_property(parts_or_species, property_name, property_values=None, single_host=True):
    '''
    Get property values, either input or stored in particle catalog.
    List-ify as necessary to match input particle catalog.

    Parameters
    ----------
    parts_or_species : dict or string or list thereof :
        catalog[s] of particles or string[s] of species
    property_name : str : options: 'center_position', 'center_velocity', 'indices'
    property_values : float/array or list thereof : property values to assign
    single_host : bool : use only the primary host (if not input any property_values)

    Returns
    -------
    property_values : float or list
    '''

    def parse_property_single(part_or_spec, property_name, property_values, single_host):
        if property_name in ['center_position', 'center_velocity']:
            if property_values is None or not len(property_values):
                if property_name == 'center_position':
                    property_values = part_or_spec.host_positions
                elif property_name == 'center_velocity':
                    # default to the primary host
                    property_values = part_or_spec.host_velocities

                if property_values is None or not len(property_values):
                    raise ValueError('no input {} and no {} in input catalog'.format(
                                     property_name, property_name))

                if single_host:
                    property_values = property_values[0]  # use omly the primary host

        if isinstance(property_values, list):
            raise ValueError('input list of {}s but input single catalog'.format(property_name))

        return property_values

    assert property_name in ['center_position', 'center_velocity', 'indices']

    if isinstance(parts_or_species, list):
        # input list of particle catalogs
        if (property_values is None or not len(property_values) or
                not isinstance(property_values, list)):
            property_values = [property_values for _ in parts_or_species]

        if len(property_values) != len(parts_or_species):
            raise ValueError('number of input {}s not match number of input catalogs'.format(
                             property_name))

        for i, part_or_spec in enumerate(parts_or_species):
            property_values[i] = parse_property_single(
                part_or_spec, property_name, property_values[i], single_host)
    else:
        # input single particle catalog
        property_values = parse_property_single(
            parts_or_species, property_name, property_values, single_host)

    return property_values


#===================================================================================================
# id <-> index conversion
#===================================================================================================
def assign_id_to_index(
    part, species=['all'], id_name='id', id_min=0, store_as_dict=False, print_diagnostic=True):
    '''
    Assign, to particle dictionary, arrays that points from object id to species kind and index in
    species array.
    This is useful for analyses multi-species catalogs with intermixed ids.
    Do not assign pointers for ids below id_min.

    Parameters
    ----------
    part : dict : catalog of particles of various species
    species : str or list : name[s] of species to use: 'all' = use all in particle dictionary
    id_name : str : key name for particle id
    id_min : int : minimum id in catalog
    store_as_dict : bool : whether to store id-to-index pointer as dict instead of array
    print_diagnostic : bool : whether to print diagnostic information
    '''
    Say = ut.io.SayClass(assign_id_to_index)

    # get list of species that have valid id key
    species = parse_species(part, species)
    for spec in species:
        assert id_name in part[spec]

    # get list of all ids
    ids_all = []
    for spec in species:
        ids_all.extend(part[spec][id_name])
    ids_all = np.array(ids_all, dtype=part[spec][id_name].dtype)

    if print_diagnostic:
        # check if duplicate ids within species
        for spec in species:
            masks = (part[spec][id_name] >= id_min)
            total_number = np.sum(masks)
            unique_number = np.unique(part[spec][id_name][masks]).size
            if total_number != unique_number:
                Say.say('species {} has {} ids that are repeated'.format(
                        spec, total_number - unique_number))

        # check if duplicate ids across species
        if len(species) > 1:
            masks = (ids_all >= id_min)
            total_number = np.sum(masks)
            unique_number = np.unique(ids_all[masks]).size
            if total_number != unique_number:
                Say.say('across all species, {} ids are repeated'.format(
                        total_number - unique_number))

        Say.say('maximum id = {}'.format(ids_all.max()))

    part.id_to_index = {}

    if store_as_dict:
        # store pointers as a dictionary
        # store overall dictionary (across all species) and dictionary within each species
        for spec in species:
            part[spec].id_to_index = {}
            for part_i, part_id in enumerate(part[spec][id_name]):
                if part_id in part.id_to_index:
                    # redundant ids - add to existing entry as list
                    if isinstance(part.id_to_index[part_id], tuple):
                        part.id_to_index[part_id] = [part.id_to_index[part_id]]
                    part.id_to_index[part_id].append((spec, part_i))

                    if part_id in part[spec].id_to_index:
                        if np.isscalar(part[spec].id_to_index[part_id]):
                            part[spec].id_to_index[part_id] = [part[spec].id_to_index[part_id]]
                        part[spec].id_to_index[part_id].append(part_i)

                else:
                    # new id - add as new entry
                    part.id_to_index[part_id] = (spec, part_i)
                    part[spec].id_to_index[part_id] = part_i

            # convert lists to arrays
            dtype = part[spec][id_name].dtype
            for part_id in part[spec].id_to_index:
                if isinstance(part[spec].id_to_index[part_id], list):
                    part[spec].id_to_index[part_id] = np.array(
                        part[spec].id_to_index[part_id], dtype=dtype)

    else:
        # store pointers as arrays
        part.id_to_index['species'] = np.zeros(ids_all.max() + 1, dtype='|S6')
        dtype = ut.array.parse_data_type(ids_all.max() + 1)
        part.id_to_index['index'] = ut.array.get_array_null(ids_all.max() + 1, dtype=dtype)

        for spec in species:
            masks = (part[spec][id_name] >= id_min)
            part.id_to_index['species'][part[spec][id_name][masks]] = spec
            part.id_to_index['index'][part[spec][id_name][masks]] = ut.array.get_arange(
                part[spec][id_name], dtype=dtype)[masks]


#===================================================================================================
# position, velocity
#===================================================================================================
def get_center_positions(
    part, species=['star', 'dark', 'gas'], part_indicess=None, method='center-of-mass',
    center_number=1, exclusion_distance=200, center_positions=None, distance_max=Inf,
    compare_centers=False, return_array=True):
    '''
    Get position[s] of center of mass [kpc comoving] using iterative zoom-in on input species.

    Parameters
    ----------
    part : dict : dictionary of particles
    species : str or list : name[s] of species to use: 'all' = use all in particle dictionary
    part_indicess : array or list of arrays : indices of particle to use to define center
        use this to include only particles that you know are relevant
    method : str : method of centering: 'center-of-mass', 'potential'
    center_number : int : number of centers to compute
    exclusion_distance : float :
        radius around previous center to cut before finding next center [kpc comoving]
    center_position : array-like : initial center position[s] to use
    distance_max : float : maximum radius to consider initially
    compare_centers : bool : whether to run sanity check to compare centers via zoom v potential
    return_array : bool :
        whether to return single array instead of array of arrays, if center_number = 1

    Returns
    -------
    center_positions : array or array of arrays: position[s] of center[s] [kpc comoving]
    '''
    Say = ut.io.SayClass(get_center_positions)

    assert method in ['center-of-mass', 'potential']

    species = parse_species(part, species)
    part_indicess = parse_property(species, 'indices', part_indicess)
    if center_positions is None or np.ndim(center_positions) == 1:
        # list-ify center_positions
        center_positions = [center_positions for _ in range(center_number)]
    if np.shape(center_positions)[0] != center_number:
        raise ValueError('! input center_positions = {} but also input center_number = {}'.format(
            center_positions, center_number))

    if method == 'potential':
        if len(species) > 1:
            Say.say('! using only first species = {} for centering via potential'.format(
                species[0]))

        if 'potential' not in part[species[0]]:
            Say.say('! {} does not have potential, using center-of-mass zoom instead'.format(
                species[0]))
            method = 'center-of-mass'

    if method == 'potential':
        # use single (first) species
        spec_i = 0
        spec_name = species[spec_i]
        part_indices = parse_indices(spec_name, part_indicess[spec_i])
        for center_i, center_position in enumerate(center_positions):
            if center_i > 0:
                # cull out particles near previous center
                distances = get_distances_wrt_center(
                    part, spec_name, part_indices, center_positions[center_i - 1],
                    total_distance=True, return_array=True)
                # exclusion distance in [kpc comoving]
                part_indices = part_indices[
                    distances > (exclusion_distance * part.info['scalefactor'])]

            if center_position is not None and distance_max > 0 and distance_max < Inf:
                # impose distance cut around input center
                part_indices = get_indices_within_coordinates(
                    part, spec_name, [0, distance_max], center_position, part_indicess=part_indices,
                    return_array=True)

            part_index = np.nanargmin(part[spec_name]['potential'][part_indices])
            center_positions[center_i] = part[spec_name]['position'][part_index]
    else:
        for spec_i, spec_name in enumerate(species):
            part_indices = parse_indices(part[spec_name], part_indicess[spec_i])

            if spec_i == 0:
                positions = part[spec_name]['position'][part_indices]
                masses = part[spec_name]['mass'][part_indices]
            else:
                positions = np.concatenate(
                    [positions, part[spec_name]['position'][part_indices]])
                masses = np.concatenate([masses, part[spec_name]['mass'][part_indices]])

        for center_i, center_position in enumerate(center_positions):
            if center_i > 0:
                # remove particles near previous center
                distances = ut.coordinate.get_distances(
                    positions, center_positions[center_i - 1], part.info['box.length'],
                    part.snapshot['scalefactor'], total_distance=True)  # [kpc physical]
                masks = (distances > (exclusion_distance * part.info['scalefactor']))
                positions = positions[masks]
                masses = masses[masks]

            center_positions[center_i] = ut.coordinate.get_center_position_zoom(
                positions, masses, part.info['box.length'], center_position=center_position,
                distance_max=distance_max)

    center_positions = np.array(center_positions)

    if compare_centers:
        position_dif_max = 1  # [kpc comoving]

        if 'potential' not in part[species[0]]:
            Say.say('! {} not have potential, cannot compare against zoom center-of-mass'.format(
                    species[0]))
            return center_positions

        if method == 'potential':
            method_other = 'center-of-mass'
        else:
            method_other = 'potential'

        center_positions_other = get_center_positions(
            part, species, part_indicess, method_other, center_number, exclusion_distance,
            center_positions, distance_max, compare_centers=False, return_array=False)

        position_difs = np.abs(center_positions - center_positions_other)

        for pi, position_dif in enumerate(position_difs):
            if np.max(position_dif) > position_dif_max:
                Say.say('! offset center positions')
                Say.say('center position via {}: '.format(method), end='')
                ut.io.print_array(center_positions[pi], '{:.3f}')
                Say.say('center position via {}: '.format(method_other), end='')
                ut.io.print_array(center_positions_other[pi], '{:.3f}')
                Say.say('position difference: ', end='')
                ut.io.print_array(position_dif, '{:.3f}')

    if return_array and center_number == 1:
        center_positions = center_positions[0]

    return center_positions


def get_center_velocities(
    part, species_name='star', part_indices=None, distance_max=15, center_positions=None,
    return_array=True):
    '''
    Get velocity[s] [km / s] of center of mass of input species.

    Parameters
    ----------
    part : dict : dictionary of particles
    species_name : str : name of particle species to use
    part_indices : array : indices of particle to use to define center
        use this to exclude particles that you know are not relevant
    distance_max : float : maximum radius to consider [kpc physical]
    center_positions : array or list of arrays: center position[s] [kpc comoving]
        if None, will use default center position[s] in catalog
    return_array : bool :
        whether to return single array instead of array of arrays, if input single center position

    Returns
    -------
    center_velocities : array or array of arrays : velocity[s] of center of mass [km / s]
    '''
    center_positions = parse_property(part, 'center_position', center_positions, single_host=False)
    part_indices = parse_indices(part[species_name], part_indices)

    distance_max /= part.snapshot['scalefactor']  # convert to [kpc comoving] to match positions

    center_velocities = np.zeros(center_positions.shape, part[species_name]['velocity'].dtype)

    for center_i, center_position in enumerate(center_positions):
        center_velocities[center_i] = ut.coordinate.get_center_velocity(
            part[species_name]['velocity'][part_indices],
            part[species_name]['mass'][part_indices],
            part[species_name]['position'][part_indices],
            center_position, distance_max, part.info['box.length'])

    if return_array and len(center_velocities) == 1:
        center_velocities = center_velocities[0]

    return center_velocities


def get_distances_wrt_center(
    part, species=['star'], part_indicess=None, center_position=None, rotation=None,
    coordinate_system='cartesian', total_distance=False, return_array=True):
    '''
    Get distances (scalar or vector) between input particles and center_position (input or stored
    in particle catalog).

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species : str or list : name[s] of particle species to compute
    part_indicess : array or list : indices[s] of particles to compute, one array per input species
    center_position : array : position of center [kpc comoving]
        if None, will use default center position in particle catalog
    rotation : bool or array : whether to rotate particles
        two options:
        (a) if input array of eigen-vectors, will define rotation axes for all species
        (b) if True, will rotate to align with principal axes defined by input species
    coordinate_system : str : which coordinates to get distances in:
        'cartesian' (default), 'cylindrical', 'spherical'
    total_distance : bool : whether to compute total/scalar distance
    return_array : bool : whether to return single array instead of dict if input single species

    Returns
    -------
    dist : array (object number x dimension number) or dict thereof : [kpc physical]
        3-D distance vectors aligned with default x,y,z axes OR
        3-D distance vectors aligned with major, medium, minor axis OR
        2-D distance vectors along major axes and along minor axis OR
        1-D scalar distances
    OR
    dictionary of above for each species
    '''
    assert coordinate_system in ('cartesian', 'cylindrical', 'spherical')

    species = parse_species(part, species)

    center_position = parse_property(part, 'center_position', center_position)
    part_indicess = parse_property(species, 'indices', part_indicess)

    dist = {}

    for spec_i, spec in enumerate(species):
        part_indices = parse_indices(part[spec], part_indicess[spec_i])

        dist[spec] = ut.coordinate.get_distances(
            part[spec]['position'][part_indices], center_position, part.info['box.length'],
            part.snapshot['scalefactor'], total_distance)  # [kpc physical]

        if not total_distance:
            if rotation is not None:
                if rotation is True:
                    # get principal axes stored in particle dictionary
                    if (len(part[spec].host_rotation_tensors) and
                            len(part[spec].host_rotation_tensors[0])):
                        rotation_tensor = part[spec].host_rotation_tensors[0]
                    else:
                        raise ValueError('! cannot find principal_axes_tensor in species dict')
                elif len(rotation):
                    # use input rotation vectors
                    rotation_tensor = rotation

                dist[spec] = ut.coordinate.get_coordinates_rotated(dist[spec], rotation_tensor)

            if coordinate_system in ['cylindrical', 'spherical']:
                dist[spec] = ut.coordinate.get_positions_in_coordinate_system(
                    dist[spec], 'cartesian', coordinate_system)

    if return_array and len(species) == 1:
        dist = dist[species[0]]

    return dist


def get_velocities_wrt_center(
    part, species=['star'], part_indicess=None, center_velocity=None, center_position=None,
    rotation=False, coordinate_system='cartesian', total_velocity=False, return_array=True):
    '''
    Get velocities (either scalar or vector) between input particles and center_velocity
    (input or stored in particle catalog).

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species : str or list : name[s] of particle species to get
    part_indicess : array or list : indices[s] of particles to select, one array per input species
    center_velocity : array : center velocity [km / s]
        if None, will use default center velocity in catalog
    center_position : array : center position [kpc comoving], to use in computing Hubble flow
        if None, will use default center position in catalog
    rotation : bool or array : whether to rotate particles
        two options:
        (a) if input array of eigen-vectors, will define rotation axes for all species
        (b) if True, will rotate to align with principal axes defined by input species
    coordinate_system : str : which coordinates to get positions in:
        'cartesian' (default), 'cylindrical', 'spherical'
    total_velocity : bool : whether to compute total/scalar velocity
    return_array : bool : whether to return array (instead of dict) if input single species

    Returns
    -------
    vel : array or dict thereof :
        velocities (object number x dimension number, or object number) [km / s]
    '''
    assert coordinate_system in ('cartesian', 'cylindrical', 'spherical')

    species = parse_species(part, species)

    center_velocity = parse_property(part, 'center_velocity', center_velocity)
    center_position = parse_property(part, 'center_position', center_position)
    part_indicess = parse_property(species, 'indices', part_indicess)

    vel = {}
    for spec_i, spec in enumerate(species):
        part_indices = parse_indices(part[spec], part_indicess[spec_i])

        vel[spec] = ut.coordinate.get_velocity_differences(
            part[spec]['velocity'][part_indices], center_velocity,
            part[spec]['position'][part_indices], center_position, part.info['box.length'],
            part.snapshot['scalefactor'], part.snapshot['time.hubble'], total_velocity)

        if not total_velocity:
            if rotation is not None:
                if rotation is True:
                    # get principal axes stored in particle dictionary
                    if (len(part[spec].host_rotation_tensors) and
                            len(part[spec].host_rotation_tensors[0])):
                        rotation_tensor = part[spec].host_rotation_tensors[0]
                    else:
                        raise ValueError('! cannot find principal_axes_tensor in species dict')
                elif len(rotation):
                    # use input rotation vectors
                    rotation_tensor = rotation

                vel[spec] = ut.coordinate.get_coordinates_rotated(vel[spec], rotation_tensor)

            if coordinate_system in ('cylindrical', 'spherical'):
                # need to compute distance vectors
                distances = ut.coordinate.get_distances(
                    part[spec]['position'][part_indices], center_position,
                    part.info['box.length'], part.snapshot['scalefactor'])  # [kpc physical]

                if rotation is not None:
                    # need to rotate distances too
                    distances = ut.coordinate.get_coordinates_rotated(distances, rotation_tensor)

                vel[spec] = ut.coordinate.get_velocities_in_coordinate_system(
                    vel[spec], distances, 'cartesian', coordinate_system)

    if return_array and len(species) == 1:
        vel = vel[species[0]]

    return vel


def get_orbit_dictionary(
    part, species=['star'], part_indicess=None, center_position=None, center_velocity=None,
    return_single=True):
    '''
    Get dictionary of orbital parameters.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species : str or list : name[s] of particle species to compute
    part_indicess : array or list : indices[s] of particles to select, one array per input species
    center_position : array : center (reference) position
    center_position : array : center (reference) velociy
    return_single : bool :
        whether to return single dict instead of dict of dicts, if single species

    Returns
    -------
    orb : dict : dictionary of orbital properties, one for each species (unless scalarize is True)
    '''
    species = parse_species(part, species)

    center_position = parse_property(part, 'center_position', center_position)
    center_velocity = parse_property(part, 'center_velocity', center_velocity)
    part_indicess = parse_property(species, 'indices', part_indicess)

    orb = {}
    for spec_i, spec in enumerate(species):
        part_indices = parse_indices(part[spec], part_indicess[spec_i])

        distance_vectors = ut.coordinate.get_distances(
            part[spec]['position'][part_indices], center_position, part.info['box.length'],
            part.snapshot['scalefactor'])

        velocity_vectors = ut.coordinate.get_velocity_differences(
            part[spec]['velocity'][part_indices], center_velocity,
            part[spec]['position'][part_indices], center_position,
            part.info['box.length'], part.snapshot['scalefactor'], part.snapshot['time.hubble'])

        orb[spec] = orbit.get_orbit_dictionary(distance_vectors, velocity_vectors)

    if return_single and len(species) == 1:
        orb = orb[species[0]]

    return orb


#===================================================================================================
# subsample
#===================================================================================================
def get_indices_within_coordinates(
    part, species=['star'],
    distance_limitss=[], center_position=None,
    velocity_limitss=[], center_velocity=None,
    rotation=None, coordinate_system='cartesian',
    part_indicess=None, return_array=True):
    '''
    Get indices of particles that are within distance and/or velocity coordinate limits from center
    (either input or stored in particle catalog).

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species : str or list : name[s] of particle species to use
    distance_limitss : list or list of lists:
        min and max distance[s], relative to center, to get particles [kpc physical]
        default is 1-D list, but can be 2-D or 3-D list to select separately along dimensions
        if 2-D or 3-D, need to input *signed* limits
    center_position : array : center position [kpc comoving]
        if None, will use default center position in particle catalog
    velocity_limitss : list or list of lists:
        min and max velocities, relative to center, to get particles [km / s]
        default is 1-D list, but can be 2-D or 3-D list to select separately along dimensions
        if 2-D or 3-D, need to input *signed* limits
    center_velocity : array : center velocity [km / s]
        if None, will use default center velocity in particle catalog
    rotation : bool or array : whether to rotate particle coordinates
        two options:
        (a) if input array of eigen-vectors, will use to define rotation axes for all species
        (b) if True, will rotate to align with principal axes defined by each input species
    coordinate_system : str : which coordinates to get positions in:
        'cartesian' (default), 'cylindrical', 'spherical'
    part_indicess : array : prior indices[s] of particles to select, one array per input species
    return_array : bool : whether to return single array instead of dict, if input single species

    Returns
    -------
    part_index : dict or array : array or dict of arrays of indices of particles in region
    '''
    assert coordinate_system in ['cartesian', 'cylindrical', 'spherical']

    species = parse_species(part, species)
    center_position = parse_property(part, 'center_position', center_position)
    if velocity_limitss is not None and len(velocity_limitss):
        center_velocity = parse_property(part, 'center_velocity', center_velocity)
    part_indicess = parse_property(species, 'indices', part_indicess)

    part_index = {}
    for spec_i, spec in enumerate(species):
        part_indices = parse_indices(part[spec], part_indicess[spec_i])

        if len(part_indices) and distance_limitss is not None and len(distance_limitss):
            distance_limits_dimen = np.ndim(distance_limitss)

            if distance_limits_dimen == 1:
                total_distance = True
            elif distance_limits_dimen == 2:
                total_distance = False
                assert len(distance_limitss) in [2, 3]
            else:
                raise ValueError('! cannot parse distance_limitss = {}'.format(distance_limitss))

            if (distance_limits_dimen == 1 and distance_limitss[0] <= 0 and
                    distance_limitss[1] >= Inf):
                pass  # null case, no actual limits imposed, so skip rest
            else:
                """
                # an attempt to be clever, but gains seem modest
                distances = np.abs(coordinate.get_position_difference(
                    part[spec]['position'] - center_position,
                    part.info['box.length'])) * part.snapshot['scalefactor']  # [kpc physical]

                for dimension_i in range(part[spec]['position'].shape[1]):
                    masks *= ((distances[:, dimension_i] < np.max(distance_limits)) *
                              (distances[:, dimension_i] >= np.min(distance_limits)))
                    part_indices[spec] = part_indices[spec][masks]
                    distances = distances[masks]

                distances = np.sum(distances ** 2, 1)  # assume 3-d position
                """

                distancess = get_distances_wrt_center(
                    part, spec, part_indices, center_position, rotation, coordinate_system,
                    total_distance)

                if distance_limits_dimen == 1:
                    # distances are absolute
                    masks = (
                        (distancess >= np.min(distance_limitss)) *
                        (distancess < np.max(distance_limitss))
                    )
                elif distance_limits_dimen == 2:
                    if len(distance_limitss) == 2:
                        # distances are signed
                        masks = (
                            (distancess[0] >= np.min(distance_limitss[0])) *
                            (distancess[0] < np.max(distance_limitss[0])) *
                            (distancess[1] >= np.min(distance_limitss[1])) *
                            (distancess[1] < np.max(distance_limitss[1]))
                        )
                    elif distance_limits_dimen == 3:
                        # distances are signed
                        masks = (
                            (distancess[0] >= np.min(distance_limitss[0])) *
                            (distancess[0] < np.max(distance_limitss[0])) *
                            (distancess[1] >= np.min(distance_limitss[1])) *
                            (distancess[1] < np.max(distance_limitss[1]))
                            (distancess[2] >= np.min(distance_limitss[2])) *
                            (distancess[2] < np.max(distance_limitss[2]))
                        )

                part_indices = part_indices[masks]

        if len(part_indices) and velocity_limitss is not None and len(velocity_limitss):
            velocity_limits_dimen = np.ndim(velocity_limitss)

            if velocity_limits_dimen == 1:
                return_total_velocity = True
            elif velocity_limits_dimen == 2:
                return_total_velocity = False
                assert len(velocity_limitss) in [2, 3]
            else:
                raise ValueError('! cannot parse velocity_limitss = {}'.format(velocity_limitss))

            if (velocity_limits_dimen == 1 and velocity_limitss[0] <= 0 and
                    velocity_limitss[1] >= Inf):
                pass  # null case, no actual limits imposed, so skip rest
            else:
                velocitiess = get_velocities_wrt_center(
                    part, spec, part_indices, center_velocity, center_position, rotation,
                    coordinate_system, return_total_velocity)

                if velocity_limits_dimen == 1:
                    # velocities are absolute
                    masks = (
                        (velocitiess >= np.min(velocity_limitss)) *
                        (velocitiess < np.max(velocity_limitss))
                    )
                elif velocity_limits_dimen == 2:
                    if len(velocity_limitss) == 2:
                        # velocities are signed
                        masks = (
                            (velocitiess[0] >= np.min(velocity_limitss[0])) *
                            (velocitiess[0] < np.max(velocity_limitss[0])) *
                            (velocitiess[1] >= np.min(velocity_limitss[1])) *
                            (velocitiess[1] < np.max(velocity_limitss[1]))
                        )
                    elif len(velocity_limitss) == 3:
                        # velocities are signed
                        masks = (
                            (velocitiess[0] >= np.min(velocity_limitss[0])) *
                            (velocitiess[0] < np.max(velocity_limitss[0])) *
                            (velocitiess[1] >= np.min(velocity_limitss[1])) *
                            (velocitiess[1] < np.max(velocity_limitss[1]))
                            (velocitiess[2] >= np.min(velocity_limitss[2])) *
                            (velocitiess[2] < np.max(velocity_limitss[2]))
                        )

                part_indices = part_indices[masks]

        part_index[spec] = part_indices

    if return_array and len(species) == 1:
        part_index = part_index[species[0]]

    return part_index


def get_indices_id_kind(
    part, species=['star'], id_kind='unique', part_indicess=None, return_array=True):
    '''
    Get indices of particles that either are unique (no other particles of same species have
    same id) or multiple (other particle of same species has same id).

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species : str or list : name[s] of particle species
    split_kind : str : id kind of particles to get: 'unique', 'multiple'
    part_indicess : array : prior indices[s] of particles to select, one array per input species
    return_array : bool : whether to return single array instead of dict, if input single species

    Returns
    -------
    part_index : dict or array : array or dict of arrays of indices of particles of given split kind
    '''
    species = parse_species(part, species)
    part_indicess = parse_property(species, 'indices', part_indicess)

    assert id_kind in ['unique', 'multiple']

    part_index = {}
    for spec_i, spec in enumerate(species):
        part_indices = parse_indices(part[spec], part_indicess[spec_i])

        _pids, piis, counts = np.unique(
            part[spec]['id'][part_indices], return_index=True, return_counts=True)

        pis_unsplit = np.sort(part_indices[piis[counts == 1]])

        if id_kind == 'unique':
            part_index[spec] = pis_unsplit
        elif id_kind == 'multiple':
            part_index[spec] = np.setdiff1d(part_indices, pis_unsplit)
        else:
            raise ValueError('! not recognize id_kind = {}'.format(id_kind))

    if return_array and len(species) == 1:
        part_index = part_index[species[0]]

    return part_index


#===================================================================================================
# halo/galaxy major/minor axes
#===================================================================================================
def get_principal_axes(
    part, species_name='star', distance_max=Inf, mass_percent=None, age_percent=None, age_limits=[],
    center_positions=None, center_velocities=None, part_indices=None, return_array=True,
    print_results=True):
    '''
    Get reverse-sorted eigen-vectors, eigen-values, and axis ratios of principal axes of
    each host galaxy/halo.
    Ensure that principal axes are oriented so median v_phi > 0.

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species : str or list : name[s] of particle species to use
    distance_max : float : maximum distance to select particles [kpc physical]
    mass_percent : float : keep particles within the distance that encloses mass percent [0, 100]
        of all particles within distance_max
    age_percent : float : use the youngest age_percent of particles within distance cut
    age_limits : float : use only particles within age limits
    center_positions : array or array of arrays : position[s] of center[s] [kpc comoving]
    center_velocities : array or array of arrays : velocity[s] of center[s] [km / s]
    part_indices : array : indices[s] of particles to select
    return_array : bool :
        whether to return single array for each property, instead of array of arrays, if single host
    print_results : bool : whether to print axis ratios

    Returns
    -------
    principal_axes = {
        'rotation.tensor': array : rotation vectors that define max, med, min axes
        'eigen.values': array : eigen-values of max, med, min axes
        'axis.ratios': array : ratios of principal axes
    }
    '''
    Say = ut.io.SayClass(get_principal_axes)

    center_positions = parse_property(part, 'center_position', center_positions, single_host=False)
    center_velocities = parse_property(
        part, 'center_velocity', center_velocities, single_host=False)
    part_indices = parse_indices(part[species_name], part_indices)

    principal_axes = {
        'rotation.tensor': [],
        'eigen.values': [],
        'axis.ratios': [],
    }

    for center_i, center_position in enumerate(center_positions):
        distance_vectors = ut.coordinate.get_distances(
            part[species_name]['position'][part_indices], center_position, part.info['box.length'],
            part.snapshot['scalefactor'])  # [kpc physical]

        distances = np.sqrt(np.sum(distance_vectors ** 2, 1))
        masks = (distances < distance_max)

        if mass_percent:
            distance_percent = ut.math.percentile_weighted(
                distances[masks], mass_percent,
                part[species_name].prop('mass', part_indices[masks]))
            masks *= (distances < distance_percent)

        if age_percent or (age_limits is not None and len(age_limits)):
            if 'form.scalefactor' not in part[species_name]:
                raise ValueError('! input age constraints but age not in {} catalog'.format(
                    species_name))

            if age_percent and (age_limits is not None and len(age_limits)):
                Say.say('input both age_percent and age_limits, using only age_percent')

            if age_percent:
                age_max = ut.math.percentile_weighted(
                    part[species_name].prop('age', part_indices[masks]), age_percent,
                    part[species_name].prop('mass', part_indices[masks]))
                age_limits_use = [0, age_max]
            else:
                age_limits_use = age_limits

            Say.say('using {} particles with age = {} Gyr'.format(
                species_name, ut.array.get_limits_string(age_limits_use)))
            masks *= ((part[species_name].prop('age', part_indices) >= min(age_limits_use)) *
                      (part[species_name].prop('age', part_indices) < max(age_limits_use)))

        rotation_tensor, eigen_values, axis_ratios = ut.coordinate.get_principal_axes(
            distance_vectors[masks], part[species_name].prop('mass', part_indices[masks]),
            print_results)

        # test if need to flip a principal axis to ensure that net v_phi > 0
        velocity_vectors = ut.coordinate.get_velocity_differences(
            part[species_name].prop('velocity', part_indices[masks]), center_velocities[center_i])
        velocity_vectors_rot = ut.coordinate.get_coordinates_rotated(
            velocity_vectors, rotation_tensor)
        distance_vectors_rot = ut.coordinate.get_coordinates_rotated(
            distance_vectors[masks], rotation_tensor)
        velocity_vectors_cyl = ut.coordinate.get_velocities_in_coordinate_system(
            velocity_vectors_rot, distance_vectors_rot, 'cartesian', 'cylindrical')
        if np.median(velocity_vectors_cyl[:, 2]) < 0:
            rotation_tensor[1] *= -1  # flip so net v_phi is positive

        principal_axes['rotation.tensor'].append(rotation_tensor)
        principal_axes['eigen.values'].append(eigen_values)
        principal_axes['axis.ratios'].append(axis_ratios)

    for k in principal_axes:
        principal_axes[k] = np.array(principal_axes[k])

    if return_array and np.shape(center_positions)[0] == 1:
        for k in principal_axes:
            principal_axes[k] = principal_axes[k][0]

    return principal_axes


#===================================================================================================
# halo/galaxy radius
#===================================================================================================
def get_halo_properties(
    part, species=['dark', 'star', 'gas'], virial_kind='200m',
    distance_limits=[10, 600], distance_bin_width=0.02, distance_scaling='log',
    center_position=None, return_array=True, print_results=True):
    '''
    Compute halo radius according to virial_kind.
    Return this radius, the mass from each species within this radius, and particle indices within
    this radius (if get_part_indices).

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species : str or list : name[s] of particle species to use: 'all' = use all in dictionary
    virial_kind : str : virial overdensity definition
      '200m' -> average density is 200 x matter
      '200c' -> average density is 200 x critical
      'vir' -> average density is Bryan & Norman
      'fof.100m' -> edge density is 100 x matter, for FoF(ll=0.168)
      'fof.60m' -> edge density is 60 x matter, for FoF(ll=0.2)
    distance_limits : list : min and max distance to consider [kpc physical]
    distance_bin_width : float : width of distance bin
    distance_scaling : str : scaling of distance: 'log', 'linear'
    center_position : array : center position to use
        if None, will use default center position in catalog
    return_array : bool : whether to return array (instead of dict) if input single species
    print_results : bool : whether to print radius and mass

    Returns
    -------
    halo_prop : dict : dictionary of halo properties:
        radius : float : halo radius [kpc physical]
        mass : float : mass within radius [M_sun]
        indices : array : indices of partices within radius (if get_part_indices)
    '''
    distance_limits = np.asarray(distance_limits)

    Say = ut.io.SayClass(get_halo_properties)

    species = parse_species(part, species)
    center_position = parse_property(part, 'center_position', center_position)

    HaloProperty = halo_property.HaloPropertyClass(part.Cosmology, part.snapshot['redshift'])

    DistanceBin = ut.binning.DistanceBinClass(
        distance_scaling, distance_limits, width=distance_bin_width, dimension_number=3)

    overdensity, reference_density = HaloProperty.get_overdensity(virial_kind, units='kpc physical')
    virial_density = overdensity * reference_density

    mass_cum_in_bins = np.zeros(DistanceBin.number)
    distancess = []

    for spec_i, spec in enumerate(species):
        distances = ut.coordinate.get_distances(
            part[spec]['position'], center_position, part.info['box.length'],
            part.snapshot['scalefactor'], total_distance=True)  # [kpc physical]
        distancess.append(distances)
        mass_in_bins = DistanceBin.get_histogram(distancess[spec_i], False, part[spec]['mass'])

        # get mass within distance minimum, for computing cumulative values
        distance_indices = np.where(distancess[spec_i] < np.min(distance_limits))[0]
        mass_cum_in_bins += (np.sum(part[spec]['mass'][distance_indices]) +
                             np.cumsum(mass_in_bins))

    if part.info['baryonic'] and len(species) == 1 and species[0] == 'dark':
        # correct for baryonic mass if analyzing only dark matter in baryonic simulation
        Say.say('! using only dark particles, so correcting for baryonic mass')
        mass_factor = 1 + part.Cosmology['omega_baryon'] / part.Cosmology['omega_matter']
        mass_cum_in_bins *= mass_factor

    # cumulative densities in bins
    density_cum_in_bins = mass_cum_in_bins / DistanceBin.volumes_cum

    # get smallest radius that satisfies virial density
    for d_bin_i in range(DistanceBin.number - 1):
        if (density_cum_in_bins[d_bin_i] >= virial_density and
                density_cum_in_bins[d_bin_i + 1] < virial_density):
            # interpolate in log space
            log_halo_radius = np.interp(
                np.log10(virial_density), np.log10(density_cum_in_bins[[d_bin_i + 1, d_bin_i]]),
                DistanceBin.log_maxs[[d_bin_i + 1, d_bin_i]])
            halo_radius = 10 ** log_halo_radius
            break
    else:
        Say.say('! could not determine halo R_{}'.format(virial_kind))
        if density_cum_in_bins[0] < virial_density:
            Say.say('distance min = {:.1f} kpc already is below virial density = {}'.format(
                distance_limits.min(), virial_density))
            Say.say('decrease distance_limits')
        elif density_cum_in_bins[-1] > virial_density:
            Say.say('distance max = {:.1f} kpc still is above virial density = {}'.format(
                distance_limits.max(), virial_density))
            Say.say('increase distance_limits')
        else:
            Say.say('not sure why!')

        return

    # get maximum of V_circ = sqrt(G M(< r) / r)
    vel_circ_in_bins = ut.constant.km_per_kpc * np.sqrt(
        ut.constant.grav_kpc_msun_sec * mass_cum_in_bins / DistanceBin.maxs)
    vel_circ_max = np.max(vel_circ_in_bins)
    vel_circ_max_radius = DistanceBin.maxs[np.argmax(vel_circ_in_bins)]

    halo_mass = 0
    part_indices = {}
    for spec_i, spec in enumerate(species):
        masks = (distancess[spec_i] < halo_radius)
        halo_mass += np.sum(part[spec]['mass'][masks])
        part_indices[spec] = ut.array.get_arange(part[spec]['mass'])[masks]

    if print_results:
        Say.say(
            'R_{} = {:.1f} kpc\n  M_{} = {} M_sun, log = {}\n  V_max = {:.1f} km/s'.format(
                virial_kind, halo_radius, virial_kind,
                ut.io.get_string_from_numbers(halo_mass, 2),
                ut.io.get_string_from_numbers(np.log10(halo_mass), 2),
                vel_circ_max)
        )

    halo_prop = {}
    halo_prop['radius'] = halo_radius
    halo_prop['mass'] = halo_mass
    halo_prop['vel.circ.max'] = vel_circ_max
    halo_prop['vel.circ.max.radius'] = vel_circ_max_radius
    if return_array and len(species) == 1:
        part_indices = part_indices[species[0]]
    halo_prop['indices'] = part_indices

    return halo_prop


def get_galaxy_properties(
    part, species_name='star', edge_kind='mass.percent', edge_value=90,
    distance_max=20, distance_bin_width=0.02, distance_scaling='log', center_position=None,
    axis_kind='', rotation_tensor=None, rotation_distance_max=20,
    other_axis_distance_limits=None, part_indices=None, print_results=True):
    '''
    Compute galaxy radius according to edge_kind.
    Return this radius, the mass from species within this radius, particle indices within this
    radius, and rotation vectors (if applicable).

    Parameters
    ----------
    part : dict : catalog of particles at snapshot
    species_name : str : name of particle species to use
    edge_kind : str : method to define galaxy radius
        'mass.percent' = radius at which edge_value (percent) of stellar mass within distance_max
        'density' = radius at which density is edge_value [log(M_sun / kpc^3)]
    edge_value : float : value to use to define galaxy radius
    mass_percent : float : percent of mass (out to distance_max) to define radius
    distance_max : float : maximum distance to consider [kpc physical]
    distance_bin_width : float : width of distance bin
    distance_scaling : str : distance bin scaling: 'log', 'linear'
    axis_kind : str : 'major', 'minor', 'both'
    rotation_tensor : array : rotation vectors that define principal axes
    rotation_distance_max : float :
        maximum distance to use in defining rotation vectors of principal axes [kpc physical]
    other_axis_distance_limits : float :
        min and max distances along other axis[s] to keep particles [kpc physical]
    center_position : array : center position [kpc comoving]
        if None, will use default center position in catalog
    part_indices : array : star particle indices (if already know which ones are close)
    print_results : bool : whether to print radius and mass of galaxy

    Returns
    -------
    gal_prop : dict : dictionary of galaxy properties:
        radius or radius.major & radius.minor : float : galaxy radius[s] [kpc physical]
        mass : float : mass within radius[s] [M_sun]
        indices : array : indices of partices within radius[s] (if get_part_indices)
        rotation.vectors : array : eigen-vectors that defined rotation
    '''

    def get_radius_mass_indices(
        masses, distances, distance_scaling, distance_limits, distance_bin_width, dimension_number,
        edge_kind, edge_value):
        '''
        Utility function.
        '''
        Say = ut.io.SayClass(get_radius_mass_indices)

        DistanceBin = ut.binning.DistanceBinClass(
            distance_scaling, distance_limits, width=distance_bin_width,
            dimension_number=dimension_number)

        # get masses in distance bins
        mass_in_bins = DistanceBin.get_histogram(distances, False, masses)

        if edge_kind == 'mass.percent':
            # get mass within distance minimum, for computing cumulative values
            d_indices = np.where(distances < np.min(distance_limits))[0]
            log_masses_cum = ut.math.get_log(np.sum(masses[d_indices]) + np.cumsum(mass_in_bins))

            log_mass = np.log10(edge_value / 100) + log_masses_cum.max()

            try:
                # interpolate in log space
                log_radius = np.interp(log_mass, log_masses_cum, DistanceBin.log_maxs)
            except ValueError:
                Say.say('! could not find object radius - increase distance_max')
                return

        elif edge_kind == 'density':
            log_density_in_bins = ut.math.get_log(mass_in_bins / DistanceBin.volumes)
            # use only bins with defined density (has particles)
            d_bin_indices = np.arange(DistanceBin.number)[np.isfinite(log_density_in_bins)]
            # get smallest radius that satisfies density threshold
            for d_bin_ii, d_bin_i in enumerate(d_bin_indices):
                d_bin_i_plus_1 = d_bin_indices[d_bin_ii + 1]
                if (log_density_in_bins[d_bin_i] >= edge_value and
                        log_density_in_bins[d_bin_i_plus_1] < edge_value):
                    # interpolate in log space
                    log_radius = np.interp(
                        edge_value, log_density_in_bins[[d_bin_i_plus_1, d_bin_i]],
                        DistanceBin.log_maxs[[d_bin_i_plus_1, d_bin_i]])
                    break
            else:
                Say.say('! could not find object radius - increase distance_max')
                return

        radius = 10 ** log_radius

        masks = (distances < radius)
        mass = np.sum(masses[masks])
        indices = ut.array.get_arange(masses)[masks]

        return radius, mass, indices

    # start function
    Say = ut.io.SayClass(get_galaxy_properties)

    distance_min = 0.001  # [kpc physical]
    distance_limits = [distance_min, distance_max]

    if edge_kind == 'mass.percent':
        # dealing with cumulative value - stable enough to decrease bin with
        distance_bin_width *= 0.1

    center_position = parse_property(part, 'center_position', center_position)

    if part_indices is None or not len(part_indices):
        part_indices = ut.array.get_arange(part[species_name]['position'].shape[0])

    distance_vectors = ut.coordinate.get_distances(
        part[species_name]['position'][part_indices], center_position,
        part.info['box.length'], part.snapshot['scalefactor'])  # [kpc physical]
    distances = np.sqrt(np.sum(distance_vectors ** 2, 1))  # 3-D distance

    masses = part[species_name].prop('mass', part_indices)

    if axis_kind:
        # radius along 2-D major axes (projected radius) or along 1-D minor axis (height)
        assert axis_kind in ['major', 'minor', 'both']

        if rotation_tensor is None or not len(rotation_tensor):
            if (len(part[species_name].host_rotation_tensors) and
                    len(part[species_name].host_rotation_tensors[0])):
                # use only the primary host
                rotation_tensor = part[species_name].host_rotation_tensors[0]
            else:
                masks = (distances < rotation_distance_max)
                rotation_tensor = ut.coordinate.get_principal_axes(
                    distance_vectors[masks], masses[masks])[0]

        distance_vectors = ut.coordinate.get_coordinates_rotated(
            distance_vectors, rotation_tensor=rotation_tensor)

        distances_cyl = ut.coordinate.get_positions_in_coordinate_system(
            distance_vectors, 'cartesian', 'cylindrical')
        major_distances, minor_distances = distances_cyl[:, 0], distances_cyl[:, 1]
        minor_distances = np.abs(minor_distances)  # need only absolute distances

        if axis_kind in ['major', 'minor']:
            if axis_kind == 'minor':
                dimension_number = 1
                distances = minor_distances
                other_distances = major_distances
            elif axis_kind == 'major':
                dimension_number = 2
                distances = major_distances
                other_distances = minor_distances

            if (other_axis_distance_limits is not None and
                    (min(other_axis_distance_limits) > 0 or max(other_axis_distance_limits) < Inf)):
                masks = ((other_distances >= min(other_axis_distance_limits)) *
                         (other_distances < max(other_axis_distance_limits)))
                distances = distances[masks]
                masses = masses[masks]
    else:
        # spherical average
        dimension_number = 3

    gal_prop = {}

    if axis_kind == 'both':
        # first get 3-D radius
        galaxy_radius_3d, _galaxy_mass_3d, indices = get_radius_mass_indices(
            masses, distances, distance_scaling, distance_limits, distance_bin_width, 3,
            edge_kind, edge_value)

        galaxy_radius_major = galaxy_radius_3d
        axes_mass_dif = 1

        # then iterate to get both major and minor axes
        while axes_mass_dif > 0.005:
            # get 1-D radius along minor axis
            masks = (major_distances < galaxy_radius_major)
            galaxy_radius_minor, galaxy_mass_minor, indices = get_radius_mass_indices(
                masses[masks], minor_distances[masks], distance_scaling, distance_limits,
                distance_bin_width, 1, edge_kind, edge_value)

            # get 2-D radius along major axes
            masks = (minor_distances < galaxy_radius_minor)
            galaxy_radius_major, galaxy_mass_major, indices = get_radius_mass_indices(
                masses[masks], major_distances[masks], distance_scaling, distance_limits,
                distance_bin_width, 2, edge_kind, edge_value)

            axes_mass_dif = (abs(galaxy_mass_major - galaxy_mass_minor) /
                             (0.5 * (galaxy_mass_major + galaxy_mass_minor)))

        indices = (major_distances < galaxy_radius_major) * (minor_distances < galaxy_radius_minor)

        gal_prop['radius.major'] = galaxy_radius_major
        gal_prop['radius.minor'] = galaxy_radius_minor
        gal_prop['mass'] = galaxy_mass_major
        gal_prop['log mass'] = np.log10(galaxy_mass_major)
        gal_prop['rotation.tensor'] = rotation_tensor
        gal_prop['indices'] = part_indices[indices]

        if print_results:
            Say.say('R_{:.0f} along major, minor axes = {:.2f}, {:.2f} kpc physical'.format(
                    edge_value, galaxy_radius_major, galaxy_radius_minor))

    else:
        galaxy_radius, galaxy_mass, indices = get_radius_mass_indices(
            masses, distances, distance_scaling, distance_limits, distance_bin_width,
            dimension_number, edge_kind, edge_value)

        gal_prop['radius'] = galaxy_radius
        gal_prop['mass'] = galaxy_mass
        gal_prop['log mass'] = np.log10(galaxy_mass)
        gal_prop['indices'] = part_indices[indices]

        if print_results:
            Say.say('R_{:.0f} = {:.2f} kpc physical'.format(edge_value, galaxy_radius))

    if print_results:
        Say.say('M_star = {:.2e} M_sun, log = {:.2f}'.format(
                gal_prop['mass'], gal_prop['log mass']))

    return gal_prop


#===================================================================================================
# profiles of properties
#===================================================================================================
class SpeciesProfileClass(ut.binning.DistanceBinClass):
    '''
    Get profiles of either histogram/sum or stastitics (such as average, median) of given
    property for given particle species.

    __init__ is defined via ut.binning.DistanceBinClass
    '''

    def get_profiles(
        self, part, species=['all'],
        property_name='', property_statistic='sum', weight_by_mass=False,
        center_position=None, center_velocity=None, rotation=None,
        other_axis_distance_limits=None, property_select={}, part_indicess=None):
        '''
        Parse inputs into either get_sum_profiles() or get_statistics_profiles().
        If know what you want, can skip this and jump to those functions.

        Parameters
        ----------
        part : dict : catalog of particles
        species : str or list : name[s] of particle species to compute mass from
        property_name : str : name of property to get statistics of
        property_statistic : str : statistic to get profile of:
            'sum', 'sum.cum', 'density', 'density.cum', 'vel.circ'
        weight_by_mass : bool : whether to weight property by species mass
        center_position : array : position of center
        center_velocity : array : velocity of center
        rotation : bool or array : whether to rotate particles - two options:
          (a) if input array of eigen-vectors, will define rotation axes
          (b) if True, will rotate to align with principal axes stored in species dictionary
        other_axis_distance_limits : float :
            min and max distances along other axis[s] to keep particles [kpc physical]
        property_select : dict : (other) properties to select on: names as keys and limits as values
        part_indicess : array (species number x particle number) :
            indices of particles from which to select

        Returns
        -------
        pros : dict : dictionary of profiles for each particle species
        '''
        if ('sum' in property_statistic or 'vel.circ' in property_statistic or
                'density' in property_statistic):
            pros = self.get_sum_profiles(
                part, species, property_name, center_position, rotation, other_axis_distance_limits,
                property_select, part_indicess)
        else:
            pros = self.get_statistics_profiles(
                part, species, property_name, weight_by_mass, center_position, center_velocity,
                rotation, other_axis_distance_limits, property_select, part_indicess)

        for k in pros:
            if '.cum' in property_statistic or 'vel.circ' in property_statistic:
                pros[k]['distance'] = pros[k]['distance.cum']
                pros[k]['log distance'] = pros[k]['log distance.cum']
            else:
                pros[k]['distance'] = pros[k]['distance.mid']
                pros[k]['log distance'] = pros[k]['log distance.mid']

        return pros

    def get_sum_profiles(
        self, part, species=['all'], property_name='mass', center_position=None,
        rotation=None, other_axis_distance_limits=None, property_select={}, part_indicess=None):
        '''
        Get profiles of summed quantity (such as mass or density) for given property for each
        particle species.

        Parameters
        ----------
        part : dict : catalog of particles
        species : str or list : name[s] of particle species to compute mass from
        property_name : str : property to get sum of
        center_position : list : center position
        rotation : bool or array : whether to rotate particles - two options:
          (a) if input array of eigen-vectors, will define rotation axes
          (b) if True, will rotate to align with principal axes stored in species dictionary
        other_axis_distance_limits : float :
            min and max distances along other axis[s] to keep particles [kpc physical]
        property_select : dict : (other) properties to select on: names as keys and limits as values
        part_indicess : array (species number x particle number) :
            indices of particles from which to select

        Returns
        -------
        pros : dict : dictionary of profiles for each particle species
        '''
        if 'gas' in species and 'consume.time' in property_name:
            pros_mass = self.get_sum_profiles(
                part, species, 'mass', center_position, rotation, other_axis_distance_limits,
                property_select, part_indicess)

            pros_sfr = self.get_sum_profiles(
                part, species, 'sfr', center_position, rotation, other_axis_distance_limits,
                property_select, part_indicess)

            pros = pros_sfr
            for k in pros_sfr['gas']:
                if 'distance' not in k:
                    pros['gas'][k] = pros_mass['gas'][k] / pros_sfr['gas'][k] / 1e9

            return pros

        pros = {}

        Fraction = ut.math.FractionClass()

        if np.isscalar(species):
            species = [species]
        if species == ['baryon']:
            # treat this case specially for baryon fraction
            species = ['gas', 'star', 'dark', 'dark2']
        species = parse_species(part, species)

        center_position = parse_property(part, 'center_position', center_position)
        part_indicess = parse_property(species, 'indices', part_indicess)

        assert 0 < self.dimension_number <= 3

        for spec_i, spec in enumerate(species):
            part_indices = part_indicess[spec_i]
            if part_indices is None or not len(part_indices):
                part_indices = ut.array.get_arange(part[spec].prop(property_name))

            if property_select:
                part_indices = catalog.get_indices_catalog(
                    part[spec], property_select, part_indices)

            prop_values = part[spec].prop(property_name, part_indices)

            if self.dimension_number == 3:
                # simple case: profile using scalar distance
                distances = ut.coordinate.get_distances(
                    part[spec]['position'][part_indices], center_position, part.info['box.length'],
                    part.snapshot['scalefactor'], total_distance=True)  # [kpc physical]

            elif self.dimension_number in [1, 2]:
                # other cases: profile along R (2 major axes) or Z (minor axis)
                if rotation is not None and not isinstance(rotation, bool) and len(rotation):
                    rotation_tensor = rotation
                elif (len(part[spec].host_rotation_tensors) and
                        len(part[spec].host_rotation_tensors[0])):
                    rotation_tensor = part[spec].host_rotation_tensors[0]
                else:
                    raise ValueError('want 2-D or 1-D profile but no means to define rotation')

                distancess = get_distances_wrt_center(
                    part, spec, part_indices, center_position, rotation_tensor,
                    coordinate_system='cylindrical')
                # ensure all distances are positive definite
                distancess = np.abs(distancess)

                if self.dimension_number == 1:
                    # compute profile along minor axis (Z)
                    distances = distancess[:, 1]
                    other_distances = distancess[:, 0]
                elif self.dimension_number == 2:
                    # compute profile along major axes (R)
                    distances = distancess[:, 0]
                    other_distances = distancess[:, 1]

                if (other_axis_distance_limits is not None and
                        (min(other_axis_distance_limits) > 0 or
                         max(other_axis_distance_limits) < Inf)):
                    masks = ((other_distances >= min(other_axis_distance_limits)) *
                             (other_distances < max(other_axis_distance_limits)))
                    distances = distances[masks]
                    prop_values = prop_values[masks]

            pros[spec] = self.get_sum_profile(distances, prop_values)  # defined in DistanceBinClass

        props = [pro_prop for pro_prop in pros[species[0]] if 'distance' not in pro_prop]
        props_dist = [pro_prop for pro_prop in pros[species[0]] if 'distance' in pro_prop]

        if property_name == 'mass':
            # create dictionary for baryonic mass
            if 'star' in species or 'gas' in species:
                spec_new = 'baryon'
                pros[spec_new] = {}
                for spec in np.intersect1d(species, ['star', 'gas']):
                    for pro_prop in props:
                        if pro_prop not in pros[spec_new]:
                            pros[spec_new][pro_prop] = np.array(pros[spec][pro_prop])
                        elif 'log' in pro_prop:
                            pros[spec_new][pro_prop] = ut.math.get_log(
                                10 ** pros[spec_new][pro_prop] +
                                10 ** pros[spec][pro_prop])
                        else:
                            pros[spec_new][pro_prop] += pros[spec][pro_prop]

                for pro_prop in props_dist:
                    pros[spec_new][pro_prop] = pros[species[0]][pro_prop]
                species.append(spec_new)

            if len(species) > 1:
                # create dictionary for total mass
                spec_new = 'total'
                pros[spec_new] = {}
                for spec in np.setdiff1d(species, ['baryon', 'total']):
                    for pro_prop in props:
                        if pro_prop not in pros[spec_new]:
                            pros[spec_new][pro_prop] = np.array(pros[spec][pro_prop])
                        elif 'log' in pro_prop:
                            pros[spec_new][pro_prop] = ut.math.get_log(
                                10 ** pros[spec_new][pro_prop] +
                                10 ** pros[spec][pro_prop])
                        else:
                            pros[spec_new][pro_prop] += pros[spec][pro_prop]

                for pro_prop in props_dist:
                    pros[spec_new][pro_prop] = pros[species[0]][pro_prop]
                species.append(spec_new)

                # create mass fraction wrt total mass
                for spec in np.setdiff1d(species, ['total']):
                    for pro_prop in ['sum', 'sum.cum']:
                        pros[spec][pro_prop + '.fraction'] = Fraction.get_fraction(
                            pros[spec][pro_prop], pros['total'][pro_prop])

                        if spec == 'baryon':
                            # units of cosmic baryon fraction
                            pros[spec][pro_prop + '.fraction'] /= (
                                part.Cosmology['omega_baryon'] / part.Cosmology['omega_matter'])

            # create circular velocity = sqrt (G m(< r) / r)
            for spec in species:
                pros[spec]['vel.circ'] = halo_property.get_circular_velocity(
                    pros[spec]['sum.cum'], pros[spec]['distance.cum'])

        return pros

    def get_statistics_profiles(
        self, part, species=['all'], property_name='', weight_by_mass=True,
        center_position=None, center_velocity=None, rotation=None, other_axis_distance_limits=None,
        property_select={}, part_indicess=None):
        '''
        Get profiles of statistics (such as median, average) for given property for each
        particle species.

        Parameters
        ----------
        part : dict : catalog of particles
        species : str or list : name[s] of particle species to compute mass from
        property_name : str : name of property to get statistics of
        weight_by_mass : bool : whether to weight property by species mass
        center_position : array : position of center
        center_velocity : array : velocity of center
        rotation : bool or array : whether to rotate particles - two options:
          (a) if input array of eigen-vectors, will define rotation axes
          (b) if True, will rotate to align with principal axes stored in species dictionary
        other_axis_distance_limits : float :
            min and max distances along other axis[s] to keep particles [kpc physical]
        property_select : dict : (other) properties to select on: names as keys and limits as values
        part_indicess : array or list : indices of particles from which to select

        Returns
        -------
        pros : dict : dictionary of profiles for each particle species
        '''
        pros = {}

        species = parse_species(part, species)

        center_position = parse_property(part, 'center_position', center_position)
        if 'velocity' in property_name:
            center_velocity = parse_property(part, 'center_velocity', center_velocity)
        part_indicess = parse_property(species, 'indices', part_indicess)

        assert 0 < self.dimension_number <= 3

        for spec_i, spec in enumerate(species):
            prop_test = property_name
            if 'velocity' in prop_test:
                prop_test = 'velocity'  # treat velocity specially because compile below
            assert part[spec].prop(prop_test) is not None

            part_indices = part_indicess[spec_i]
            if part_indices is None or not len(part_indices):
                part_indices = ut.array.get_arange(part[spec].prop(property_name))

            if property_select:
                part_indices = catalog.get_indices_catalog(
                    part[spec], property_select, part_indices)

            masses = None
            if weight_by_mass:
                masses = part[spec].prop('mass', part_indices)

            if 'velocity' in property_name:
                distance_vectors = ut.coordinate.get_distances(
                    part[spec]['position'][part_indices], center_position,
                    part.info['box.length'], part.snapshot['scalefactor'])  # [kpc physical]

                velocity_vectors = ut.coordinate.get_velocity_differences(
                    part[spec]['velocity'][part_indices], center_velocity,
                    part[spec]['position'][part_indices], center_position, part.info['box.length'],
                    part.snapshot['scalefactor'], part.snapshot['time.hubble'])

                # defined in DistanceBinClass
                pro = self.get_velocity_profile(distance_vectors, velocity_vectors, masses)

                pros[spec] = pro[property_name.replace('host.', '')]
                for prop in pro:
                    if 'velocity' not in prop:
                        pros[spec][prop] = pro[prop]
            else:
                prop_values = part[spec].prop(property_name, part_indices)

                if self.dimension_number == 3:
                    # simple case: profile using total distance [kpc physical]
                    distances = ut.coordinate.get_distances(
                        part[spec]['position'][part_indices], center_position,
                        part.info['box.length'], part.snapshot['scalefactor'], total_distance=True)
                elif self.dimension_number in [1, 2]:
                    # other cases: profile along R (2 major axes) or Z (minor axis)
                    if rotation is not None and not isinstance(rotation, bool) and len(rotation):
                        rotation_tensor = rotation
                    elif (len(part[spec].host_rotation_tensors) and
                            len(part[spec].host_rotation_tensors[0])):
                        rotation_tensor = part[spec].host_rotation_tensors[0]
                    else:
                        raise ValueError('want 2-D or 1-D profile but no means to define rotation')

                    distancess = get_distances_wrt_center(
                        part, spec, part_indices, center_position, rotation_tensor, 'cylindrical')
                    distancess = np.abs(distancess)

                    if self.dimension_number == 1:
                        # compute profile alongminor axis (Z)
                        distances = distancess[:, 1]
                        other_distances = distancess[:, 0]
                    elif self.dimension_number == 2:
                        # compute profile along 2 major axes (R)
                        distances = distancess[:, 0]
                        other_distances = distancess[:, 1]

                    if (other_axis_distance_limits is not None and
                        (min(other_axis_distance_limits) >= 0 or
                         max(other_axis_distance_limits) < Inf)):
                        masks = ((other_distances >= min(other_axis_distance_limits)) *
                                 (other_distances < max(other_axis_distance_limits)))
                        distances = distances[masks]
                        masses = masses[masks]
                        prop_values = prop_values[masks]

                # defined in DistanceBinClass
                pros[spec] = self.get_statistics_profile(distances, prop_values, masses)

        return pros
