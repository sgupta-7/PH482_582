'''
Find nearest neighbors.

@author: Andrew Wetzel <arwetzel@gmail.com>
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import numpy as np
from scipy import spatial
# local ----
from . import array, binning, coordinate, io


class NeighborClass(io.SayClass):
    '''
    Find neighbors using k-d tree, chaining mesh, or direct n^2.
    '''

    def get_neighbors(
        self, center_positions, neig_positions, neig_number_max=1000,
        distance_limits=[1e-5, 10000], periodic_length=None, scalefactor=None, is_angular=False,
        find_kind='kd-tree', neig_ids=None, print_diagnostics=True):
        '''
        Find neighbors within distance limits around centers (up to neig_number_max for kd-tree).

        Parameters
        ----------
        center_positions : array :
            positions around which to get neighbors (object number x dimension number)
        neig_positions : array : positions of neighbors (object number x dimension number)
        neig_number_max : int : maximum number of neighbors per center
        distance_limits : list or list of lists : neighbor distance limits [physical]:
            note: set non-zero distance lower limit if want not to select self
        periodic_length : float : periodic length [comoving] (if none, do not use periodic)
        scalefactor : float : use to convert input distance limits to [comoving] to match positions
        is_angular : bool : whether positions are angular (RA & dec)
        find_kind : str : neighbor finding method: kd-tree, mesh, direct
        neig_ids : array : ids of neighbors (to return instead of neighbor selection indices)
        print_diagnostics : bool : whether to print diagnostics along the way

        Returns
        -------
        distances : list of arrays : distances of neighbors for each center position
        neig_indices: list of arrays :
            indices (of neig_positions array) or ids (of input neig_ids array) of neighbors
            for each center position
        '''
        # ensure that position arrays are object number x dimension number
        if np.ndim(center_positions) == 1:
            center_positions = np.array([center_positions])
        if np.ndim(neig_positions) == 1:
            neig_positions = np.array([neig_positions])
        if center_positions.shape[1] != neig_positions.shape[1]:
            raise ValueError('center_positions.shape[1] = {} != neig_positions.shape[1] {}'.format(
                             center_positions.shape[1], neig_positions.shape[1]))

        self.center_number = center_positions.shape[0]
        self.neig_number = neig_positions.shape[0]
        self.dimension_number = center_positions.shape[1]
        self.dimension_indices = array.get_arange(self.dimension_number)
        self.neig_number_max = int(neig_number_max)
        self.periodic_length = periodic_length
        self.print_diagnostics = print_diagnostics
        if self.print_diagnostics:
            self.say('neighbor finding via ' + find_kind)
            self.say('center number = {}, neighbor number = {}'.format(
                     self.center_number, self.neig_number))
            if periodic_length:
                self.say('using periodic boundary of length = {:.3f}'.format(self.periodic_length))

        # check distance limits
        distance_limits = np.array(distance_limits)
        if scalefactor:
            distance_limits /= scalefactor  # convert to [comoving] to match positions

        if distance_limits.min() < 0:
            raise ValueError('distance lower limit = {} not valid'.format(distance_limits.min()))

        if periodic_length and distance_limits.max() >= 0.5 * periodic_length:
            self.say('! input neighbor max distance = {:.3f}'.format(distance_limits.max()))
            self.say('but periodic_length = {:.3f}, so this is > periodic_length/2'.format(
                     periodic_length))
            self.say('searching for neighbors only to distance = periodic_length/2 = {:.3f}'.format(
                     0.5 * periodic_length))
            distance_limits = np.clip(distance_limits, 0, 0.5 * periodic_length)

        if self.print_diagnostics:
            self.say('keeping neighbors within distance limits = [{}, {}]'.format(
                io.get_string_from_numbers(distance_limits.min(), 3, strip=True),
                io.get_string_from_numbers(distance_limits.max(), 3, strip=True))
            )

        # make distance_limits an object number x 2 array
        if distance_limits.size == 2:
            self.distance_limits = np.zeros((self.center_number, 2))
            self.distance_limits[:, 0] += distance_limits.min()
            self.distance_limits[:, 1] += distance_limits.max()
            self.is_distance_adaptive = False
        elif distance_limits.shape[1] == self.center_number and distance_limits.shape[0] == 2:
            self.distance_limits = np.transpose(distance_limits)
            self.is_distance_adaptive = True
            self.hit_adaptive_distance_limit = False
        elif distance_limits.shape[0] == self.center_number and distance_limits.shape[1] == 2:
            self.distance_limits = distance_limits
            self.is_distance_adaptive = True
            self.hit_adaptive_distance_limit = False
        else:
            raise ValueError('distance_limits.shape = {} not valid'.format(distance_limits.shape))

        self.distance_min = self.distance_limits.min()
        self.distance_max = self.distance_limits.max()

        # check if postions are angular
        self.is_angular = is_angular
        if self.is_angular:
            self.say('computing 2-D angular distances')
            if self.dimension_number != 2:
                raise ValueError('positions flagged as angular but have {} dimensions'.format(
                                 self.dimension_number))
            if periodic_length not in [360, 180, 2 * np.pi, np.pi]:
                self.say('positions flagged as angular, but weird periodic length = {:.1f}'.format(
                         periodic_length))
        else:
            if periodic_length in [360, 180, 2 * np.pi, np.pi]:
                self.say('positions flagged as not angular, but periodic length = {:.1f}'.format(
                         periodic_length))

        # find neighbors
        if find_kind == 'kd-tree':
            if self.is_angular:
                neig_distancess, neig_indicess = self.get_neighbors_kdtree_angular(
                    center_positions, neig_positions)
            else:
                neig_distancess, neig_indicess = self.get_neighbors_kdtree(
                    center_positions, neig_positions)
        elif find_kind == 'mesh':
            neig_distancess, neig_indicess = self.get_neighbors_mesh(
                center_positions, neig_positions)
        elif find_kind == 'direct':
            neig_distancess, neig_indicess = self.get_neighbors_direct(
                center_positions, neig_positions)
        else:
            raise ValueError('not recognize find_kind = ' + find_kind)

        # if input scale-factor, rescale distances to physical
        if scalefactor:
            for ci in range(len(neig_distancess)):
                neig_distancess[ci] *= scalefactor  # convert to [physical]

        if np.concatenate(neig_distancess).size:
            if self.print_diagnostics:
                ds = np.concatenate(neig_distancess)
                self.say('got {:d} neighbor distances in range = [{}, {}]'.format(
                    np.concatenate(neig_distancess).size,
                    io.get_string_from_numbers(ds.min(), 3, strip=True),
                    io.get_string_from_numbers(ds.max(), 3, strip=True))
                )
        else:
            self.say('! got no neighbors in distance limits = [{:.3e}, {:.3e}]'.format(
                     distance_limits.min(), distance_limits.max()))

        # check if neighbor list is saturated at neig_number_max
        neig_number_got_max = 0
        neig_number_max_distance_min = np.Inf
        for ci in range(len(neig_distancess)):
            if neig_distancess[ci].size >= self.neig_number_max:
                if neig_distancess[ci][-1] < neig_number_max_distance_min:
                    neig_number_max_distance_min = neig_distancess[ci][-1]
            elif neig_distancess[ci].size > neig_number_got_max:
                neig_number_got_max = neig_distancess[ci].size

        if neig_number_max_distance_min < np.Inf:
            self.say(
                '! reached neig_number_max = {} at min distance = {:.2f} - increase it!'.format(
                    self.neig_number_max, neig_number_max_distance_min))
        elif self.print_diagnostics:
            self.say('got maximum number of neighbors per center = {}'.format(neig_number_got_max))

        # apply mapping to go from neighbor index to input neighbor id array, if defined
        if neig_ids is not None:
            for ci in range(len(neig_indicess)):
                if neig_indicess[ci].size:
                    neig_indicess[ci] = neig_ids[neig_indicess[ci]]

        return neig_distancess, neig_indicess

    def get_neighbors_kdtree(self, center_positions, neig_positions):
        '''
        Get distances and indices of neighbors (neig_positions array), using k-d tree,
        with modifications to deal with periodic boundaries.

        Parameters
        ----------
        center_positions : array :
            positions around which to get neighbors (object number x dimension number)
        neig_positions : array : neighbors positions (object number x dimension number)
        '''
        neig_number_max = self.neig_number_max + 1  # accomodate k-d tree selecting self as neighbor

        # check need for periodic buffering
        neig_positions_buffer = neig_positions
        if self.periodic_length:
            positions_max = np.max(center_positions, 0)
            positions_min = np.min(center_positions, 0)
            # check if any centers are within distance_max from edge
            if ((positions_max.max() + self.distance_max) > self.periodic_length or
                    (positions_min.min() - self.distance_max) < 0):
                # add buffer volumes where centers are within distance_max of edge
                offsets = []
                for dimension_i in range(self.dimension_number):
                    offsets.append([0])
                    if positions_min[dimension_i] < self.distance_max:
                        offsets[dimension_i].append(-1)
                    if positions_max[dimension_i] > (self.periodic_length - self.distance_max):
                        offsets[dimension_i].append(1)

                # create buffer neighbors beyond cube edge
                neig_indices_orig = array.get_arange(self.neig_number)
                neig_indices_buffer = array.get_arange(self.neig_number)
                neig_positions_buffer = np.array(neig_positions)

                buffer_number = 0  # count how many buffer region have neighbors within distance_max
                if self.dimension_number == 3:
                    for offset_0 in offsets[0]:
                        # dimension 0
                        if offset_0 == 0:
                            neig_indices_near_edge_0 = neig_indices_orig
                        elif offset_0 == 1:
                            neig_indices_near_edge_0 = neig_indices_orig[
                                neig_positions[:, 0] < self.distance_max]
                        elif offset_0 == -1:
                            neig_indices_near_edge_0 = neig_indices_orig[
                                neig_positions[:, 0] > self.periodic_length - self.distance_max]

                        for offset_1 in offsets[1]:
                            # dimension 1
                            if offset_1 == 0:
                                neig_indices_near_edge_1 = neig_indices_near_edge_0
                            elif offset_1 == 1:
                                neig_indices_near_edge_1 = neig_indices_near_edge_0[
                                    neig_positions[neig_indices_near_edge_0, 1] <
                                    self.distance_max]
                            elif offset_1 == -1:
                                neig_indices_near_edge_1 = neig_indices_near_edge_0[
                                    neig_positions[neig_indices_near_edge_0, 1] >
                                    self.periodic_length - self.distance_max]

                            for offset_2 in offsets[2]:
                                # dimension 2
                                if offset_0 == offset_1 == offset_2 == 0:
                                    continue  # this is original volume adding to
                                elif offset_2 == 0:
                                    neig_indices_near_edge = neig_indices_near_edge_1
                                elif offset_2 == 1:
                                    neig_indices_near_edge = neig_indices_near_edge_1[
                                        neig_positions[neig_indices_near_edge_1, 2] <
                                        self.distance_max]
                                elif offset_2 == -1:
                                    neig_indices_near_edge = neig_indices_near_edge_1[
                                        neig_positions[neig_indices_near_edge_1, 2] >
                                        self.periodic_length - self.distance_max]

                                if neig_indices_near_edge.size:
                                    buffer_number += 1
                                    neig_poss_temp = np.array(
                                        neig_positions[neig_indices_near_edge])
                                    neig_poss_temp[:, 0] += offset_0 * self.periodic_length
                                    neig_poss_temp[:, 1] += offset_1 * self.periodic_length
                                    neig_poss_temp[:, 2] += offset_2 * self.periodic_length
                                    # append positions and indices to original volume + buffer array
                                    neig_positions_buffer = np.append(
                                        neig_positions_buffer, neig_poss_temp, 0)
                                    neig_indices_buffer = np.append(
                                        neig_indices_buffer, neig_indices_near_edge)

                elif self.dimension_number == 2:
                    for offset_0 in offsets[0]:
                        # dimension 0
                        if offset_0 == 0:
                            neig_indices_near_edge_0 = neig_indices_orig
                        elif offset_0 == 1:
                            neig_indices_near_edge_0 = neig_indices_orig[
                                neig_positions[:, 0] < self.distance_max]
                        elif offset_0 == -1:
                            neig_indices_near_edge_0 = neig_indices_orig[
                                neig_positions[:, 0] > self.periodic_length - self.distance_max]

                        for offset_1 in offsets[1]:
                            # dimension 1
                            if offset_0 == offset_1 == 0:
                                continue  # this is original volume adding to
                            elif offset_1 == 0:
                                neig_indices_near_edge = neig_indices_near_edge_0
                            elif offset_1 == 1:
                                neig_indices_near_edge = neig_indices_near_edge_0[
                                    neig_positions[neig_indices_near_edge_0, 1] <
                                    self.distance_max]
                            elif offset_1 == -1:
                                neig_indices_near_edge = neig_indices_near_edge_0[
                                    neig_positions[neig_indices_near_edge_0, 1] >
                                    self.periodic_length - self.distance_max]

                            if neig_indices_near_edge.size:
                                buffer_number += 1
                                neig_poss_temp = np.array(neig_positions[neig_indices_near_edge])
                                neig_poss_temp[:, 0] += offset_0 * self.periodic_length
                                neig_poss_temp[:, 1] += offset_1 * self.periodic_length
                                # append positions & indices to original volume + buffer array
                                neig_positions_buffer = np.append(
                                    neig_positions_buffer, neig_poss_temp, 0)
                                neig_indices_buffer = np.append(
                                    neig_indices_buffer, neig_indices_near_edge)

                if self.print_diagnostics:
                    self.say(
                        'creating {} neighbors in {} buffer regions for {} total neighbors'.format(
                            neig_indices_buffer.size - self.neig_number, buffer_number,
                            neig_indices_buffer.size))
        # end periodic buffering

        # create kd-tree
        KDTree = spatial.cKDTree(neig_positions_buffer)

        neig_distancess = [np.array([], dtype=np.float32) for _ in range(self.center_number)]
        neig_indicess = [np.array([], dtype=np.int32) for _ in range(self.center_number)]

        # find neighbors around each center
        if self.is_distance_adaptive:
            # input different distance limits for each center
            # so search using different neighbor distance limits & maximum number for each
            # scale neighbor maximum number by volume, according to each center's distance_max
            # assuming that input num_max corresponds to the largest distance_limits
            neig_number_maxs = np.ceil(
                neig_number_max *
                (self.distance_limits[:, 1] / self.distance_max) ** self.dimension_number
            ).astype(np.int32)

            for ci in range(self.center_number):
                neig_distancess[ci], neig_indicess[ci] = KDTree.query(
                    center_positions[ci], neig_number_maxs[ci])
                # distance_upper_bound=self.distance_limits[ci, 1])
                if np.isscalar(neig_distancess[ci]):
                    neig_distancess[ci] = [neig_distancess[ci]]
                    neig_indicess[ci] = [neig_indicess[ci]]

                neig_distancess[ci] = np.array(neig_distancess[ci], dtype=np.float32)
                neig_indicess[ci] = np.array(neig_indicess[ci], dtype=np.int32)
                if self.periodic_length:
                    # re-normalize indices of buffer neighbors back to original neighbor list
                    reals = (neig_indicess[ci] != neig_positions_buffer.shape[0])
                    neig_indicess[ci][reals] = neig_indices_buffer[neig_indicess[ci][reals]]
                # remove neighbors (& self) outside of distance limits
                masks = (neig_distancess[ci] >= self.distance_limits[ci, 0])
                masks *= (neig_distancess[ci] < self.distance_limits[ci, 1])
                neig_distancess[ci] = neig_distancess[ci][masks][:self.neig_number_max]
                neig_indicess[ci] = neig_indicess[ci][masks][:self.neig_number_max]
                if (len(neig_distancess[ci]) >= neig_number_maxs[ci] and
                        not self.hit_adaptive_distance_limit):
                    self.say('! filled adaptive neig_number_max list: increase neig_number_max')
                    self.hit_adaptive_distance_limit = True
        else:
            # input single distance limits, so for all centers use same search distance and number
            neig_distancess_raw, neig_indicess_raw = KDTree.query(
                center_positions, neig_number_max,  # distance_upper_bound=self.distance_max
            )
            neig_distancess_raw = neig_distancess_raw.astype(np.float32)
            neig_indicess_raw = neig_indicess_raw.astype(np.int32)
            if self.periodic_length and neig_positions_buffer.size > neig_positions.size:
                # re-normalize indices of buffer neighbors
                for ci in range(self.center_number):
                    reals = (neig_indicess_raw[ci] != neig_positions_buffer.shape[0])
                    neig_indicess_raw[ci, reals] = neig_indices_buffer[neig_indicess_raw[ci, reals]]

            # convert from padded array to apaptive list
            for ci in range(self.center_number):
                masks = (neig_distancess_raw[ci] >= self.distance_limits[ci, 0])
                masks *= (neig_distancess_raw[ci] < self.distance_limits[ci, 1])
                neig_distancess[ci] = neig_distancess_raw[ci][masks][:self.neig_number_max]
                neig_indicess[ci] = neig_indicess_raw[ci][masks][:self.neig_number_max]

        return neig_distancess, neig_indicess

    def get_neighbors_kdtree_angular(self, center_positions, neig_positions, is_recursive=False):
        '''
        Get neighbor distances and indices, using k-d tree for angular separations.

        Parameters
        ----------
        center_positions : array :
            positions around which to get neighbors (object number x dimension number)
        neig_positions : array : neighbors positions (object number x dimension number)
        is_recursive : bool : whether function is being called periodically
        '''
        neig_number_max = self.neig_number_max + 1  # deal with k-d tree self selection

        if self.periodic_length in [360, 180, 90]:
            ang_scale = np.pi / 180
        elif self.periodic_length in [2 * np.pi, np.pi]:
            ang_scale = 1
        else:
            raise ValueError('not recognize periodic angle = {}'.format(self.periodic_length))

        # not same as self because this function gets call recursively
        center_number = center_positions.shape[0]
        neig_number = neig_positions.shape[0]

        dec_max = abs(np.concatenate((center_positions[:, 1], neig_positions[:, 1]))).max()
        # maximum distance, in RA units, scaled to maximum dec, so should get everything
        distance_max_temp = self.distance_max / np.cos(ang_scale * dec_max)
        cis = array.get_arange(center_number)
        nis = array.get_arange(neig_number)

        neig_indicess = np.zeros((center_number, neig_number_max), np.int32) - 1
        neig_distancess = np.zeros((center_number, neig_number_max), np.float32)
        neig_distancess += np.Inf  # do this way to keep as float32

        # deal with those far from periodic edge
        cis_safe = cis  # farther than distance_max from edge
        nis_safe = nis  # farther than 2 * distance_max from edge
        cis_safe = array.get_indices(
            center_positions[:, 0], [distance_max_temp, self.periodic_length - distance_max_temp],
            cis_safe)
        nis_safe = array.get_indices(
            neig_positions[:, 0],
            [2 * distance_max_temp, self.periodic_length - 2 * distance_max_temp], nis_safe)

        KDTree = spatial.cKDTree(neig_positions)
        neig_distancess[cis_safe], neig_indicess[cis_safe] = KDTree.query(
            center_positions[cis_safe], neig_number_max, distance_upper_bound=distance_max_temp)

        if cis_safe.size != cis.size and nis_safe.size != nis.size:
            # treat those near an edge
            cis_edge = np.setdiff1d(cis, cis_safe)
            nis_edge = np.setdiff1d(nis, nis_safe)
            self.say('{} total | {} safe | {} near edge'.format(
                     cis.size, cis_safe.size, cis_edge.size))
            # deal with those at edge by shifting positions by 2 * distance_max
            cen_poss_edge = center_positions[cis_edge].copy()
            neig_poss_edge = neig_positions[nis_edge].copy()
            cen_poss_edge[(cen_poss_edge[:, 0] + distance_max_temp >
                           self.periodic_length), 0] -= self.periodic_length
            cen_poss_edge[:, 0] += 2 * distance_max_temp
            neig_poss_edge[(neig_poss_edge[:, 0] + 2 * distance_max_temp >
                            self.periodic_length), 0] -= self.periodic_length
            neig_poss_edge[:, 0] += 2 * distance_max_temp
            neig_distancess[cis_edge], neig_iis = self.get_neighbors_kdtree_angular(
                cen_poss_edge, neig_poss_edge, is_recursive=True)

            # fix neighbor id pointers
            for cii in range(cis_edge.size):
                niis_real = neig_iis[cii][neig_distancess[cis_edge[cii]] < np.Inf]
                neig_indicess[cis_edge[cii]][:niis_real.size] = nis_edge[niis_real]

        if not is_recursive:
            if neig_distancess[:, -1].min() < np.Inf:
                self.say(('! reached neig number max = {} before compute true angular distance\n' +
                          '  minimum distance(number_max) = {:.2f}').format(
                         self.neig_number_max, neig_distancess[:, -1].min()))

            # excise extra neighbor slot created to deal with self selection
            neig_distancess = neig_distancess[:, :-1]
            neig_indicess = neig_indicess[:, :-1]

            # fix angular separations & convert padded arrays to adaptive list
            neig_indicess_t = neig_indicess
            neig_distancess = [np.array([], dtype=np.float32) for _ in cis]
            neig_indicess = [np.array([], dtype=np.int32) for _ in cis]
            neig_iis_dummy = array.get_arange(self.neig_number_max)
            for ci in cis:
                neig_iis = neig_iis_dummy[neig_indicess_t[ci] < neig_number]
                if neig_iis.size:
                    neig_indicess[ci] = neig_indicess_t[ci, neig_iis]
                    neig_distancess[ci] = coordinate.get_distances_angular(
                        center_positions[ci], neig_positions[neig_indicess[ci]],
                        self.periodic_length)

                    # now that have computed real distances, keep only those within limits
                    masks = (neig_distancess[ci] >= self.distance_limits[ci, 0])
                    masks *= (neig_distancess[ci] < self.distance_limits[ci, 1])
                    neig_distancess[ci] = neig_distancess[ci][masks][:self.neig_number_max]
                    neig_indicess[ci] = neig_indicess[ci][masks][:self.neig_number_max]
                    if neig_indicess[ci].size:
                        # if neighbors within distance limits, sort by actual angular distance
                        neig_iis_sort = np.argsort(neig_distancess[ci])
                        neig_indicess[ci] = neig_indicess[ci][neig_iis_sort]
                        neig_distancess[ci] = neig_distancess[ci][neig_iis_sort]

        return neig_distancess, neig_indicess

    def get_neighbors_mesh(self, center_positions, neig_positions):
        '''
        Get neighbor distances and indices, using chaining mesh.

        Parameters
        ----------
        center_positions : array :
            positions around which to get neighbors (object number x dimension number)
        neig_positions : array : neighbors positions (object number x dimension number)
        '''

        def make_mesh(positions, mesh_number):
            '''
            Get dictionary list of position ids in each mesh cell, mesh size.

            Parameters
            ----------
            positions: array (object number x dimension number)
            mesh_number : int : number of mesh cells per dimension
            '''
            dimension_number = positions.shape[1]
            if self.periodic_length:
                x_min, x_max = 0, self.periodic_length
            else:
                x_min, x_max = positions.min(), positions.max() * 1.001

            mesh_poss, mesh_width = np.linspace(x_min, x_max, mesh_number + 1, True, True)
            mesh_bin_indices = binning.get_bin_indices(positions.flatten(), mesh_poss)
            mesh_bin_indices.shape = positions.shape
            mesh_position_indices = {}
            mesh_range = range(mesh_number)
            if dimension_number == 3:
                for x in mesh_range:
                    for y in mesh_range:
                        for z in mesh_range:
                            mesh_position_indices[(x, y, z)] = []
            elif dimension_number == 2:
                for x in mesh_range:
                    for y in mesh_range:
                        mesh_position_indices[(x, y)] = []
            for mii in range(mesh_bin_indices.shape[0]):
                mesh_position_indices[tuple(mesh_bin_indices[mii])].append(mii)

            return mesh_position_indices, mesh_width

        number_per_mesh = 8  # number of neighbor positions per mesh (assuming average density)

        # make lists to store neighbor indices and distances
        neig_distancess = [np.array([], np.float32) for _ in range(self.center_number)]
        neig_indicess = [np.array([], np.int32) for _ in range(self.center_number)]

        # number of mesh points per dimension (make sure is even)
        mesh_number = 2 * int((self.center_number / number_per_mesh /
                               2 ** self.dimension_number) ** (1 / self.dimension_number))
        self.say('using {} mesh cells per dimension'.format(mesh_number))

        mesh_position_indices, _ = make_mesh(center_positions, mesh_number)
        mesh_neig_indices, mesh_width = make_mesh(neig_positions, mesh_number)
        loop_number = int(self.distance_max / mesh_width) + 1
        mesh_loops = range(-loop_number, loop_number + 1)
        loop_number_2 = (loop_number + 1) ** 2

        for mesh_point in mesh_position_indices.keys():
            if len(mesh_position_indices[mesh_point]) > 0:
                neig_indices_mesh = []  # indices of position array
                if self.dimension_number == 3:
                    for x in mesh_loops:
                        xx = (x + mesh_point[0]) % mesh_number
                        for y in mesh_loops:
                            yy = (y + mesh_point[1]) % mesh_number
                            for z in mesh_loops:
                                if x ** 2 + y ** 2 + z ** 2 < loop_number_2:
                                    zz = (z + mesh_point[2]) % mesh_number
                                    neig_indices_mesh.extend(mesh_neig_indices[(xx, yy, zz)])

                elif self.dimension_number == 2:
                    for x in mesh_loops:
                        xx = (x + mesh_point[0]) % mesh_number
                        for y in mesh_loops:
                            if x ** 2 + y ** 2 < loop_number_2:
                                yy = (y + mesh_point[1]) % mesh_number
                                neig_indices_mesh.extend(mesh_neig_indices[(xx, yy)])

                if len(neig_indices_mesh) > 1:
                    # neighbor list has more than just self
                    neig_indices_mesh = np.array(neig_indices_mesh, np.int32)
                    nis_mesh = array.get_arange(neig_indices_mesh)
                    # loop over object indices in mesh point
                    for ci in mesh_position_indices[mesh_point]:
                        if self.is_angular:
                            distances_ci = coordinate.get_distances_angular(
                                center_positions[ci], neig_positions[neig_indices_mesh],
                                self.periodic_length)
                        else:
                            distances_ci = coordinate.get_distances(
                                neig_positions[neig_indices_mesh], center_positions[ci],
                                self.periodic_length, total_distance=True)

                        # keep neighbors within distance limits
                        masks = (distances_ci >= self.distance_limits[ci, 0])
                        masks *= (distances_ci < self.distance_limits[ci, 1])
                        nis_real = nis_mesh[masks]
                        distances_ci = distances_ci[masks]
                        # sort neighbors by distance
                        niis_sort = np.argsort(distances_ci)[:self.neig_number_max]
                        neig_indicess[ci] = neig_indices_mesh[nis_real[niis_sort]]
                        neig_distancess[ci] = distances_ci[niis_sort]

        return neig_distancess, neig_indicess

    def get_neighbors_direct(self, center_positions, neighbor_positions):
        '''
        Get neighbor distances and indices, using direct n^2 calcuation.

        Parameters
        ----------
        center_positions : array :
            positions around which to count neighbors (object number x dimension number)
        neighbor_positions : array : neighbors positions (object number x dimension number)
        '''
        # make lists to store neighbor distances and indices
        neig_distancess = [np.array([], dtype=np.float32) for _ in range(self.center_number)]
        neig_indicess = [np.array([], dtype=np.int32) for _ in range(self.center_number)]

        neig_indices_all = array.get_arange(self.neig_number)

        for ci in range(self.center_number):
            if self.is_angular:
                distances_ci = coordinate.get_distances_angular(
                    neighbor_positions, center_positions[ci], self.periodic_length,
                    total_distance=True)
            else:
                distances_ci = coordinate.get_distances(
                    neighbor_positions, center_positions[ci], self.periodic_length,
                    total_distance=True)

            # keep neighbors closer than distance_max that are not self
            masks = (distances_ci >= self.distance_limits[ci, 0])
            masks *= (distances_ci < self.distance_limits[ci, 1])
            distances_ci = distances_ci[masks]
            neig_indices_real = neig_indices_all[masks]

            # sort neighbors by distance
            niis_sort = np.argsort(distances_ci)[:self.neig_number_max]
            neig_indicess[ci] = neig_indices_real[niis_sort]
            neig_distancess[ci] = distances_ci[niis_sort]

        return neig_distancess, neig_indicess


Neighbor = NeighborClass()


def test(center_positions, neighbor_positions, distance_limits=[1e-6, 10], neig_number_max=50,
         periodic_length=None, is_angular=False):
    '''
    Test different neighbor finding methods.
    '''
    find_kinds = ['kd-tree', 'mesh', 'direct']

    if not periodic_length or is_angular:
        find_kinds.remove('mesh')

    neigs = []
    for find_kind in find_kinds:
        neigs.append(Neighbor.get_neighbors(
            center_positions, neighbor_positions, distance_limits, neig_number_max, periodic_length,
            is_angular, find_kind))

    print(find_kinds[0], find_kinds[1])
    print(np.sum(np.concatenate(neigs[0][1]) != np.concatenate(neigs[1][1])))
    print(np.max(np.abs(np.concatenate(neigs[0][0]) - np.concatenate(neigs[1][0]))))

    if len(neigs) > 2:
        print(find_kinds[0], find_kinds[2])
        print(np.sum(np.concatenate(neigs[0][1]) != np.concatenate(neigs[2][1])))
        print(np.max(np.abs(np.concatenate(neigs[0][0]) - np.concatenate(neigs[2][0]))))
        print(find_kinds[1], find_kinds[2])
        print(np.sum(np.concatenate(neigs[1][1]) != np.concatenate(neigs[2][1])))
        print(np.max(np.abs(np.concatenate(neigs[1][0]) - np.concatenate(neigs[2][0]))))

    return neigs
