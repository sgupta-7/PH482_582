'''
Utilities for setting up and running simulations with Gizmo or CART.

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
# local ----
from . import basic as ut


#===================================================================================================
# snapshots
#===================================================================================================
class SnapshotClass(dict, ut.io.SayClass):
    '''
    Dictionary class to store/print redshifts, scale-factors, times, to use for simulation.
    '''

    def __init__(self):
        self.info = {'scaling': None}

    def get_snapshot_indices(self, time_kind='redshift', values=[], round_kind='near'):
        '''
        Get index[s] in snapshot list where values are closest to, using input round_kind.

        Parameters
        ----------
        time_kind : str : time kind for values: 'redshift', 'scalefactor', 'time'
        values_test : float or array : redshift[s] / scale-factor[s] / time[s] to get index of
        round_kind : str : method to identify nearest snapshot: 'up', 'down', 'near'

        Returns
        -------
        snapshot_indices : int or array : index number[s] of snapshot file
        '''
        snapshot_values = np.sort(self[time_kind])  # sort redshifts because are in reverse order

        scalarize = False
        if np.isscalar(values):
            values = [values]
            scalarize = True  # if input scalar value, return scalar value (instead of array)

        assert (np.min(values) >= np.min(snapshot_values) and
                np.max(values) <= np.max(snapshot_values))

        snapshot_indices = ut.binning.get_bin_indices(
            values, snapshot_values, round_kind=round_kind)

        if time_kind == 'redshift':
            # because had to sort redshifts in increasing order, have to reverse indices
            snapshot_indices = (snapshot_values.size - 1) - snapshot_indices

        if scalarize:
            snapshot_indices = snapshot_indices[0]

        return snapshot_indices

    def parse_snapshot_values(self, snapshot_value_kind, snapshot_values, verbose=True):
        '''
        Convert input snapshot value[s] to snapshot index[s].

        Parameters
        ----------
        snapshot_value_kind : str : kind of number that am supplying:
            'redshift', 'scalefactor', 'time', 'index'
        snapshot_value : int or float or array thereof: corresponding value[s]
        verbose : bool : whether to print conversions

        Returns
        -------
        snapshot_index : int : index[s] of snapshot file
        '''
        if snapshot_value_kind in ['redshift', 'scalefactor', 'time']:
            snapshot_indices = self.get_snapshot_indices(snapshot_value_kind, snapshot_values)
            snapshot_time_kind = snapshot_value_kind
            snapshot_time_values = self[snapshot_value_kind][snapshot_indices]
            self.say('* input {} = {}:'.format(
                     ut.array.scalarize(snapshot_value_kind),
                     ut.array.scalarize(snapshot_values)), verbose, end='')
        else:
            snapshot_indices = snapshot_values
            snapshot_time_kind = 'redshift'
            snapshot_time_values = self['redshift'][snapshot_indices]

        if np.isscalar(snapshot_indices) or len(snapshot_indices) == 1:
            self.say('using snapshot index = {}, {} = {:.3f}\n'.format(
                ut.array.scalarize(snapshot_indices), snapshot_time_kind, snapshot_time_values),
                verbose)
        else:
            self.say('* using snapshot indices = {}'.format(
                ut.array.scalarize(snapshot_indices)), verbose)
            self.say('{}s = {}\n'.format(snapshot_time_kind, snapshot_time_values), verbose)

        return snapshot_indices

    def get_redshifts(self, scaling, redshift_limts, number):
        '''
        Get list of redshifts spaced linearly or logarithmically in scale-factor.

        Parameters
        ----------
        scaling : str : scaling of bins: 'log', 'linear'
            note: compute linear or log spacings in terms of *scale-factor*, not redshift
        redshift_limits: list : min and max limits for redshifts
        number : int : number of redshifts

        Returns
        -------
        redshifts : array
        '''
        redshift_limits = np.array([np.max(redshift_limts), np.min(redshift_limts)])
        scale_limits = np.sort(1 / (1 + redshift_limits))

        if 'log'in scaling:
            scalefactors = np.logspace(
                np.log10(scale_limits.min()), np. log10(scale_limits.max()), number)
        else:
            scalefactors = np.linspace(scale_limits.min(), scale_limits.max(), number)

        return 1 / scalefactors - 1

    def generate_snapshots(
        self, redshifts=[30, 25, 20, 19, 18, 17, 16, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        number=601, scaling='linear', redshifts_append=[99, 19], redshifts_hires=[0],
        factor_hires=10, Cosmology=None):
        '''
        Assign scale-factors, redshifts, [ages, time spacings] to self.
        Use to determine snapshots for simulation.

        Parameters
        ----------
        redshifts : list
            note: if two values, treat at limits. if several values, treat its min, max as limits
            and ensure that other values are in return array
        number : int : number of redshifts
        scaling : str : scaling of bins: 'log', 'linear'
            note: compute linear or log spacings in *scale-factor*, not redshift (better behaved)
        redshifts_append : list : redshifts to append to list (but not use to compute spacings)
        redshifts_hires : list : redshifts at which to sub-sample at higher resolution
        factor_hires : int : factor by which to sub-sample at higher time resolution
        Cosmology : cosmology class (if want to generate times corresponding to scale-factors)
        '''
        redshifts_ensure = np.sort(redshifts)[::-1]  # sort from early to late time
        redshifts_append = np.sort(redshifts_append)[::-1]
        redshifts_hires = np.sort(redshifts_hires)[::-1]

        redshift_limits = np.array([redshifts_ensure.max(), redshifts_ensure.min()])

        number_spacing = number - redshifts_append.size - redshifts_hires.size * (factor_hires - 1)

        self['redshift'] = self.get_redshifts(scaling, redshift_limits, number_spacing)

        if redshifts_ensure.size > 2:
            # ensure that input redshifts are in list
            for redshift_ensure in redshifts_ensure[1:-1]:
                z_i = self.get_snapshot_indices('redshift', redshift_ensure)
                self['redshift'][z_i] = redshift_ensure

            for z_ensure_i in range(redshifts_ensure.size - 1):
                z_ensure_hi = redshifts_ensure[z_ensure_i]
                z_ensure_lo = redshifts_ensure[z_ensure_i + 1]
                zi_hi = self.get_snapshot_indices('redshift', z_ensure_hi)
                zi_lo = self.get_snapshot_indices('redshift', z_ensure_lo)
                self['redshift'][zi_hi: zi_lo + 1] = self.get_redshifts(
                    scaling, [z_ensure_hi, z_ensure_lo], zi_lo - zi_hi + 1)

        # append redshifts outside of spaced limits
        if redshifts_append.size:
            redshifts_append = np.sort(redshifts_append)[::-1]

            redshifts_hi = redshifts_append[redshifts_append > self['redshift'].max()]
            if redshifts_hi.size:
                self['redshift'] = np.concatenate([redshifts_hi, self['redshift']])

            redshifts_lo = redshifts_append[redshifts_append < self['redshift'].min()]
            if redshifts_lo.size:
                self['redshift'] = np.concatenate([self['redshift'], redshifts_lo])

        # sample input redshifts at higher time resolution
        if redshifts_hires.size:
            for redshift_hires in redshifts_hires:
                z_i = self.get_snapshot_indices('redshift', redshift_hires)
                redshifts_hires = self.get_redshifts(
                    scaling, [self['redshift'][z_i - 1], self['redshift'][z_i]], factor_hires + 1)
                redshifts_mod = np.zeros(self['redshift'].size + factor_hires - 1)
                redshifts_mod[:z_i - 1] = self['redshift'][:z_i - 1]
                redshifts_mod[z_i - 1: z_i - 1 + factor_hires] = redshifts_hires[:-1]
                redshifts_mod[z_i - 1 + factor_hires:] = self['redshift'][z_i:]
                self['redshift'] = redshifts_mod

        self['scalefactor'] = 1 / (1 + self['redshift'])

        # scalefactor_wids = np.zeros(self['scalefactor'].size)
        # scalefactor_wids[1:] = self['scalefactor'][1:] - self['scalefactor'][:-1]

        if Cosmology is not None:
            self._assign_snapshot_times(Cosmology)

        self['index'] = np.arange(self['redshift'].size)

        self.info['scaling'] = scaling

    def _assign_snapshot_times(self, Cosmology):
        '''
        Assign ages and time spacings to self.

        Parameters
        ----------
        Cosmology : cosmology class (to generate times corresponding to scale-factors)
        '''
        self['time'] = Cosmology.get_time(self['redshift'])  # [Gyr]
        self['time.width'] = np.zeros(self['time'].size)
        self['time.width'][1:] = (self['time'][1:] - self['time'][:-1]) * 1000  # [Myr]

        self.info['source'] = Cosmology.source

    def read_snapshots(self, file_name='snapshot_times.txt', directory='.'):
        '''
        Read scale-factors, [redshifts, times, time spacings] from file.
        Assign to self dictionary.

        Parameters
        ----------
        file_name : str : name of file that contains list of snapshots
        directory : str : directory of snapshot file
        '''
        path_file_name = ut.io.get_path(directory) + file_name

        if 'times' in file_name:
            snap = np.loadtxt(path_file_name, comments='#', dtype=[
                ('index', np.int32),
                ('scalefactor', np.float32),
                ('redshift', np.float32),
                ('time', np.float32),  # [Gyr]
                ('time.width', np.float32),
            ])
            for k in snap.dtype.names:
                self[k] = snap[k]

        elif 'scale-factors' in file_name:
            scalefactors = np.loadtxt(path_file_name, np.float32)
            self['index'] = np.arange(scalefactors.size, dtype=np.int32)
            self['scalefactor'] = scalefactors
            self['redshift'] = 1 / self['scalefactor'] - 1

        self.say('* reading:  {}\n'.format(path_file_name.strip('./')))

    def print_snapshots(
        self, write_file=False, print_times=False, file_name='snapshot_times.txt', directory='.',
        subsample_factor=0, redshift_max=None):
        '''
        Print snapshot time information from self to screen or file.

        Parameters
        ----------
        write_file : bool : whether to write to file
        print_times : bool : whether to print scale-factor + redshfit + time + time width
        file_name : str : name for snapshot file
        directory : str : directory for snapshot file
        subsample_factor : int : factor by which to subsample snapshot times
        '''
        file_out = None
        if write_file:
            path_file_name = ut.io.get_path(directory) + file_name
            file_out = open(path_file_name, 'w')

        Write = ut.io.WriteClass(file_out)

        snapshot_indices = np.array(self['index'])

        if subsample_factor > 1:
            # sort backwards in time to ensure get snapshot at z = 0
            snapshot_indices = snapshot_indices[::-1]
            snapshot_indices = snapshot_indices[::subsample_factor]
            snapshot_indices = snapshot_indices[::-1]

        if redshift_max:
            snapshot_indices = snapshot_indices[self['redshift'][snapshot_indices] <= redshift_max]

        if print_times:
            Write.write('# {} snapshots'.format(self['scalefactor'].size))
            if self.info['scaling']:
                Write.write('# {} scaling in scale-factor'.format(self.info['scaling']))
            if subsample_factor > 1:
                Write.write('# subsampling every {} snapshots'.format(subsample_factor))
            Write.write('# times assume cosmology from {}'.format(self.info['source']))
            Write.write('# i scale-factor redshift time[Gyr] time_width[Myr]')
            for snap_i in snapshot_indices:
                if 'time' in self:
                    Write.write('{:3d} {:9.7f} {:11.8f} {:11.8f} {:9.5f}'.format(
                                snap_i, self['scalefactor'][snap_i], self['redshift'][snap_i],
                                self['time'][snap_i], self['time.width'][snap_i]))
                else:
                    Write.write('{:3d} {:9.7f} {:11.8f}'.format(
                                snap_i, self['scalefactor'][snap_i], self['redshift'][snap_i]))
        else:
            for snap_i in snapshot_indices:
                Write.write('{:9.7f}'.format(self['scalefactor'][snap_i]))

        if file_out is not None:
            file_out.close()


Snapshot = SnapshotClass()


def read_snapshot_times(directory='.'):
    '''
    Within imput directory, search for and read snapshot file,
    that contains scale-factors[, redshifts, times, time spacings].
    Return as dictionary.

    Parameters
    ----------
    directory : str : directory where snapshot time/scale-factor file is

    Returns
    -------
    Snapshot : dictionary class : snapshot information
    '''
    Snapshot = SnapshotClass()

    try:
        try:
            Snapshot.read_snapshots('snapshot_times.txt', directory)
        except OSError:
            Snapshot.read_snapshots('snapshot_scale-factors.txt', directory)
    except OSError:
        raise OSError('cannot find file of snapshot times in {}'.format(directory))

    return Snapshot


#===================================================================================================
# particle/cell properties
#===================================================================================================
class ParticlePropertyClass(ut.io.SayClass):
    '''
    Calculate properties (such as mass, size) of particles and cells in a simulation of given size,
    number of particles, and cosmology.
    '''

    def __init__(self, Cosmology):
        '''
        Store variables from cosmology class and spline for converting between virial density
        definitions.

        Parameters
        ----------
        Cosmology : class : cosmology information
        '''
        self.Cosmology = Cosmology

    def get_particle_mass(self, simulation_length, number_per_dimension, particle_kind='combined'):
        '''
        Get particle mass [M_sun] for given cosmology.

        Parameters
        ----------
        simulation_length : float : box length [kpc comoving]
        number_per_dimension : int : number of particles per dimension
        particle_kind : str : 'combined' = dark + gas (for n-body only), 'dark', 'gas'

        Returns
        -------
        mass : float [M_sun]
        '''
        mass = (ut.constant.density_critical_0 * self.Cosmology['hubble'] ** 2 *
                self.Cosmology['omega_matter'] * (simulation_length / number_per_dimension) ** 3)

        if particle_kind == 'combined':
            pass
        elif particle_kind == 'dark':
            mass *= 1 - self.Cosmology['omega_baryon'] / self.Cosmology['omega_matter']
        elif particle_kind == 'gas':
            mass *= self.Cosmology['omega_baryon'] / self.Cosmology['omega_matter']
        else:
            raise ValueError('not recognize particle_kind = {}'.format(particle_kind))

        return mass

    def get_cell_length(
        self, simulation_length=25000 / 0.7, grid_root_number=7, grid_refine_number=8,
        redshift=None, units='kpc comoving'):
        '''
        Get length of grid cell at refinement level,
        in units [comoving or physical] corresponding to simulation_length.

        Parameters
        ----------
        simulation_length : float : box length [kpc comoving]
        grid_root_number : int : number of root grid refinement levels
        grid_refine_number : int : number of adaptive refinement levels
        redshift : float
        units : str : 'kpc[/h]', 'pc/h', 'cm' + 'comoving' or 'physical'

        Returns
        -------
        length : float : size of cell
        '''
        length = simulation_length / 2 ** (grid_root_number + grid_refine_number)
        if units[:3] == 'kpc':
            pass
        elif units[:2] == 'pc':
            length *= ut.constant.kilo
        elif units[:2] == 'cm':
            length *= ut.constant.cm_per_kpc
        else:
            raise ValueError('not recognize units = {}'.format(units))

        if '/h' in units:
            length *= self.Cosmology['hubble']

        if 'physical' in units:
            if redshift is None:
                raise ValueError('need to input redshift to scale to physical length')
            else:
                length /= (1 + redshift)
        elif 'comoving' in units:
            pass
        else:
            raise ValueError('need to specify comoving or physical in units = {}'.format(units))

        return length

    def get_gas_mass_per_cell(
        self, number_density, simulation_length=25000 / 0.7, grid_root_number=7,
        grid_refine_number=8, redshift=0, units='M_sun'):
        '''
        Get mass in cell of given size at given number density.

        Parameters
        ----------
        number_density : float : hydrogen number density [cm^-3 physical]
        simulation_length : float : box length [kpc comoving]
        grid_root_number : int : number of root grid refinement levels
        grid_refine_number : int : number of adaptive refinement levels
        redshift : float
        units : str : mass units: g, M_sun, M_sun/h

        Returns
        -------
        mass : float : mass of cell
        '''
        cell_length = self.cell_length(
            simulation_length, grid_root_number, grid_refine_number, redshift, 'cm physical')
        mass = ut.constant.proton_mass * number_density * cell_length ** 3

        if 'g' in units:
            pass
        elif 'M_sun' in units:
            mass /= ut.constant.sun_mass
        else:
            raise ValueError('not recognize units = {}'.format(units))

        if '/h' in units[-2:]:
            mass *= self.Cosmology['hubble']

        return mass

    def get_star_formation_rate_density(
        self, gas_density, timescale=4e9 / 1.5, slope=1.5, gas_density_0=0.01):
        '''
        Standard model for star formation rate density in CART.

        SFR_density = gas_density / timescale * (gas_density / density_0) ^ (1 - slope)

        Parameters
        ----------
        gas_density : float : gas density [M_sun / pc^3]
        timescale : float : SFR (cloud collapse) timescale (includes efficiency parameter [Gyr]
        slope : float
        gas_density_0 : float : density_0 [M_sun / pc^3]

        Returns
        -------
        star formation rate : float
        '''
        return gas_density / timescale * (gas_density / gas_density_0) ** (1 - slope)

    def get_star_form_mass(
        self, number_density=1, sfr_duration=3e7, sfr_timescale=4e9 / 1.5, sfr_slope=1.5,
        gas_density_0=0.01, simulation_length=25000 / 0.7, grid_root_number=7, grid_refine_number=8,
        redshift=0):
        '''
        .
        '''
        cell_gas_mass = self.gas_mass_per_cell(
            number_density, simulation_length, grid_root_number, grid_refine_number, redshift,
            units='M_sun')
        cell_length = self.cell_length(
            simulation_length, grid_root_number, grid_refine_number, redshift, units='pc physical')
        cell_gas_density = cell_gas_mass / cell_length ** 3
        sfr_density = self.star_formation_rate_density(
            cell_gas_density, sfr_timescale, sfr_slope, gas_density_0)

        return sfr_density * sfr_duration * cell_length ** 3
