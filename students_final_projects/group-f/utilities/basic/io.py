'''
Utility functions for reading, writing, printing.

@author: Andrew Wetzel <arwetzel@gmail.com>
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import os
import glob
import time
import datetime
import numpy as np
import h5py


#===================================================================================================
# print at run-time
#===================================================================================================
class ListPropClass:
    '''
    Print self and attributes in nice format.

    Use by importing this class into another class.
    '''

    def __repr__(self):
        return '< Instance of {}:\n{}>'.format(self.__module__ + '.' + self.__class__.__name__,
                                               self.get_attr_names())

    def get_attr_names(self):
        result = ''
        for attr in self.__dict__.keys():
            if attr[:2] == '__':
                result += '  self.{} = <built-in>\n'.format(attr)
            else:
                result += '  self.{} = {}\n'.format(attr, self.__dict__[attr])

        return result


class SayClass(ListPropClass):
    '''
    Print comments and diagnostics at run time in nice format.

    Use by importing this class into antother class or declaring within a function.
    '''
    _is_first_print = True

    def __init__(self, func=None):
        '''
        Parameters
        ----------
        func : function (to get its name for printing), if not using SayClass within a class
        '''
        if func is not None:
            self.func_name = func.__module__ + '.' + func.__name__
        elif 'SayClass' in str(self.__class__):
            raise ValueError('need to pass function to get name')

    def say(self, string, verbose=True, end='\n'):
        '''
        Print string in nice format.

        Parameters
        ----------
        string : str : str to print
        verbose : bool : whether to print
        end : str : end of string to pass to print()
        '''
        if not verbose:
            return

        if string:
            if string[0] != '!' and string[0] != '*' and string[0] != '#':
                string = '  ' + string

            if self._is_first_print:
                print()
                if 'SayClass' in str(self.__class__):
                    print('# in {}():'.format(self.func_name))
                else:
                    print('# in {}():'.format(self.__module__ + '.' +
                                              self.__class__.__name__.replace('Class', '')))

                self._is_first_print = False

        print(string, end=end)

        os.sys.stdout.flush()


class WriteClass:
    '''
    Class to store behavior for printing - either print to stdout or write to file or both.
    '''

    def __init__(self, file_out=None, print_stdout=False):
        '''
        Parameters
        ----------
        file_out : file object : output file to write to
        print_also : whether to print to stout in addition to writing to file
        '''
        self.file_out = file_out
        self.print_stdout = print_stdout

    def write(self, string, print_stdout=None):
        '''
        Print string to stdout or to file, depending on whether self.file_out is defined.

        Parameters
        ----------
        string : str : str to write
        print_also : whether to print to stout in addition to writing to file
        '''
        if self.file_out:
            self.file_out.write(string + '\n')
            if print_stdout is None:
                if self.print_stdout:
                    print(string)
            elif print_stdout:
                print(string)
        else:
            print(string)


def print_flush(string, end='\n'):
    '''
    Print and instantly flush std output.

    Parameters
    ----------
    string : str
    '''
    print(string, end=end)

    os.sys.stdout.flush()


def get_string_from_numbers(values, digits=3, exponential=None, strip=False):
    '''
    Get string of number[s] in nice format.

    Parameters
    ----------
    value[s] : int/float or list thereof
    digits : int : number of digits after period
    exponential : bool : whether to use exponential (instead of float) notation
      if None, choose automatically
    strip : bool : whether to strip trailing 0s (and .)

    Returns
    -------
    string
    '''
    if not np.isscalar(values):
        # multiple values, need to parse
        strings = ''
        for value in values:
            strings += get_string_from_numbers(value, digits) + ', '
        return strings[:-2]

    # single value
    value = values

    if not np.isfinite(value):
        string = '{}'
    elif exponential is False:
        # use float format
        string = '{{:.{}f}}'.format(digits)
    elif exponential or np.abs(value) >= 1e5 or 0 < np.abs(value) <= 1e-4:
        # use exponential format
        exponential = True
        string = '{{:.{}e}}'.format(digits)
    elif isinstance(value, int) or (isinstance(value, np.ndarray) and 'int' in str(value.dtype)):
        string = '{:d}'
    else:
        string = '{{:.{}f}}'.format(digits)

    if exponential:
        string = string.format(value)
        string = string.replace('+', '')
        string = string.replace('e0', 'e')
        string = string.replace('e-0', 'e-')

        if strip:
            strings = string.split('e')
            strings[0] = strings[0].rstrip('0').rstrip('.')
            string = strings[0] + 'e' + strings[1]

    else:
        string = string.format(values)

        if strip:
            string = string.rstrip('0').rstrip('.')

    return string


def print_array(values, form='{:.3f}', delimeter=', ', end='\n'):
    '''
    Print values of array in nice format.

    Parameters
    ----------
    values : array-like : values
    form : str : format to print
    delimeter : str : delimeter to print between numbers
    end : str : ending of string to pass to print()
    '''
    if np.isscalar(values):
        string = form
    else:
        string = form + delimeter

    for value in values[:-1]:
        print(string.format(value), end='')
    print(form.format(values[-1]), end=end)


#===================================================================================================
# print within batch script
#===================================================================================================
class SubmissionScriptClass():
    '''
    Helper functions and run-time information for batch submission script.
    '''

    def __init__(self, scheduler_name='slurm', node_number=None, mpi_number_per_node=None):
        '''
        Get type of batch scheduler.
        Print pre-run information on job settings.
        Only need to import parameters if using PBS scheduler.

        Parameters
        ----------
        scheduler_name : str : 'slurm' or 'pbs'
        node_number : int : number of nodes
        mpi_number_per_node : int : number of MPI tasks per node
        '''
        assert scheduler_name in ('slurm', 'pbs')

        self.scheduler_name = scheduler_name

        os.system('date')
        print('')

        if 'slurm' in self.scheduler_name:
            self.node_number = int(os.environ['SLURM_NNODES'])
            self.mpi_number_per_node = int(os.environ['SLURM_TASKS_PER_NODE'])
            self.mpi_number = self.mpi_number_per_node * self.node_number

            if 'SLURM_CPUS_PER_TASK' in os.environ:
                # copy to value to OMP_NUM_THREADS to enable OpenMP in Gizmo
                os.environ['OMP_NUM_THREADS'] = os.environ['SLURM_CPUS_PER_TASK']
                self.omp_number = int(os.environ['SLURM_CPUS_PER_TASK'])
                if self.omp_number < 1:
                    self.omp_number = 1
            else:
                self.omp_number = 1

            print('job name = {}'.format(os.environ['SLURM_JOB_NAME']))
            print('job id = {}\n'.format(os.environ['SLURM_JOB_ID']))
            print('using the following resources:')
            print('  {} nodes'.format(os.environ['SLURM_NNODES']))
            print('  {} MPI tasks per node, {} MPI tasks total'.format(
                os.environ['SLURM_TASKS_PER_NODE'], os.environ['SLURM_NTASKS']))
            if self.omp_number > 1:
                print('  {} OpenMP threads per MPI task'.format(os.environ['OMP_NUM_THREADS']))

        elif 'pbs' in self.scheduler_name:
            assert node_number and mpi_number_per_node
            self.node_number = node_number
            self.mpi_number_per_node = mpi_number_per_node
            self.mpi_number = self.mpi_number_per_node * self.node_number
            self.omp_number = int(os.environ['OMP_NUM_THREADS'])

            print('job name = {}'.format(os.environ['PBS_JOBNAME']))
            print('job id = {}\n'.format(os.environ['PBS_JOBID']))
            print('using the following resources:')
            print('  {} nodes'.format(self.node_number))
            print('  {} MPI tasks per node, {} MPI tasks total'.format(
                self.mpi_number_per_node, self.mpi_number))
            if self.omp_number >= 1:
                print('  {} OpenMP threads per MPI task'.format(self.omp_number))
                if self.omp_number < 1:
                    self.omp_number = 1
            else:
                self.omp_number = 1  # ensure 1 for sanity

        self.core_number = self.mpi_number * self.omp_number
        print('  {} cores\n'.format(self.core_number))
        os.sys.stdout.flush()

        self.time_ini = time.time()

    def print_runtime(self):
        '''
        Print run time information of job.
        '''
        time_dif = time.time() - self.time_ini
        time_dif_str = str(datetime.timedelta(seconds=time_dif))

        print('')
        print('wall time: {:.0f} sec = {:.2f} day = {}'.format(
              time_dif, time_dif / 3600 / 24, time_dif_str.split('.')[0]))
        print('cpu time: {:.1f} hr\n'.format(time_dif * self.core_number / 3600))
        os.sys.stdout.flush()

        os.system('date')

    def check_existing_job(self, file_name, submit_before=False, time_delay=4):
        '''
        Check if defined/input jobid that needs to finish before this starts.

        Parameters
        ----------
        file_name : str : name of submission script file
        submit_before : bool :
            whether to submit a follow-up job before this one starts (to wait in queue)
        time_delay : float :
            how long [hr] to wait if follow-up tries to start before current finishes
        '''
        import subprocess

        if 'slurm' in self.scheduler_name:
            if len(os.sys.argv) > 1 and int(os.sys.argv[1]) > 0:
                # ensure previous job id finished before start this job
                prev_job_id = str(os.sys.argv[1])
                print('checking if previous job id = {} still is running'.format(prev_job_id))
                processes = str(subprocess.check_output('squeue -u $(id -un)', shell=True), 'utf')
                if prev_job_id in processes:
                    print('! previous job id = {} not finished - bailing!'.format(prev_job_id))
                    # make next job wait to avoid rapid string of restarts
                    #restart_time = time.strftime('%Y%m%d%H%M',
                    # time.localtime(time.time() + time_delay * 3600))
                    os.system('sbatch {} {}'.format(file_name, prev_job_id))
                    exit()
                print('')
            elif submit_before:
                # submit next job, to wait for this one to finish
                os.system('sbatch {} {}'.format(file_name, os.environ['SLURM_JOB_ID']))

        elif 'pbs' in self.scheduler_name:
            if 'jobid' in os.environ and os.environ['jobid']:
                # ensure previous job id finished before start this job
                prev_job_id = os.environ['jobid']
                print('checking if previous job id = {} still is running'.format(prev_job_id))
                processes = str(subprocess.check_output('qstat -u $(id -un)', shell=True), 'utf')
                if prev_job_id in processes:
                    print('! previous job id = {} not finished - bailing!'.format(prev_job_id))
                    # make next job wait to avoid rapid string of restarts
                    restart_time = time.strftime(
                        '%Y%m%d%H%M', time.localtime(time.time() + time_delay * 3600))
                    os.system('qsub -v jobid={} -a {} {}'.format(
                        prev_job_id, restart_time, file_name))
                    exit()
                print('')
            elif submit_before:
                # submit next job, to wait for this one to finish
                os.system('qsub -v jobid={} {}'.format(os.environ['PBS_JOBID'], file_name))

    def get_restart_flag(self, restart_from_snapshot=False):
        '''
        Parameters
        ----------
        restart_from_snapshot : bool : whether to restart from snapshot file[s]

        Returns
        -------
        execute_command_restart_flag : str : flag to add to executable
        '''
        restart_file_name_base = 'output/restartfiles/restart.*'
        execute_command_restart_flag = ''

        if restart_from_snapshot:
            execute_command_restart_flag = ' 2'
            print('restarting from snapshot file[s]\n')
        else:
            # check if restart files exist
            restart_file_names_tot = glob.glob(restart_file_name_base)

            if len(restart_file_names_tot) > 0:
                execute_command_restart_flag = ' 1'
                restart_file_names = [rfn for rfn in restart_file_names_tot if '.bak' not in rfn]
                restart_backup_file_names = [rfn for rfn in restart_file_names_tot if '.bak' in rfn]
                print('restarting from restart files: found {} regular, {} backup'.format(
                      len(restart_file_names), len(restart_backup_file_names)))
                if (len(restart_file_names) != self.mpi_number and
                        len(restart_backup_file_names) != self.mpi_number):
                    print('! number of restart files != number of MPI tasks = {}'.format(
                        self.mpi_number))
                print('')

        return execute_command_restart_flag


#===================================================================================================
# paths and file names
#===================================================================================================
def get_path(directories, create_path=False, remove_period=False):
    '''
    Get path to directory[s], safely including trailing /.

    Parameters
    ----------
    directories : str or list thereof : name[s] of directory[s]
    create_path : bool : whether to create path if it does not exist
    remove_period : bool : whether to remove preceding '.' of current directory

    Returns
    -------
    directories : str, or list of strings
    '''

    def get_path_single(directory, create_path, remove_period):
        if directory == '.' and remove_period:
            directory_return = ''
        elif directory[-1] != '/':
            directory_return = directory + '/'
        else:
            directory_return = directory

        if create_path:
            if not os.path.exists(directory_return):
                os.makedirs(directory_return)

        return directory_return

    if np.isscalar(directories):
        # single directory
        directories_return = get_path_single(directories, create_path, remove_period)
    else:
        # list of directories
        directories_return = list(directories)  # ensure is list
        for di in range(len(directories_return)):
            directories_return[di] = get_path_single(
                directories_return[di], create_path, remove_period)

    return directories_return


def get_numbers_in_string(string, scalarize=False):
    '''
    Get list of int and float numbers in string.

    Parameters
    ----------
    string : str
    scalarize : bool : whether to return scalar value if only one number

    Returns
    -------
    numbers : int[s] and/or float[s]
    '''
    numbers = []
    number = ''

    for ci, char in enumerate(string):
        if char.isdigit():
            number += char
        elif char == '.':
            if (number and ci > 0 and string[ci - 1].isdigit() and len(string) > ci + 1 and
                    string[ci + 1].isdigit()):
                number += char

        if number and ((not char.isdigit() and not char == '.') or ci == len(string) - 1):
            if '.' in number:
                numbers.append(float(number))
            else:
                numbers.append(int(number))
            number = ''

    if scalarize and len(numbers) == 1:
        numbers = numbers[0]

    return numbers


def get_file_names(file_name_base, number_type=None, sort_reverse=False, verbose=True):
    '''
    Get all file name[s] (with full path) with given name base
    [and number in each file name, if number_type defined], sorted.

    Parameters
    ----------
    file_name_base : str : base name of file, with full/relative path, using * as wildcard
    number_type : dtype : type of number to get in file name (get final one of given type in name)
        options: None, int, float, (int, float), numbers.Real
    sort_reverse : bool : whether to return list of file names and numbers in reverse order
    verbose: bool : whether to print diagnostics

    Returns
    -------
    path_file_names : str : name[s] of file[s], with full path
    [file_numbers] : int[s] and/or float[s] : number in file name
    '''
    Say = SayClass(get_file_names)

    # get all file names matching string in directory
    path_file_names = glob.glob(file_name_base)
    if not path_file_names:
        Say.say('! cannot find files with base name:  {}'.format(file_name_base), verbose)
        if number_type:
            return [], []
        else:
            return path_file_names

    path_file_names.sort()

    if number_type:
        # for file names with number, get final number of given type in each file name
        file_numbers = []
        for path_file_name in path_file_names:

            file_name = path_file_name
            if '/' in file_name:
                file_name = file_name.split('/')[-1]
            if '.hdf5' in file_name:
                file_name = file_name.replace('.hdf5', '')

            file_numbers_t = get_numbers_in_string(file_name, scalarize=False)

            for file_number_t in reversed(file_numbers_t):
                if isinstance(file_number_t, number_type):
                    file_numbers.append(file_number_t)
                    break
            else:
                raise OSError(
                    'no number of type {} in file: {}'.format(number_type, path_file_name))

        file_numbers = np.array(file_numbers); 

        if sort_reverse:
            path_file_names = path_file_names[::-1]
            file_numbers = file_numbers[::-1]; 

        return path_file_names, file_numbers

    else:
        if len(path_file_names) > 1 and sort_reverse:
            path_file_names = path_file_names[::-1]; 

        return path_file_names


def get_file_names_nearest_number(
    file_name_base, numbers=None, number_type=float, sort_kind='', arrayize=False,
    dif_tolerance=0.1):
    '''
    Get file name (with full path) and number of each file whose number in its name is closest to
    input number[s].

    Parameters
    ----------
    file_name_base : str : base name of file, with full path, using * as wildcard
    numbers : float/int or list : number[s] (such as scale-factors)
    number_type : dtype : type of number in file name to get
        options: None, int, float
    sort_kind : str : way to sort file names/numbers
        options: '' = none, 'forward', 'reverse'
    arrayize : bool : whether to force return as array, even if single element
    dif_tolerance : float : tolerance for warning flag in number rounding

    Returns
    -------
    file_names : str : name[s] of file[s], with full path
    file_numbers : int[s] and/or float[s] : number in each file name
    '''
    if np.isscalar(numbers):
        numbers = [numbers]
    numbers = np.array(numbers)

    if sort_kind:
        numbers = np.sort(numbers)
        if sort_kind == 'reverse':
            numbers = numbers[::-1]

    # get all file names and numbers matching string in directory
    file_names_read, file_numbers_read = get_file_names(file_name_base, number_type)
    file_names = []
    file_numbers = []
    for number in numbers:
        number_difs = abs(number_type(number) - file_numbers_read)
        near_i = np.nanargmin(number_difs)
        # warn if number of file is too far from input value
        if number_difs[near_i] > dif_tolerance:
            print('! input number = {}, but nearest file number = {}'.format(
                  number, file_numbers_read[near_i]))
        file_names.append(file_names_read[near_i])
        file_numbers.append(file_numbers_read[near_i])

    if numbers.size == 1 and not arrayize:
        # if input scalar number, return as scalar
        file_names = file_names[0]
        file_numbers = file_numbers[0]
    else:
        file_names, file_numbers = np.array(file_names), np.array(file_numbers)

    return file_names, file_numbers


def get_file_names_intersect(
    file_name_bases=[], numbers_in=[], number_type=float, sort_kind='', arrayize=False):
    '''
    Get number for each file that exists in all file names with given base whose number in its name
    is closest to input number[s].

    Parameters
    ----------
    file_name_bases: str or list : base name[s] of files, with full path, using * as wildcard
    numbers_in : float or list : number[s] (such as scale-factors)
    number_type : dtype : what kind of number in file name to get
        options: None, int, float
    sort_kind: str : how to sort file names/numbers
        options: '' = none, forward, reverse
    arrayize : bool : whether to force return as list/array, even if single element

    Returns
    -------
    file_names : str : name[s] of file[s], with full path
    file_numbers : int[s] and/or float[s] : number in each file name
    '''
    if np.isscalar(file_name_bases):
        file_name_bases = [file_name_bases]
    file_name_bases = np.array(file_name_bases)

    if numbers_in is None or numbers_in == []:
        file_numbers = get_file_names(file_name_bases[0], number_type)[1]
    else:
        file_numbers = get_file_names_nearest_number(
            file_name_bases[0], numbers_in, number_type, arrayize=arrayize)[1]

    for file_i in range(1, file_name_bases.size):
        file_numbers_t = get_file_names(file_name_bases[file_i], number_type)[1]
        file_numbers = np.intersect1d(file_numbers, file_numbers_t)

    return get_file_names_nearest_number(
        file_name_bases[0], file_numbers, number_type, sort_kind, arrayize)


def rename_files(directory, string_old='', string_new=''):
    '''
    For all file names containing string_old, rename string_old to string_old.

    Parameters
    ----------
    directory : str
    string_old : str : str of file name to replace (can use *)
    string_new : str : str with which to replace it (can also use *)
    '''
    directory = get_path(directory)
    file_names = glob.os.listdir(directory)

    if not file_names:
        print('found no files in directory: ' + directory)

    if '*' in string_old and '*' in string_new:
        strings_old = string_old.split('*')
        strings_new = string_new.split('*')
    else:
        strings_old = [string_old]
        strings_new = [string_new]

    if len(strings_old) != len(strings_new):
        raise ValueError('length of strings_old = {} not match strings_new = {}'.format(
                         strings_old, strings_new))

    for file_name in file_names:
        file_name_new = file_name
        string_in_file = [False for _ in range(len(strings_old))]
        for si in range(len(strings_old)):
            if strings_old[si] in file_name:
                string_in_file[si] = True
                file_name_new = file_name_new.replace(strings_old[si], strings_new[si])

        if np.min(string_in_file) and file_name_new != file_name:
            print('in', directory, 'rename', file_name, 'to', file_name_new)
            file_name = directory + file_name
            file_name_new = directory + file_name_new
            glob.os.rename(file_name, file_name_new)


#===================================================================================================
# read/write to/from file
#===================================================================================================
def file_hdf5(file_name_base, dict_or_array_to_write=None, verbose=True):
    '''
    Write or read to/from file in HDF5 format.
    Assumes data is numpy array[s] or dictionary thereof.

    Parameters
    ----------
    file_name_base : str : base name for file (without '.hdf5')
    dict_or_array_to_write : any : dictionary or single array to write
    verbose : bool : whether to print each property read/written

    Returns
    -------
    [dict_in] : dictionary of dataset names + arrays
    '''
    Say = SayClass(file_hdf5)

    array_name_default = 'array'  # every HDF5 dataset needs a name

    if '.hdf5' not in file_name_base:
        file_name_base += '.hdf5'

    if dict_or_array_to_write is not None:
        # write to file
        if not isinstance(dict_or_array_to_write, dict):
            dict_out = {array_name_default: dict_or_array_to_write}
        else:
            dict_out = dict_or_array_to_write

        # create file
        with h5py.File(file_name_base, 'w') as file_out:
            Say.say('write file: {}'.format(file_name_base), verbose)

            for key in dict_out:
                key_use = key
                if '/' in key:
                    key_use = key_use.replace('/', '_div.by_')

                if dict_out[key].dtype == 'O':
                    # general numpy data type - assume is variable length array
                    for element in dict_out[key]:
                        if len(element):
                            break
                    dtype = h5py.special_dtype(vlen=element.dtype)
                    dset = file_out.create_dataset(key_use, (len(dict_out[key]),), dtype=dtype)
                    for ie, element in enumerate(dict_out[key]):
                        dset[ie] = element
                else:
                    file_out.create_dataset(key_use, data=dict_out[key])

                Say.say('  {} | {}, shape = {}'.format(
                    key_use, dict_out[key].dtype, dict_out[key].shape), verbose)

    else:
        # read from file
        with h5py.File(file_name_base, 'r') as file_in:
            Say.say('read file: {}'.format(file_name_base.split('/')[-1]), verbose)

            dict_in = {}
            keys = list(file_in.keys())

            for key in keys:
                key_use = key
                if '_div.by_' in key:
                    key_use = key_use.replace('_div.by_', '/')

                dict_in[key_use] = np.array(file_in.get(key))

                Say.say('  {} | {}, shape = {}'.format(
                    key, dict_in[key_use].dtype, dict_in[key_use].shape), verbose)

        return dict_in


def file_pickle(file_name, object_to_write=None, protocol=None):
    '''
    Write or read to/from file in pickle format.

    Parameters
    ----------
    file_name : str : name of file (without '.pkl')
    object_to_write : any : object to write
    protocal : int : pickle protocal (file type) to use. 2 = compatible with Python 2.3+

    Returns
    -------
    [obj] : python object
    '''
    import pickle

    Say = SayClass(file_pickle)

    if '.pkl' not in file_name:
        file_name += '.pkl'

    if object_to_write is not None:
        # write to file
        with open(file_name, 'wb') as file_out:
            pickle.dump(object_to_write, file_out, protocol)

        Say.say('wrote file:  {}'.format(file_name))

    else:
        # read from file
        with open(file_name, 'rb') as file_in:
            obj = pickle.load(file_in)

        Say.say('read file:  {}\n'.format(file_name.split('/')[-1]))

        return obj


#===================================================================================================
# run function in parallel
#===================================================================================================
def run_in_parallel(func, args_list, kwargs={}, thread_number=4, synchronize=False, verbose=False):
    '''
    Run function in parallel, looping over inputs.

    Parameters
    ----------
    func : function object
    args_list : list : list of arguments to func, with one bundle per parallelized iteration
    kwargs : dict : keyword arguments to send to the function (same for all threads)
    thread_number : int : number of parallel threads to use
    synchronize : bool : whether to synchronize parallel tasks,
        wait for each thread bundle to complete before starting new bundle
    verbose : bool : whether to print name of function and its inputs
    '''
    if thread_number > len(args_list):
        thread_number = len(args_list)  # threads <= input argument combinations

    # initiate threads, if asking for > 1
    if thread_number > 1:
        import multiprocessing as mp
        pool = mp.Pool(thread_number)

        if synchronize:
            parallel_func = pool.apply
        else:
            parallel_func = pool.apply_async

    for args in args_list:
        if verbose:
            print_flush('running:  {}{}'.format(func.__name__, args))

        if thread_number > 1:
            parallel_func(func, args, kwargs)
        else:
            func(*args, **kwargs)

    # close threads
    if thread_number > 1:
        pool.close()
        pool.join()


def run_parallel_function_old(func, args, thread_number=4, print_diagnostic=False):
    '''
    ARCHIVE
    Run function in parallel, looping over inputs.

    Parameters
    ----------
    func : function object
    args : list : list of inputs to func
    thread_number : int : number of threads to run
    print_diagnostic : bool : whether to print diagnostic information
    '''
    if thread_number > len(args):
        thread_number = len(args)  # threads <= input argument combinations

    if thread_number == 1:
        for i in range(len(args)):
            func(*args[i])
            if print_diagnostic:
                print_flush('ran: ' + args[i])
    else:
        child_number = 0
        for i in range(len(args)):
            proc_id = os.fork()  # return 0 to child process and proc id of child to parent process
            if proc_id:
                child_number += 1
                if child_number >= thread_number:
                    os.wait()  # wait for completion of child process
                    child_number -= 1
            else:
                func(args[i])  # if child process, execute command
                if print_diagnostic:
                    print_flush('ran: ' + args[i])
                os.sys.exit(0)

        while child_number > 0:
            os.wait()
            child_number -= 1


def run_parallel_command_line_old(commands=[''], thread_number=4, print_diagnostic=False):
    '''
    ARCHIVE
    Run list of commands via command line in parallel.

    Parameters
    ----------
    commands : str or list : command[s] to run
    thread_number : int : number of threads to run in parallel
    print_diagnostic : bool : whether to print diagnostic information
    '''
    if np.isscalar(commands):
        commands = [commands]
    commands = np.array(commands)

    if thread_number > commands.size:
        thread_number = commands.size  # wasteful to use more parallel threads than commands

    if thread_number == 1:
        for ci in range(commands.size):
            os.system(commands[ci])
            if print_diagnostic:
                print_flush('ran ' + commands[ci])
    else:
        child_number = 0
        for ci in range(commands.size):
            proc_id = os.fork()  # return 0 to child process and proc id of child to parent process
            if proc_id:
                child_number += 1
                if child_number >= thread_number:
                    os.wait()  # wait for completion of child process
                    child_number -= 1
            else:
                os.system(commands[ci])  # if child process, execute command
                if print_diagnostic:
                    print_flush('ran ' + commands[ci])
                os.sys.exit(0)

        while child_number > 0:
            os.wait()
            child_number -= 1
