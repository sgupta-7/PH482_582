'''
Utility functions for run-time diagnostics.

@author: Andrew Wetzel

If in IPython, use timeit to time function and prun to profile function.
'''


# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import importlib


#===================================================================================================
# timing, profiling
#===================================================================================================
def line_profile_func(func, args):
    '''
    Print time to run each line.
    *** Might have problems after reloading code.

    Parameters
    ----------
    func : function, with preceding module name, without trailing parentheses
    args : its arguments, all within ()
    '''
    import line_profiler  # @UnresolvedImport

    importlib.reload(line_profiler)
    ppp = line_profiler.LineProfiler(func)
    ppp.runcall(func, *args)
    ppp.print_stats()


def time_this(func):
    '''
    Decorator that reports execution time.
    '''
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('# run time = {:.3f} sec'.format(end - start))
        return result

    return wrapper
