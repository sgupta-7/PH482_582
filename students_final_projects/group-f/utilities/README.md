# Description

Python package of general low-level utility functions that are useful in analyzing various datasets, in particular, particle data and catalogs of galaxies/halos from cosmological simulations.


# Content

## basic/
* lowest-level utilities - higher-level utilities rely on these
* array.py - create, manipulate, diagnostics of arrays
* binning.py - binning of arrays
* constants.py - physical constants and unit conversions, primarily in cgs
* coordinate.py - manipuate / convert positions and velocities
* diagnostic.py - run-time diagnostics
* io.py - read, write, print during run time
* math.py - math and function fitting
* neighbor.py - find nearest neighbors
* statistic.py - compute statistics

## catalog.py
* analysis specific to catalogs of halos/galaxies

## cosmic_structure.py
* compute mass variance on different scales, characteristic halo collapse mass, and halo mass function, using linear density field theory

## cosmology.py
* calculate various cosmology values, such as cosmic density, distance, age, volume

## halo_property.py
* calculate halo properties at different radii, convert between virial definitions

## orbit.py
* compute orbital quantities such as peri/apo-centric distance, orbital time, given a gravitational potential

## particle.py
* analysis specific to N-body particle data

## plot.py
* supplementary routines for plotting with matplotlib

## simulation.py
* set up and run simulations


# Requirements
python 3, numpy, scipy, h5py, matplotlib.
I develop this package using Python 3.7 and recommend that you use it too.


# Licensing

Copyright 2014-2019 by Andrew Wetzel.

In summary, you are free to use, edit, share, and do whatever you want. But please keep me informed. Have fun!

Less succinctly, this software is governed by the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.