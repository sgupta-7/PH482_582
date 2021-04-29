# Description

Python package for reading and analyzing Rockstar halo catalogs and merger trees.


---
# Requirements

This package relies on and requires my [utilities/](https://bitbucket.org/awetzel/utilities) python package for low-level utility functions.

I develop this package using Python 3.7 and recommend that you use it too.


---
# Contents

## rockstar_io.py
* read rockstar files, convert to hdf5 format, assign particle species to dark-matter halos

## rockstar_analysis.py
* high-level analysis and plotting of rockstar halos/galaxies

## rockstar_select.py
* select halos from large simulations for generating initial conditions for zoom-in

## rockstar_tutorial.ipynb
* ipython/jupyter notebook tutorial for using this package, reading and analyzing halo catalogs and merger trees


---
# Units

Unless otherwise noted, all quantities are in (or converted to during read-in) these units (and combinations thereof):

* mass [M_sun]
* position [kpc comoving]
* distance, radius [kpc physical]
* velocity [km / s]
* time [Gyr]
* elemental abundance [(linear) mass fraction]
* metallicity [log10(mass_fraction / mass_fraction_solar)], assuming Asplund et al 2009 for Solar


---
# License

Copyright 2014-2019 by Andrew Wetzel <arwetzel@gmail.com>.

In summary, you are free to use, edit, share, and do whatever you want. But please keep me informed. Have fun!

Less succinctly, this software is governed by the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE aAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.