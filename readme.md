XMM — Probabilistic Models for Continuous Motion Recognition and Mapping
===========================================

XMM is a portable, cross-platform C++ library that implements Gaussian Mixture Models and Hidden Markov Models for recognition and regression. The XMM library was developed for movement interaction in creative applications and implements an interactive machine learning workflow with fast training and continuous, real-time inference.

### Contact

Jules Françoise: <jules.francoise@ircam.fr>

### author

This code has been initially authored by <a href="http://julesfrancoise.com">Jules Françoise</a> during his PhD thesis, supervised by <a href="frederic-bevilacqua.net">Frederic Bevilacqua</a>, in the <a href="http://ismm.ircam.fr">Sound Music Movement Interaction</a> team of the <a href="http://www.ircam.fr/stms.html?&L=1">STMS Lab</a> - IRCAM - CNRS - UPMC (2011-2015).

### Copyright

Copyright (C) 2015 UPMC, Ircam-Centre Pompidou.

### Licence

This project is released under the <a href="http://www.gnu.org/licenses/gpl-3.0.en.html">GPLv3</a> license.
For commercial applications, a proprietary license is available upon request to Frederick Rousseau <frederick.rousseau@ircam.fr>.

XMM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

XMM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with XMM.  If not, see <http://www.gnu.org/licenses/>.

### Citing this work

If you use this code for research purposes, please cite one of the following publications:

- J. Françoise, N. Schnell, R. Borghesi, and F. Bevilacqua, Probabilistic Models for Designing Motion and Sound Relationships. In Proceedings of the 2014 International Conference on New Interfaces for Musical Expression, NIME’14, London, UK, 2014.
- J. Françoise, N. Schnell, and F. Bevilacqua, A Multimodal Probabilistic Model for Gesture-based Control of Sound Synthesis. In Proceedings of the 21st ACM international conference on Multimedia (MM’13), Barcelona, Spain, 2013.

### Dependencies

This software uses the open-source library <a href="http://libjson.sourceforge.net/">libJSON</a> for JSON file I/O.

## Download

The source code is available on __Github__: https://github.com/Ircam-RnD/xmm

For the <a href="https://cycling74.com/">Cycling'74 Max</a> externals, see the MuBu collection of Max objects on the ISMM team website: http://ismm.ircam.fr/mubu

## Documentation

The full documentation is available on Github Pages: http://ircam-rnd.github.io/xmm/

## Compilation and Usage

### Dependencies

The library depends on the <a href="http://libjson.sourceforge.net/">libjson</a> c++ library for JSON file I/O. A modified version of the library is distributed with this source code.
The library depends on the <a href="https://github.com/philsquared/Catch">Catch</a> unit-test framework.
The library uses PTHREADS for parallel training of models with multiple classes.

### Compiling as a static/dynamic library
#### XCode

See the xcode project in "ide/xcode/"

#### CMake

The library can be built using <a href="http://www.cmake.org/">CMake</a>.
In the root directory, type the following command to generate the Makefiles:
```
cmake . -G"Unix Makefiles"
```
The following commands can be used to build the static library, run the unit tests, and generate the documentation:
```
make
make test
make doc
```
#### Usage

The header file "xmm.h" includes all useful headers of the library.
To enable parallel training, define the preprocessor macro "USE_PTHREAD" and link with the pthread library.

### Building the Python Library
#### Dependencies

* <a href="http://www.doxygen.org/">doxygen</a>
* <a href="http://www.swig.org/">swig</a>
* <a href="http://www.numpy.org/">Numpy</a>
* <a href="http://matplotlib.org/">Matplotlib</a> (for plotting utilities)

#### Building

The python module can be built using <a href="http://www.cmake.org/">CMake</a>.
In the python directory, type the following command to generate the Makefiles and build the python module:
```
cmake . -G"Unix Makefiles"
make
```
The module should be installed in "${xmm_root}/python/bin/"
