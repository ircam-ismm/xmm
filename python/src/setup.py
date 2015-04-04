#!/usr/bin/env python
# encoding: utf-8
"""
setup.py

Setup script for SWIG-Python wrapping

Contact:
- Jules Françoise <jules.francoise@ircam.fr>

This code has been initially authored by Jules Françoise
<http://julesfrancoise.com> during his PhD thesis, supervised by Frédéric
Bevilacqua <href="http://frederic-bevilacqua.net>, in the Sound Music
Movement Interaction team <http://ismm.ircam.fr> of the
STMS Lab - IRCAM, CNRS, UPMC (2011-2015).

Copyright (C) 2015 UPMC, Ircam-Centre Pompidou.

This File is part of XMM.

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
"""

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.
numpy_include = numpy.get_include()

src_dir = '../src/models/'
rtml_dir = '../src/core/'

# xmm_lib extension module
xmm_module = Extension('_xmm',
                        define_macros=[('USE_PTHREAD', None)],
                        extra_compile_args = ["-Wno-reorder", "-Wno-unused-private-field", "-Wno-dynamic-class-memaccess"],
                        extra_link_args = ["-lpthread"],
                        sources=['build/xmm_wrap.cxx', rtml_dir + 'json_utilities.cpp',
                                    rtml_dir + 'label.cpp', rtml_dir + 'phrase.cpp',
                                    rtml_dir + 'training_set.cpp',
                                    rtml_dir + 'probabilistic_model.cpp', rtml_dir + 'gaussian_distribution.cpp',
                                    src_dir + 'gmm.cpp', src_dir + 'gmm_group.cpp',
                                    src_dir + 'hmm.cpp', src_dir + 'hierarchical_hmm.cpp', 
                                    src_dir + 'kmeans.cpp'],
                        include_dirs = [numpy_include, '../dependencies/libjson/src/', src_dir, rtml_dir],
                        library_dirs = ['../dependencies/libjson/bin'],
                        libraries = ['json']
                        )
# xmm_lib setup
setup (name        = 'xmm',
       version     = '0.3',
       author      = "Jules Francoise <jules.francoise@ircam.fr>",
       description = """Multimodal Hidden Markov Models Library""",
       ext_modules = [xmm_module],
       py_modules  = ["xmm"],
       )