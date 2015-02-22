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
                        extra_link_args = ["-lpthread"],
                        sources=['xmm_wrap.cxx', rtml_dir + 'json_utilities.cpp',
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