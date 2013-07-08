from distutils.core import setup, Extension
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

src_dir = '../src/'
rtml_dir = '../src/rtml/'

# mhmm_lib extension module
mhmm_module = Extension('_mhmm',
                           extra_compile_args = [],#'-O0'],
                           sources=['mhmm_wrap.cxx', rtml_dir + 'utility.cpp'],
                           include_dirs = [numpy_include, src_dir, rtml_dir]
                           )
# mhmm_lib setup
setup (name        = 'mhmm',
       version     = '0.1',
       author      = "Herve",
       description = """Real-Time Machine Learning Library""",
       ext_modules = [mhmm_module],
       py_modules  = ["mhmm"],
       )