from distutils.core import setup
from Cython.Build import cythonize

import numpy
import os

os.chdir('src/')

setup(
    name='Cython RBFOpt Utils',
    ext_modules=cythonize(["cython_rbfopt/rbfopt_utils.pyx", "cython_rbfopt/rbfopt_aux_problems.pyx"]),
    include_dirs=[numpy.get_include()],
)
