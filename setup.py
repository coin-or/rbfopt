from setuptools import setup
from Cython.Build import cythonize

import numpy
import os

setup(
    name='Cython RBFOpt Utils',
    ext_modules=cythonize([os.path.normpath('src/cython_rbfopt/rbfopt_utils.pyx'), os.path.normpath('src/cython_rbfopt/rbfopt_aux_problems.pyx')]),
    include_dirs=[numpy.get_include()],
)
