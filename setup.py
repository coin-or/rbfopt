from __future__ import print_function
from __future__ import absolute_import

from setuptools import setup
import unittest

def readme():
    with open('README.rst') as f:
        return f.read()

def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite
                
setup(name='rbfopt',
      version='3.1.0',
      description='Library for black-box (derivative-free) optimization',
      long_description=readme(),
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Mathematics'
      ],
      url='https://github.com/coin-or/rbfopt',
      author='Giacomo Nannicini',
      author_email='giacomo.n@gmail.com',
      license='Revised BSD',
      package_dir={'': 'src'},
      packages=['rbfopt'],
      package_data={'rbfopt': ['doc/*.rst', 'doc/conf.py', 'doc/Makefile',
                               'doc/make.bat', 'examples/*.py']},
      install_requires=['numpy', 'scipy', 'pyomo'],
      test_suite='setup.my_test_suite',
      scripts=['bin/rbfopt_cl_interface.py', 'bin/rbfopt_test_interface.py'],
      zip_safe=False)

