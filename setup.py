from __future__ import print_function
from __future__ import absolute_import

from setuptools import setup
import unittest
import io
import os
import re

POST_VERSION = ''

def readme():
    with open('README.rst') as f:
        return f.read()

def readpath(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = readpath(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

if (__name__ == '__main__'):
    setup(name='rbfopt',
          version=find_version('src/rbfopt', '__init__.py') + POST_VERSION,
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
          author_email='nannicini@us.ibm.com',
          license='Revised BSD',
          package_dir={'': 'src'},
          packages=['rbfopt'],
          package_data={'rbfopt': ['doc/*.rst', 'doc/conf.py', 'doc/Makefile',
                                   'doc/make.bat', 'examples/*.py']},
          install_requires=['numpy', 'scipy', 'pyomo'],
          setup_requires=['nose2>=0.11.0'],
          test_suite='nose2.collector.collector',
          scripts=['bin/rbfopt_cl_interface.py',
                   'bin/rbfopt_test_interface.py'],
          zip_safe=False)

