..
	File:      README.rst
	Author(s): Giacomo Nannicini
        University of Southern California
	g.nannicini@usc.edu

	(C) Copyright Singapore University of Technology and Design 2015.
	(C) Copyright International Business Machines Corporation 2016.
	You should have received a copy of the license with this code.
	Research partially supported by SUTD-MIT International Design Center.

=================
Table of contents
=================
	
This software is released under the Revised BSD License. By using
this software, you are implicitly accepting the terms of the license.

RBFOpt is a Python library for black-box optimization (also known as
derivative-free optimization). It is developed for Python 3 but
currently runs on Python 2.7 as well. This README contains
installation instructions and a brief overview. More details can be
found in the user manual.

Contents of this directory:

* AUTHORS: Authors of the library.
* CHANGELOG: Changelog.
* LICENSE: Licensing information.
* MANIFEST.in: List of additional files to be included in archives.
* README.rst: This file.
* VERSION: Version of the library.
* manual.pdf: User manual.
* requirements.txt: List of dependencies for this project.
* setup.cfg: Configuration file for setup.py.
* setup.py: Setup file.
* tox.ini: Configuration file for tox.
* bin/

  * rbfopt_cl_interface.py: Script for the command-line interface,
    to run the library on a user-defined black-box function
    implemented in a user-specified file.
  * rbfopt_test_interface.py: Script to test the library on a
    global optimization test set.

* src/

  * rbfopt/
  
    * rbfopt_black_box.py: Description of an abstract black-box
      function.
    * rbfopt_algorithm.py: Main optimization algorithm, both
      serial and parallel.
    * rbfopt_aux_problems.py: Interface for the auxiliary problems
      solved during the optimization process.
    * rbfopt_degreeX_models.py: PyOmo models for the auxiliary
      problems necessary for RBF functions with minimum required
      polynomial degree X.
    * rbfopt_refinement: Routines for refinement phase.
    * rbfopt_settings.py: Global and algorithmic settings.
    * rbfopt_test_functions.py: Mathematical test functions.
    * rbfopt_user_black_box.py: A black-box class constructed from
      user data.
    * rbfopt_utils.py: Utility routines.

    * doc/

      * conf.py: Configuration file for Sphinx.
      * Makefile: Makefile (for Linux/Mac) to build the
	documentation.
      * make.bat: Batch file (for Windows) to build the
	documentation.
      * \*.rst: ReStructured Text files for the documentation.

    * examples/

      * rbfopt_black_box_example.py: Example of an implementation
	of a simple black-box function.
	  
* tests/

  * context.py: Configuration file for nose.
  * test_functions.py: Global optimization test functions.
  * test_rbfopt_algorithm.py: Testing module for
    rbfopt_algorithm.py (regular unit tests).
  * slow_test_rbfopt_algorithm.py: Testing module for
    rbfopt_algorithm.py (additional, slow tests).
  * test_rbfopt_aux_problems.py: Testing module for
    rbfopt_aux_problems.py.
  * test_rbfopt_degreeX_models.py: Testing module for
    rbfopt_degreeX_models.py.
  * test_rbfopt_env.py: Environment variables for testing
    environment.
  * test_rbfopt_mwe.py: Test the minimal working example given in the
    documentation.
  * test_rbfopt_refinement: Testing module for rbfopt_refinement.py
  * test_rbfopt_settings.py: Testing module for rbfopt_settings.py.
  * test_rbfopt_utils.py: Testing module for rbfopt_utils.py.

=========================
Installation requirements
=========================

This package requires the following software:

* Python version >= 3.7
* NumPy version >= 1.11.0
* SciPy version >= 0.17.0
* Pyomo version >= 5.1.1

The software has been tested with the versions indicated above. It may
work with earlier version and should work with subsequent version, if
they are backward compatible. In particular, the software is known to
work with Pyomo version 4 and earlier versions of Scipy.

The code is developed for Python 3.7, but it currently also runs on
Python 2.7. Since Python 2.7 has reached end-of-life in January 2020,
we recommend using Python 3.7 or higher.

The easiest, and recommended, way to install the package is via the
Python module manager pip. The code is on PyPI, therefore it can be
installed from PyPI using::

  pip install rbfopt

You can install from source, downloading an archive or cloning from
git (for example if you want to use a development version that is not
released on PyPI yet), using the command::

  pip install .

You may need the -e switch to install in a virtual environment. To
build the documentation, you also need numpydoc::

  pip install numpydoc

On Windows systems, we recommend `WinPython
<http://winpython.sourceforge.net/>`_, which comes with NumPy, SciPy
and pip already installed. After installing WinPython, it is typically
necessary to update the PATH environment variable. The above command
using pip to install missing libraries has been successfully tested on
a fresh WinPython installation.

RBFOpt requires the solution of convex and nonconvex nonlinear
programs (NLPs), as well as nonconvex mixed-integer nonlinear programs
(MINLPs) if some of the decision variables (design parameters) are
constrained to be integer. Solution of these subproblems is performed
through Pyomo, which in principle supports any solver with an AMPL
interface (.nl file format). The code is setup to employ Bonmin and
Ipopt, that are open-source, with a permissive license, and available
through the COIN-OR repository. The end-users are responsible for
checking that they have the right to use these solvers. To use
different solvers, a few lines of the source code have to be modified:
ask for help on GitHub or on the mailing list, see below.

To obtain pre-compiled binaries for Bonmin and Ipopt for several
platforms, we suggest having a look at the AMPL `opensource solvers
<http://ampl.com/products/solvers/open-source/>`_ (also `here
<http://ampl.com/dl/open/>`_) for static binaries. **Note:** These
binaries might be outdated: better performance can sometimes be
obtained compiling Bonmin from scratch (Bonmin contains Ipopt as
well), especially if compiling with a different solver for linear
systems rather than the default Mumps, e.g., ma27.  Bonmin and Ipopt
must be compiled with ASL support.

In case any of the packages indicated above is missing, some features
may be disabled, not function properly, or the software may not run at
all.

=============================================
Installation instructions and getting started
=============================================

1) Install the package with pip as indicated above. This will install
   the two executable Python scripts rbfopt_cl_interface.py and
   rbfopt_test_interface.py in your bin/ directory (whatever is used
   by pip for this purpose), as well as the module files in your
   site-packages directory.

2) Make sure Bonmin and Ipopt are in your path; otherwise, use the
   options minlp_solver_path and nlp_solver_path in RbfoptSettings to
   indicate the full path to the solvers. If you use RBFOpt as a
   library and create your own RbfoptSettings object, these options
   can be given as::

     import rbfopt
     settings = rbfopt.RbfoptSettings(minlp_solver_path='full/path/to/bonmin', nlp_solver_path='full/path/to/ipopt')

   If you use the command-line tools, you can simply provide the option preceded by double hyphen, as in::

     rbfopt_test_interface.py --minlp_solver_path='full/path/to/bonmin' branin

3) Enjoy!

4) You can test the installation by running::

     rbfopt_test_interface.py branin

   See::

     rbfopt_test_interface.py --help

   for more details on command-line options for the testing tool.

   Many more test functions, with different characteristics, are
   implemented in the file rbfopt_test_functions.py. They can all be
   used for testing.

5) Unit tests for the library can be executed by running::

     nose2

   or::

     python setup.py test

   from the current (main) directory. If some of the tests fail, the
   library may or may not work correctly. Some of the test failures
   are relatively harmless. You are advised to contact the mailing
   list (see below) if you are unsure about some test failure.

   Additional slow tests, that check if various parametrizations of
   the optimization algorithm can solve some global optimization
   problems, are found in the file slow_test_rbfopt_algorithm.py,
   which is ignored by nose by default. To execute these tests, run::

     nose2 tests.slow_test_rbfopt_algorithm
   
=======================
Minimal working example
=======================

After installation, the easiest way to optimize a function is to use
the RbfoptUserBlackBox class to define a black-box, and execute
RbfoptAlgorithm on it. This is a minimal example to optimize the
3-dimensional function defined below::

  import rbfopt
  import numpy as np
  def obj_funct(x):
    return x[0]*x[1] - x[2]
  
  bb = rbfopt.RbfoptUserBlackBox(3, np.array([0] * 3), np.array([10] * 3),
                                 np.array(['R', 'I', 'R']), obj_funct)
  settings = rbfopt.RbfoptSettings(max_evaluations=50)
  alg = rbfopt.RbfoptAlgorithm(settings, bb)
  val, x, itercount, evalcount, fast_evalcount = alg.optimize()

Another possibility is to define your own class derived from
RbfoptBlackBox in a separate file, and execute the command-line
interface on the file. An example is provided under
src/rbfopt/examples, in the file rbfopt_black_box_example.py. This can
be executed with::

  rbfopt_cl_interface.py src/rbfopt/examples/rbfopt_black_box_example.py

=====================
Parallel optimization
=====================

RBFOpt supports asynchronous parallel optimization using Python's
multiprocessing library. This mode is enabled whenever the parameter
num_cpus is set to a value greater than 1. Black-box function
evaluations as well as some of the heaviest computatations carried out
by the algorithm will then be executed in parallel. Since the parallel
computations are asynchronous, determinism cannot be guaranteed: in
other words, if you execute the parallel optimizer twice in a row, you
may (and often will) get different results, even if you provide the
same random seed. This is because the order in which the computations
will be completed may change, and this may impact the course of the
algorithm.

The default parameters of the algorithm are optimized for the serial
optimization mode. For recommendations on what parameters to use with
the parallel optimizer, feel free to ask on the mailing list.

Note that the parallel optimizer is oblivious of the system-wide
settings for executing linear algebra routines (BLAS) in parallel. We
recommend setting the number of threads for BLAS to 1 when using the
parallel optimizer, see the next section.

==========================
Known issues with OpenBLAS
==========================

We are aware of an issue when launching multiple distinct processes
that use RBFOpt and the NumPy implementation is configured to use
OpenBLAS in parallel: in this case, on rare occasions we have observed
that some processes may get stuck forever when computing matrix-vector
multiplications. The problem can be fixed by setting the number of
threads for OpenBLAS to 1. We do not know if the same issue occurs
with other parallel implementations of BLAS.

For this reason, and because parallel BLAS uses resources suboptimally
when used in conjunction with the parallel optimizer of RBFOpt (if
BLAS runs in parallel, each thread of the parallel optimizer would
spawn multiple threads to run BLAS, therefore disregarding the option
num_cpus), RBFOpt attempts to set the number of BLAS threads to 1 at
run time.

All scripts (rbfopt_cl_interface.py and rbfopt_test_interface.py) set
the environment variables OMP_NUM_THREADS to 1. Furthermore, the
rbfopt module does the same when imported for the first time.

Note that these settings are only effective if the environment
variable is set *before* NumPy is imported; otherwise, they are
ignored. If you are facing the same issue, we recommend setting
environment variable OMP_NUM_THREADS to 1. In Python, this can be done
with::

  import os
  os.environ['OMP_NUM_THREADS'] = '1'

=============
Documentation
=============

The documentation for the code can be built using Sphinx with the
numpydoc extension. numpydoc can be installed with pip::

  pip install numpydoc

After that, the directory src/rbfopt/doc/ contains a Makefile (on
Windows, use make.bat) and the Sphinx configuration file conf.py.

You can build the HTML documentation (recommended) with::

  make html

The output will be located in _build/html/ and the index can be found
in _build/html/index.html.

A PDF version of the documentation (much less readable than the HTML
version) can be built using the command::

  make latexpdf

An online version of the documentation for the latest master branch of
the code, and for the latest stable release, are available on
ReadTheDocs for the `latest
<http://rbfopt.readthedocs.org/en/latest/>`_ and `stable
<http://rbfopt.readthedocs.org/en/stable/>`_ version.

=============
Citing RBFOpt
=============

If you use RBFOpt in one of your projects or papers, please cite the
following papers (this is the only way in which the authors get
credit):

* A. Costa and G. Nannicini. RBFOpt: an open-source library for
  black-box optimization with costly function
  evaluations. Mathematical Programming Computation,
  10(4):597–629, 2018. (The paper can be downloaded as: `Optimization
  Online paper 4538
  <http://www.optimization-online.org/DB_HTML/2014/09/4538.html>`_)

* G. Nannicini. On the implementation of a global optimization method
  for mixed-variable problems. Open Journal of Mathematical
  Optimization, 2(1), 2021. (Download link: `OJMO
  <https://ojmo.centre-mersenne.org/articles/OJMO_2021__2__A1_0/>`_)

======================================
RBFOpt for hyperparameter optimization
======================================

RBFOpt is used in `IBM Watson Studio AutoAI
<https://www.ibm.com/cloud/watson-studio/autoai>`_. For a discussion
on the application of RBFOpt to hyperparameter optimization in machine
learning, besides the aforementioned paper published in `OJMO
<https://ojmo.centre-mersenne.org/articles/OJMO_2021__2__A1_0/>`_, see
the paper:

* G. I. Diaz, A. Fokoue-Nkoutche, G. Nannicini and H. Samulowitz. An
  effective algorithm for hyperparameter optimization of neural
  networks. IBM Journal of Research and Development 61, no. 4/5
  (2017): 9-1. (Download link: `IBM Journal of R&D
  <https://ieeexplore.ieee.org/document/8030298>`_)

=======
Support
=======

If you believe there is a bug or an issue, please open an issue on
GitHub.  If you have a general question, please use GitHub's
"Discussions" feature (the tab can be opened at the top of the page).
