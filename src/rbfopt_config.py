"""Configuration file for RBFOpt.

This modules contains the system-wide options for RBFOpt. The values
below can be changed at runtime, but it makes more sense to use this
as a static configuration file.

Attributes
----------

MINLP_SOLVER_NAME : str
    Name of the MINLP solver.

MINLP_SOLVER_PATH : str 
    MINLP solver executable, including path if the executable is not
    in your path. Use '/' to delimit directories, even on Windows, to
    avoid issues with escaping backward slashes.

MINLP_SOLVER_OPTIONS : List[(str, Any)]
    Options passed on to the MINLP solver, in the form ('option_name',
    value)

MINLP_SOLVER_RAND_SEED_OPTION : str or None
    Option name to initialize MINLP solver's random seed. Set to None
    if option is not supported.

MINLP_SOLVER_MAX_SEED : int
    Maximum integer that can be used to seed the MINLP solver's random
    seed.

NLP_SOLVER_NAME : str
    Name of the NLP solver.

NLP_SOLVER_PATH : str 
    NLP solver executable, including path if the executable is not
    in your path. Use '/' to delimit directories, even on Windows, to
    avoid issues with escaping backward slashes.

NLP_SOLVER_OPTIONS : List[(str, Any)]
    Options passed on to the NLP solver, in the form ('option_name',
    value)

NLP_SOLVER_RAND_SEED_OPTION : str or None
    Option name to initialize NLP solver's random seed. Set to None
    if option is not supported.

NLP_SOLVER_MAX_SEED : int
    Maximum integer that can be used to seed the NLP solver's random
    seed.

GAMMA : float
    Parameter gamma for some of the radial basis functions. This is
    essentially a scaling parameter and it is very hard to control at
    runtime, so we treat it as a constant.

LOCAL_SEARCH_THRESHOLD : float
    Threshold used to determines what is a local search. If the
    scaling factor used in the computation of f_n^* is less than this
    value, it is assumed that the search is a local search.

DISTANCE_SHIFT : float
    Shift of the argument of the square root operator in the
    computation of distances, in order to prevent computing the
    derivative of sqrt(0) if we are evaluating a point that coincides
    with one of the nodes. Note that this shift is usually unnecessary
    in the continuous setting, but for all-integer problems, it is
    common that in Branch-and-Bound we end up exactly on one of the
    nodes. This should be a very small value, so that it is
    essentially ignored unless the argument of the square root is
    zero.

MAX_RANDOM_INIT : int
    Maximum number of trials for the random initialization strategies,
    in case they generate a linearly dependent set of samples. After
    this number of trials, the initialization algorithm will bail out.


Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2015.
(C) Copyright International Business Machines Corporation 2016.
Research partially supported by SUTD-MIT International Design Center.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


MINLP_SOLVER_NAME = 'bonmin'
# If the path contains directories, use forward slashes '/' to
# separate them, even on Windows.
MINLP_SOLVER_PATH = 'bonmin'
MINLP_SOLVER_OPTIONS = [('bonmin.num_resolve_at_root', 10),
                        ('bonmin.num_retry_unsolved_random_point', 5),
                        ('bonmin.num_resolve_at_infeasibles', 5),
                        ('bonmin.algorithm', 'B-BB'),
                        ('bonmin.time_limit', 45),
                        ('max_cpu_time', 20),
                        ('max_iter', 1000)]
MINLP_SOLVER_RAND_SEED_OPTION = 'bonmin.random_generator_seed'
MINLP_SOLVER_MAX_SEED = 2147983646

NLP_SOLVER_NAME = 'ipopt'
# If the path contains directories, use forward slashes '/' to
# separate them, even on Windows.
NLP_SOLVER_PATH = 'ipopt'
NLP_SOLVER_OPTIONS = [('acceptable_tol', 1.0e-3),
                      ('honor_original_bounds', 'no'),
                      ('max_cpu_time', 20),
                      ('max_iter', 1000)]
NLP_SOLVER_RAND_SEED_OPTION = None
NLP_SOLVER_MAX_SEED = 2147983646

GAMMA = 1.0

LOCAL_SEARCH_THRESHOLD = 0.25

DISTANCE_SHIFT = 1.0e-40

MAX_RANDOM_INIT = 50
