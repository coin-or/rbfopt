#!/usr/bin/env python3
"""Optimize a test function taken as a black box with RBFOpt.

This module contains the routines used to construct a black-box object
using the standard test functions of the module `test_functions`. It
can be used as an interface to test RBFOpt.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
(C) Copyright International Business Machines Corporation 2017.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
# We must set the threading options before numpy is loaded, otherwise
# there might be issues when running several processes in parallel.
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import ast
import numpy as np
import rbfopt
from rbfopt import RbfoptSettings
from rbfopt import RbfoptAlgorithm
from rbfopt import RbfoptBlackBox
from rbfopt.rbfopt_test_functions import TestBlackBox, TestNoisyBlackBox, TestEnlargedBlackBox

def register_options(parser):
    """Add options to the command line parser.

    Register all the options for the optimization algorithm.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser.

    See also
    --------   
    :class:`rbfopt_settings.RbfoptSettings` for a detailed description of
    all the command line options.
    """
    # Algorithmic settings
    algset = parser.add_argument_group('Algorithmic settings')
    # Get default values from here
    default = RbfoptSettings()
    attrs = vars(default)
    docstring = default.__doc__
    param_docstring = docstring[docstring.find('Parameters'):
                                docstring.find('Attributes')].split(' : ')
    param_name = [val.split(' ')[-1].strip() for val in param_docstring[:-1]]
    param_type = [val.split('\n')[0].strip() for val in param_docstring[1:]]
    param_help = [' '.join(line.strip() for line in val.split('\n')[1:-2])
                  for val in param_docstring[1:]]
    # We extract the default from the docstring in case it is
    # necessary, but we use the actual default from the object above.
    param_default = [val.split(' ')[-1].rstrip('.').strip('\'') 
                     for val in param_help]
    for i in range(len(param_name)):
        if (param_type[i] == 'float'):
            type_fun = float
        elif (param_type[i] == 'int'):
            type_fun = int
        elif (param_type[i] == 'bool'):
            type_fun = ast.literal_eval
        else:
            type_fun = str
        algset.add_argument('--' + param_name[i], action = 'store',
                            dest = param_name[i],
                            type = type_fun,
                            help = param_help[i],
                            default = getattr(default, param_name[i]))
# -- end function

def rbfopt_test_interface(args, black_box):
    """Interface for test functions.
    
    Optimize the specified objective function using the algorithmic
    options given on the command line.

    Parameters
    ----------
    args : Dict[string]
        A dictionary containing the values of the parameters in a
        format args['name'] = value. 

    black_box : :class:`rbfopt_black_box.RbfoptBlackBox`
        The black box to be optimized.
    """
    if (not isinstance(black_box, RbfoptBlackBox)):
        raise ValueError('The specified module does not contain a ' +
                         'valid BlackBox instance')

    settings = RbfoptSettings.from_dictionary(args)
    settings.print(sys.stdout)
    alg = RbfoptAlgorithm(settings = settings, black_box = black_box)
    result = alg.optimize()
    print('RbfoptAlgorithm.optimize() returned ' + 
          'function value {:.15f}'.format(result[0]))
    for (i, val) in enumerate(result[1]):
        print('x{:<4d}: {:16.6f}'.format(i, val))
# -- end function

if (__name__ == "__main__"):
    if (sys.version_info[0] <= 2 and sys.version_info[1] < 7):
        print('Error: this software requires Python 2.7 or later')
        exit()
    # Create command line parsers
    parser = argparse.ArgumentParser(description='Test RBF method')
    # Add the main test function option
    parser.add_argument('function', action='store', 
                        metavar='function_name',
                        help='test function to optimize')
    parser.add_argument('--dimension_multiplier', action='store',
                        type=int, dest='dimension_multiplier',
                        default=1, help='Multiply dimension of test ' +
                        'function by this factor. Default 1')
    parser.add_argument('--noisy_objfun_rel_error', action='store',
                        type=float, dest='noisy_objfun_rel_error',
                        default=0.0, help='The maximum relative ' +
                        'error by which the noisy version of the ' +
                        'objective function is affected. Default 0.0.')
    parser.add_argument('--noisy_objfun_abs_error', action='store',
                        type=float, dest='noisy_objfun_abs_error',
                        default=0.0, help='The maximum absolute ' +
                        'error by which the noisy version of the ' +
                        'objective function is affected. Default 0.0.')
    # Add additional options to parser and parse arguments
    register_options(parser)
    args = parser.parse_args()
    noisy = (True if (args.noisy_objfun_rel_error > 0 or
                      args.noisy_objfun_abs_error > 0) 
             else False)
    enlarged = True if args.dimension_multiplier > 1 else False
    # Ensure seed is set before we initialize black box
    np.random.seed(args.rand_seed)
    if (enlarged):
        bb = TestEnlargedBlackBox(args.function, args.dimension_multiplier)
    else:
        bb = TestBlackBox(args.function)
    if (noisy):
        bb = TestNoisyBlackBox(bb, args.noisy_objfun_rel_error,
                               args.noisy_objfun_abs_error)
    # Obtain parameters in dictionary format for easier unpacking
    dict_args = vars(args)
    del dict_args['function']
    del dict_args['dimension_multiplier']
    del dict_args['noisy_objfun_rel_error']
    del dict_args['noisy_objfun_abs_error']
    dict_args['target_objval'] = bb._function.optimum_value

    rbfopt_test_interface(dict_args, bb)
    
