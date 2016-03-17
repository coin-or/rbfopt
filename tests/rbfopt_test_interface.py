"""Optimize given test function as a black box.

This module contains the routines used to construct a black-box object
using the standard test functions of the module `test_functions`. It
can be used as an interface to test RBFOpt.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
Research partially supported by SUTD-MIT International Design Center.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import argparse
import random
import test_rbfopt_env
import rbfopt_cl_interface
import test_functions
from rbfopt_black_box import BlackBox

class TestBlackBox(BlackBox):
    """A black-box constructed from a known test function.

    Parameters
    ----------
    name : string
        The name of the function to be implemented.
    """
    def __init__(self, name):
        """Constructor.
        """
        try:
            self._function = getattr(test_functions, name.lower())
        except AttributeError:
            raise ValueError('Function ' + name + ' not implemented')

    def get_dimension(self):
        return self._function.dimension

    def get_var_lower(self):
        return self._function.var_lower

    def get_var_upper(self):
        return self._function.var_upper

    def get_integer_vars(self):
        return self._function.integer_vars

    def evaluate(self, point):
        return self._function.evaluate(point)

    def evaluate_fast(self, point):
        raise NotImplementedError('evaluate_fast() not implemented')

    def has_evaluate_fast(self):
        return False
# -- end class

class TestNoisyBlackBox(BlackBox):
    """A noisy black-box constructed from a known test function.

    Parameters
    ----------
    name : string
        The name of the function to be implemented.

    max_rel_error: float
        Maximum relative error.

    max_abs_error: float
        Maximum absolute error.
    """
    def __init__(self, name, max_rel_error = 0.1, max_abs_error = 0.1):
        """Constructor.
        """
        assert(max_rel_error >= 0.0)
        assert(max_abs_error >= 0.0)
        try:
            self._function = getattr(test_functions, name.lower())
        except AttributeError:
            raise ValueError('Function ' + name + ' not implemented')
        self._max_rel_error = max_rel_error
        self._max_abs_error = max_abs_error

    def get_dimension(self):
        return self._function.dimension

    def get_var_lower(self):
        return self._function.var_lower

    def get_var_upper(self):
        return self._function.var_upper

    def get_integer_vars(self):
        return self._function.integer_vars

    def evaluate(self, point):
        return self._function.evaluate(point)

    def evaluate_fast(self, point):
        value = self._function.evaluate(point)
        rel_noise = random.uniform(-self._max_rel_error, self._max_rel_error)
        abs_noise = random.uniform(-self._max_abs_error, self._max_abs_error)
        return (value + rel_noise*abs(value) + abs_noise)
        
    def has_evaluate_fast(self):
        return True
# -- end class

if (__name__ == "__main__"):
    if (sys.version_info[0] >= 3):
        print('Error: Python 3 is currently not tested.')
        print('Please use Python 2.7')
        exit()
    # Create command line parsers
    parser = argparse.ArgumentParser(description = 'Test RBF method')
    # Add the main test function option
    parser.add_argument('function', action = 'store', 
                        metavar = 'function_name',
                        help = 'test function to optimize')
    # Add additional options to parser and parse arguments
    rbfopt_cl_interface.register_options(parser)
    args = parser.parse_args()
    noisy = (True if (args.fast_objfun_rel_error > 0 or
                      args.fast_objfun_abs_error > 0) 
             else False)

    if (noisy):
        bb = TestNoisyBlackBox(args.function, args.fast_objfun_rel_error,
                               args.fast_objfun_abs_error)
    else:
        bb = TestBlackBox(args.function)

    # Obtain parameters in dictionary format for easier unpacking
    dict_args = vars(args)
    del dict_args['function']
    dict_args['target_objval'] = bb._function.optimum_value

    rbfopt_cl_interface.rbfopt_cl_interface(dict_args, bb)
    
