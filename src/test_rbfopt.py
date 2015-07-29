"""Optimize given test function as a black box.

This module contains the routines used to construct a black-box object
using the standard test functions of the module `test_functions`

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
import rbfopt
import rbfopt_cl_interface
import test_functions

class NoisyFunction:
    """Adds noise to an existing function.

    A class that adds relative and absolute noise to a given
    function. The noise is an additive random variable with uniform
    distribution by default, and the random numbers are generated with
    the standard "random" module.

    Parameters
    ----------

    function: Callable[List[float]]
        The function to which noise is added.

    max_rel_error: float
        Maximum relative error.

    max_abs_error: float
        Maximum absolute error.
    """

    def __init__(self, function, max_rel_error = 0.1, max_abs_error = 0.1):
        """Constructor.
        """
        assert(max_rel_error >= 0.0)
        assert(max_abs_error >= 0.0)
        self._function = function
        self._max_rel_error = max_rel_error
        self._max_abs_error = max_abs_error

    def evaluate(self, x):
        """Function evaluation with noise.

        Evaluate the function at the specified point, then add noise.

        Parameters
        ----------

        x: List[float]
            The point at which the function is evaluated.

        Returns
        -------
        float
            The value of the function after noise is introduced.
        """
        value = self._function(x)
        rel_noise = random.uniform(-self._max_rel_error, self._max_rel_error)
        abs_noise = random.uniform(-self._max_abs_error, self._max_abs_error)
        return (value + rel_noise*abs(value) + abs_noise)

# -- end class

class TestFunctionBlackBox:
    """Black-box for test functions.

    A class that implements the necessary attributes to mimick the
    `black_box.BlackBox` class, when using one of the standard test
    functions.

    Attributes
    ----------

    dimension : int
        Dimension of the problem.
        
    var_lower : List[float]
        Lower bounds of the decision variables

    var_upper : List[float]
        Upper bounds of the decision variables.

    evaluate : Callable[List[float]]
        The function implementing the black-box.

    evaluate_fast : Callable[List[float]]
        The function implementing a faster, potentially noisy version
        of the black-box, or None if not available.

    integer_vars : List[int]
        A list of indices of the variables that must assume integer
        values.

    optimum_value : float
        The value of the optimum for the function.

    See Also
    --------
    The `black_box` module.
    """

    def __init__(self):
        self.dimension = None
        self.var_lower = None
        self.var_lower = None
        self.optimum_value = None
        self.evaluate = None
        self.evaluate_fast = None

# -- end class

def select_function(function_name):
    """Choose test function.

    Return the appropriate test function, interpreting the name.

    Parameters
    ----------
    function_name : str
        The name of the test function to be used.

    Raises
    ------
    AttributeError
        If the function does not exist.
    """
    # Pick appropriate function from the test set
    try:
        function = getattr(test_functions, function_name.lower())
        return(function)
    except AttributeError:
        raise ValueError('Function ' + function_name + ' not implemented')

# -- end function

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
    function = select_function(args.function)
    noisy = (True if (args.fast_objfun_rel_error > 0 or
                      args.fast_objfun_abs_error > 0) 
             else False)
    if (noisy):
        noisy_fun = NoisyFunction(function.evaluate,
                                  args.fast_objfun_rel_error,
                                  args.fast_objfun_abs_error)
        fast_objfun = noisy_fun.evaluate
    else:
        fast_objfun = None

    black_box = TestFunctionBlackBox()
    black_box.dimension = function.dimension
    black_box.var_lower = function.var_lower
    black_box.var_upper = function.var_upper
    black_box.integer_vars = function.integer_vars
    black_box.evaluate = function.evaluate
    black_box.evaluate_fast = fast_objfun

    # Obtain parameters in dictionary format for easier unpacking
    dict_args = vars(args)
    del dict_args['function']
    dict_args['target_objval'] = function.optimum_value


    rbfopt_cl_interface.rbfopt_cl_interface(dict_args, black_box)
