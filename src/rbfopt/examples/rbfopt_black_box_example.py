"""Black-box function.

This module contains the definition of the black box function that is
optimized by RBFOpt, when using the default command line
interface. The user can implement a similar class to define their own
function.

We provide here an example for a function of dimension 3 that returns
the power of the sum of the three variables, with pre-determined
exponent.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
(C) Copyright International Business Machines Corporation 2016.
Research partially supported by SUTD-MIT International Design Center.

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import rbfopt

class RbfoptBlackBox(rbfopt.RbfoptBlackBox):
    """Example of a black-box function that can be optimized. 

    A class that implements the necessary methods to describe a
    black-box function. The user can implement a similar class and use
    it to compute the function that must be optimized. The attributes
    and functions below are required.

    Attributes
    ----------

    dimension : int
        Dimension of the problem.
        
    var_lower : 1D numpy.ndarray[float]
        Lower bounds of the decision variables.

    var_upper : 1D numpy.ndarray[float]
        Upper bounds of the decision variables.

    var_type : 1D numpy.ndarray[char]
        An array of length equal to dimension, specifying the type of
        each variable. Possible types are 'R' for real (continuous)
        variables, 'I' for integer (discrete, ordered) variables, 'C'
        for categorical (discrete, unordered) variables. Bounds for
        categorical variables are interpreted the same way as for
        integer variables, but categorical variables are handled
        differently by the optimization algorithm; e.g., a categorical
        variable with bounds [2, 4] can take the value 2, 3 or 4.

    integer_vars : 1D numpy.ndarray[int]
        A list of indices of the variables that must assume integer
        values.

    exponent : float
        The power to which the sum of the variables should be
        raised.

    Parameters
    ----------
    exponent : float
        The power to which the sum of the variables should be
        raised. Should be nonnegative.

    See also
    --------
    :class:`rbfopt_black_box.BlackBox`

    """

    def __init__(self, exponent=2):
        """Constructor.
        """
        assert(exponent >= 0)
        self.exponent = exponent

        # Set required data
        self.dimension = 3

        self.var_lower = np.array([0, 0, 0])
        self.var_upper = np.array([10, 10, 10])

        self.var_type = np.array(['I', 'I', 'R'])
    # -- end function

    def get_dimension(self):
        """Return the dimension of the problem.

        Returns
        -------
        int
            The dimension of the problem.
        """
        return self.dimension
    # -- end function
    
    def get_var_lower(self):        
        """Return the array of lower bounds on the variables.

        Returns
        -------
        List[float]
            Lower bounds of the decision variables.
        """
        return self.var_lower
    # -- end function
        
    def get_var_upper(self):
        """Return the array of upper bounds on the variables.

        Returns
        -------
        List[float]
            Upper bounds of the decision variables.
        """
        return self.var_upper
    # -- end function

    def get_var_type(self):
        """Return the type of each variable.
        
        Returns
        -------
        1D numpy.ndarray[char]
            An array of length equal to dimension, specifying the type
            of each variable. Possible types are 'R' for real
            (continuous) variables, 'I' for integer (discrete)
            variables, 'C' for categorical (discrete,
            unordered). Bounds for categorical variables are
            interpreted the same way as for integer variables, but
            categorical variables are handled differently by the
            optimization algorithm; e.g., a categorical variable with
            bounds [2, 4] can take the value 2, 3 or 4.

        """
        return self.var_type
    # -- end function
    
    def evaluate(self, x):

        """Evaluate the black-box function.
        
        Parameters
        ----------
        x : List[float]
            Value of the decision variables.

        Returns
        -------
        float
            Value of the function at x.

        """
        assert(len(x) == self.dimension)
        return np.sum(x)**self.exponent        
    # -- end function
    
    def evaluate_noisy(self, x):
        """Evaluate a noisy approximation of the black-box function.

        Returns an approximation of the value of evaluate(), hopefully
        much more quickly, and provides error bounds on the
        evaluation. If has_evaluate_noisy() returns False, this
        function will never be queried and therefore it does not have
        to return any value.

        Parameters
        ----------
        x : List[float]
            Value of the decision variables.

        Returns
        -------
        float
            Approximate value of the function at x.

        """
        raise NotImplementedError('evaluate_noisy not available')
    # -- end function

    def has_evaluate_noisy(self):
        """Indicate whether evaluate_noisy is available.

        Indicate if a noisy but potentially noisy version of evaluate
        is available through the function evaluate_noisy. If True, such
        function will be used to try to accelerate convergence of the
        optimization algorithm. If False, the function evaluate_noisy
        will never be queried.

        Returns
        -------
        bool
            Is evaluate_noisy available?
        """
        return False
    # -- end function

# -- end class
