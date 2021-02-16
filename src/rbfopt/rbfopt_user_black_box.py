"""Black-box function from user data.

This module contains the definition of a black box function
constructed from user data that can be optimized by RBFOpt.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright International Business Machines Corporation 2017.

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import rbfopt.rbfopt_black_box as bb


class RbfoptUserBlackBox(bb.RbfoptBlackBox):
    """A black-box function from user data that can be optimized.

    A class that implements the necessary methods to describe the
    black-box function to be minimized, and gets all the required data
    from the user.

    Parameters
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


    obj_funct : Callable[1D numpy.ndarray[float]]
        The function to optimize. Must take a numpy array as argument,
        and return a float.

    obj_funct_noisy : Callable[1D numpy.ndarray[float]] or None
        The noisy but fast version of the function to optimize. If
        given, it must take a numpy array as argument, and return a
        numpy array with three floats, in the following order: the
        approximate function value, its lower variation, and its upper
        variation, where where lower <= 0 and upper >= 0 and the true
        function value is contained between value + lower and value +
        upper. If it is None, we assume that there is no fast version
        of the objective function.
        

    See also
    --------
    :class:`rbfopt_black_box.BlackBox`

    """

    def __init__(self, dimension, var_lower, var_upper, var_type,
                 obj_funct, obj_funct_noisy=None):
        """Constructor.
        """
        assert(len(var_lower) == dimension)
        assert(len(var_upper) == dimension)
        assert(len(var_type) == dimension)

        self.dimension = dimension
        self.var_lower = np.array(var_lower)
        self.var_upper = np.array(var_upper)
        self.var_type = np.array(var_type)
        self.obj_funct = obj_funct
        self.obj_funct_noisy = obj_funct_noisy
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
        return self.obj_funct(x)
    # -- end function
    
    def evaluate_noisy(self, x):
        """Evaluate a fast approximation of the black-box function.

        Returns an approximation of the value of evaluate(), hopefully
        much more quickly, and provides error bounds on the
        evaluation. If has_evaluate_noisy() returns False, this
        function will never be queried and therefore it does not have
        to return any value.

        Parameters
        ----------
        x : 1D numpy.ndarray[float]
            Value of the decision variables.

        Returns
        -------
        1D numpy.ndarray[float]
            A numpy array with three floats (value, lower, upper)
            containing the approximate value of the function at x, the
            lower error bound, and the upper error bound, such that
            the true function value is contained between value + lower
            and value + upper. Hence, lower should be <= 0 while upper
            should be >= 0.

        """
        assert(len(x) == self.dimension)
        if (self.obj_funct_noisy is None):
            raise NotImplementedError('evaluate_noisy not available')
        else:
            return self.obj_funct_noisy(x)
        
    # -- end function

    def has_evaluate_noisy(self):
        """Indicate whether evaluate_noisy is available.

        Indicate if a fast but potentially noisy version of evaluate
        is available through the function evaluate_noisy. If True, such
        function will be used to try to accelerate convergence of the
        optimization algorithm. If False, the function evaluate_noisy
        will never be queried.

        Returns
        -------
        bool
            Is evaluate_noisy available?
        """
        return (self.obj_funct_noisy is not None)
    # -- end function

# -- end class
