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

import rbfopt_black_box as bb

class BlackBox(bb.BlackBox):
    """Example of a black-box function that can be optimized. 

    A class that implements the necessary methods to describe a
    black-box function. The user can implement a similar class and use
    it to compute the function that must be optimized. The attributs
    and functions below are required.

    Attributes
    ----------

    dimension : int
        Dimension of the problem.
        
    var_lower : List[float]
        Lower bounds of the decision variables.

    var_upper : List[float]
        Upper bounds of the decision variables.

    integer_vars : List[int]
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
    """

    def __init__(self, exponent = 1):
        """Constructor.
        """
        assert(exponent >= 0)
        self.exponent = exponent

        # Set required data
        self.dimension = 3

        self.var_lower = [0, 0, 0]
        self.var_upper = [10, 10, 10]

        self.integer_vars = [0, 1]
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

    def get_integer_vars(self):
        """Return the list of integer variables.
        
        Returns
        -------
        List[int]
            A list of indices of the variables that must assume
            integer values. Can be empty.
        """
        return self.integer_vars
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
        return (x[0] + x[1] + x[2])**self.exponent        
    # -- end function
    
    def evaluate_fast(self, x):
        """Evaluate a fast approximation of the black-box function.

        Returns an approximation of the value of evaluate(), hopefully
        much more quickly. If has_evaluate_fast() returns False, this
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
        raise NotImplementedError('evaluate_fast not available')
    # -- end function

    def has_evaluate_fast(self):
        """Indicate whether evaluate_fast is available.

        Indicate if a fast but potentially noisy version of evaluate
        is available through the function evaluate_fast. If True, such
        function will be used to try to accelerate convergence of the
        optimization algorithm. If False, the function evaluate_fast
        will never be queried.

        Returns
        -------
        bool
            Is evaluate_fast available?
        """
        return False
    # -- end function

# -- end class
