"""Black-box function.

This file contains the definition of the black box function that is
optimized by RBFOpt, when using the default command line interface.

We provide here an example for a function of dimension 3 that
returns the sum of the three variables.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
Research partially supported by SUTD-MIT International Design Center.
"""

class BlackBox:
    """Black-box. To be reimplemented by user.
    
    A class that implements the necessary methods to describe a
    black-box function. The user can reimplement this class and use it
    to compute the function that must be optimized.


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
    """

    dimension = 3

    var_lower = [0, 0, 0]
    var_upper = [10, 10, 10]

    integer_vars = [0, 1]

    def evaluate(self, x):
        """
        Evaluate the black-box function.
        
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
        return (x[0] + x[1] + x[2])

    # If there is a noisy version of the black-box function, it should
    # follow the same format as the "evaluate" function.
    evaluate_fast = None

