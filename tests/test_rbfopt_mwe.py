"""Test minimal working example in RBFOpt.

This module tests correctness of a minimal working example.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright International Business Machines Corporation 2018.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import rbfopt
import numpy as np


def obj_funct(x):
    return x[0]*x[1] - x[2]

class TestMinimalWorkingExample(unittest.TestCase):
    """Test the minimal working example given in the documentation."""

    def test_minimal_working_example(self):
        """Check solution of minimal working example.

        Construct the minimal working example described in the README,
        and verify that it is correctly solved.
        """
        bb = rbfopt.RbfoptUserBlackBox(3, np.array([0] * 3),
                                       np.array([10] * 3),
                                       np.array(['R', 'I', 'R']),
                                       obj_funct)
        for rbf in ['linear', 'cubic', 'thin_plate_spline', 'multiquadric',
                    'gaussian']:
            settings = rbfopt.RbfoptSettings(max_evaluations=50,
                                             rbf=rbf)
            alg = rbfopt.RbfoptAlgorithm(settings, bb)
            val, x, itercount, evalcount, fast_evalcount = alg.optimize()
            self.assertTrue(evalcount == 50,
                            msg='Did not use the full evaluation budget' +
                            ' with rbf ' + rbf)
            self.assertLessEqual(val, 9.95, msg='Did not find optimum' +
                                 ' with rbf ' + rbf)
                         
    # -- end function
# -- end class

if (__name__ == '__main__'):
    unittest.main()
