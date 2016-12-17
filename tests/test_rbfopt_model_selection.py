"""Test the model selection routines in RBFOpt.

This module contains unit tests for the module rbfopt_model_selection.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright International Business Machines Corporation 2016.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import numpy as np
try:
    import cplex 
    cpx_available = True
except ImportError:
    cpx_available = False
try:
    import cylp.cy
    clp_available = True
except ImportError:
    clp_available = False
import test_rbfopt_env
import rbfopt_model_selection as ms
import rbfopt_utils as ru
from rbfopt_settings import RbfSettings

class TestModelSelection(unittest.TestCase):
    """Test the rbfopt_model_selection module using available solvers."""

    def setUp(self):
        """Determine which model selection solvers should be tested."""
        self.solver = ['numpy']
        if cpx_available:
            self.solver.append('cplex')
        if clp_available:
            self.solver.append('clp')
        self.n = 3
        self.k = 10
        self.var_lower = np.array([i for i in range(self.n)])
        self.var_upper = np.array([i + 10 for i in range(self.n)])
        self.node_pos = np.array([self.var_lower, self.var_upper,
                         [1, 2, 3], [9, 5, 8.8], [5.5, 7, 12],
                         [3.2, 10.2, 4], [2.1, 1.1, 7.4], [6.6, 9.1, 2.0],
                         [10, 8.8, 11.1], [7, 7, 7]])
        self.node_val = np.array([2*i for i in range(self.k)])
    # -- end function        

    def test_get_best_rbf_model(self):
        """Test the get_best_rbf_model function.

        This is the main function of the module, which employs all
        other functions. We test it on a set of pre-generated data to
        verify that we get the expected response.
        """
        for solver in self.solver:
            settings = RbfSettings(model_selection_solver = solver)
            res = ms.get_best_rbf_model(settings, self.n, self.k, 
                                        self.node_pos, self.node_val,
                                        self.k)
            self.assertEqual(res, 'linear')
    # -- end function

    def test_get_model_quality_estimate_clp(self):
        """Test the get_model_quality_estimate_clp function.
        """
        if clp_available:
            settings = RbfSettings(model_selection_solver = 'clp')
            res = ms.get_model_quality_estimate_clp(settings, self.n, self.k, 
                                                    self.node_pos, 
                                                    self.node_val, self.k)
            self.assertAlmostEqual(res, 7.95598007028)
    # -- end function

    def test_get_model_quality_estimate_cpx(self):
        """Test the get_model_quality_estimate_cpx function.
        """
        if cpx_available:
            settings = RbfSettings(model_selection_solver = 'cplex')
            res = ms.get_model_quality_estimate_cpx(settings, self.n, self.k, 
                                                    self.node_pos, 
                                                    self.node_val, self.k)
            self.assertAlmostEqual(res, 7.95598007028)
    # -- end function

    def test_get_model_quality_estimate(self):
        """Test the get_model_quality_estimate function.
        """
        settings = RbfSettings(model_selection_solver = 'numpy')
        res = ms.get_model_quality_estimate(settings, self.n, self.k,
                                            self.node_pos, self.node_val,
                                            self.k)
        self.assertAlmostEqual(res, 7.95598007028)
    # -- end function

# -- end class

if (__name__ == '__main__'):
    unittest.main()
