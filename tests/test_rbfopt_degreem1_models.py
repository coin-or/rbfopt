"""Test the successful creation of Pyomo 1-degree models in RBFOpt.

This module contains unit tests for the module rbfopt_degree1_models.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright International Business Machines Corporation 2016.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import rbfopt
import numpy as np
import pyomo.environ
import rbfopt.rbfopt_utils as ru
import rbfopt.rbfopt_degreem1_models as dm1
from rbfopt.rbfopt_settings import RbfoptSettings

class TestGaussianModels(unittest.TestCase):
    """Test the rbfopt_degreem1_models module using gaussian RBF."""

    def setUp(self):
        """Generate data to simulate an optimization problem."""
        np.random.seed(71294123)
        self.settings = RbfoptSettings(rbf = 'gaussian')
        self.n = 3
        self.k = 5
        self.var_lower = np.array([i for i in range(self.n)])
        self.var_upper = np.array([i + 10 for i in range(self.n)])
        self.node_pos = np.array([self.var_lower, self.var_upper,
                         [1, 2, 3], [9, 5, 8.8], [5.5, 7, 12]])
        self.node_val = np.array([2*i for i in range(self.k)])
        Amat = [[  1.00000000e+00,   0.00000000e+00,   4.97870684e-02,
                   0.00000000e+00,   0.00000000e+00],
                [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
                   0.00000000e+00,   0.00000000e+00],
                [  4.97870684e-02,   0.00000000e+00,   1.00000000e+00,
                   0.00000000e+00,   0.00000000e+00],
                [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                   1.00000000e+00,   3.12996279e-12],
                [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                   3.12996279e-12,   1.00000000e+00]]
        self.Amat = np.matrix(Amat)
        self.Amatinv = self.Amat.getI()
        self.rbf_lambda = np.array([-0.19964314,  2.        ,  4.00993965,
                                    6.        ,  8.        ])
        self.rbf_h = np.array([])
        self.integer_vars = np.array([1])
    # -- end function        

    def test_create_min_rbf_model(self):
        """Test the create_min_rbf_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = dm1.create_min_rbf_model(self.settings, self.n, self.k,
                                         self.var_lower, self.var_upper,
                                         self.integer_vars, None,
                                         self.node_pos,
                                         self.rbf_lambda, self.rbf_h)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)
        model = dm1.create_min_rbf_model(
            self.settings, 10, 20, np.array([0] * 10),np.array([1] * 10),
            np.array([i for i in range(10)]),
            (np.array([0]), np.array([]),
             [(0, 0, np.array([i for i in range(10)]))]),
            np.random.randint(0, 2, size=(20, 10)),
            np.random.uniform(size=20), np.array([]))
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)


    def test_create_max_one_over_mu_model(self):
        """Test the create_max_one_over_mu_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = dm1.create_max_one_over_mu_model(self.settings, self.n, self.k,
                                                 self.var_lower, self.var_upper,
                                                 self.integer_vars, None,
                                                 self.node_pos, self.Amat)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_max_h_k_model(self):
        """Test the create_max_h_k_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = dm1.create_max_h_k_model(self.settings, self.n, self.k,
                                         self.var_lower, self.var_upper,
                                         self.integer_vars, None,
                                         self.node_pos, self.rbf_lambda,
                                         self.rbf_h, self.Amat, -1)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_min_bump_model(self):
        """Test the create_min_bump_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        Phimat = self.Amat[:self.k, :self.k]
        Pmat = self.Amat[:self.k, self.k:]
        node_err_bounds = np.array([[- 2, + 2] for i in range(self.k)])
        model = dm1.create_min_bump_model(self.settings, self.n, self.k, 
                                         Phimat, Pmat, self.node_val,
                                         node_err_bounds)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_maximin_dist_model(self):
        """Test the create_maximin_dist_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = dm1.create_maximin_dist_model(self.settings, self.n,
                                              self.k, self.var_lower,
                                              self.var_upper,
                                              self.integer_vars,
                                              None,self.node_pos)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_min_msrsm_model(self):
        """Test the create_min_msrsm_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = dm1.create_min_msrsm_model(self.settings, self.n,
                                           self.k, self.var_lower,
                                           self.var_upper,
                                           self.integer_vars,
                                           None, self.node_pos,
                                           self.rbf_lambda, self.rbf_h,
                                           0.5, 0.0, 1.0,
                                           min(self.node_val),
                                           max(self.node_val))
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

# -- end class

