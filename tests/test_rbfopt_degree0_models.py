"""Test the successful creation of Pyomo 0-degree models in RBFOpt.

This module contains unit tests for the module rbfopt_degree0_models.

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
import rbfopt.rbfopt_degree0_models as d0
from rbfopt.rbfopt_settings import RbfoptSettings

class TestMultiquadricModels(unittest.TestCase):
    """Test the rbfopt_degree0_models module using multiquadric RBF."""

    def setUp(self):
        """Generate data to simulate an optimization problem."""
        np.random.seed(71294123)
        self.settings = RbfoptSettings(rbf = 'multiquadric')
        self.n = 3
        self.k = 5
        self.var_lower = np.array([i for i in range(self.n)])
        self.var_upper = np.array([i + 10 for i in range(self.n)])
        self.node_pos = np.array([self.var_lower, self.var_upper,
                         [1, 2, 3], [9, 5, 8.8], [5.5, 7, 12]])
        self.node_val = np.array([2*i for i in range(self.k)])
        Amat = [[1.0, 17.349351572897476, 1.9999999999999998,
                 12.009995836801943, 12.932517156377562, 1.0],
                [17.349351572897476, 1.0, 15.620499351813308,
                 6.945502141674135, 6.103277807866851, 1.0],
                [1.9999999999999998, 15.620499351813308, 1.0,
                 10.374969879474351, 11.280514172678478, 1.0],
                [12.009995836801943, 6.945502141674135, 10.374969879474351,
                 1.0, 5.243090691567331, 1.0], 
                [12.932517156377562, 6.103277807866851, 11.280514172678478,
                 5.243090691567331, 1.0, 1.0], 
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]
        self.Amat = np.matrix(Amat)
        self.Amatinv = self.Amat.getI()
        self.rbf_lambda = np.array([1.981366489986409, 0.6262004309283905,
                                    -1.8477896263093248, -0.10028069928913483,
                                    -0.65949659531634])
        self.rbf_h = np.array([0.5833631458309435])
        self.integer_vars = np.array([1])
    # -- end function        

    def test_create_min_rbf_model(self):
        """Test the create_min_rbf_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d0.create_min_rbf_model(self.settings, self.n, self.k,
                                        self.var_lower, self.var_upper,
                                        self.integer_vars, None, self.node_pos,
                                        self.rbf_lambda, self.rbf_h)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)
        model = d0.create_min_rbf_model(
            self.settings, 10, 20, np.array([0] * 10),np.array([1] * 10),
            np.array([i for i in range(10)]),
            (np.array([0]), np.array([]),
             [(0, 0, np.array([i for i in range(10)]))]),
            np.random.randint(0, 2, size=(20, 10)),
            np.random.uniform(size=20), np.array([-1]))
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)
        

    def test_create_max_one_over_mu_model(self):
        """Test the create_max_one_over_mu_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d0.create_max_one_over_mu_model(self.settings, self.n, self.k,
                                                self.var_lower, self.var_upper,
                                                self.integer_vars, None,
                                                self.node_pos, self.Amat)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_max_h_k_model(self):
        """Test the create_max_h_k_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d0.create_max_h_k_model(self.settings, self.n, self.k,
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
        model = d0.create_min_bump_model(self.settings, self.n, self.k, 
                                         Phimat, Pmat, self.node_val,
                                         node_err_bounds)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_maximin_dist_model(self):
        """Test the create_maximin_dist_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d0.create_maximin_dist_model(self.settings, self.n, self.k,
                                             self.var_lower, self.var_upper, 
                                             self.integer_vars, None,
                                             self.node_pos)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_min_msrsm_model(self):
        """Test the create_min_msrsm_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d0.create_min_msrsm_model(self.settings, self.n, self.k,
                                          self.var_lower, self.var_upper,
                                          self.integer_vars, None,
                                          self.node_pos, self.rbf_lambda,
                                          self.rbf_h, 0.5, 0.0, 1.0,
                                          min(self.node_val),
                                          max(self.node_val))
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

# -- end class

class TestLinearModels(unittest.TestCase):
    """Test the rbfopt_degree0_models module using linear RBF."""

    def setUp(self):
        """Generate data to simulate an optimization problem."""
        self.settings = RbfoptSettings(rbf = 'linear')
        self.n = 3
        self.k = 5
        self.var_lower = np.array([i for i in range(self.n)])
        self.var_upper = np.array([i + 10 for i in range(self.n)])
        self.node_pos = np.array([self.var_lower, self.var_upper,
                         [1, 2, 3], [9, 5, 8.8], [5.5, 7, 12]])
        self.node_val = np.array([2*i for i in range(self.k)])
        Amat = [[0.0, 17.320508075688775, 1.7320508075688772,
                 11.968291440301744, 12.893796958227627, 1.0],
                [17.320508075688775, 0.0, 15.588457268119896,
                 6.873136110975833, 6.020797289396148, 1.0],
                [1.7320508075688772, 15.588457268119896, 0.0,
                 10.32666451474047, 11.236102527122116, 1.0],
                [11.968291440301744, 6.873136110975833, 
                 10.32666451474047, 0.0, 5.146843692983108, 1.0],
                [12.893796958227627, 6.020797289396148,
                 11.236102527122116, 5.146843692983108, 0.0, 1.0], 
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]
        self.Amat = np.matrix(Amat)
        self.Amatinv = self.Amat.getI()
        self.rbf_lambda = np.array([1.1704846814048488, 0.5281643269521171,
                                    -0.9920149389974761, -0.1328847504999134,
                                    -0.5737493188595765])
        self.rbf_h = np.array([1.5583564301976252])
        self.integer_vars = np.array([1])
    # -- end function        

    def test_create_min_rbf_model(self):
        """Test the create_min_rbf_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d0.create_min_rbf_model(self.settings, self.n, self.k,
                                        self.var_lower, self.var_upper,
                                        self.integer_vars, None, self.node_pos,
                                        self.rbf_lambda, self.rbf_h)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)
        model = d0.create_min_rbf_model(
            self.settings, 10, 20, np.array([0] * 10),np.array([1] * 10),
            np.array([i for i in range(10)]),
            (np.array([0]), np.array([]),
             [(0, 0, np.array([i for i in range(10)]))]),
            np.random.randint(0, 2, size=(20, 10)),
            np.random.uniform(size=20), np.array([-1]))
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)


    def test_create_max_one_over_mu_model(self):
        """Test the create_max_one_over_mu_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d0.create_max_one_over_mu_model(self.settings, self.n, self.k,
                                                self.var_lower, self.var_upper,
                                                self.integer_vars, None,
                                                self.node_pos, self.Amat)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_max_h_k_model(self):
        """Test the create_max_h_k_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d0.create_max_h_k_model(self.settings, self.n, self.k,
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
        model = d0.create_min_bump_model(self.settings, self.n, self.k, 
                                         Phimat, Pmat, self.node_val,
                                         node_err_bounds)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_maximin_dist_model(self):
        """Test the create_maximin_dist_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d0.create_maximin_dist_model(self.settings, self.n, self.k,
                                             self.var_lower, self.var_upper, 
                                             self.integer_vars, None,
                                             self.node_pos)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_min_msrsm_model(self):
        """Test the create_min_msrsm_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d0.create_min_msrsm_model(self.settings, self.n, self.k,
                                          self.var_lower, self.var_upper,
                                          self.integer_vars, None,
                                          self.node_pos, self.rbf_lambda,
                                          self.rbf_h, 0.5, 0.0, 1.0,
                                          min(self.node_val),
                                          max(self.node_val))
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

# -- end class
