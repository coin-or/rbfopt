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
import rbfopt.rbfopt_degree1_models as d1
from rbfopt.rbfopt_settings import RbfoptSettings

class TestCubicModels(unittest.TestCase):
    """Test the rbfopt_degree1_models module using cubic RBF."""

    def setUp(self):
        """Generate data to simulate an optimization problem."""
        np.random.seed(71294123)
        self.settings = RbfoptSettings(rbf = 'cubic')
        self.n = 3
        self.k = 5
        self.var_lower = np.array([i for i in range(self.n)])
        self.var_upper = np.array([i + 10 for i in range(self.n)])
        self.node_pos = np.array([self.var_lower, self.var_upper,
                         [1, 2, 3], [9, 5, 8.8], [5.5, 7, 12]])
        self.node_val = np.array([2*i for i in range(self.k)])
        Amat = [[0.0, 5196.152422706633, 5.196152422706631,
                 1714.338065908822, 2143.593744305343, 0.0, 1.0, 2.0, 1.0],
                [5196.152422706633, 0.0, 3787.995116153135, 324.6869498824983,
                 218.25390174061036, 10.0, 11.0, 12.0, 1.0],
                [5.196152422706631, 3787.995116153135, 0.0, 1101.235503851924,
                 1418.557944049167, 1.0, 2.0, 3.0, 1.0], 
                [1714.338065908822, 324.6869498824983, 1101.235503851924, 
                 0.0, 136.3398894271225, 9.0, 5.0, 8.8, 1.0],
                [2143.593744305343, 218.25390174061036, 1418.557944049167,
                 136.3398894271225, 0.0, 5.5, 7.0, 12.0, 1.0],
                [0.0, 10.0, 1.0, 9.0, 5.5, 0.0, 0.0, 0.0, 0.0], 
                [1.0, 11.0, 2.0, 5.0, 7.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 12.0, 3.0, 8.8, 12.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
        self.Amat = np.matrix(Amat)
        self.Amatinv = self.Amat.getI()
        self.rbf_lambda = np.array([-0.02031417613815348, -0.0022571306820170587,
                                    0.02257130682017054, 6.74116235140294e-18,
                                    -1.0962407017011667e-18])
        self.rbf_h = np.array([-0.10953754862932995, 0.6323031632900591,
                               0.5216788297837124, 9.935450288253636])
        self.integer_vars = np.array([1])
    # -- end function        

    def test_create_min_rbf_model(self):
        """Test the create_min_rbf_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d1.create_min_rbf_model(self.settings, self.n, self.k,
                                        self.var_lower, self.var_upper,
                                        self.integer_vars, None, self.node_pos,
                                        self.rbf_lambda, self.rbf_h)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)
        model = d1.create_min_rbf_model(
            self.settings, 10, 20, np.array([0] * 10),np.array([1] * 10),
            np.array([i for i in range(10)]),
            (np.array([0]), np.array([]),
             [(0, 0, np.array([i for i in range(10)]))]),
            np.random.randint(0, 2, size=(20, 10)),
            np.random.uniform(size=20), np.random.uniform(size=11))
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)


    def test_create_max_one_over_mu_model(self):
        """Test the create_max_one_over_mu_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d1.create_max_one_over_mu_model(self.settings, self.n, self.k,
                                                self.var_lower, self.var_upper,
                                                self.integer_vars, None,
                                                self.node_pos, self.Amat)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_max_h_k_model(self):
        """Test the create_max_h_k_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d1.create_max_h_k_model(self.settings, self.n, self.k,
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
        model = d1.create_min_bump_model(self.settings, self.n, self.k, 
                                         Phimat, Pmat, self.node_val,
                                         node_err_bounds)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_maximin_dist_model(self):
        """Test the create_maximin_dist_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d1.create_maximin_dist_model(self.settings, self.n, self.k,
                                             self.var_lower, self.var_upper, 
                                             self.integer_vars, None,
                                             self.node_pos)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_min_msrsm_model(self):
        """Test the create_min_msrsm_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d1.create_min_msrsm_model(self.settings, self.n, self.k,
                                          self.var_lower, self.var_upper,
                                          self.integer_vars, None,
                                          self.node_pos, self.rbf_lambda,
                                          self.rbf_h, 0.5, 0.0, 1.0,
                                          min(self.node_val),
                                          max(self.node_val))
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

# -- end class

class TestThinPlateSplineModels(unittest.TestCase):
    """Test the rbfopt_degree1_models module using thin plate splines."""

    def setUp(self):
        """Generate data to simulate an optimization problem."""
        self.settings = RbfoptSettings(rbf = 'thin_plate_spline')
        self.n = 3
        self.k = 5
        self.var_lower = np.array([i for i in range(self.n)])
        self.var_upper = np.array([i + 10 for i in range(self.n)])
        self.node_pos = np.array([self.var_lower, self.var_upper,
                         [1, 2, 3], [9, 5, 8.8], [5.5, 7, 12]])
        self.node_val = np.array([2*i for i in range(self.k)])
        Amat = [[0.0, 855.5673711984304, 1.6479184330021641,
                 355.55903306222723, 425.059078986427, 0.0, 1.0, 2.0, 1.0],
                [855.5673711984304, 0.0, 667.4069653658767, 91.06079221519477,
                 65.07671378607489, 10.0, 11.0, 12.0, 1.0],
                [1.6479184330021641, 667.4069653658767, 0.0,
                 248.97553659741263, 305.415419302314, 1.0, 2.0, 3.0, 1.0],
                [355.55903306222723, 91.06079221519477, 248.97553659741263,
                 0.0, 43.40078293199628, 9.0, 5.0, 8.8, 1.0],
                [425.059078986427, 65.07671378607489, 305.415419302314,
                 43.40078293199628, 0.0, 5.5, 7.0, 12.0, 1.0], 
                [0.0, 10.0, 1.0, 9.0, 5.5, 0.0, 0.0, 0.0, 0.0],
                [1.0, 11.0, 2.0, 5.0, 7.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 12.0, 3.0, 8.8, 12.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
        self.Amat = np.matrix(Amat)
        self.Amatinv = self.Amat.getI()
        self.rbf_lambda = np.array([-0.1948220562664489, -0.02164689514071656,
                                    0.21646895140716543, 2.4492621453325443e-18,
                                    3.4694803106897584e-17])
        self.rbf_h = np.array([-0.047916449337864896, -0.42012611088687196,
                               1.072728018406163, 16.43832406902896])
        self.integer_vars = np.array([1])
    # -- end function        

    def test_create_min_rbf_model(self):
        """Test the create_min_rbf_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d1.create_min_rbf_model(self.settings, self.n, self.k,
                                        self.var_lower, self.var_upper,
                                        self.integer_vars, None, self.node_pos,
                                        self.rbf_lambda, self.rbf_h)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)
        model = d1.create_min_rbf_model(
            self.settings, 10, 20, np.array([0] * 10),np.array([1] * 10),
            np.array([i for i in range(10)]),
            (np.array([0]), np.array([]),
             [(0, 0, np.array([i for i in range(10)]))]),
            np.random.randint(0, 2, size=(20, 10)),
            np.random.uniform(size=20), np.random.uniform(size=11))
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)


    def test_create_max_one_over_mu_model(self):
        """Test the create_max_one_over_mu_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d1.create_max_one_over_mu_model(self.settings, self.n, self.k,
                                                self.var_lower, self.var_upper,
                                                self.integer_vars, None,
                                                self.node_pos, self.Amat)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_max_h_k_model(self):
        """Test the create_max_h_k_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d1.create_max_h_k_model(self.settings, self.n, self.k,
                                        self.var_lower, self.var_upper,
                                        self.integer_vars, None, self.node_pos,
                                        self.rbf_lambda, self.rbf_h,
                                        self.Amat, -1)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_min_bump_model(self):
        """Test the create_min_bump_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        Phimat = self.Amat[:self.k, :self.k]
        Pmat = self.Amat[:self.k, self.k:]
        node_err_bounds = np.array([[- 2, + 2] for i in range(self.k)])
        model = d1.create_min_bump_model(self.settings, self.n, self.k, 
                                         Phimat, Pmat, self.node_val,
                                         node_err_bounds)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_maximin_dist_model(self):
        """Test the create_maximin_dist_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d1.create_maximin_dist_model(self.settings, self.n, self.k,
                                             self.var_lower, self.var_upper, 
                                             self.integer_vars, None,
                                             self.node_pos)
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)

    def test_create_min_msrsm_model(self):
        """Test the create_min_msrsm_model function.

        This test simply checks whether the function returns a valid
        pyomo.ConcreteModel object.
        """
        model = d1.create_min_msrsm_model(self.settings, self.n, self.k,
                                          self.var_lower, self.var_upper,
                                          self.integer_vars, None,
                                          self.node_pos, self.rbf_lambda,
                                          self.rbf_h, 0.5, 0.0, 1.0,
                                          min(self.node_val),
                                          max(self.node_val))
        self.assertIsInstance(model, pyomo.environ.ConcreteModel)


# -- end class
