"""Test the rbfopt_algorithm module in RBFOpt.

This module contains unit tests for the module rbfopt_algorithm, that
implements the main optimization algorithm.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright International Business Machines Corporation 2016.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import time
import tempfile
import os
import numpy as np
import rbfopt
import rbfopt.rbfopt_algorithm as ra
import rbfopt.rbfopt_test_functions as tf
from rbfopt.rbfopt_settings import RbfoptSettings
from rbfopt.rbfopt_black_box import RbfoptBlackBox

class TestGutmann(unittest.TestCase):
    """Test Gutmann's algorithm on a small set of problems."""

    rand_seeds = [512319876, 231974198, 908652418]

    def setUp(self):
        """Initialize random seed."""
        np.random.seed(71294123)

    def test_time_limit(self):
        """Verify that time limits are satisfied (Gutmann)."""
        bb = tf.TestBlackBox('hartman6')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving hartman6 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='Gutmann',
                                      global_search_method='solver',
                                      target_objval=optimum,
                                      eps_opt=0.0,
                                      max_clock_time=2.0,
                                      rand_seed=seed)
            start_time = time.time()
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            tot_time = time.time() - start_time
            msg = 'Time limit exceeded with Gutmann algorithm'
            self.assertLessEqual(tot_time, 5.0, msg=msg)
    # -- end function
# -- end class

class TestGutmannParallel(unittest.TestCase):
    """Test Gutmann's algorithm in parallel on a small set of problems."""

    rand_seeds = [512319876, 231974198, 908652418]

    def test_time_limit(self):
        """Verify that time limits are satisfied (Gutmann parallel)."""
        bb = tf.TestBlackBox('hartman6')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving hartman6 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='Gutmann',
                                      global_search_method='sampling',
                                      num_cpus=2,
                                      target_objval=optimum,
                                      eps_opt=0.0,
                                      max_clock_time=2.0,
                                      rand_seed=seed)
            start_time = time.time()
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            tot_time = time.time() - start_time
            msg = 'Time limit exceeded with Gutmann algorithm'
            self.assertLessEqual(tot_time, 5.0, msg=msg)
    # -- end function
# -- end class

class TestMSRSM(unittest.TestCase):
    """Test MSRSM algorithm on a small set of problems."""

    rand_seeds = [512319876, 231974198, 908652418]

    def test_time_limit(self):
        """Verify that time limits are satisfied (MSRSM)."""
        bb = tf.TestBlackBox('hartman6')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving hartman6 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      global_search_method='solver',
                                      target_objval=optimum,
                                      eps_opt=0.0,
                                      max_clock_time=2.0,
                                      rand_seed=seed)
            start_time = time.time()
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            tot_time = time.time() - start_time
            msg = 'Time limit exceeded with MSRSM algorithm'
            self.assertLessEqual(tot_time, 5.0, msg=msg)
    # -- end function
# -- end class

class TestMSRSMParallel(unittest.TestCase):
    """Test MSRSM algorithm in parallel on a small set of problems."""

    rand_seeds = [512319876, 231974198, 908652418]

    def test_time_limit(self):
        """Verify that time limits are satisfied (MSRSM parallel)."""
        bb = tf.TestBlackBox('hartman6')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving hartman6 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      global_search_method='genetic',
                                      num_cpus=2,
                                      target_objval=optimum,
                                      eps_opt=0.0,
                                      max_clock_time=2.0,
                                      rand_seed=seed)
            start_time = time.time()
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            tot_time = time.time() - start_time
            msg = 'Time limit exceeded with MSRSM algorithm in parallel'
            self.assertLessEqual(tot_time, 5.0, msg=msg)
    # -- end function
# -- end class

class TestState(unittest.TestCase):
    """Test load/save state methods."""

    rand_seeds = [512319876, 231974198, 908652418]
    eps_opt = 0.05

    def test_state_reload(self):
        """Check solution of ex8_1_4 after state save/reload."""
        bb = tf.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        handle, filename = tempfile.mkstemp()
        for seed in self.rand_seeds:
            print()
            print('Solving ex8_1_4 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      rbf='linear',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize(5)
            alg.save_to_file(filename)
            alg_reload = ra.RbfoptAlgorithm.load_from_file(filename)
            res = alg_reload.optimize()
            msg = 'Could not solve ex8_1_4 after reload'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
        os.close(handle)
        os.remove(filename)
    # -- end function
# -- end class

class TestBlackBoxFixed(RbfoptBlackBox):
    """A black-box constructed from a known test function.

    Parameters
    ----------
    name : string
        The name of the function to be implemented.
    """
    def __init__(self, name, num_fixed_vars):
        """Constructor.
        """
        try:
            self._function = getattr(tf, name.lower())
        except AttributeError:
            raise ValueError('Function ' + name + ' not implemented')
        # Generate lower and upper bounds for fixed variables
        self.fixed_var_lower = np.random.rand(1, num_fixed_vars)[0]*10 - 5
        self.fixed_var_upper = self.fixed_var_lower
        self.num_fixed_vars = num_fixed_vars
        self.var_pos = np.random.permutation(self._function.dimension +
                                             num_fixed_vars)
        

    def get_dimension(self):
        return self._function.dimension + self.num_fixed_vars

    def get_var_lower(self):
        return np.array([self._function.var_lower[i] 
                         if i < self._function.dimension else
                         self.fixed_var_lower[i - self._function.dimension]
                         for i in self.var_pos])

    def get_var_upper(self):
        return np.array([self._function.var_upper[i] 
                         if i < self._function.dimension else
                         self.fixed_var_upper[i - self._function.dimension]
                         for i in self.var_pos])

    def get_var_type(self):
        return np.array([self._function.var_type[i]
                         if i < self._function.dimension else 'R'
                         for i in self.var_pos])

    def evaluate(self, point):
        # Objective is shifted by the sum of all fixed variables
        orig_point = np.array([point[np.where(self.var_pos == i)[0][0]] 
                               for i in range(self._function.dimension)])
        return (self._function.evaluate(orig_point) + 
                sum([point[np.where(self.var_pos == i)[0][0]] for i in 
                     range(self._function.dimension, 
                           self._function.dimension + self.num_fixed_vars)]))
        
    def evaluate_noisy(self, point):
        raise NotImplementedError('evaluate_noisy() not implemented')

    def has_evaluate_noisy(self):
        return False
        
    def get_obj_shift(self):
        # We simply sum all fixed variables
        var_lower = self.get_var_lower()
        return sum([var_lower[np.where(self.var_pos == i)[0][0]] for i in 
                    range(self._function.dimension, 
                          self._function.dimension + self.num_fixed_vars)])    
# -- end class


class TestFixedVariables(unittest.TestCase):
    """Test problems with fixed variables."""
    rand_seeds = [512319876, 231974198, 908652418]
    eps_opt = 0.05

    def test_branin_fixed(self):
        """Check solution of branin with fixed variables."""
        for seed in self.rand_seeds:
            bb = TestBlackBoxFixed('branin', 4)
            optimum = bb._function.optimum_value + bb.get_obj_shift()
            print()
            print('Solving branin with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve branin with fixed variables'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_prob03_fixed(self):
        """Check solution of prob03 with fixed variables."""
        for seed in self.rand_seeds:
            bb = TestBlackBoxFixed('prob03', 5)
            optimum = bb._function.optimum_value + bb.get_obj_shift()
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve prob03 with fixed variables'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function
# -- end class
