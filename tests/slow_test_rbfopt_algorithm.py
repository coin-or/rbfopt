"""Test the rbfopt_algorithm module in RBFOpt - slow tests.

This module contains the slow tests for the module rbfopt_algorithm,
that implements the main optimization algorithm. These tests may not
be necessary for the average user and they can take several minutes.

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
import rbfopt
import numpy as np
import rbfopt.rbfopt_algorithm as ra
import rbfopt.rbfopt_test_functions as tf
from rbfopt.rbfopt_settings import RbfoptSettings
from rbfopt.rbfopt_black_box import RbfoptBlackBox

class TestGutmann(unittest.TestCase):
    """Test Gutmann's algorithm on a small set of problems."""

    rand_seeds = [512319876, 231974198, 908652418]
    eps_opt = 0.05

    def setUp(self):
        """Initialize random seed."""
        np.random.seed(71294123)

    def test_gutmann_ex8_1_4(self):
        """Check solution of ex8_1_4 with Gutmann's method, genetic."""
        bb = tf.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving ex8_1_4 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='Gutmann',
                                      global_search_method='genetic',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve ex8_1_4 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_gutmann_prob03(self):
        """Check solution of prob03 with Gutmann's method, solver."""
        bb = tf.TestBlackBox('prob03')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='Gutmann',
                                      rbf='cubic',
                                      global_search_method='solver',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve prob03 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_gutmann_ex8_1_4_log(self):
        """Check solution of ex8_1_4 with Gutmann, log scaling, infstep.

        Sampling-based global search.

        """
        bb = tf.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving ex8_1_4 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='Gutmann',
                                      rbf='multiquadric',
                                      global_search_method='sampling',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      function_scaling='log',
                                      do_infstep=True,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve ex8_1_4 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_gutmann_st_miqp3_noisy(self):
        """Check solution of noisy st_miqp3 with Gutmann, genetic."""
        bb = tf.TestNoisyBlackBox(tf.TestBlackBox('st_miqp3'), 0.1, 0.01)
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving st_miqp3 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='Gutmann',
                                      global_search_method='genetic',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve st_miqp3 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_gutmann_branin_noisy_with_init(self):
        """Check solution of noisy branin with Gutmann, solver."""
        bb = tf.TestNoisyBlackBox(tf.TestBlackBox('branin'), 0.1, 0.01)
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving branin with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='Gutmann',
                                      global_search_method='solver',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      rand_seed=seed)
            init_node_pos = [[0, 0], [-2, 2], [5, 10]]
            init_node_val = [bb._function.evaluate(x) for x in init_node_pos]
            alg = ra.RbfoptAlgorithm(settings, bb, init_node_pos, 
                                     init_node_val)
            res = alg.optimize()
            msg = ('Could not solve noisy branin with init and ' +
                   'Gutmann\'s algorithm')
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_gutmann_branin_cat_noisy_with_init(self):
        """Check solution of noisy branin_cat with Gutmann, solver."""
        bb = tf.TestNoisyBlackBox(tf.TestBlackBox('branin_cat'), 0.1, 0.01)
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving branin_cat with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='Gutmann',
                                      global_search_method='solver',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      rand_seed=seed)
            init_node_pos = [[0, 0, 0], [-2, 2, 1], [5, 10, 2]]
            init_node_val = [bb._function.evaluate(x) for x in init_node_pos]
            alg = ra.RbfoptAlgorithm(settings, bb, init_node_pos, 
                                     init_node_val)
            res = alg.optimize()
            msg = ('Could not solve noisy branin with init and ' +
                   'Gutmann\'s algorithm')
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

# -- end class

class TestGutmannParallel(unittest.TestCase):
    """Test Gutmann's algorithm in parallel on a small set of problems."""

    rand_seeds = [512319876, 231974198, 908652418]
    eps_opt = 0.2

    def test_gutmann_parallel_ex8_1_4(self):
        """Check solution of ex8_1_4 with Gutmann's method, solver."""
        bb = tf.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving ex8_1_4 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='Gutmann',
                                      rbf='gaussian',
                                      global_search_method='solver',
                                      target_objval=optimum,
                                      max_stalled_iterations=20,
                                      eps_impr=0.05,
                                      refinement_frequency=6,
                                      eps_opt=self.eps_opt,
                                      max_iterations=500,
                                      max_evaluations=500,
                                      max_fraction_discarded=0.2,
                                      num_cpus=2,
                                      max_clock_time=1200,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve ex8_1_4 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_gutmann_parallel_prob03(self):
        """Check solution of prob03 with Gutmann's method, sampling."""
        bb = tf.TestBlackBox('prob03')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='Gutmann',
                                      global_search_method='sampling',
                                      rbf='cubic',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=500,
                                      max_evaluations=500,
                                      num_cpus=2,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve prob03 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_gutmann_parallel_ex8_1_4_infstep(self):
        """Check solution of ex8_1_4 with Gutmann, infstep.

        Genetic algorithm."""
        bb = tf.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving ex8_1_4 with infstep and random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='Gutmann',
                                      global_search_method='genetic',
                                      rbf='multiquadric',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=500,
                                      max_evaluations=500,
                                      num_cpus=2,
                                      do_infstep=True,
                                      refinement_frequency=5,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve ex8_1_4 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_gutmann_parallel_st_miqp3_noisy(self):
        """Check solution of noisy st_miqp3 with Gutmann, solver."""
        bb = tf.TestNoisyBlackBox(tf.TestBlackBox('st_miqp3'), 0.1, 0.01)
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving st_miqp3 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='Gutmann',
                                      global_search_method='solver',
                                      rbf_shape_parameter=0.01,
                                      rbf='gaussian',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=500,
                                      max_evaluations=500,
                                      num_cpus=2,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve st_miqp3 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

# -- end class

class TestMSRSM(unittest.TestCase):
    """Test MSRSM algorithm on a small set of problems."""

    rand_seeds = [512319876, 231974198, 908652418]
    eps_opt = 0.05

    def test_msrsm_ex8_1_4(self):
        """Check solution of ex8_1_4 with the MSRSM algorithm, sampling."""
        bb = tf.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving ex8_1_4 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      global_search_method='sampling',
                                      rbf='linear',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve hartman3 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_msrsm_prob03(self):
        """Check solution of prob03 with the MSRSM algorithm, genetic."""
        bb = tf.TestBlackBox('prob03')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      global_search_method='genetic',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve prob03 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_msrsm_prob03_no_local_search(self):
        """Check solution of prob03 with MSRSM and no local search.

        Sampling solution of global search problems.

        """
        bb = tf.TestBlackBox('prob03')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(rbf='cubic',
                                      global_search_method='sampling',
                                      algorithm='MSRSM',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      local_search_box_scaling=10000,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve prob03 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_msrsm_st_miqp3_noisy(self):
        """Check solution of noisy st_miqp3 with MSRSM, genetic."""
        bb = tf.TestNoisyBlackBox(tf.TestBlackBox('st_miqp3'), 0.1, 0.01)
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving st_miqp3 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      global_search_method='genetic',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve st_miqp3 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_msrsm_branin_noisy_with_init(self):
        """Check solution of noisy branin with MSRSM, sampling."""
        bb = tf.TestNoisyBlackBox(tf.TestBlackBox('branin'), 0.1, 0.01)
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving branin with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      global_search_method='sampling',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      rand_seed=seed)
            init_node_pos = [[0, 0], [-2, 2], [5, 10], [-2.5, 1]]
            alg = ra.RbfoptAlgorithm(settings, bb, init_node_pos)
            res = alg.optimize()
            msg = ('Could not solve noisy branin with init and ' +
                   'MSRSM algorithm')
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_msrsm_branin_cat_noisy_with_init(self):
        """Check solution of noisy branin with MSRSM, sampling."""
        bb = tf.TestNoisyBlackBox(tf.TestBlackBox('branin_cat'), 0.1, 0.01)
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving branin_cat with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      global_search_method='sampling',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=200,
                                      max_evaluations=300,
                                      rand_seed=seed)
            init_node_pos = [[0, 0, 0]]
            alg = ra.RbfoptAlgorithm(settings, bb, init_node_pos)
            res = alg.optimize()
            msg = ('Could not solve noisy branin with init and ' +
                   'MSRSM algorithm')
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

# -- end class

class TestMSRSMParallel(unittest.TestCase):
    """Test MSRSM algorithm in parallel on a small set of problems."""

    rand_seeds = [512319876, 231974198, 908652418]
    eps_opt = 0.2

    def test_msrsm_parallel_ex8_1_4(self):
        """Check solution of ex8_1_4 with the MSRSM algorithm, sampling."""
        bb = tf.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving ex8_1_4 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      global_search_method='sampling',
                                      rbf='linear',
                                      rbf_shape_parameter=0.01,
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=500,
                                      max_evaluations=500,
                                      num_cpus=2,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve ex8_1_4 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function


    def test_msrsm_parallel_nvs09_cat(self):
        """Check solution of nvs09_cat with the MSRSM algorithm, sampling."""
        bb = tf.TestBlackBox('nvs09_cat')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving nvs09_cat with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      global_search_method='sampling',
                                      rbf='linear',
                                      rbf_shape_parameter=0.01,
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=500,
                                      max_evaluations=500,
                                      num_cpus=2,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve nvs09_cat with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_msrsm_parallel_prob03(self):
        """Check solution of prob03 with the MSRSM algorithm, genetic."""
        bb = tf.TestBlackBox('prob03')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      global_search_method='genetic',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=500,
                                      max_evaluations=500,
                                      num_cpus=4,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve prob03 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_msrsm_parallel_st_miqp3_no_local_search(self):
        """Check solution of st_miqp3 with MSRSM and no local search.

        Solver solution of global search problems."""
        bb = tf.TestBlackBox('st_miqp3')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving st_miqp3 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(rbf='cubic',
                                      global_search_method='genetic',
                                      algorithm='MSRSM',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=500,
                                      max_evaluations=500,
                                      num_cpus=4,
                                      local_search_box_scaling=10000,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve st_miqp3 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

    def test_msrsm_parallel_prob03_noisy(self):
        """Check solution of noisy prob03 with MSRSM, sampling."""
        bb = tf.TestNoisyBlackBox(tf.TestBlackBox('prob03'), 0.1, 0.01)
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfoptSettings(algorithm='MSRSM',
                                      global_search_method='sampling',
                                      target_objval=optimum,
                                      eps_opt=self.eps_opt,
                                      max_iterations=500,
                                      max_evaluations=500,
                                      num_cpus=4,
                                      rand_seed=seed)
            alg = ra.RbfoptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve prob03 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero
                                else self.eps_opt)
            self.assertLessEqual(res[0], target, msg=msg)
    # -- end function

# -- end class
