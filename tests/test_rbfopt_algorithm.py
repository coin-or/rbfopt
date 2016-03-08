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
import test_rbfopt_env
from rbfopt_settings import RbfSettings
import rbfopt_test_interface as ti
import test_functions
import rbfopt_algorithm as ra

def create_black_box(function, evaluate_fast = None):
    """Return a TestFunctionBlackBox object with given function.
    
    Given a function from test_functions, this method creates a
    black-box object setting all appropriate members. The
    evaluate_fast will be set to None by default, otherwise a
    different value has to be specified.
    """
    black_box = ti.TestFunctionBlackBox()
    black_box.dimension = function.dimension
    black_box.var_lower = function.var_lower
    black_box.var_upper = function.var_upper
    black_box.integer_vars = function.integer_vars
    black_box.evaluate = function.evaluate
    black_box.evaluate_fast = evaluate_fast
    return black_box
# -- end function

class TestGutmann(unittest.TestCase):
    """Test Gutmann's algorithm on a small set of problems."""

    rand_seeds = [512319876412, 231974198123, 90865241837]
    eps_opt = 0.05        

    def test_gutmann_goldsteinprice(self):
        """Check solution of goldsteinprice with Gutmann's method."""
        function = test_functions.hartman3
        bb = create_black_box(function)
        optimum = function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving goldsteinprice with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings = settings,
                                  dimension = bb.dimension, 
                                  var_lower = bb.var_lower,
                                  var_upper = bb.var_upper,
                                  objfun = bb.evaluate,
                                  objfun_fast = bb.evaluate_fast,
                                  integer_vars = bb.integer_vars)
            res = alg.optimize()
            msg = 'Could not solve goldstein with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_gutmann_prob03(self):
        """Check solution of prob03 with Gutmann's method."""
        function = test_functions.prob03
        bb = create_black_box(function)
        optimum = function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   rbf = 'cubic',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings = settings,
                                  dimension = bb.dimension, 
                                  var_lower = bb.var_lower,
                                  var_upper = bb.var_upper,
                                  objfun = bb.evaluate,
                                  objfun_fast = bb.evaluate_fast,
                                  integer_vars = bb.integer_vars)
            res = alg.optimize()
            msg = 'Could not solve prob03 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_gutmann_branin_log(self):
        """Check solution of branin with Gutmann, log scaling, infstep."""
        function = test_functions.branin
        bb = create_black_box(function)
        optimum = function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving branin with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   rbf = 'multiquadric',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   function_scaling = 'log',
                                   do_infstep = True,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings = settings,
                                  dimension = bb.dimension, 
                                  var_lower = bb.var_lower,
                                  var_upper = bb.var_upper,
                                  objfun = bb.evaluate,
                                  objfun_fast = bb.evaluate_fast,
                                  integer_vars = bb.integer_vars)
            res = alg.optimize()
            msg = 'Could not solve branin with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_gutmann_st_miqp1_noisy(self):
        """Check solution of noisy st_miqp1 with Gutmann."""
        function = test_functions.st_miqp1
        noisy_fun = ti.NoisyFunction(function.evaluate, 0.1, 0.01)
        bb = create_black_box(function, noisy_fun.evaluate)
        optimum = function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving st_miqp1 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   fast_objfun_rel_error = 0.1,
                                   fast_objfun_abs_error = 0.01,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings = settings,
                                  dimension = bb.dimension, 
                                  var_lower = bb.var_lower,
                                  var_upper = bb.var_upper,
                                  objfun = bb.evaluate,
                                  objfun_fast = bb.evaluate_fast,
                                  integer_vars = bb.integer_vars)
            res = alg.optimize()
            msg = 'Could not solve st_miqp1 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_time_limit(self):
        """Verify that time limits are satisfied."""
        function = test_functions.hartman6
        bb = create_black_box(function)
        optimum = function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving hartman6 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_clock_time = 2.0,
                                   rand_seed = seed)
            start_time = time.time()
            alg = ra.OptAlgorithm(settings = settings,
                                  dimension = bb.dimension, 
                                  var_lower = bb.var_lower,
                                  var_upper = bb.var_upper,
                                  objfun = bb.evaluate,
                                  objfun_fast = bb.evaluate_fast,
                                  integer_vars = bb.integer_vars)
            res = alg.optimize()
            tot_time = time.time() - start_time
            msg = 'Time limit exceeded with Gutmann algorithm'
            self.assertLessEqual(tot_time, 4.0, msg = msg)
    # -- end function
# -- end class

class TestMSRSM(unittest.TestCase):
    """Test MSRSM algorithm on a small set of problems."""

    rand_seeds = [512319876412, 231974198123, 90865241837]
    eps_opt = 0.05

    def test_msrsm_hartman3(self):
        """Check solution of hartman3 with the MSRSM algorithm."""
        function = test_functions.hartman3
        bb = create_black_box(function)
        optimum = function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving hartman3 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM', 
                                   rbf = 'linear',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings = settings,
                                  dimension = bb.dimension, 
                                  var_lower = bb.var_lower,
                                  var_upper = bb.var_upper,
                                  objfun = bb.evaluate,
                                  objfun_fast = bb.evaluate_fast,
                                  integer_vars = bb.integer_vars)
            res = alg.optimize()
            msg = 'Could not solve hartman3 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_msrsm_nvs03(self):
        """Check solution of nvs03 with the MSRSM algorithm."""
        function = test_functions.nvs03
        bb = create_black_box(function)
        optimum = function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving nvs03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM', 
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings = settings,
                                  dimension = bb.dimension, 
                                  var_lower = bb.var_lower,
                                  var_upper = bb.var_upper,
                                  objfun = bb.evaluate,
                                  objfun_fast = bb.evaluate_fast,
                                  integer_vars = bb.integer_vars)
            res = alg.optimize()
            msg = 'Could not solve nvs03 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_msrsm_nvs06_no_local_search(self):
        """Check solution of nvs06 with MSRSM and no local search."""
        function = test_functions.nvs06
        bb = create_black_box(function)
        optimum = function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving nvs06 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(rbf = 'cubic',
                                   algorithm = 'MSRSM', 
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   local_search_box_scaling = 10000,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings = settings,
                                  dimension = bb.dimension, 
                                  var_lower = bb.var_lower,
                                  var_upper = bb.var_upper,
                                  objfun = bb.evaluate,
                                  objfun_fast = bb.evaluate_fast,
                                  integer_vars = bb.integer_vars)
            res = alg.optimize()
            msg = 'Could not solve nvs06 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_msrsm_goldsteinprice_noisy(self):
        """Check solution of noisy goldsteinprice with MSRSM."""
        function = test_functions.goldsteinprice
        noisy_fun = ti.NoisyFunction(function.evaluate, 0.1, 0.01)
        bb = create_black_box(function, noisy_fun.evaluate)
        optimum = function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving goldsteinprice with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM', 
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   fast_objfun_rel_error = 0.1,
                                   fast_objfun_abs_error = 0.01,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings = settings,
                                  dimension = bb.dimension, 
                                  var_lower = bb.var_lower,
                                  var_upper = bb.var_upper,
                                  objfun = bb.evaluate,
                                  objfun_fast = bb.evaluate_fast,
                                  integer_vars = bb.integer_vars)
            res = alg.optimize()
            msg = 'Could not solve goldsteinprice with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_time_limit(self):
        """Verify that time limits are satisfied."""
        function = test_functions.hartman6
        bb = create_black_box(function)
        optimum = function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving hartman6 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM', 
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_clock_time = 2.0,
                                   rand_seed = seed)
            start_time = time.time()
            alg = ra.OptAlgorithm(settings = settings,
                                  dimension = bb.dimension, 
                                  var_lower = bb.var_lower,
                                  var_upper = bb.var_upper,
                                  objfun = bb.evaluate,
                                  objfun_fast = bb.evaluate_fast,
                                  integer_vars = bb.integer_vars)
            res = alg.optimize()
            tot_time = time.time() - start_time
            msg = 'Time limit exceeded with MSRSM algorithm'
            self.assertLessEqual(tot_time, 4.0, msg = msg)
    # -- end function
# -- end class

class TestState(unittest.TestCase):
    """Test load/save state methods."""

    rand_seeds = [512319876412, 231974198123, 90865241837]
    eps_opt = 0.05

    def test_state_reload(self):
        """Check solution of hartman3 after state save/reload."""
        function = test_functions.hartman3
        bb = create_black_box(function)
        optimum = function.optimum_value
        handle, filename = tempfile.mkstemp()
        for seed in self.rand_seeds:
            print()
            print('Solving hartman3 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM', 
                                   rbf = 'linear',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings = settings,
                                  dimension = bb.dimension, 
                                  var_lower = bb.var_lower,
                                  var_upper = bb.var_upper,
                                  objfun = bb.evaluate,
                                  objfun_fast = bb.evaluate_fast,
                                  integer_vars = bb.integer_vars)
            res = alg.optimize(5)
            alg.save_to_file(filename)
            alg_reload = ra.OptAlgorithm.load_from_file(filename,
                                                        bb.evaluate,
                                                        bb.evaluate_fast)
            res = alg_reload.optimize()
            msg = 'Could not solve hartman3 after reload'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
        os.remove(filename)
    # -- end function
# -- end class

if (__name__ == '__main__'):
    unittest.main()
