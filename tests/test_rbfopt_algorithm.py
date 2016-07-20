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

class TestGutmann(unittest.TestCase):
    """Test Gutmann's algorithm on a small set of problems."""

    rand_seeds = [512319876, 231974198, 908652418]
    eps_opt = 0.05        

    def test_gutmann_ex8_1_4(self):
        """Check solution of ex8_1_4 with Gutmann's method, genetic."""
        bb = ti.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving ex8_1_4 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   global_search_method = 'genetic',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve ex8_1_4 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_gutmann_prob03(self):
        """Check solution of prob03 with Gutmann's method, traditional."""
        bb = ti.TestBlackBox('prob03')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   rbf = 'cubic',
                                   global_search_method = 'traditional',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve prob03 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_gutmann_ex8_1_4_log(self):
        """Check solution of ex8_1_4 with Gutmann, log scaling, infstep.

        Traditional global search.

        """
        bb = ti.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving branin with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   rbf = 'multiquadric',
                                   global_search_method = 'traditional',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   function_scaling = 'log',
                                   do_infstep = True,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve branin with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_gutmann_st_miqp3_noisy(self):
        """Check solution of noisy st_miqp3 with Gutmann, genetic."""
        bb = ti.TestNoisyBlackBox('st_miqp3', 0.1, 0.01)
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving st_miqp3 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   global_search_method = 'genetic',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   fast_objfun_rel_error = 0.1,
                                   fast_objfun_abs_error = 0.01,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve st_miqp3 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_time_limit(self):
        """Verify that time limits are satisfied."""
        bb = ti.TestBlackBox('hartman6')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving hartman6 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   global_search_method = 'genetic',
                                   target_objval = optimum,
                                   eps_opt = 0.0,
                                   max_clock_time = 2.0,
                                   rand_seed = seed)
            start_time = time.time()
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            tot_time = time.time() - start_time
            msg = 'Time limit exceeded with Gutmann algorithm'
            self.assertLessEqual(tot_time, 5.0, msg = msg)
    # -- end function
# -- end class

class TestGutmannParallel(unittest.TestCase):
    """Test Gutmann's algorithm in parallel on a small set of problems."""

    rand_seeds = [512319876, 231974198, 908652418]
    eps_opt = 0.05        

    def test_gutmann_parallel_ex8_1_4(self):
        """Check solution of ex8_1_4 with Gutmann's method, traditional."""
        bb = ti.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving goldsteinprice with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   global_search_method = 'traditional',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   num_cpus = 4,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve ex8_1_4 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_gutmann_parallel_prob03(self):
        """Check solution of prob03 with Gutmann's method, genetic."""
        bb = ti.TestBlackBox('prob03')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   global_search_method = 'genetic',
                                   rbf = 'cubic',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   num_cpus = 2,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve prob03 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_gutmann_parallel_ex8_1_4_log(self):
        """Check solution of ex8_1_4 with Gutmann, log scaling, infstep.

        Genetic algorithm."""
        bb = ti.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving branin with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   global_search_method = 'genetic',
                                   rbf = 'multiquadric',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   num_cpus = 2,
                                   function_scaling = 'log',
                                   do_infstep = True,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve branin with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_gutmann_parallel_st_miqp3_noisy(self):
        """Check solution of noisy st_miqp3 with Gutmann, traditional."""
        bb = ti.TestNoisyBlackBox('st_miqp3', 0.1, 0.01)
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving st_miqp3 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   global_search_method = 'traditional',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   num_cpus = 2,
                                   fast_objfun_rel_error = 0.1,
                                   fast_objfun_abs_error = 0.01,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve st_miqp3 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_time_limit(self):
        """Verify that time limits are satisfied."""
        bb = ti.TestBlackBox('hartman6')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving hartman6 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'Gutmann', 
                                   global_search_method = 'traditional',
                                   num_cpus = 2,
                                   target_objval = optimum,
                                   eps_opt = 0.0,
                                   max_clock_time = 2.0,
                                   rand_seed = seed)
            start_time = time.time()
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            tot_time = time.time() - start_time
            msg = 'Time limit exceeded with Gutmann algorithm'
            self.assertLessEqual(tot_time, 5.0, msg = msg)
    # -- end function
# -- end class

class TestMSRSM(unittest.TestCase):
    """Test MSRSM algorithm on a small set of problems."""

    rand_seeds = [512319876, 231974198, 908652418]
    eps_opt = 0.05

    def test_msrsm_ex8_1_4(self):
        """Check solution of ex8_1_4 with the MSRSM algorithm, traditional."""
        bb = ti.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving ex8_1_4 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM', 
                                   global_search_method = 'traditional',
                                   rbf = 'linear',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve hartman3 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_msrsm_prob03(self):
        """Check solution of prob03 with the MSRSM algorithm, genetic."""
        bb = ti.TestBlackBox('prob03')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM', 
                                   global_search_method = 'genetic',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve prob03 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_msrsm_prob03_no_local_search(self):
        """Check solution of prob03 with MSRSM and no local search.

        Traditional method.

        """
        bb = ti.TestBlackBox('prob03')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(rbf = 'cubic',
                                   global_search_method = 'traditional',
                                   algorithm = 'MSRSM', 
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   local_search_box_scaling = 10000,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve prob03 with Gutmann\'s algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_msrsm_st_miqp3_noisy(self):
        """Check solution of noisy st_miqp3 with MSRSM, genetic."""
        bb = ti.TestNoisyBlackBox('st_miqp3', 0.1, 0.01)
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving goldsteinprice with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM',
                                   global_search_method = 'genetic',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   fast_objfun_rel_error = 0.1,
                                   fast_objfun_abs_error = 0.01,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve goldsteinprice with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_time_limit(self):
        """Verify that time limits are satisfied."""
        bb = ti.TestBlackBox('hartman6')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving hartman6 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM', 
                                   target_objval = optimum,
                                   eps_opt = 0.0,
                                   max_clock_time = 2.0,
                                   rand_seed = seed)
            start_time = time.time()
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            tot_time = time.time() - start_time
            msg = 'Time limit exceeded with MSRSM algorithm'
            self.assertLessEqual(tot_time, 5.0, msg = msg)
    # -- end function
# -- end class

class TestMSRSMParallel(unittest.TestCase):
    """Test MSRSM algorithm in parallel on a small set of problems."""

    rand_seeds = [512319876, 231974198, 908652418]
    eps_opt = 0.05

    def test_msrsm_parallel_ex8_1_4(self):
        """Check solution of ex8_1_4 with the MSRSM algorithm, genetic."""
        bb = ti.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving ex8_1_4 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM', 
                                   global_search_method = 'genetic',
                                   rbf = 'linear',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   num_cpus = 4,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve ex8_1_4 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_msrsm_parallel_prob03(self):
        """Check solution of prob03 with the MSRSM algorithm, traditional."""
        bb = ti.TestBlackBox('prob03')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM', 
                                   global_search_method = 'traditional',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   num_cpus = 4,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve prob03 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_msrsm_parallel_st_miqp3_no_local_search(self):
        """Check solution of st_miqp3 with MSRSM and no local search.

        Genetic algorithm."""
        bb = ti.TestBlackBox('st_miqp3')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving nvs06 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(rbf = 'cubic',
                                   global_search_method = 'genetic',
                                   algorithm = 'MSRSM', 
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   num_cpus = 4,
                                   local_search_box_scaling = 10000,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve st_miqp3 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_msrsm_parallel_prob03_noisy(self):
        """Check solution of noisy prob03 with MSRSM, traditional."""
        bb = ti.TestNoisyBlackBox('prob03', 0.1, 0.01)
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving prob03 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM', 
                                   global_search_method = 'traditional',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   num_cpus = 4,
                                   fast_objfun_rel_error = 0.1,
                                   fast_objfun_abs_error = 0.01,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            msg = 'Could not solve prob03 with MSRSM algorithm'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
    # -- end function

    def test_time_limit(self):
        """Verify that time limits are satisfied."""
        bb = ti.TestBlackBox('hartman6')
        optimum = bb._function.optimum_value
        for seed in self.rand_seeds:
            print()
            print('Solving hartman6 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM',
                                   num_cpus = 2,
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_clock_time = 2.0,
                                   rand_seed = seed)
            start_time = time.time()
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize()
            tot_time = time.time() - start_time
            msg = 'Time limit exceeded with MSRSM algorithm in parallel'
            self.assertLessEqual(tot_time, 5.0, msg = msg)
    # -- end function
# -- end class

class TestState(unittest.TestCase):
    """Test load/save state methods."""

    rand_seeds = [512319876, 231974198, 908652418]
    eps_opt = 0.05

    def test_state_reload(self):
        """Check solution of ex8_1_4 after state save/reload."""
        bb = ti.TestBlackBox('ex8_1_4')
        optimum = bb._function.optimum_value
        handle, filename = tempfile.mkstemp()
        for seed in self.rand_seeds:
            print()
            print('Solving ex8_1_4 with random seed ' +
                  '{:d}'.format(seed))
            settings = RbfSettings(algorithm = 'MSRSM', 
                                   rbf = 'linear',
                                   target_objval = optimum,
                                   eps_opt = self.eps_opt,
                                   max_iterations = 200,
                                   max_evaluations = 300,
                                   rand_seed = seed)
            alg = ra.OptAlgorithm(settings, bb)
            res = alg.optimize(5)
            alg.save_to_file(filename)
            alg_reload = ra.OptAlgorithm.load_from_file(filename)
            res = alg_reload.optimize()
            msg = 'Could not solve ex8_1_4 after reload'
            target = optimum + (abs(optimum)*self.eps_opt if
                                abs(optimum) > settings.eps_zero 
                                else self.eps_opt) 
            self.assertLessEqual(res[0], target, msg = msg)
        os.close(handle)
        os.remove(filename)
    # -- end function
# -- end class

if (__name__ == '__main__'):
    unittest.main()
