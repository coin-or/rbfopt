"""Test the module rbfopt_utils in RBFOpt.

This module contains unit tests for the module rbfopt_utils.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright International Business Machines Corporation 2016.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import math
import random
import numpy as np
import test_rbfopt_env
import rbfopt_utils as ru
import rbfopt_config as config
from rbfopt_settings import RbfSettings

class TestUtils(unittest.TestCase):
    """Test the rbfopt_utils module."""

    def setUp(self):
        """Initialize data used by several functions."""
        self.rbf_types = [rbf_type for rbf_type in RbfSettings._allowed_rbf
                          if rbf_type != 'auto']
    # -- end function

    def test_get_rbf_function(self):
        """Check that all RBFs are properly computed at 0 and at 1."""
        settings = RbfSettings()
        # Set up values of the RBF at 0 and at 1
        rbf_values = dict()
        rbf_values['linear'] = (0.0, 1.0)
        rbf_values['multiquadric'] = (config.GAMMA, 
                                      math.sqrt(1 + config.GAMMA**2))
        rbf_values['cubic'] = (0.0, 1.0)
        rbf_values['thin_plate_spline'] = (0.0, 0.0)
        for rbf_type in self.rbf_types:
            settings.rbf = rbf_type
            rbf = ru.get_rbf_function(settings)
            rbf_at_0, rbf_at_1 = rbf_values[rbf_type]
            msg = 'RBF {:s} is not {:f} at 0'.format(rbf_type, rbf_at_0)
            self.assertEqual(rbf_at_0, rbf(0.0), msg = msg)
            msg = 'RBF {:s} is not {:f} at 1'.format(rbf_type, rbf_at_1)
            self.assertEqual(rbf_at_1, rbf(1.0), msg = msg)
    # -- end function

    def test_get_degree_polynomial(self):
        """Verify that the degree is always between 0 and 1."""
        settings = RbfSettings()
        for rbf_type in self.rbf_types:
            settings.rbf = rbf_type
            degree = ru.get_degree_polynomial(settings)
            self.assertTrue(degree == 0 or degree == 1)
    # -- end function

    def test_get_size_P_matrix(self):
        """Verify that the size is always between 0 and n+1."""
        settings = RbfSettings()
        for rbf_type in self.rbf_types:
            settings.rbf = rbf_type
            for n in range(20):
                size = ru.get_size_P_matrix(settings, n)
                self.assertTrue(0 <= size <= n + 1)
    # -- end function            

    def test_get_all_corners(self):
        """Check that all corners of a box are properly returned."""
        var_lower = [-1, 0, 1]
        var_upper = [1, 2, 3]
        corners = ru.get_all_corners(var_lower, var_upper)
        self.assertItemsEqual([[-1, 0, 1], [-1, 0, 3], [-1, 2, 1], [-1, 2, 3],
                               [1, 0, 1], [1, 0, 3], [1, 2, 1], [1, 2, 3]],
                              corners)
    # -- end function

    def test_get_lower_corners(self):
        """Check that the lower corners of a box are properly returned."""
        var_lower = [-1, 0, 1]
        var_upper = [1, 2, 3]
        corners = ru.get_lower_corners(var_lower, var_upper)
        self.assertItemsEqual([[-1, 0, 1], [-1, 0, 3], [-1, 2, 1], 
                               [1, 0, 1]], corners)
    # -- end function

    def test_get_random_corners(self):
        """Check that random corners of a box are properly returned."""
        var_lower = [-1, 0, 1]
        var_upper = [1, 2, 3]
        all_corners = [[-1, 0, 1], [-1, 0, 3], [-1, 2, 1], [-1, 2, 3],
                       [1, 0, 1], [1, 0, 3], [1, 2, 1], [1, 2, 3]]
        for i in range(10):
            corners = ru.get_random_corners(var_lower, var_upper)
            for corner in corners:
                self.assertIn(corner, all_corners)
            self.assertEqual(len(corners), 4)
    # -- end function

    def test_get_lhd_points(self):
        """Check that latin hypercube designs have the correct size."""
        var_lower = [-1, 0, 1]
        var_upper = [1, 2, 3]
        corners = ru.get_lhd_maximin_points(var_lower, var_upper)
        self.assertEqual(len(corners), 4)
        corners = ru.get_lhd_corr_points(var_lower, var_upper)
        self.assertEqual(len(corners), 4)
    # -- end function

    def test_initialize_nodes(self):
        """Test initialization methods for the sample points.

        This method verifies that returned sets of points have at
        least n+1 points, and integer variables are integer.
        """
        var_lower = [-1, 0, 1]
        var_upper = [1, 2, 3]
        integer_vars = [1, 2]
        for method in RbfSettings._allowed_init_strategy:
            settings = RbfSettings(init_strategy = method)
            points = ru.initialize_nodes(settings, var_lower, var_upper,
                                         integer_vars)
            msg = ('Number of points returned by {:s}'.format(method) +
                   ' is insufficient')
            self.assertGreaterEqual(len(points), 4, msg = msg)
            for point in points:
                for index in integer_vars:
                    self.assertEqual(point[index] - round(point[index]), 0)
    # -- end function

    def test_round_integer_vars(self):
        """Verify that some fractional points are properly rounded."""
        point = [0.1, 2.3, -3.5, 4.6]
        ru.round_integer_vars(point, [0, 2])
        self.assertListEqual(point, [0.0, 2.3, -4.0, 4.6],
                             msg = 'Failed when integer_vars is subset')
        point = [0.1, 2.3, -3.5, 4.6]
        ru.round_integer_vars(point, [])
        self.assertListEqual(point, [0.1, 2.3, -3.5, 4.6],
                             msg = 'Failed when integer_vars is empty')
        point = [0.1, 2.3, -3.5, 4.6]
        ru.round_integer_vars(point, [0, 1, 2, 3])
        self.assertListEqual(point, [0.0, 2.0, -4.0, 5.0],
                             msg = 'Failed when integer_vars is everything')
    # -- end function

    def test_round_integer_bounds(self):
        """Verify that some fractional bounds are properly rounded."""
        var_lower = [-0.1, 2.3, -3.5, 4.6]        
        var_upper = [2.5, 3.0, -1.2, 4.6]
        ru.round_integer_bounds(var_lower, var_upper, [0, 2])
        self.assertListEqual(var_lower, [-1.0, 2.3, -4.0, 4.6],
                             msg = 'Failed when integer_vars is subset')
        self.assertListEqual(var_upper, [3.0, 3.0, -1.0, 4.6],
                             msg = 'Failed when integer_vars is subset')
        var_lower = [-0.1, 2.3, -3.5, 4.6]        
        var_upper = [2.5, 3.0, -1.2, 4.6]
        ru.round_integer_bounds(var_lower, var_upper, [])
        self.assertListEqual(var_lower, [-0.1, 2.3, -3.5, 4.6],
                             msg = 'Failed when integer_vars is empty')
        self.assertListEqual(var_upper, [2.5, 3.0, -1.2, 4.6],
                             msg = 'Failed when integer_vars is empty')
        var_lower = [-0.1, 2.3, -3.5, 4.6]        
        var_upper = [2.5, 3.0, -1.2, 4.6]
        ru.round_integer_bounds(var_lower, var_upper, [0, 1, 2, 3])
        self.assertListEqual(var_lower, [-1.0, 2.0, -4.0, 4.0],
                             msg = 'Failed when integer_vars is everything')
        self.assertListEqual(var_upper, [3.0, 3.0, -1.0, 5.0],
                             msg = 'Failed when integer_vars is everything')
    # -- end function

    def test_norm(self):
        """Verify that norm is 0 at 0 and correct for some other vectors."""
        self.assertEqual(ru.norm([0 for i in range(10)]), 0.0,
                         msg = 'Norm is not zero at zero')
        self.assertEqual(ru.norm([-1 for i in range(9)]), 3.0,
                         msg = 'Norm is not 3.0 at {-1}^9')
        self.assertEqual(ru.norm([-2 + i for i in range(5)]), math.sqrt(10),
                         msg = 'Norm is not sqrt{10} at [-2, -1, 0, 1, 2]')
    # -- end function

    def test_distance(self):
        """Verify that distance is 0 iff two points are the same."""
        self.assertEqual(ru.distance([i*5 for i in range(15)],
                                     [i*5 for i in range(15)]), 0.0,
                         msg = 'Distance is not zero at equal points')
        self.assertNotEqual(ru.distance([i*5 for i in range(15)],
                                        [i*5 + 0.001 for i in range(15)]), 
                            0.0, msg = 'Distance is nonzero at diff points')
        self.assertNotEqual(ru.distance([-i*5 for i in range(15)],
                                        [-i*5 + 0.001 for i in range(15)]), 
                            0.0, msg = 'Distance is nonzero at diff points')
    # -- end function

    def test_get_min_distance(self):
        """Test some extreme cases for get_min_distance."""
        self.assertEqual(ru.get_min_distance([i for i in range(5)],
                                             [[i+j for i in range(5)]
                                              for j in range(10)]), 0.0)
    # -- end function
    def test_get_min_distance_index(self):
        """Test some extreme cases for get_min_distance_index."""
        self.assertEqual(ru.get_min_distance_index([i for i in range(5)],
                                                   [[i+j for i in range(5)]
                                                    for j in range(-2, 3)]), 
                         2)
        self.assertEqual(ru.get_min_distance_index([i+0.01 for i in range(5)],
                                                   [[i+j for i in range(5)]
                                                    for j in range(-3, 2)]), 
                         3)
    # -- end function

    def test_bulk_get_min_distance(self):
        """Verify that bulk version returns the same as regular version.
    
        This function checks that the bulk version of get_min_distance
        on a number of randomly generated points returns the same
        result as the regular version.
        """
        for i in range(50):
            dim = random.randint(1, 20)
            num_points_1 = random.randint(10, 50)
            num_points_2 = random.randint(10, 50)
            points = [[random.uniform(-100, 100) for j in range(dim)]
                      for k in range(num_points_1)]
            other_points = [[random.uniform(-100, 100) for j in range(dim)]
                            for k in range(num_points_2)]
            dist1 = [ru.get_min_distance(point, other_points)
                     for point in points]
            dist2 = ru.bulk_get_min_distance(points, other_points)
            for j in range(num_points_1):
                msg = 'Failed random test {:d} point {:d}'.format(i, j)
                self.assertAlmostEqual(dist1[j], dist2[j], 12, msg = msg)
    # -- end function

    def test_get_rbf_matrix(self):
        """Test basic properties of the RBF matrix (e.g. symmetry, size).

        Verify that the RBF matrix is symmetric and it has the correct
        size for all types of RBF.
        """
        settings = RbfSettings()
        for i in range(50):
            dim = random.randint(1, 20)
            num_points = random.randint(10, 50)
            node_pos = [[random.uniform(-100, 100) for j in range(dim)]
                      for k in range(num_points)]
            # Possible shapes of the matrix
            for rbf_type in self.rbf_types:
                settings.rbf = rbf_type
                mat = ru.get_rbf_matrix(settings, dim, num_points, node_pos)
                self.assertIsInstance(mat, np.matrix)
                self.assertAlmostEqual(np.max(mat - mat.transpose()), 0.0,
                                       msg = 'RBF matrix is not symmetric')
                size = num_points + 1
                if (ru.get_degree_polynomial(settings) > 0):
                    size += dim ** ru.get_degree_polynomial(settings)
                self.assertEqual(mat.shape, (size, size))
        # Check that exception is raised for unknown RBF types
        settings.rbf = 'unknown'
        self.assertRaises(ValueError, ru.get_rbf_matrix, settings, 
                          dim, num_points, node_pos)
    # -- end function

    def test_transform_function_values(self):
        """Test all codomain transformation strategies.

        This will verify that the transformation strategies always
        produce valid results and can handle extreme cases.
        """
        settings = RbfSettings()
        settings.fast_objfun_rel_error = 0.01
        settings.fast_objfun_abs_error = 0.01
        list_scaling = [val for val in RbfSettings._allowed_function_scaling
                        if val != 'auto']
        list_clipping = [val for val in RbfSettings._allowed_dynamism_clipping
                         if val != 'auto']
        transf = ru.transform_function_values
        # Create list of values to test: node_val and corresponding
        # fast_node_index
        to_test = [([0, -100, settings.dynamism_threshold * 10], []),
                   ([0.0], [0]), 
                   ([0.0 for i in range(10)], [8, 9]), 
                   ([100.0 for i in range(10)], [i for i in range(10)]),
                   ([10**i for i in range(-20, 20)], []),
                   ([-10**i for i in range(-20, 20)] + 
                    [10**i for i in range(-20, 20)], 
                    [i for i in range (50, 60)])]
        for scaling in list_scaling:
            for clipping in list_clipping:
                header = '({:s}, {:s}):'.format(scaling, clipping)
                for (node_val, fast_node_index) in to_test:
                    settings.function_scaling = scaling
                    settings.dynamism_clipping = clipping
                    (scaled, minval, maxval,
                     errbounds) = transf(settings, node_val, min(node_val),
                                         max(node_val), fast_node_index)
                    # Check that the number of scaled values is the
                    # same as the number of input values
                    msg = 'Number of output values is different from input'
                    self.assertEqual(len(scaled), len(node_val),
                                     msg = header + msg)
                    msg = 'Dynamism threshold was not enforced'
                    v1 = abs(min(scaled))
                    v2 = abs(max(scaled))
                    c1 = v1 > 1.0e-10 and v2/v1 <= settings.dynamism_threshold
                    c2 = v1 <= 1.0e-10 and v2 <= settings.dynamism_threshold
                    self.assertTrue(clipping == 'off' or c1 or c2,
                                    msg = header + msg)
                    for (i, j) in enumerate(fast_node_index):
                        msg = 'Fast_node_index have wrong sign'
                        self.assertLessEqual(errbounds[i][0], 0, msg = msg)
                        self.assertGreaterEqual(errbounds[i][1], 0, msg = msg)
                    msg = ('Min/Max of scaled values inconsistent with ' +
                           'returned scaled_min and scaled_max')
                    self.assertEqual(min(scaled), minval, msg = header + msg)
                    self.assertEqual(max(scaled), maxval, msg = header + msg)
        # -- end for
    # -- end function

    def test_transform_domain(self):
        """Check that affine transformation does not hang on limit case.

        Further check that 'off' transformation returns the point as
        is, and unimplemented strategies raise a ValueError.
        """
        settings = RbfSettings()
        settings.domain_scaling = 'affine'
        var_lower = [i for i in range(5)] + [i for i in range(5)]
        var_upper = [i for i in range(5)] + [i + 10 for i in range(5)]
        point = [i for i in range(5)] + [i + 2*i for i in range(5)]
        # Test what happend when lower and upper bounds coincide
        transf_point = ru.transform_domain(settings, var_lower, 
                                           var_upper, point)
        orig_point = ru.transform_domain(settings, var_lower, var_upper,
                                         transf_point, True)
        for i in range(10):
            msg = 'Exceeding lower bound on affine domain scaling'
            self.assertLessEqual(0.0, transf_point[i], msg = msg)
            msg = 'Exceeding upper bound on affine domain scaling'
            self.assertLessEqual(transf_point[i], 1.0, msg = msg)
            msg = 'Doubly transformed point does not match original'
            self.assertAlmostEqual(point[i], orig_point[i], 12, msg = msg)
        # Check that 'off' scaling does not do anything
        settings.domain_scaling = 'off'
        transf_point = ru.transform_domain(settings, var_lower, 
                                           var_upper, point)
        for i in range(10):
            msg = 'Transformed point with \'off\' does not match original'
            self.assertEqual(point[i], transf_point[i], msg = msg)
        # Check that unimplemented strategies are rejected
        settings.domain_scaling = 'test'
        self.assertRaises(ValueError, ru.transform_domain, settings, 
                          var_lower, var_upper, point)
    # -- end function

    def test_transform_domain_bounds(self):
        """Check that domain bounds are consistent."""
        list_scaling = [val for val in RbfSettings._allowed_domain_scaling 
                        if val != 'auto']
        for scaling in list_scaling:
            settings = RbfSettings(domain_scaling = scaling)
            # Test limit case with empty bounds
            vl, vu = ru.transform_domain_bounds(settings, [], [])
            msg = 'Failed transform_domain_bounds on empty bounds'
            self.assertEqual(len(vl), 0, msg = msg)
            self.assertEqual(len(vu), 0, msg = msg)
            msg = 'Bounds inconsistent with random bounds'
            for i in range(10):
                dim = random.randint(0, 20)
                var_lower = [random.uniform(-100, 100) for j in range(dim)]
                var_upper = [var_lower[j] + random.uniform(0, 100)
                             for j in range(dim)]
                vl, vu = ru.transform_domain_bounds(settings, var_lower,
                                                    var_upper)
                self.assertEqual(len(vl), len(var_lower), msg = msg)
                self.assertEqual(len(vu), len(var_upper), msg = msg)
                for j in range(dim):
                    self.assertLessEqual(vl[j], vu[j], msg = msg)
    # -- end function

    def test_get_sigma_n(self):
        """Check that sigma_n is always within the bounds [0, k-1]."""
        for k in range(0, 1000, 50):
            for num_global_searches in range(0, 10, 2):
                for current_step in range(1, num_global_searches):
                    for num_initial_points in range(0, k):
                        i = ru.get_sigma_n(k, current_step, 
                                           num_global_searches,
                                           num_initial_points)
                        self.assertTrue(0 <= i < k, 
                                        msg = 'sigma_n out of bounds')
    # -- end function

    def test_get_fmax_current_iter(self):
        """Verify get_fmax_current_iter is resilient to limit cases.

        This function tests whether correct values are returned when
        there is a single-element list of node values, and when the
        list of node values is exactly the minimum required k + 1.
        """
        settings = RbfSettings()
        fun = ru.get_fmax_current_iter
        self.assertEqual(fun(settings, 0, 1, 1, [1]), 1, 
                         msg = 'Failed on single-element list')
        self.assertEqual(fun(settings, 10, 11, 5, [i for i in range(11)]),
                         10, msg = 'Failed on n == k + 1')
    # -- end function

    def test_get_min_bump_node(self):
        """Verify get_min_bump_node is resilient to limit cases.

        Verify that when fast_node_index is empty, (None, +inf) is
        returned, and for a small problem with 3 variables and 5
        nodes, the corect answer is reached when there is only one
        possible point that could be replaced.

        """
        settings = RbfSettings(rbf = 'cubic')
        ind, bump = ru.get_min_bump_node(settings, 1, 10, np.matrix((1,1)), 
                                         [0] * 10, [], [], 0)
        self.assertIsNone(ind, msg = 'Failed whith empty list')
        self.assertEqual(bump, float('+inf'), msg = 'Failed whith empty list')

        n = 3
        k = 5
        var_lower = [i for i in range(n)]
        var_upper = [i + 10 for i in range(n)]
        node_pos = [var_lower, var_upper,
                    [1, 2, 3], [9, 5, 8.8], [5.5, 7, 12]]
        node_val = [2*i for i in range(k)]
        fast_node_index = [i for i in range(k)]
        fast_node_err_bounds = [(-1, +1) for i in range(k)]
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
        Amat = np.matrix(Amat)
        for j in range(k):
            ind, bump = ru.get_min_bump_node(settings, n, k, Amat, node_val,
                                             fast_node_index,
                                             fast_node_err_bounds,
                                             node_val[j] - 0.5)
            self.assertEqual(ind, j, msg = 'Only one point is a candidate' +
                             'for replacement, but it was not returned!')
    # -- end function

    def test_get_bump_new_node(self):
        """Verify bumpiness is constant under the right conditions.

        This function tests the bumpiness of the interpolant model,
        when adding an interpolation point at a location with function
        value increasing very fast. This should give increasing
        bumpiness, and that's what we check.

        """
        settings = RbfSettings(rbf = 'cubic')
        n = 3
        k = 5
        var_lower = [i for i in range(n)]
        var_upper = [i + 10 for i in range(n)]
        node_pos = [var_lower, var_upper,
                    [1, 2, 3], [9, 5, 8.8], [5.5, 7, 12]]
        node_val = [2*i for i in range(k)]
        fast_node_index = [i for i in range(k)]
        fast_node_err_bounds = [(-1, +1) for i in range(k)]
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
        Amat = np.matrix(Amat)
        fast_node_index = [0, 1]
        fast_node_err_bounds = [(-1, 1), (-1, 1)]
        # Previous bumpiness
        new_node = [(var_lower[i] + var_upper[i])/2 for i in range(n)]
        bump = 0.0
        for i in range(5):
            # Set increasing target values
            target_val = 5 + i*10000
            # Compute new bumpiness
            nbump = ru.get_bump_new_node(settings, n, k, node_pos, node_val, 
                                         new_node, fast_node_index,
                                         fast_node_err_bounds, target_val)
            self.assertGreaterEqual(nbump, bump,
                                    msg = 'Bumpiness not increasing')
            # Store new bumpiness
            bump = nbump
    # -- end function

# -- end class

if (__name__ == '__main__'):
    # Set random seed for testing environment
    random.seed(test_rbfopt_env.rand_seed)
    np.random.seed(test_rbfopt_env.rand_seed)
    unittest.main()
