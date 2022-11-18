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
import rbfopt
import numpy as np
import rbfopt.rbfopt_utils as ru
from rbfopt.rbfopt_settings import RbfoptSettings

class TestUtils(unittest.TestCase):
    """Test the rbfopt_utils module."""

    def setUp(self):
        """Initialize data used by several functions."""
        np.random.seed(71294123)
        self.rbf_types = [rbf_type for rbf_type
                          in RbfoptSettings._allowed_rbf
                          if rbf_type != 'auto']
    # -- end function

    def test_get_rbf_function(self):
        """Check that all RBFs are properly computed at 0 and at 1."""
        settings = RbfoptSettings()
        # Set up values of the RBF at 0 and at 1
        rbf_values = dict()
        rbf_values['linear'] = (0.0, 1.0)
        rbf_values['multiquadric'] = (
            np.sqrt(settings.rbf_shape_parameter**2),
            np.sqrt(1+settings.rbf_shape_parameter**2))
        rbf_values['cubic'] = (0.0, 1.0)
        rbf_values['thin_plate_spline'] = (0.0, 0.0)
        rbf_values['gaussian'] = (1.0, np.exp(-settings.rbf_shape_parameter))
        for rbf_type in self.rbf_types:
            settings.rbf = rbf_type
            rbf = ru.get_rbf_function(settings)
            rbf_at_0, rbf_at_1 = rbf_values[rbf_type]
            msg='RBF {:s} is not {:f} at 0'.format(rbf_type, rbf_at_0)
            self.assertEqual(rbf_at_0, rbf(0.0), msg=msg)
            msg='RBF {:s} is not {:f} at 1'.format(rbf_type, rbf_at_1)
            self.assertEqual(rbf_at_1, rbf(1.0), msg=msg)
    # -- end function

    def test_get_degree_polynomial(self):
        """Verify that the degree is always between 0 and 1."""
        settings = RbfoptSettings()
        for rbf_type in self.rbf_types:
            settings.rbf = rbf_type
            degree = ru.get_degree_polynomial(settings)
            self.assertTrue(-1 <= degree <= 1)
    # -- end function

    def test_get_size_P_matrix(self):
        """Verify that the size is always between 0 and n+1."""
        settings = RbfoptSettings()
        for rbf_type in self.rbf_types:
            settings.rbf = rbf_type
            for n in range(20):
                size = ru.get_size_P_matrix(settings, n)
                self.assertTrue(0 <= size <= n + 1)
    # -- end function            

    def test_get_all_corners(self):
        """Check that all corners of a box are properly returned."""
        var_lower = np.array([-1, 0, 1])
        var_upper = np.array([1, 2, 3])
        corners = ru.get_all_corners(var_lower, var_upper)
        self.assertTrue(sorted([[-1, 0, 1], [-1, 0, 3], [-1, 2, 1], 
                                [-1, 2, 3], [1, 0, 1], [1, 0, 3], 
                                [1, 2, 1], [1, 2, 3]]) ==
                        sorted(corners.tolist()))
    # -- end function

    def test_get_lower_corners(self):
        """Check that the lower corners of a box are properly returned."""
        var_lower = np.array([-1, 0, 1])
        var_upper = np.array([1, 2, 3])
        corners = ru.get_lower_corners(var_lower, var_upper)
        self.assertTrue(sorted([[-1, 0, 1], [-1, 0, 3], 
                                [-1, 2, 1], [1, 0, 1]]) == 
                        sorted(corners.tolist()))
    # -- end function

    def test_get_random_corners(self):
        """Check that random corners of a box are properly returned."""
        var_lower = np.array([-1, 0, 1])
        var_upper = np.array([1, 2, 3])
        all_corners = [[-1, 0, 1], [-1, 0, 3], [-1, 2, 1], [-1, 2, 3],
                       [1, 0, 1], [1, 0, 3], [1, 2, 1], [1, 2, 3]]
        for i in range(10):
            corners = ru.get_random_corners(var_lower, var_upper)
            for corner in corners:
                self.assertIn(corner.tolist(), all_corners)
            self.assertEqual(len(corners), 4)
    # -- end function

    def test_get_lhd_points(self):
        """Check that latin hypercube designs have the correct size."""
        var_lower = np.array([-1, 0, 1], dtype=np.float_)
        var_upper = np.array([1, 2, 3], dtype=np.float_)
        corners = ru.get_lhd_maximin_points(
            var_lower, var_upper, np.array([0]), 4)
        self.assertEqual(len(corners), 4)
        corners = ru.get_lhd_corr_points(
            var_lower, var_upper, np.array([]), 5)
        self.assertEqual(len(corners), 5)
    # -- end function

    def test_get_num_init_samples(self):
        """Test edge cases when getting initial number of samples."""
        n=10
        settings = RbfoptSettings(init_strategy='lhd_maximin',
                                  init_sample_fraction=1.0,)
        self.assertGreater(ru.get_num_init_samples(settings, 0), 0)
        set1 = RbfoptSettings(init_strategy='lhd_maximin',
                              init_sample_increase_parallel=0.0,
                              init_sample_fraction=1.0,
                              num_cpus=4)
        self.assertEqual(ru.get_num_init_samples(settings, n),
                         ru.get_num_init_samples(set1, n))
        set2 = RbfoptSettings(init_strategy='lhd_maximin',
                              init_sample_increase_parallel=1.0,
                              init_sample_fraction=1.0,
                              num_cpus=4)
        self.assertLess(ru.get_num_init_samples(set1, n),
                        ru.get_num_init_samples(set2, n))
    # -- end function

    def test_get_min_num_init_samples_parallel(self):
        """Test edge cases when getting initial number of samples."""
        settings = RbfoptSettings(init_strategy='lhd_maximin',
                                  init_sample_fraction=1.0,
                                  init_sample_increase_parallel=1.0,
                                  num_cpus=4)
        self.assertLess(ru.get_min_num_init_samples_parallel(settings, 10),
                        ru.get_num_init_samples(settings, 10))
    # -- end function

    def test_initialize_nodes(self):
        """Test initialization methods for the sample points.

        This method verifies that returned sets of points have at
        least n+1 points, and integer variables are integer.
        """
        var_lower = np.array([-1, 0, 1], dtype=np.float_)
        var_upper = np.array([1, 2, 3], dtype=np.float_)
        integer_vars = np.array([1, 2])
        for method in RbfoptSettings._allowed_init_strategy:
            settings = RbfoptSettings(init_strategy=method,
                                      init_sample_fraction=1.0)
            points = ru.initialize_nodes(settings, var_lower, var_upper,
                                         integer_vars, None)
            msg=('Number of points returned by {:s}'.format(method) +
                   ' is insufficient')
            self.assertGreaterEqual(len(points), 4, msg=msg)
            for point in points:
                for index in integer_vars:
                    self.assertEqual(point[index] - round(point[index]), 0)
    # -- end function

    def test_initialize_nodes_midpoint(self):
        """Test initialization methods for the sample points.

        This method verifies that returned sets of points have at
        least n+1 points, and integer variables are integer.
        """
        var_lower = np.array([-1, 0, 1], dtype=np.float_)
        var_upper = np.array([1, 2, 3], dtype=np.float_)
        integer_vars = np.array([1, 2])
        midpoint = np.array([0, 1, 2])
        for method in RbfoptSettings._allowed_init_strategy:
            settings = RbfoptSettings(init_strategy=method,
                                      init_include_midpoint=True,
                                      init_sample_fraction=1.0)
            points = ru.initialize_nodes(settings, var_lower, var_upper,
                                         integer_vars, None)
            msg=('Number of points returned by {:s}'.format(method) +
                   ' is insufficient')
            self.assertGreaterEqual(len(points), 4, msg=msg)
            dist = np.linalg.norm(points - midpoint, axis=1)
            self.assertEqual(np.min(dist), 0.0,
                             msg='Did not find midpoint')
    # -- end function

    def test_round_integer_vars(self):
        """Verify that some fractional points are properly rounded."""
        point = np.array([0.1, 2.3, -3.5, 4.6])
        ru.round_integer_vars(point, np.array([0, 2]))
        self.assertListEqual(point.tolist(), [0.0, 2.3, -4.0, 4.6],
                             msg='Failed when integer_vars is subset')
        point = np.array([0.1, 2.3, -3.5, 4.6])
        ru.round_integer_vars(point, np.array([]))
        self.assertListEqual(point.tolist(), [0.1, 2.3, -3.5, 4.6],
                             msg='Failed when integer_vars is empty')
        point = np.array([0.1, 2.3, -3.5, 4.6])
        ru.round_integer_vars(point, np.array([0, 1, 2, 3]))
        self.assertListEqual(point.tolist(), [0.0, 2.0, -4.0, 5.0],
                             msg='Failed when integer_vars is everything')
    # -- end function

    def test_round_integer_bounds(self):
        """Verify that some fractional bounds are properly rounded."""
        var_lower = np.array([-0.1, 2.3, -3.5, 4.6])
        var_upper = np.array([2.5, 3.0, -1.2, 4.6])
        ru.round_integer_bounds(var_lower, var_upper, np.array([0, 2]))
        self.assertListEqual(var_lower.tolist(), [-1.0, 2.3, -4.0, 4.6],
                             msg='Failed when integer_vars is subset')
        self.assertListEqual(var_upper.tolist(), [3.0, 3.0, -1.0, 4.6],
                             msg='Failed when integer_vars is subset')
        var_lower = np.array([-0.1, 2.3, -3.5, 4.6])
        var_upper = np.array([2.5, 3.0, -1.2, 4.6])
        ru.round_integer_bounds(var_lower, var_upper, np.array([]))
        self.assertListEqual(var_lower.tolist(), [-0.1, 2.3, -3.5, 4.6],
                             msg='Failed when integer_vars is empty')
        self.assertListEqual(var_upper.tolist(), [2.5, 3.0, -1.2, 4.6],
                             msg='Failed when integer_vars is empty')
        var_lower = np.array([-0.1, 2.3, -3.5, 4.6])
        var_upper = np.array([2.5, 3.0, -1.2, 4.6])
        ru.round_integer_bounds(var_lower, var_upper, np.array([0, 1, 2, 3]))
        self.assertListEqual(var_lower.tolist(), [-1.0, 2.0, -4.0, 4.0],
                             msg='Failed when integer_vars is everything')
        self.assertListEqual(var_upper.tolist(), [3.0, 3.0, -1.0, 5.0],
                             msg='Failed when integer_vars is everything')
    # -- end function

    def test_expand_categorical_variables(self):
        """Verify that expansion is correct and does not fail at limit."""
        points = np.array([[1, 0, 2], [1, 3, 1], [3, 2, 0]])
        categorical = np.array([1])
        not_categorical = np.array([0, 2])
        categorical_expansion = [(1, 0, np.array([2, 3, 4, 5]))]
        expanded = ru.expand_categorical_vars(
            points, categorical, not_categorical, categorical_expansion)
        msg = 'Expansion with 1 categorical variables'
        self.assertTrue(
            np.array_equal(expanded, np.array([[ 1,  2,  1,  0,  0,  0],
                                               [ 1,  1,  0,  0,  0,  1],
                                               [ 3,  0,  0,  0,  1,  0]])),
            msg=msg)
        
        categorical = np.array([0, 2])
        not_categorical = np.array([1])
        categorical_expansion = [(0, 0, np.array([1, 2, 3, 4])),
                                 (2, 0, np.array([5, 6, 7]))]
        expanded = ru.expand_categorical_vars(
            points, categorical, not_categorical, categorical_expansion)
        msg = 'Expansion with 2 categorical variables'
        self.assertTrue(
            np.array_equal(expanded, np.array([[0, 0, 1, 0, 0, 0, 0, 1],
                                               [3, 0, 1, 0, 0, 0, 1, 0],
                                               [2, 0, 0, 0, 1, 1, 0, 0]])),
            msg=msg)
        
        categorical = np.array([0, 2])
        not_categorical = np.array([1])
        categorical_expansion = [(0, 0, np.array([1, 2, 3, 4])),
                                 (2, 0, np.array([5, 6, 7]))]
        expanded = ru.expand_categorical_vars(
            np.array([[1, 0, 2]]), categorical, not_categorical,
            categorical_expansion)
        msg = 'Expansion with 2 categorical variables'
        self.assertTrue(
            np.array_equal(expanded, np.array([[0, 0, 1, 0, 0, 0, 0, 1]])),
            msg=msg)
    # -- end function

    def test_compress_categorical_variables(self):
        """Verify that compression is correct and does not fail at limit."""
        points = np.array([[ 1,  2,  1,  0,  0,  0],
                           [ 1,  1,  0,  0,  0,  1],
                           [ 3,  0,  0,  0,  1,  0]])
        categorical = np.array([1])
        not_categorical = np.array([0, 2])
        categorical_expansion = [(1, 0, np.array([2, 3, 4, 5]))]
        compressed = ru.compress_categorical_vars(
            points, categorical, not_categorical, categorical_expansion)
        msg = 'Compression of 1 variable'
        self.assertTrue(
            np.array_equal(compressed,
                           np.array([[1, 0, 2], [1, 3, 1], [3, 2, 0]])),
            msg=msg)
        points = np.array([[0, 0, 1, 0, 0, 0, 0, 1],
                           [3, 0, 1, 0, 0, 0, 1, 0],
                           [2, 0, 0, 0, 1, 1, 0, 0]])
        categorical = np.array([0, 2])
        not_categorical = np.array([1])
        categorical_expansion = [(0, 0, np.array([1, 2, 3, 4])),
                                 (2, 0, np.array([5, 6, 7]))]
        compressed = ru.compress_categorical_vars(
            points, categorical, not_categorical, categorical_expansion)
        msg = 'Compression of 2 variables'
        self.assertTrue(
            np.array_equal(compressed,
                           np.array([[1, 0, 2], [1, 3, 1], [3, 2, 0]])),
            msg=msg)
        points = np.array([[0, 0, 1, 0, 0, 0, 0, 1]])
        categorical = np.array([0, 2])
        not_categorical = np.array([1])
        categorical_expansion = [(0, 0, np.array([1, 2, 3, 4])),
                                 (2, 0, np.array([5, 6, 7]))]
        msg = 'Compression of 2 variables'
        compressed = ru.compress_categorical_vars(
            points, categorical, not_categorical, categorical_expansion)
        self.assertTrue(
            np.array_equal(compressed, np.array([[1, 0, 2]])),
            msg=msg)
    # -- end function

    def test_compress_categorical_bounds(self):
        """Verify that compression is correct and does not fail at limit."""
        var_lower = np.array([10, -10, 0, 0, 0, 0])
        var_upper = np.array([20, 10, 1, 1, 1, 1])
        categorical = np.array([1])
        not_categorical = np.array([0, 2])
        categorical_expansion = [(1, 2, np.array([2, 3, 4, 5]))]
        compressed = ru.compress_categorical_bounds(
            var_lower, categorical, not_categorical, categorical_expansion)
        msg = 'Compression of bounds for 1 shifted variable'
        self.assertTrue(
            np.array_equal(compressed, np.array([10, 2, -10])), msg=msg)
        compressed = ru.compress_categorical_bounds(
            var_upper, categorical, not_categorical, categorical_expansion)
        self.assertTrue(
            np.array_equal(compressed, np.array([20, 5, 10])), msg=msg)

        var_lower = np.array([10, -10, 0, 0, 0, 0])
        var_upper = np.array([20, 10, 1, 1, 1, 1])
        categorical = np.array([0, 1])
        not_categorical = np.array([2, 3])
        categorical_expansion = [(0, -2, np.array([2])),
                                 (1, -1, np.array([3, 4, 5]))]
        compressed = ru.compress_categorical_bounds(
            var_lower, categorical, not_categorical, categorical_expansion)
        msg = 'Compression of bounds for 2 vars, one fixed'
        self.assertTrue(
            np.array_equal(compressed, np.array([-2, -1, 10, -10])), msg=msg)
        compressed = ru.compress_categorical_bounds(
            var_upper, categorical, not_categorical, categorical_expansion)
        self.assertTrue(
            np.array_equal(compressed, np.array([-2, 1, 20, 10])), msg=msg)

        var_lower = np.array([10, -10, 0])
        var_upper = np.array([20, 10, 1])
        categorical = np.array([])
        not_categorical = np.array([0, 1, 2])
        categorical_expansion = []
        compressed = ru.compress_categorical_bounds(
            var_lower, categorical, not_categorical, categorical_expansion)
        msg = 'Compression of bounds for 0 vars'
        self.assertTrue(
            np.array_equal(compressed, np.array([10, -10, 0])), msg=msg)
        compressed = ru.compress_categorical_bounds(
            var_upper, categorical, not_categorical, categorical_expansion)
        self.assertTrue(
            np.array_equal(compressed, np.array([20, 10, 1])), msg=msg)
    # -- end function


    def test_norm(self):
        """Verify that norm is 0 at 0 and correct for some other vectors."""
        self.assertEqual(ru.norm(np.array([0 for i in range(10)])), 0.0,
                         msg='Norm is not zero at zero')
        self.assertEqual(ru.norm(np.array([-1 for i in range(9)])), 3.0,
                         msg='Norm is not 3.0 at {-1}^9')
        self.assertEqual(ru.norm(np.array([-2 + i for i in range(5)])), math.sqrt(10),
                         msg='Norm is not sqrt{10} at [-2, -1, 0, 1, 2]')
    # -- end function

    def test_distance(self):
        """Verify that distance is 0 iff two points are the same."""
        self.assertEqual(ru.distance(np.array([i*5 for i in range(15)]),
                                     np.array([i*5 for i in range(15)])), 0.0,
                         msg='Distance is not zero at equal points')
        self.assertNotEqual(ru.distance(np.array([i*5 for i in range(15)]),
                                        np.array([i*5 + 0.001 for i in range(15)])),
                            0.0, msg='Distance is nonzero at diff points')
        self.assertNotEqual(ru.distance(np.array([-i*5 for i in range(15)]),
                                        np.array([-i*5 + 0.001 for i in range(15)])),
                            0.0, msg='Distance is nonzero at diff points')
    # -- end function

    def test_get_min_distance(self):
        """Test some extreme cases for get_min_distance."""
        self.assertEqual(ru.get_min_distance(np.array([i for i in range(5)]),
                                             np.array([[i+j for i in range(5)]
                                              for j in range(10)])), 0.0)
    # -- end function
    def test_get_min_distance_and_index(self):
        """Test some extreme cases for get_min_distance_index."""
        d, i = ru.get_min_distance_and_index(np.array([i for i in range(5)]),
                                             np.array([[i+j for i in range(5)]
                                                       for j in range(-2, 3)]))
        self.assertEqual(i, 2)
        d, i = ru.get_min_distance_and_index(np.array([i+0.01 for 
                                                       i in range(5)]),
                                             np.array([[i+j for i in range(5)]
                                                       for j in range(-3, 2)]))
        self.assertEqual(i, 3)
    # -- end function

    def test_bulk_get_min_distance(self):
        """Verify that bulk version returns the same as regular version.
    
        This function checks that the bulk version of get_min_distance
        on a number of randomly generated points returns the same
        result as the regular version.
        """
        for i in range(50):
            dim = np.random.randint(1, 20)
            num_points_1 = np.random.randint(10, 50)
            num_points_2 = np.random.randint(10, 50)
            points = np.random.uniform(-100, 100, size=(num_points_1,dim))
            other_points = np.random.uniform(-100, 100,
                                             size=(num_points_2,dim))
            dist1 = [ru.get_min_distance(point, other_points)
                     for point in points]
            dist2 = ru.bulk_get_min_distance(points, other_points)
            for j in range(num_points_1):
                msg='Failed random test {:d} point {:d}'.format(i, j)
                self.assertAlmostEqual(dist1[j], dist2[j], 12, msg=msg)
    # -- end function

    def test_get_rbf_matrix(self):
        """Test basic properties of the RBF matrix (e.g. symmetry, size).

        Verify that the RBF matrix is symmetric and it has the correct
        size for all types of RBF.
        """
        settings = RbfoptSettings()
        for i in range(50):
            dim = np.random.randint(1, 20)
            num_points = np.random.randint(10, 50)
            node_pos = np.random.uniform(-100, 100, size=(num_points,dim))
            # Possible shapes of the matrix
            for rbf_type in self.rbf_types:
                settings.rbf = rbf_type
                mat = ru.get_rbf_matrix(settings, dim, num_points, node_pos)
                self.assertIsInstance(mat, np.ndarray)
                self.assertEqual(len(mat.shape), 2)
                self.assertAlmostEqual(np.max(mat - mat.transpose()), 0.0,
                                       msg='RBF matrix is not symmetric')
                size = num_points
                if (ru.get_degree_polynomial(settings) >= 0):
                    size += 1
                if (ru.get_degree_polynomial(settings) > 0):
                    size += dim ** ru.get_degree_polynomial(settings)
                self.assertEqual(mat.shape, (size, size))
        # Check that exception is raised for unknown RBF types
        settings.rbf = 'unknown'
        self.assertRaises(ValueError, ru.get_rbf_matrix, settings, 
                          dim, num_points, node_pos)
    # -- end function

    def test_rbf_interpolation(self):
        """Test interpolation conditions.

        Verify that the RBF interpolates at points.
        """
        settings = RbfoptSettings()
        for i in range(20):
            dim = np.random.randint(1, 20)
            num_points = np.random.randint(dim+1, 50)
            node_pos = np.random.uniform(-100, 100, size=(num_points,dim))
            node_val = np.random.uniform(0, 100, num_points)
            # Possible shapes of the matrix
            for rbf_type in self.rbf_types:
                settings.rbf = rbf_type
                mat = ru.get_rbf_matrix(settings, dim, num_points, node_pos)
                rbf_l, rbf_h = ru.get_rbf_coefficients(
                    settings, dim, num_points, mat, node_val)
                for i in range(num_points):
                    value = ru.evaluate_rbf(settings, node_pos[i], dim,
                                            num_points, node_pos, rbf_l, rbf_h)
                    self.assertAlmostEqual(value, node_val[i], places=3,
                                           msg='Interpolation failed' +
                                           'with rbf ' + rbf_type)
    # -- end function

    def test_rbf_interpolation_cat(self):
        """Test interpolation conditions with categorical variables.

        Verify that the RBF interpolates at points.
        """
        settings = RbfoptSettings()
        for i in range(20):
            dim = np.random.randint(5, 15)
            cat_dim = 8
            # We need enough points to ensure the system is not singular
            num_points = np.random.randint(2*(dim+cat_dim), 60)
            node_pos = np.hstack(
                (np.random.uniform(-100, 100, size=(num_points,dim)),
                 np.zeros(shape=(num_points,cat_dim))))
            # Pick random categorical values
            for j in range(num_points):                
                node_pos[j, dim + np.random.choice(4)] = 1
                node_pos[j, dim + 4 + np.random.choice(4)] = 1
            categorical_info = (np.array([0, 1]),
                                np.array([j+2 for j in range(dim)]),
                                [(0, 0, np.array([dim + j for j in range(4)])),
                                 (1, 0, np.array([dim+4+j for j in range(4)]))])
            node_val = np.random.uniform(0, 100, num_points)
            # Possible shapes of the matrix
            for rbf_type in self.rbf_types:
                settings.rbf = rbf_type
                mat = ru.get_rbf_matrix(settings, dim+cat_dim, num_points,
                                        node_pos)
                rbf_l, rbf_h = ru.get_rbf_coefficients(
                    settings, dim+cat_dim, num_points, mat, node_val,
                    categorical_info)
                for i in range(num_points):
                    value = ru.evaluate_rbf(settings, node_pos[i], dim+cat_dim,
                                            num_points, node_pos, rbf_l, rbf_h)
                    self.assertAlmostEqual(value, node_val[i], places=4,
                                           msg='Interpolation failed ' +
                                           'with rbf ' + rbf_type)
    # -- end function

    def test_transform_function_values(self):
        """Test all codomain transformation strategies.

        This will verify that the transformation strategies always
        produce valid results and can handle extreme cases.
        """
        settings = RbfoptSettings()
        list_scaling = [val for val in RbfoptSettings._allowed_function_scaling
                        if val != 'auto']
        list_clipping = [val for val in 
                         RbfoptSettings._allowed_dynamism_clipping
                         if val != 'auto']
        transf = ru.transform_function_values
        # Create list of values to test: node_val and corresponding
        # node_err_bound
        to_test = [(np.array([0, -100, settings.dynamism_threshold * 10]), 
                    np.array([[0,0], [0,0], [0,0]])),
                   (np.array([0.0]), np.array([[0, 0]])),
                   (np.array([0.0 for i in range(10)]),
                    np.array([[-i, i] for i in range(10)])),
                   (np.array([100.0 for i in range(10)]), 
                    np.array([[-1,1] for i in range(10)])),
                   (np.array([10.0**i for i in range(-20, 20)]),
                    np.array([[0,0] for i in range(-20, 20)])),
                   (np.append(np.array([-10.0**i for i in range(-20, 20)]),
                              np.array([10.0**i for i in range(-20, 20)])),
                    np.array([[-2**i,2**i] for i in range(-40, 40)]))]
        for scaling in list_scaling:
            for clipping in list_clipping:
                header = '({:s}, {:s}):'.format(scaling, clipping)
                for (node_val, node_err_bounds) in to_test:
                    settings.function_scaling = scaling
                    settings.dynamism_clipping = clipping
                    (scaled, minval, maxval, errbounds,
                     rescale_func) = transf(settings, node_val, min(node_val),
                                            max(node_val), node_err_bounds)
                    # Check that the number of scaled values is the
                    # same as the number of input values
                    msg='Number of output values is different from input'
                    self.assertEqual(len(scaled), len(node_val),
                                     msg=header + msg)
                    msg='Dynamism threshold was not enforced'
                    v1 = abs(min(scaled))
                    v2 = abs(max(scaled))
                    c1 = v1 > 1.0e-10 and v2/v1 <= settings.dynamism_threshold
                    c2 = v1 <= 1.0e-10 and v2 <= settings.dynamism_threshold
                    self.assertTrue(clipping == 'off' or c1 or c2,
                                    msg=header + msg)
                    for i in range(len(node_val)):
                        msg='Fast_node_index have wrong sign'
                        self.assertLessEqual(errbounds[i][0], 0, msg=msg)
                        self.assertGreaterEqual(errbounds[i][1], 0, msg=msg)
                    msg=('Min/Max of scaled values inconsistent with ' +
                           'returned scaled_min and scaled_max')
                    self.assertEqual(min(scaled), minval, msg=header + msg)
                    self.assertEqual(max(scaled), maxval, msg=header + msg)
        # -- end for
    # -- end function

    def test_transform_domain(self):
        """Check that affine transformation does not hang on limit case.

        Further check that 'off' transformation returns the point as
        is, and unimplemented strategies raise a ValueError.
        """
        settings = RbfoptSettings()
        settings.domain_scaling = 'affine'
        var_lower = np.array([i for i in range(5)] + [i for i in range(5)])
        var_upper = np.array([i for i in range(5)] + [i + 10 for i in range(5)])
        point = np.array([i for i in range(5)] + [i + 2*i for i in range(5)])
        # Test what happend when lower and upper bounds coincide
        transf_point = ru.transform_domain(settings, var_lower, var_upper,
                                           np.array([]), point)
        orig_point = ru.transform_domain(settings, var_lower, var_upper,
                                         np.array([]), transf_point, True)
        for i in range(10):
            msg='Exceeding lower bound on affine domain scaling'
            self.assertLessEqual(0.0, transf_point[i], msg=msg)
            msg='Exceeding upper bound on affine domain scaling'
            self.assertLessEqual(transf_point[i], 1.0, msg=msg)
            msg='Doubly transformed point does not match original'
            self.assertAlmostEqual(point[i], orig_point[i], 12, msg=msg)
        # Check that 'off' scaling does not do anything
        settings.domain_scaling = 'off'
        transf_point = ru.transform_domain(settings, var_lower, var_upper,
                                           np.array([]), point)
        for i in range(10):
            msg='Transformed point with \'off\' does not match original'
            self.assertEqual(point[i], transf_point[i], msg=msg)
        # Check that with integer variables, we do not do anything
        settings.domain_scaling = 'affine'
        transf_point = ru.transform_domain(settings, var_lower, var_upper,
                                           np.array([0]), point)
        for i in range(10):
            msg=('Transformed integer point with \'affine\' does not ' +
                 'match original')
            self.assertEqual(point[i], transf_point[i], msg=msg)
        # Check that unimplemented strategies are rejected
        settings.domain_scaling = 'test'
        self.assertRaises(ValueError, ru.transform_domain, settings, 
                          var_lower, var_upper, np.array([]), point)
    # -- end function

    def test_transform_domain_bounds(self):
        """Check that domain bounds are consistent."""
        list_scaling = [val for val in RbfoptSettings._allowed_domain_scaling 
                        if val != 'auto']
        for scaling in list_scaling:
            settings = RbfoptSettings(domain_scaling = scaling)
            # Test limit case with empty bounds
            vl, vu = ru.transform_domain_bounds(settings, np.array([]), np.array([]))
            msg='Failed transform_domain_bounds on empty bounds'
            self.assertEqual(len(vl), 0, msg=msg)
            self.assertEqual(len(vu), 0, msg=msg)
            msg='Bounds inconsistent with random bounds'
            for i in range(10):
                dim = np.random.randint(0, 20)
                var_lower = np.random.uniform(-100, 100, dim) 
                var_upper = var_lower + np.random.uniform(0, 100, dim)
                vl, vu = ru.transform_domain_bounds(settings, var_lower,
                                                    var_upper)
                self.assertEqual(len(vl), len(var_lower), msg=msg)
                self.assertEqual(len(vu), len(var_upper), msg=msg)
                for j in range(dim):
                    self.assertLessEqual(vl[j], vu[j], msg=msg)
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
                                        msg='sigma_n out of bounds')
    # -- end function

    def test_get_fmax_current_iter(self):
        """Verify get_fmax_current_iter is resilient to limit cases.

        This function tests whether correct values are returned when
        there is a single-element list of node values, and when the
        list of node values is exactly the minimum required k + 1.
        """
        settings = RbfoptSettings(init_sample_fraction=0.75)
        fun = ru.get_fmax_current_iter
        self.assertEqual(fun(settings, 0, 1, 1, np.array([1])), 1,
                         msg='Failed on single-element list')
        self.assertEqual(fun(settings, 10, 11, 5, 
                             np.array([i for i in range(11)])),
                         10, msg='Failed on n == k + 1')
    # -- end function

    def test_init_points_cleanup(self):
        """Verify that init_points_cleanup removes points too close.

        Test that only points with distance larger than min_dist are
        returned.
        """
        settings = RbfoptSettings(min_dist=1.0e-5)
        points = np.array([[0, 0], [0, 0], [0, 0]])
        ret = ru.init_points_cleanup(settings, points)
        self.assertListEqual(ret.tolist(), [0],
                             msg='Returned coinciding points')
        points = np.array([[0, 0, 0], [0, 1, 1], [0, 1.0e-5, 0]])
        ret = ru.init_points_cleanup(settings, points)
        self.assertListEqual(ret.tolist(), [0, 1],
                             msg='Returned coinciding points')
# -- end class

class TestModelSelection(unittest.TestCase):
    """Test the model selection functions."""

    def setUp(self):
        """Determine which model selection solvers should be tested."""
        self.n = 3
        self.k = 10
        self.var_lower = np.array([i for i in range(self.n)])
        self.var_upper = np.array([i + 10 for i in range(self.n)])
        self.node_pos = np.array([self.var_lower, self.var_upper, 
                                  [1, 2, 3], [9, 5, 8.8], [5.5, 7, 12],
                                  [3.2, 10.2, 4], [2.1, 1.1, 7.4],
                                  [6.6, 9.1, 2.0], [10, 8.8, 11.1],
                                  [7, 7, 7]])
        self.node_val = np.array([2*i*i for i in range(self.k)])

    # -- end function        

    def test_get_best_rbf_model(self):
        """Test the get_best_rbf_model function.
        """
        settings = RbfoptSettings()
        rbf, gamma = ru.get_best_rbf_model(settings, self.n, self.k, 
                                           self.node_pos, self.node_val,
                                           self.k)
        self.assertTrue(rbf == 'linear' or
                        (rbf == 'multiquadric' and gamma == 0.1),
                        msg='Did not obtain expected model')
    # -- end function

    def test_get_model_quality_estimate(self):
        """Test the get_model_quality_estimate function.
        """
        for rbf in ['cubic', 'thin_plate_spline', 'multiquadric',
                    'linear', 'gaussian']:
            settings = RbfoptSettings(rbf=rbf)
            error = ru.get_model_quality_estimate(
                settings, self.n, self.k, self.node_pos, 
                self.node_val, self.k)
            # Create a copy of the interpolation nodes and values
            sorted_idx = self.node_val.argsort()
            sorted_node_val = self.node_val[sorted_idx]
            # Initialize the arrays used for the cross-validation
            cv_node_pos = self.node_pos[sorted_idx[1:]]
            cv_node_val = self.node_val[sorted_idx[1:]]            
            # The node that was left out
            rm_node_pos = self.node_pos[sorted_idx[0]]
            rm_node_val = self.node_val[sorted_idx[0]]
            # Estimate of the model error
            loo_error = 0.0    
            for i in range(self.k):
                # Compute the RBF interpolant with one node left out
                Amat = ru.get_rbf_matrix(settings, self.n, self.k-1, 
                                         cv_node_pos)
                rbf_l, rbf_h = ru.get_rbf_coefficients(
                    settings, self.n, self.k-1, Amat, cv_node_val)
                # Compute value of the interpolant at the removed node
                predicted_val = ru.evaluate_rbf(settings, rm_node_pos, 
                                                self.n, self.k-1, 
                                                cv_node_pos, rbf_l, rbf_h)
                # Update leave-one-out error
                loc = np.searchsorted(sorted_node_val, predicted_val)
                loo_error += abs(loc - i)
                # Update the node left out
                if (i < self.k - 1):
                    tmp = cv_node_pos[i].copy()
                    cv_node_pos[i] = rm_node_pos
                    rm_node_pos = tmp
                    cv_node_val[i], rm_node_val = rm_node_val, cv_node_val[i]
            self.assertAlmostEqual(loo_error, error, 
                                   msg='Model selection procedure ' +
                                   'miscomputed the error')
            # -- end for
        # -- end for
    # -- end function
# -- end class
