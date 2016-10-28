"""Test the module rbfopt_utils in RBFOpt.

This module contains unit tests for the module rbfopt_utils.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright International Business Machines Corporation 2016.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import random
import numpy as np
import test_rbfopt_env
try:
    import cython_rbfopt.rbfopt_utils as ru
    print('Imported Cython version of rbfopt_utils')
except ImportError:
    import rbfopt_utils as ru
from rbfopt_settings import RbfSettings

class TestUtils(unittest.TestCase):
    """Test the rbfopt_utils module."""

    def setUp(self):
        """Initialize data used by several functions."""
        self.rbf_types = [rbf_type for rbf_type in RbfSettings._allowed_rbf
                          if rbf_type != 'auto']
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
