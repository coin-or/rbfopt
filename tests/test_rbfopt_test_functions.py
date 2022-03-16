"""Test the rbfopt_test_functions module in RBFOpt.

This module contains unit tests for the module rbfopt_test_functions,
that contains mathematical functions for testing purposes. These tests
mainly verify that optima coincid

Licensed under Revised BSD license, see LICENSE.
(C) Copyright International Business Machines Corporation 2016.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import rbfopt
import numpy as np
import inspect
import rbfopt.rbfopt_test_functions as tf

class TestFunctions(unittest.TestCase):
    """Test the rbfopt_test_functions module."""

    def setUp(self):
        """Initialize list of functions."""        
        clsmembers = inspect.getmembers(tf, inspect.isclass)
        excluded = ['RbfoptBlackBox', 'TestBlackBox', 'TestEnlargedBlackBox',
                    'TestNoisyBlackBox']
        self.function_list = [val[0] for val in clsmembers if
                              val[0] not in excluded]
    # -- end function

    def test_optima_default(self):
        """Ensure that optimum has correct value on unmodified functions."""
        for name in self.function_list:
            function = getattr(tf, name)
            self.assertAlmostEqual(function.evaluate(function.optimum_point),
                                   function.optimum_value,
                                   msg='Function ' + name + ': ' +
                                   'optimum does not match')

    def test_optima_enlarged(self):
        """Ensure that enlarged functions keep the optimum."""
        for name in self.function_list:
            for dim in [1, 2, 5, 10, 20]:
                bb = tf.TestEnlargedBlackBox(name, dim)
                self.assertAlmostEqual(bb.evaluate(bb.optimum_point),
                                       bb.optimum_value,
                                       msg='Function ' + name + ' dim ' +
                                       str(dim) + ': optimum does not match')
    # -- end function

    def test_optima_noisy(self):
        """Ensure that noisy functions keep the optimum."""
        for name in self.function_list:
            bb = tf.TestBlackBox(name)
            noisybb = tf.TestNoisyBlackBox(bb, 0.0, 0.1)
            self.assertAlmostEqual(
                noisybb.evaluate(bb._function.optimum_point),
                bb._function.optimum_value, delta=0.1,
                msg='Noisy function ' + name + ': absolute error exceeded')
        for name in self.function_list:
            bb = tf.TestBlackBox(name)
            noisybb = tf.TestNoisyBlackBox(bb, 0.1, 0.0)
            funval = noisybb.evaluate(bb._function.optimum_point)
            if (abs(bb._function.optimum_value) > 1.0e-8):
                error = (abs(funval - bb._function.optimum_value) /
                         abs(bb._function.optimum_value))
            else:
                error = abs(funval - bb._function.optimum_value)
            self.assertLessEqual(error, 0.1,
                                 msg='Noisy function ' + name +
                                 ': relative error exceeded')
        
    # -- end function
                    
