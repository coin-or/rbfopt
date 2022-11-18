"""Test the class RbfoptSettings.

This module contains unit tests for the class RbfoptSettings.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright International Business Machines Corporation 2016.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import ast
import rbfopt
import numpy as np
import rbfopt.rbfopt_settings as rs


class TestRbfoptSettings(unittest.TestCase):
    """Test the RbfSetting class."""

    def setUp(self):
        """Create default parameter object and read docstring."""
        np.random.seed(71294123)
        self.default = rs.RbfoptSettings()
        attrs = vars(self.default)
        docstring = self.default.__doc__
        param_docstring = docstring[docstring.find('Parameters'):
                                    docstring.find('Attributes')].split(' : ')
        self.param_name = [val.split(' ')[-1].strip()
                           for val in param_docstring[:-1]]
        self.param_type = [val.split('\n')[0].strip()
                           for val in param_docstring[1:]]
        self.param_help = [' '.join(line.strip() for line in
                                    val.split('\n')[1:-2])
                           for val in param_docstring[1:]]
        # We extract the default from the docstring, and see if
        # matches with the object created above
        self.param_default = [val.split(' ')[-1].rstrip('.').strip('\'') 
                              for val in self.param_help]
    # -- end function

    def test_param_types(self):
        """Verify that parameter types match in docstring and code."""
        # Loop through parameters and check their type
        for i in range(len(self.param_name)):
            if (self.param_type[i] == 'float'):
                type_fun = float
            elif (self.param_type[i] == 'int'):
                type_fun = int
            elif (self.param_type[i] == 'bool'):
                type_fun = bool
            else:
                type_fun = str
            message = ('Attribute {:s}'.format(self.param_name[i]) +
                       ' is not of type {:s}'.format(self.param_type[i]))
            self.assertIsInstance(getattr(self.default, self.param_name[i]),
                                  type_fun, msg = message)
    # -- end function

    def test_default_values(self):
        """Verify that default values match in docstring and code."""
        for i in range(len(self.param_name)):
            if (self.param_type[i] == 'float'):
                type_fun = float
            elif (self.param_type[i] == 'int'):
                type_fun = int
            elif (self.param_type[i] == 'bool'):
                type_fun = ast.literal_eval
            else:
                type_fun = str
            message = ('Attribute {:s}'.format(self.param_name[i]) +
                       ' has value ' + 
                       str(getattr(self.default, self.param_name[i])) + 
                       ' instead of {:s}'.format(str(self.param_default[i])))
            self.assertEqual(type_fun(self.param_default[i]), 
                             getattr(self.default, self.param_name[i]),
                             msg = message)
    # -- end function

    def test_from_dictionary(self):
        """Verify that an object can be created from a dictionary."""
        dict_settings = dict()
        # Set dictionary key/value pairs
        for i in range(len(self.param_name)):
            if (self.param_type[i] == 'float'):
                type_fun = float
            elif (self.param_type[i] == 'int'):
                type_fun = int
            elif (self.param_type[i] == 'bool'):
                type_fun = ast.literal_eval
            else:
                type_fun = str
            dict_settings[self.param_name[i]] = type_fun(self.param_default[i])
        # Create new object
        settings = rs.RbfoptSettings.from_dictionary(dict_settings)
        self.assertIsInstance(settings, rs.RbfoptSettings)
        # Verify that values are preserved
        for name in self.param_name:
            message = ('Value of parameter {:s}'.format(name) + 
                       ' not set correctly.')
            self.assertEqual(getattr(self.default, name),
                             getattr(settings, name), msg = message)
    # -- end function

    def test_invalid_parameters(self):
        """Verify that an exception is raised for invalid values."""
        for param in rs.RbfoptSettings._nonnegative_parameters:
            message = ('Negative parameter {:s}'.format(param) + 
                       ' did not raise an exception.')            
            with self.assertRaises(ValueError, msg = message):
                rs.RbfoptSettings(**{param : -0.5})
    # -- end function

# -- end class
