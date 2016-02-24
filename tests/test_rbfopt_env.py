"""Set module import path to allow easy unit tests.

This module adds the src/ directory of RBFOpt to the system path, so
that all modules can be imported directly.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright International Business Machines Corporation 2016.

"""

import sys
import os
import random
import numpy

# append module src/ directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir) + '/src/')
# Random seed for testing environment: will be set where appropriate
rand_seed = 7123694123


