import os
# We must set the threading options before numpy is loaded, otherwise
# there might be issues when running several processes in parallel.
os.environ['OMP_NUM_THREADS'] = '1'
from .rbfopt_settings import RbfoptSettings
from .rbfopt_algorithm import RbfoptAlgorithm
from .rbfopt_black_box import RbfoptBlackBox
from .rbfopt_user_black_box import RbfoptUserBlackBox

__version__ = '4.3.0'

__all__ = ['rbfopt_algorithm',
           'rbfopt_aux_problems',
           'rbfopt_black_box',
           'rbfopt_cl_interface',
           'rbfopt_degree0_models',
           'rbfopt_degree1_models',
           'rbfopt_degreem1_models',
           'rbfopt_refinement',
           'rbfopt_settings',
           'rbfopt_test_functions',
           'rbfopt_test_interface',
           'rbfopt_user_black_box',
           'rbfopt_utils']
