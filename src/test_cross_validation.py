from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import rbfopt_model_selection as ms
import random
import math
import sys
import time
import numpy as np
from rbfopt_settings import RbfSettings

def test_timing(n, k, settings, node_pos, node_val):
    """Test timing of cross validation methods.

    """
    start_time = time.time()
    ms.get_best_rbf_model(settings, n, k, node_pos, node_val,
                          int(math.floor(k*0.7)))
    end_time = time.time()
    return (end_time - start_time)

if (__name__ == "__main__"):
    n = int(sys.argv[1])
    k = int(sys.argv[2])
    trials = int(sys.argv[3])
    numpy_times = list()
    cplex_times = [0]
    for trial in range(trials):
        node_pos = [[random.uniform(0, 1) for j in range(n)] for i in range(k)]
        node_val = [random.uniform(0, 1000) for i in range(k)]
        settings = RbfSettings(rbf = 'auto', model_selection_solver = 'numpy')
        numpy_time = test_timing(int(sys.argv[1]), int(sys.argv[2]), settings,
                                 node_pos, node_val)
        numpy_times.append(numpy_time)
        # settings = RbfSettings(rbf = 'auto', model_selection_solver = 'cplex')
        # cplex_time = test_timing(int(sys.argv[1]), int(sys.argv[2]), settings,
        #                          node_pos, node_val)
        # cplex_times.append(cplex_time)
    print('{:d} {:f} {:f} {:f} {:f}'.format(k, np.mean(numpy_times), 
                                            np.std(numpy_times),
                                            np.mean(cplex_times),
                                            np.std(cplex_times)))
