import numpy as np
import rbfopt_utils as ru
import rbfopt_local_search as ls
import rbfopt_settings

if (__name__ == '__main__'):
    n = 5
    k = n*(n+1)//2 + n + 1
    settings = rbfopt_settings.RbfSettings()
    node_pos = np.array([np.random.rand(n) for i in range(k)])
    Q = np.random.rand(n, n)
    Q += Q.T
    h = np.random.rand(n)
    b = np.random.rand()
    node_val = np.array([np.dot(node, np.dot(Q, node)) + np.dot(h, node) + b
                         for node in node_pos])
    model_set, tr_radius = ls.init_trust_region(settings, n, k, node_pos, 
                                             node_pos[-1])
    var_lower = np.array([0]*n)
    var_upper = np.array([1]*n)
    Qe, he, be = ls.get_quadratic_model(settings, n, k, node_pos, node_val,
                                        model_set)
    point, val = ls.get_candidate_point(settings, n, k, var_lower,
                                        var_upper, Qe, he, be,
                                        node_pos[0], 0.1)
    print(point, val)
    integer_vars = np.array([2, 4])
    print(ls.get_integer_candidate(settings, n, k, Qe, he, be, point, 
                                   integer_vars))
