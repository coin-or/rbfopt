import numpy as np
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
    ls.get_quadratic_model(settings, n, k, node_pos, node_val,
                           node_pos[-1])
