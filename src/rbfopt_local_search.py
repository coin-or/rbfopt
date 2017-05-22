"""Routines for local search.

This module contains all functions that are necessary to implement a
trust-region based local search. The local search exploits a quadratic
model of the objective function.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright International Business Machines Corporation 2017.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import scipy.spatial as ss
try:
    import cython_rbfopt.rbfopt_utils as ru
except (ImportError, TypeError):
    import rbfopt_utils as ru
try:
    import cython_rbfopt.rbfopt_aux_problems as aux
except (ImportError, TypeError):
    import rbfopt_aux_problems as aux
import rbfopt_config as config
from rbfopt_settings import RbfSettings

def get_quadratic_model(settings, n, k, node_pos, node_val, center):
    """Compute a quadratic model of the function.

    Determine a quadratic model x^T Q x + h^T x + b of the objective
    function in an area that is centered on the given node. The
    model is computed by solving an underdetermined linear system,
    inducing sparsity.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`.
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes.

    node_val : 1D numpy.ndarray[float]
        List of values of the function at the nodes.

    center : 1D numpy.ndarray[float]
        Node that acts as a center for the quadratic model.

    Returns
    -------
    numpy.matrix
        The matrix A = [Phi P; P^T 0].

    """
    assert(isinstance(node_pos, np.ndarray))
    assert(len(node_pos)==k)
    assert(isinstance(node_val, np.ndarray))
    assert(len(np.atleast_1d(node_val))==k)
    assert(isinstance(center, np.ndarray))
    assert(len(np.atleast_1d(center))==n)
    assert(isinstance(settings, RbfSettings))
    # Find points closest to the given point
    dist = ss.distance.cdist(np.atleast_2d(center), node_pos)
    dist_order = np.lexsort((dist[0],))
    # The nodes to keep are those closest to the center
    num_to_keep = min(2*n + 1, k)
    # Position of upper diagonal elements
    upper_diag = np.array([n*i + j for i in range(n) for j in range(i, n)])
    # Determine the coefficients of the underdetermined linear system.
    # The matrix A contains the quadratic terms (including cross
    # products).
    A = np.array([np.outer(node_pos[dist_order[i]], 
                           node_pos[dist_order[i]]).ravel()[upper_diag]
                  for i in np.arange(num_to_keep)])
    # We put the node positions and a vector of ones to the right of A
    lstsq_mat = np.hstack((A, node_pos[dist_order[:num_to_keep]], 
                           np.ones((num_to_keep, 1))))
    # Solve least squares system and recover quadratic form
    x, res, rank, s = np.linalg.lstsq(lstsq_mat, 
                                      node_val[dist_order[:num_to_keep]])
    Q = np.zeros((n, n))
    Q.ravel()[upper_diag] = x[:(n*(n+1)//2)]
    #Q = (Q + Q.T)/2
    h = x[(n*(n+1)//2):(n*(n+1)//2+n)]
    b = x[-1]
    

