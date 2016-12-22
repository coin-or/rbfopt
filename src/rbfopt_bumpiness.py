"""Bumpiness functions.

This module is responsible for computing the bumpiness.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
Research partially supported by SUTD-MIT International Design Center.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import math
import numpy as np
import sys
try:
    was_there = ('cython_rbfopt.rbfopt_utils' in sys.modules.keys())
    import cython_rbfopt.rbfopt_utils as ru
    if not was_there:
        print('Imported Cython version of rbfopt_utils')
except ImportError:
    import rbfopt_utils as ru
try:
    was_there = ('cython_rbfopt.rbfopt_aux_problems' in sys.modules.keys())
    import cython_rbfopt.rbfopt_aux_problems as aux
    if not was_there:
        print('Imported Cython version of rbfopt_aux_problems')
except ImportError:
    import rbfopt_aux_problems as aux
from rbfopt_settings import RbfSettings


def get_min_bump_node(settings, n, k, Amat, node_val,
                      fast_node_index, fast_node_err_bounds,
                      target_val):


    """Compute the bumpiness obtained by moving an interpolation point.

    Compute the bumpiness of the interpolant obtained by moving a
    single node (the one that yields minimum bumpiness, which is
    determined by this function) within target_val plus or minus
    error, to target_val.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the space where the point lives.

    k : int
        Number of nodes, i.e. interpolation points.

    Amat : numpy.matrix
        The matrix A = [Phi P; P^T 0] of equation (3) in the paper by
        Costa and Nannicini.

    node_val : 1D numpy.ndarray[float]
        List of values of the function at the nodes.

    fast_node_index : 1D numpy.ndarray[int]
        List of indices of nodes whose function value should be
        considered variable withing the allowed range.

    fast_node_err_bounds : List[int]
        Allowed deviation from node values for nodes affected by
        error. This is a list of tuples (lower, upper) of the same
        length as fast_node_index.

    target_val : float
        Target function value at which we want to move the node.

    Returns
    -------
    (int, float)
        The index of the node and corresponding bumpiness value
        indicating the sought node in the list node_pos.
    """
    assert (isinstance(node_val, np.ndarray))
    assert (isinstance(fast_node_index, np.ndarray))
    assert (isinstance(settings, RbfSettings))
    assert (len(node_val) == k)
    assert (isinstance(Amat, np.matrix))
    assert (len(fast_node_index) == len(fast_node_err_bounds))

    # Extract the matrices Phi and P from
    Phimat = Amat[:k, :k]
    Pmat = Amat[:k, k:]

    min_bump_index, min_bump = None, float('Inf')
    for (pos, i) in enumerate(fast_node_index):
        # Check if we are within the allowed range
        if (node_val[i] + fast_node_err_bounds[pos][0] <= target_val and
                        node_val[i] + fast_node_err_bounds[pos][1] >= target_val):
            # If so, compute bumpiness. Save original data.
            orig_node_val = node_val[i]
            orig_node_err_bounds = fast_node_err_bounds[pos]
            # Fix this node at the target value.
            node_val[i] = target_val
            fast_node_err_bounds[pos] = (0.0, 0.0)
            # Compute RBF interpolant.
            # Get coefficients for the exact RBF first
            (rbf_l, rbf_h) = ru.get_rbf_coefficients(settings, n, k,
                                                  Amat, node_val)
            # And now the noisy version
            (rbf_l,
             rbf_h) = aux.get_noisy_rbf_coefficients(settings, n, k, Phimat,
                                                     Pmat, node_val,
                                                     fast_node_index,
                                                     fast_node_err_bounds,
                                                     rbf_l, rbf_h)
            # Restore original values
            node_val[i] = orig_node_val
            fast_node_err_bounds[pos] = orig_node_err_bounds
            # Compute bumpiness using the formula \lambda^T \Phi \lambda
            bump = np.dot(np.dot(rbf_l, Phimat), rbf_l)
            if (bump < min_bump):
                min_bump_index, min_bump = i, bump

    return (min_bump_index, min_bump)


# -- end function

def get_bump_new_node(settings, n, k, node_pos, node_val, new_node,
                      fast_node_index, fast_node_err_bounds, target_val):
    """Compute the bumpiness with a new interpolation point.

    Computes the bumpiness of the interpolant obtained by setting a
    new node in a specified location, at value target_val.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the space where the point lives.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : 2D numpy.ndarray[float]
        Location of current interpolation nodes.

    node_val : 1D numpy.ndarray[float]
        List of values of the function at the nodes.

    new_node : 1D numpy.ndarray[float]
        Location of new interpolation node.

    fast_node_index : 1D numpy.ndarray[float]
        List of indices of nodes whose function value should be
        considered variable withing the allowed range.

    fast_node_err_bounds : List[int]
        Allowed deviation from node values for nodes affected by
        error. This is a list of tuples (lower, upper) of the same
        length as fast_node_index.

    target_val : float
        Target function value at which we want to move the node.

    Returns
    -------
    float
        The bumpiness of the interpolant having a new node at the
        specified location, with value target_val.
    """
    assert(isinstance(node_pos, np.ndarray))
    assert(isinstance(node_val, np.ndarray))
    assert(isinstance(new_node, np.ndarray))
    assert(isinstance(fast_node_index, np.ndarray))
    assert(isinstance(settings, RbfSettings))
    assert(len(node_val) == k)
    assert(len(node_pos) == k)
    assert(len(fast_node_index) == len(fast_node_err_bounds))
    assert(new_node is not None)

    # Add the new node to existing ones
    n_node_pos = np.vstack((node_pos, new_node))
    n_node_val = np.append(node_val, target_val)

    # Compute the matrices necessary for the algorithm
    Amat = ru.get_rbf_matrix(settings, n, k + 1, n_node_pos)

    # Get coefficients for the exact RBF
    (rbf_l, rbf_h) = ru.get_rbf_coefficients(settings, n, k + 1, Amat,
                                          n_node_val)

    # Get RBF coefficients for noisy interpolant
    (rbf_l, rbf_h) = aux.get_noisy_rbf_coefficients(settings, n, k + 1,
                                                    Amat[:(k + 1), :(k + 1)],
                                                    Amat[:(k + 1), (k + 1):],
                                                    n_node_val,
                                                    fast_node_index,
                                                    fast_node_err_bounds,
                                                    rbf_l, rbf_h)

    bumpiness = np.dot(np.dot(rbf_l, Amat), rbf_l)

    return bumpiness

# -- end function
