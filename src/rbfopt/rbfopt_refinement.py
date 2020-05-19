"""Routines for a local search to refine the solution.

This module contains all functions that are necessary to implement a
local search to refine the solution quality. The local search exploits
a linear model of the objective function.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright International Business Machines Corporation 2017.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import numpy as np
import scipy.spatial as ss
import scipy.linalg as la
import rbfopt.rbfopt_utils as ru
from rbfopt.rbfopt_settings import RbfoptSettings


def init_refinement(settings, n, k, node_pos, center):
    """Initialize the local search model.

    Determine which nodes should be used to create a linear model of
    the objective function, and determine the initial radius of the
    search.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`.
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes.

    center : 1D numpy.ndarray[float]
        Node that acts as a center for the linear model.

    Returns
    -------
    (1D numpy.ndarray[int], float)
        Indices in node_pos of points to build the model, and initial
        radius of the local search.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix cannot be computed for numerical reasons.

    """
    assert(isinstance(node_pos, np.ndarray))
    assert(len(node_pos)==k)
    assert(k >= 2)
    assert(isinstance(center, np.ndarray))
    assert(len(np.atleast_1d(center))==n)
    assert(isinstance(settings, RbfoptSettings))
    # Find points closest to the given point
    dist = ss.distance.cdist(np.atleast_2d(center), node_pos)
    dist_order = np.argsort(dist[0])
    # The nodes to keep are those closest to the center
    num_to_keep = min(n + 1, k)
    # Build array of nodes to keep
    model_set = dist_order[np.arange(num_to_keep)]
    ref_radius = max(np.percentile(dist[0, model_set[1:]], 50),
                     settings.ref_min_radius * 
                     2**settings.ref_init_radius_multiplier)
    return (model_set, ref_radius)
# -- end function

def get_linear_model(settings, n, k, node_pos, node_val, model_set):
    """Compute a linear model of the function.

    Determine a linear model h^T x + b of the objective function in an
    area that is centered on the given node. The model is computed by
    solving a (not necessarily square) linear system, inducing
    sparsity.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`.
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes.

    node_val : 1D numpy.ndarray[float]
        List of values of the function at the nodes.

    model_set : 1D numpy.ndarray[int]
        Indices of points in node_pos to be used to compute model.

    Returns
    -------
    1D numpy.ndarray[float], float, bool
        Coefficients of the linear model h, b, and a boolean
        indicating if the linear model is underdetermined.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix cannot be computed for numerical reasons.

    """
    assert(isinstance(node_pos, np.ndarray))
    assert(len(node_pos)==k)
    assert(isinstance(node_val, np.ndarray))
    assert(len(node_val)==k)
    assert(isinstance(model_set, np.ndarray))
    assert(isinstance(settings, RbfoptSettings))
    model_size = len(model_set)
    # Determine the coefficients of the linear system.
    lstsq_mat = np.hstack((node_pos[model_set], np.ones((model_size, 1))))
    rank_deficient = False
    # Solve least squares system and recover linear form
    try:
        x, res, rank, s = np.linalg.lstsq(lstsq_mat, node_val[model_set],
                                          rcond=-1)
        if (rank < model_size):
            rank_deficient = True
    except np.linalg.LinAlgError as e:
        print('Exception raised trying to compute linear model',
              file=sys.stderr)
        print(e, file=sys.stderr)
        raise e
    h = x[:n]
    b = x[-1]
    return h, b, rank_deficient
# -- end function

def get_candidate_point(settings, n, k, var_lower, var_upper, h,
                        start_point, ref_radius):
    """Compute the next candidate point of the refinement.

    Starting from a given point, compute a descent direction and move
    in that direction to find the point with lowest value of the
    linear model, within the radius of the local search.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`.
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    var_lower : 1D numpy.ndarray[float]
        Vector of variable lower bounds.
    
    var_upper : 1D numpy.ndarray[float]
        Vector of variable upper bounds.

    h : 1D numpy.ndarray[float]
        Linear coefficients of the linear model.

    start_point : 1D numpy.ndarray[float]
        Starting point for the descent.

    ref_radius : float
        Radius of the local search.

    Returns
    -------
    (1D numpy.ndarray[float], float, float)
        Next candidate point for the search, the corresponding model
        value difference, and the norm of the gradient at the current
        point.

    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(isinstance(start_point, np.ndarray))
    assert(isinstance(h, np.ndarray))
    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(len(start_point)==n)
    assert(len(h)==n)
    assert(ref_radius>=0)
    assert(isinstance(settings, RbfoptSettings))
    grad_norm = np.sqrt(np.dot(h, h))
    # If the gradient is essentially zero, there is nothing to improve
    if (grad_norm <= settings.eps_zero):
        return (start_point, 0.0, grad_norm)
    # Determine maximum (smallest) t for line search before we exceed bounds
    max_t = ref_radius/np.sqrt(np.dot(h, h))
    loc = (h > 0) * (start_point >= var_lower + settings.min_dist)
    if (np.any(loc)):
        to_var_lower = (start_point[loc] - var_lower[loc]) / h[loc]
        max_t = min(max_t, np.min(to_var_lower))
    loc = (h < 0) * (start_point <= var_upper - settings.min_dist)
    if (np.any(loc)):
        to_var_upper = (start_point[loc] - var_upper[loc]) / h[loc]
        max_t = min(max_t, np.min(to_var_upper))
    candidate = np.clip(start_point - max_t * h, var_lower, var_upper)
    return (candidate, np.dot(h, start_point - candidate), grad_norm)
# -- end function

def get_integer_candidate(settings, n, k, h, start_point, ref_radius, 
                          candidate, integer_vars, categorical_info):
    """Get integer candidate point from a fractional point.

    Look for integer points around the given fractional point, trying
    to find one with a good value of the linear model.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`.
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    h : 1D numpy.ndarray[float]
        Linear coefficients of the model.

    start_point : 1D numpy.ndarray[float]
        Starting point for the descent.

    ref_radius : float
        Radius of the local search.

    candidate : 1D numpy.ndarray[float]
        Fractional point to being the search.

    integer_vars : 1D numpy.ndarray[int]
        Indices of the integer variables.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    Returns
    -------
    (1D numpy.ndarray[float], float)
        Next candidate point for the search, and the corresponding
        change in model value compared to the given point.
    """ 
    assert(isinstance(candidate, np.ndarray))
    assert(len(candidate) == n)
    assert(isinstance(h, np.ndarray))
    assert(len(h) == n)
    assert(isinstance(integer_vars, np.ndarray))
    assert(isinstance(settings, RbfoptSettings))
    # If there are categorical variables, they have to be dealt with
    # separately. Exclude them from the set of integer vars.
    if (categorical_info is not None and categorical_info[2]):
        categorical, not_categorical, expansion = categorical_info
        integer_vars = np.array([i for i in integer_vars
                                 if i < len(not_categorical)],
                                dtype=np.int_)
    # Compute the rounding down and up
    floor = np.floor(candidate[integer_vars])
    ceil = np.ceil(candidate[integer_vars])
    curr_point = np.copy(candidate)
    curr_point[integer_vars] = np.where(h[integer_vars] >= 0, ceil, floor)
    if (categorical_info is not None and categorical_info[2]):
        # Round in-place
        round_categorical(curr_point, categorical, not_categorical, expansion)
    best_value = np.dot(h, curr_point)
    best_point = np.copy(curr_point)
    for i in range(n * settings.ref_num_integer_candidates):
        # We round each integer variable up or down depending on its
        # fractional value and a uniform random number
        curr_point[integer_vars] = np.where(
            np.random.uniform(size=len(integer_vars)) < 
            candidate[integer_vars] - floor, ceil, floor)
        if (categorical_info is not None and categorical_info[2]):
            curr_point[len(not_categorical):] = candidate[len(not_categorical):]
            # Round in-place
            round_categorical(curr_point, categorical, not_categorical,
                              expansion)
        curr_value = np.dot(h, curr_point) 
        if (ru.distance(curr_point, start_point) <= ref_radius and 
            curr_value < best_value):
            best_value = curr_value
            best_point = np.copy(curr_point)
    return (best_point, np.dot(h, candidate) - best_value)
# -- end function

def round_categorical(point, categorical, not_categorical,
                      categorical_expansion):
    """Round categorical variables of a fractional point.
    
    Ensure categorical variables of fractional point are correctly
    rounded. Rounding is done in-place.

    Parameters
    ----------
    
    points : 1D numpy.ndarray[float]
        Point we want to round.

    categorical : 1D numpy.ndarray[int]
        Array of indices of categorical variables in original space.

    not_categorical : 1D numpy.ndarray[int]
        Array of indices of not categorical variables in original space.

    categorical_expansion : List[(int, float, 1D numpy.ndarray[int])]
        Expansion of original categorical variables into binaries.

    """
    assert(isinstance(point, np.ndarray))
    assert(isinstance(categorical, np.ndarray))
    assert(isinstance(not_categorical, np.ndarray))
    assert(categorical_expansion)
    # Ensure only one is picked for categorical variables
    for index, var_lower, expansion in categorical_expansion:
        sum_prob = np.sum(point[expansion])
        if (sum_prob == 0):
            # If there are no fractional values, pick a random value
            chosen = np.random.choice(expansion)
        else:
            # Otherwise, use probabilities based on fractional values
            chosen = np.random.choice(expansion,
                                      p=point[expansion]/sum_prob)
        point[expansion] = np.zeros(len(expansion))
        point[chosen] = 1
# -- end function


def get_model_improving_point(settings, n, k, var_lower, var_upper,
                              node_pos, model_set, start_point_index,
                              ref_radius, integer_vars, categorical_info):
    """Compute a point to improve the model used in refinement.

    Determine a point that improves the geometry of the set of points
    used to build the local search model. This point may not have a
    good objective function value, but it ensures that the model is
    well behaved.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`.
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    var_lower : 1D numpy.ndarray[float]
        Vector of variable lower bounds.
    
    var_upper : 1D numpy.ndarray[float]
        Vector of variable upper bounds.

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes.

    model_set : 1D numpy.ndarray[int]
        Indices of points in node_pos to be used to compute model.

    start_point_index : int
        Index in node_pos of the starting point for the descent.

    ref_radius : float
        Radius of the local search.

    integer_vars : 1D numpy.ndarray[int]
        Indices of the integer variables.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    Returns
    -------
    (1D numpy.ndarray[float], bool, int)
        Next candidate point to improve the model, a boolean
        indicating success, and the index of the point to replace if
        successful.

    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(isinstance(node_pos, np.ndarray))
    assert(len(node_pos)==k)
    assert(isinstance(model_set, np.ndarray))
    assert(start_point_index < k)
    assert(ref_radius>=0)
    assert(isinstance(settings, RbfoptSettings))
    # Remove the start point from the model set if necessary
    red_model_set = np.array([i for i in model_set if i != start_point_index])
    model_size = len(red_model_set)
    if (model_size == 0):
        # Unlikely, but after removing a point we may end up with not
        # enough points
        return (node_pos[start_point_index], False, start_point_index)
    # Tolerance for linearly dependent rows Determine
    # the coefficients of the directions spanned by the model
    A = node_pos[red_model_set] - node_pos[start_point_index]
    Q, R, P = la.qr(A.T, mode='full', pivoting=True)
    rank = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(
        settings.eps_linear_dependence)
    if (rank >= model_size):
        # Numerically, the rank is ok according to our tolerance.
        # Return indicating that we do not have to perform model
        # improvement.
        return (node_pos[start_point_index], False, start_point_index)
    success = False
    d = np.zeros(n)
    i = rank
    to_replace = P[i]
    while (i < model_size and not success):
        # Determine candidate direction
        d = Q[:, i].T*ref_radius
        d = np.clip(node_pos[start_point_index] + d, var_lower,
                    var_upper) - node_pos[start_point_index]
        if (categorical_info is not None and categorical_info[2]):
            candidate = node_pos[start_point_index] + d
            round_categorical(candidate, *categorical_info)
            d = candidate - node_pos[start_point_index]
        if (len(integer_vars)):
            # Zero out small directions, and increase to one nonzero
            # integer directions
            d[np.abs(d) < settings.eps_zero] = 0
            d[integer_vars] = (np.sign(d[integer_vars]) *
                               np.maximum(np.abs(d[integer_vars]),
                                          np.ones(len(integer_vars))))
            d[integer_vars] = np.around(d[integer_vars])
        # Check if rank increased
        B = np.vstack((A[P[:rank], :], d.T))
        Q2, R2, P2 = la.qr(B.T, mode='full', pivoting=True)
        new_rank = min(B.shape) - np.abs(np.diag(R2))[::-1].searchsorted(
            settings.eps_linear_dependence)
        if (new_rank > rank):
            to_replace = P[i]
            success = True
        i += 1
    return (node_pos[start_point_index] + d, success, to_replace)
# -- end function

def update_refinement_radius(settings, ref_radius, model_obj_diff,
                             real_obj_diff):
    """Update the radius fo refinement.

    Compute the updated refinement radius based on the true objective
    function difference between the old point and the new point, and
    that of the linear model. Also, determine if the new iterate
    should be accepted.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`.
        Global and algorithmic settings.

    ref_radius : float
        Current radius of the refinement region.

    model_obj_diff : float
        Difference in objective function value of the new point and
        the previous point, according to the linear model.

    real_obj_diff : float
        Difference in the real objective function value of the new
        point and the previous point.

    Returns
    -------
    (float, bool)
        Updated radius of refinement, and whether the point should be
        accepted.

    """
    assert(ref_radius >= 0)
    assert(isinstance(settings, RbfoptSettings))
    init_radius = ref_radius
    decrease = (real_obj_diff / model_obj_diff 
                if abs(model_obj_diff) > settings.eps_zero else 0)
    if (decrease <= settings.ref_acceptable_decrease_shrink):
        ref_radius *= 0.5
    elif (decrease >= settings.ref_acceptable_decrease_enlarge):
        ref_radius *= 2
    return (ref_radius, decrease >= settings.ref_acceptable_decrease_move)
# -- end function
