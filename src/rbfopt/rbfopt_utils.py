"""Utility functions.

This module contains a number of subroutines that are used by the
other modules. In particular it contains most of the subroutines that
do the calculations using numpy, as well as utility functions for
various modules.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
(C) Copyright International Business Machines Corporation 2017.

"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import sys
import os
import math
import itertools
import warnings
import collections
import logging
import bisect
import numpy as np
import scipy.spatial as ss
import scipy.linalg as la
from scipy.special import xlogy
from rbfopt.rbfopt_settings import RbfoptSettings

def get_rbf_function(settings):
    """Return a radial basis function.

    Return the radial basis function appropriate function as indicated
    by the settings.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    Returns
    ---
    Callable[numpy.ndarray]
        A callable radial basis function that can be applied on floats
        and numpy.ndarray.

    """
    assert(isinstance(settings, RbfoptSettings))
    if (settings.rbf == 'cubic'):
        return _cubic
    elif (settings.rbf == 'thin_plate_spline'):
        return _thin_plate_spline
    elif (settings.rbf == 'linear'):
        return _linear
    elif (settings.rbf == 'multiquadric'):
        mq = _MultiquadricRbf(settings.rbf_shape_parameter)
        return mq._multiquadric
    elif (settings.rbf == 'gaussian'):
        gauss = _GaussianRbf(settings.rbf_shape_parameter)
        return gauss._gaussian

# -- List of radial basis functions
def _cubic(r):
    """Cubic RBF: :math: `f(x) = x^3`"""
    return r*r*r

def _thin_plate_spline(r):
    """Thin plate spline RBF: :math: `f(x) = x^2 \log x`"""
    return r*r*xlogy(np.sign(r), r)

def _linear(r):
    """Linear RBF: :math: `f(x) = x`"""
    return r

class _MultiquadricRbf:
    def __init__(self, gamma):
        self._gamma_sq = gamma*gamma

    def _multiquadric(self, r):
        return (r*r + self._gamma_sq)**0.5
# -- end class

class _GaussianRbf:
    def __init__(self, gamma):
        self._gamma = gamma

    def _gaussian(self, r):
        return np.exp(-self._gamma * r * r)
# -- end class

# -- end list of radial basis functions


def get_degree_polynomial(settings):
    """Compute the degree of the polynomial for the interpolant.

    Return the degree of the polynomial that should be used in the RBF
    expression to ensure unisolvence and convergence of the
    optimization method.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    Returns
    -------
    int
        Degree of the polynomial

    Raises
    ------
    ValueError
        If the matrix type is not implemented.
    """
    assert(isinstance(settings, RbfoptSettings))
    if (settings.rbf == 'cubic' or settings.rbf == 'thin_plate_spline'):
        return 1
    elif (settings.rbf == 'linear' or settings.rbf == 'multiquadric'):
        return 0
    elif (settings.rbf == 'gaussian'):
        return -1
    raise ValueError('Rbf "' + settings.rbf + '" not implemented yet')

# -- end function


def get_size_P_matrix(settings, n):
    """Compute size of the P part of the RBF matrix.

    Return the number of columns in the P part of the matrix [\Phi P;
    P^T 0] that is used through the algorithm.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. number of variables.

    Returns
    -------
    int
        Number of columns in the matrix

    Raises
    ------
    ValueError
        If the matrix type is not implemented.
    """
    assert(isinstance(settings, RbfoptSettings))
    if (settings.rbf == 'cubic' or settings.rbf == 'thin_plate_spline'):
        return n+1
    elif (settings.rbf == 'linear' or settings.rbf == 'multiquadric'):
        return 1
    elif (settings.rbf == 'gaussian'):
        return 0
    raise ValueError('Rbf "' + settings.rbf + '" not implemented yet')

# -- end function


def get_all_corners(var_lower, var_upper):
    """Compute all corner points of a box.

    Compute and return all the corner points of the given box. Note
    that this number is exponential in the dimension of the problem.

    Parameters
    ----------
    var_lower : 1D numpy.ndarray[float]
        List of lower bounds of the variables.

    var_upper : 1D numpy.ndarray[float]
        List of upper bounds of the variables.

    Returns
    -------
    2D numpy.ndarray[float]
        All the corner points.
    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(len(var_lower) == len(var_upper))

    n = len(var_lower)
    node_pos = np.empty([2**n, n], np.float_)
    i = 0
    # Generate all corners
    for corner in itertools.product('lu', repeat=len(var_lower)):
        for (j, bound) in enumerate(corner):
            if bound == 'l':
                node_pos[i, j] = var_lower[j]
            else:
                node_pos[i, j] = var_upper[j]
        i += 1

    return node_pos

# -- end function


def get_lower_corners(var_lower, var_upper):
    """Compute the lower corner points of a box.

    Compute a list of (n+1) corner points of the given box, where n is
    the dimension of the space. The selected points are the bottom
    left (i.e. corresponding to the origin in the 0-1 hypercube) and
    the n adjacent ones.

    Parameters
    ----------
    var_lower : 1D numpy.ndarray[float]
        List of lower bounds of the variables.

    var_upper : 1D numpy.ndarray[float]
        List of upper bounds of the variables.

    Returns
    -------
    2D numpy.ndarray[float]
        The lower corner points.
    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(len(var_lower) == len(var_upper))

    n = len(var_lower)

    # Make sure we copy the object instead of copying just a reference
    node_pos = np.tile(var_lower, (n + 1, 1))
    # Generate adjacent corners
    for i in range(n):
        node_pos[i + 1, i] = var_upper[i]

    return node_pos

# -- end function


def get_random_corners(var_lower, var_upper):
    """Compute some randomly selected corner points of the box.

    Compute a list of (n+1) corner points of the given box, where n is
    the dimension of the space. The selected points are picked
    randomly.

    Parameters
    ----------
    var_lower : 1D numpy.ndarray[float]
        List of lower bounds of the variables.

    var_upper : 1D numpy.ndarray[float]
        List of upper bounds of the variables.

    Returns
    -------
    2D numpy.ndarray[float]
        A List of random corner points.
    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(len(var_lower) == len(var_upper))

    n = len(var_lower)
    limits = np.vstack((var_upper, var_lower))
    node_pos = np.atleast_2d(limits[np.random.randint(2, size=n), 
                                    np.arange(n)])
    while (len(node_pos) < n+1):
        point = limits[np.random.randint(2, size=n), np.arange(n)]
        if (get_min_distance(point, node_pos) > 0):
            node_pos = np.vstack((node_pos, point))

    return np.array(node_pos, np.float_)

# -- end function


def get_uniform_lhs(n, num_samples):
    """Generate random Latin Hypercube samples.

    Generate points using Latin Hypercube sampling from the uniform
    distribution in the unit hypercube.

    Parameters
    ----------
    n : int
        Dimension of the space, i.e. number of variables.

    num_samples : num_samples
        Number of samples to be generated.

    Returns
    -------
    2D numpy.ndarray[float]
        A list of n-dimensional points in the unit hypercube.
    """
    assert(n >= 0)
    assert(num_samples >= 0)

    # Generate integer LH in [0, num_samples]
    int_lh = np.array([np.random.permutation(num_samples)
                       for i in range(n)], np.float_).T
    # Map integer LH back to unit hypercube, and perturb points so that
    # they are uniformly distributed in the corresponding intervals
    lhs = (np.random.rand(num_samples, n) + int_lh) / num_samples

    return lhs

# -- end function


def get_lhd_maximin_points(var_lower, var_upper, integer_vars,
                           sample_size, num_trials=50):
    """Compute a latin hypercube design with maximin distance.

    Compute an array of points in the given box, where n is the
    dimension of the space. The selected points are picked according
    to a random latin hypercube design with maximin distance
    criterion.

    Parameters
    ----------
    var_lower : 1D numpy.ndarray[float]
        List of lower bounds of the variables.

    var_upper : 1D numpy.ndarray[float]
        List of upper bounds of the variables.

    integer_vars : 1D numpy.ndarray[int]
        A List containing the indices of the integrality constrained
        variables. If empty, all variables are assumed to be
        continuous.

    sample_size : int
        Number of points to sample.

    num_trials : int
        Maximum number of generated LHs to choose from.

    Returns
    -------
    2D numpy.ndarray[float]
        List of points in the latin hypercube design.

    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(np.issubdtype(var_lower.dtype, np.floating))
    assert(np.issubdtype(var_upper.dtype, np.floating))
    assert(len(var_lower) == len(var_upper))

    n = len(var_lower)
    if (n == 1):
        # For unidimensional problems, simply take the two endpoints
        # of the interval as starting points
        return np.vstack((var_lower, var_upper))
    # Otherwise, generate a bunch of Latin Hypercubes, and rank them
    lhs = [get_uniform_lhs(n, sample_size) for i in range(num_trials)]
    # Indices of upper triangular matrix (without the diagonal)
    indices = np.triu_indices(sample_size, 1)
    # Compute distance matrix of points to themselves, get upper
    # triangular part of the matrix, and get minimum
    dist_values = [np.amin(ss.distance.cdist(mat, mat)[indices])
                   for mat in lhs]
    lhd = lhs[dist_values.index(max(dist_values))]
    bound_shift = np.zeros(n)
    if (len(integer_vars)):
        # Expand bounds to ensure corner cases are equally selected
        bound_shift[integer_vars] = 0.499
    node_pos = (lhd * (2*bound_shift + var_upper - var_lower) +
                var_lower - bound_shift)

    return node_pos

# -- end function


def get_lhd_corr_points(var_lower, var_upper, integer_vars,
                        sample_size, num_trials=50):
    """Compute a latin hypercube design with min correlation.

    Compute a list of points in the given box, where n is the
    dimension of the space. The selected points are picked according
    to a random latin hypercube design with minimum correlation
    criterion. 

    Parameters
    ----------
    var_lower : 1D numpy.ndarray[float]
        List of lower bounds of the variables.

    var_upper : 1D numpy.ndarray[float]
        List of upper bounds of the variables.

    integer_vars : 1D numpy.ndarray[int]
        A List containing the indices of the integrality constrained
        variables. If empty, all variables are assumed to be
        continuous.

    sample_size : int
        Number of points to sample.

    num_trials : int
        Maximum number of generated LHs to choose from.

    Returns
    -------
    2D numpy.ndarray[float]
        List of points in the latin hypercube design.

    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(np.issubdtype(var_lower.dtype, np.floating))
    assert(np.issubdtype(var_upper.dtype, np.floating))
    assert(len(var_lower) == len(var_upper))

    n = len(var_lower)
    if (n == 1):
        # For unidimensional problems, simply take the two endpoints
        # of the interval as starting points
        return np.vstack((var_lower, var_upper))
    # Otherwise, generate a bunch of Latin Hypercubes, and rank them
    lhs = [get_uniform_lhs(n, sample_size) for i in range(num_trials)]
    # Indices of upper triangular matrix (without the diagonal)
    indices = np.triu_indices(n, 1)
    # Compute correlation matrix of points to themselves, get upper
    # triangular part of the matrix, and get minimum
    corr_values = [abs(np.amax(np.corrcoef(mat, rowvar = 0)[indices]))
                   for mat in lhs]
    lhd = lhs[corr_values.index(min(corr_values))]
    bound_shift = np.zeros(n)
    if (len(integer_vars)):
        # Expand bounds to ensure corner cases are equally selected
        bound_shift[integer_vars] = 0.499
    node_pos = (lhd * (2*bound_shift + var_upper - var_lower) +
                var_lower - bound_shift)

    return node_pos

# -- end function

def get_num_init_samples(settings, n):
    """Get number of initial sample points for a LHD strategy.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. number of variables.

    Returns
    -------
    int
        Number of samples, depending on problem dimension and
        settings.

    """
    assert(isinstance(settings, RbfoptSettings))
    if (settings.num_cpus > 1):
        return int(max(2, round((n + 1) * settings.init_sample_fraction *
                                (1 + settings.num_cpus *
                                 settings.init_sample_increase_parallel))))
    else:
        return int(max(2, round((n + 1) * settings.init_sample_fraction)))
# -- end function

def get_min_num_init_samples_parallel(settings, n):
    """Get the minimum number of initial sample points for paralell.

    In parallel optimization, when generating a lating hypercube
    design we may use more initial samples than strictly
    necessary. This function returns the desired/target number of
    initial samples before optimization can start, i.e., the strictly
    necessary number to create an RBF model with the given settings.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. number of variables.

    Returns
    -------
    int
        Number of samples, depending on problem dimension and
        settings.

    """
    assert(isinstance(settings, RbfoptSettings))
    return int(max(2, round((n + 1) * settings.init_sample_fraction)))

# -- end function


def initialize_nodes(settings, var_lower, var_upper, integer_vars,
                     categorical_info):
    """Compute the initial sample points.

    Compute an initial list of nodes using the initialization strategy
    indicated in the algorithmic settings.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    var_lower : 1D numpy.ndarray[float]
        List of lower bounds of the variables.

    var_upper : 1D numpy.ndarray[float]
        List of upper bounds of the variables.

    integer_vars : 1D numpy.ndarray[int]
        A List containing the indices of the integrality constrained
        variables. If empty, all variables are assumed to be
        continuous.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    Returns
    -------
    2D numpy.ndarray[float]
        Matrix containing at least n+1 corner points, one for each
        row, where n is the dimension of the space. The number and
        position of points depends on the chosen strategy.

    Raises
    ------
    RuntimeError
        If a set of feasible and linearly independent sample points
        cannot be computed within the prescribed number of iterations.

    """
    assert(isinstance(settings, RbfoptSettings))
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(isinstance(integer_vars, np.ndarray))
    assert(len(var_lower) == len(var_upper))
    assert(categorical_info is None or len(categorical_info)==3)

    sample_size = get_num_init_samples(settings, len(var_lower))

    if (categorical_info is not None and categorical_info[2] and
        settings.init_strategy in ['lhd_maximin', 'lhd_corr']):
        # Map bounds and integer variables
        var_lower = compress_categorical_bounds(var_lower,
                                                *categorical_info)
        var_upper = compress_categorical_bounds(var_upper,
                                                *categorical_info)
        integer_vars = compress_categorical_integer_vars(integer_vars,
                                                         *categorical_info)

    if (settings.init_include_midpoint):
        midpoint = (var_lower + var_upper)/2
        midpoint[integer_vars] = np.around(midpoint[integer_vars])
    # We must make sure points are linearly independent; if they are
    # not, we perform a given number of iterations
    dependent = True
    itercount = 0
    while (dependent and itercount < settings.max_random_init):
        itercount += 1
        if (settings.init_strategy == 'all_corners'):
            nodes = get_all_corners(var_lower, var_upper)
        elif (settings.init_strategy == 'lower_corners'):
            nodes = get_lower_corners(var_lower, var_upper)
        elif (settings.init_strategy == 'rand_corners'):
            nodes = get_random_corners(var_lower, var_upper)
        elif (settings.init_strategy == 'lhd_maximin'):
            nodes = get_lhd_maximin_points(var_lower, var_upper, integer_vars,
                                           sample_size)
        elif (settings.init_strategy == 'lhd_corr'):
            nodes = get_lhd_corr_points(var_lower, var_upper, integer_vars,
                                        sample_size)

        if (len(integer_vars)):
            nodes[:, integer_vars] = np.around(nodes[:, integer_vars])

        if (settings.init_include_midpoint and
            get_min_distance(midpoint, nodes) > settings.min_dist):
            nodes = np.vstack((nodes, midpoint))

        if (categorical_info is not None and categorical_info[2] and
            settings.init_strategy in ['lhd_maximin', 'lhd_corr']):
            # There are categorical variables. Unpack nodes in the new
            # representation.
            nodes = expand_categorical_vars(nodes, *categorical_info)
            norms = la.norm(nodes, axis=1)
            # Columns of categorical variables are going to be
            # linearly dependent because categorical variables add up
            # to 1. Therefore, exclude known linearly dependent
            # columns from th e calculations.
            columns = np.array([True] * nodes.shape[1])
            for index, lower, expansion in categorical_info[2]:
                columns[expansion[-1]] = False
            U, s, V = np.linalg.svd(
                nodes[norms > settings.eps_zero][:, columns])
        else:
            norms = la.norm(nodes, axis=1)
            U, s, V = np.linalg.svd(nodes[norms > settings.eps_zero])
        if (min(s) > settings.eps_linear_dependence):
            dependent = False

    if (itercount == settings.max_random_init):
        raise RuntimeError('Exceeded number of random initializations')

    return nodes

# -- end function


def round_integer_vars(point, integer_vars):
    """Round a point to the closest integer.

    Round the values of the integer-constrained variables to the
    closest integer value. The values are rounded in-place.

    Parameters
    ----------
    point : 1D numpy.ndarray[float]
        The point to be rounded.
    integer_vars : 1D numpy.ndarray[int]
        A list of indices of integer variables.
    """
    assert(isinstance(point, np.ndarray))
    assert(isinstance(integer_vars, np.ndarray))
    if (len(integer_vars)):
        point[integer_vars] = np.around(point[integer_vars])

# -- end function


def round_integer_bounds(var_lower, var_upper, integer_vars):
    """Round the variable bounds to integer values.

    Round the values of the integer-constrained variable bounds, in
    the usual way: lower bounds are rounded up, upper bounds are
    rounded down.

    Parameters
    ----------
    var_lower : 1D numpy.ndarray[float]
        List of lower bounds of the variables.

    var_upper : 1D numpy.ndarray[float]
        List of upper bounds of the variables.

    integer_vars : 1D numpy.ndarray[int]
        A list containing the indices of the integrality constrained
        variables. If empty, all variables are assumed to be
        continuous.
    """
    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))
    assert (isinstance(integer_vars, np.ndarray))
    if (len(integer_vars)):
        assert(len(var_lower) == len(var_upper))
        assert(max(integer_vars) < len(var_lower))
        var_lower[integer_vars] = np.floor(var_lower[integer_vars])
        var_upper[integer_vars] = np.ceil(var_upper[integer_vars])

# -- end function

def expand_categorical_vars(points, categorical, not_categorical,
                            categorical_expansion):
    """Expand points in original space to extended space.
    
    Expands points with integer encoding of categorical variables to
    points with unary encoding of categorical variables.

    Parameters
    ----------
    
    points : 2D numpy.ndarray[float]
        Matrix of points we want to expand.

    categorical : 1D numpy.ndarray[int]
        Array of indices of categorical variables in original space.

    not_categorical : 1D numpy.ndarray[int]
        Array of indices of not categorical variables in original space.

    categorical_expansion : List[(int, float, 1D numpy.ndarray[int])]
        Expansion of original categorical variables into binaries.

    Returns
    -------
    2D numpy.ndarray[float]
        Matrix containing points in unary encoding.
    """
    assert(isinstance(points, np.ndarray))
    assert(points.ndim==2)
    assert(isinstance(categorical, np.ndarray))
    assert(isinstance(not_categorical, np.ndarray))
    assert(len(categorical_expansion)==len(categorical))
    assert(points.shape[1]==(len(categorical)+len(not_categorical)))
    num_rows = points.shape[0]
    if (len(not_categorical)):
        expanded = points[:, not_categorical]
    else:
        expanded = np.zeros(shape=(num_rows,0))
    for index, var_lower, expansion in categorical_expansion:
        to_append = np.zeros(shape=(num_rows, len(expansion)))
        for j in range(len(expansion)):
            to_append[np.where(points[:, index] == (var_lower + j))[0], j] = 1
        expanded = np.hstack((expanded, to_append))
    return expanded
# -- end function

def compress_categorical_vars(points, categorical, not_categorical,
                              categorical_expansion):
    """Compress points in extended space to original space.
    
    Compress points with unary encoding of categorical variables to
    points with integer encoding of categorical variables.

    Parameters
    ----------
    
    points : 2D numpy.ndarray[float]
        Matrix of points we want to compress.

    categorical : 1D numpy.ndarray[int]
        Array of indices of categorical variables in original space.

    not_categorical : 1D numpy.ndarray[int]
        Array of indices of not categorical variables in original space.

    categorical_expansion : List[(int, float, 1D numpy.ndarray[int])]
        Expansion of original categorical variables into binaries.

    Returns
    -------
    2D numpy.ndarray[float]
        Matrix containing points in integer encoding.
    """
    assert(isinstance(points, np.ndarray))
    assert(points.ndim==2)
    assert(isinstance(categorical, np.ndarray))
    assert(isinstance(not_categorical, np.ndarray))
    assert(len(categorical_expansion)==len(categorical))
    num_rows = points.shape[0]
    compressed = np.zeros(
        shape=(num_rows, len(categorical)+len(not_categorical)))
    compressed[:, not_categorical] = points[:, :len(not_categorical)]
    for index, var_lower, expansion in categorical_expansion:
        for j in range(len(expansion)):
            compressed[np.where(
                points[:, expansion[j]] == 1)[0], index] = var_lower + j
    return compressed
# -- end function

def compress_categorical_bounds(bounds, categorical, not_categorical,
                                categorical_expansion):
    """Compress bounds vector in extended space to original space.
    
    Compress bounds for points with unary encoding of categorical
    variables to bounds for points with integer encoding of
    categorical variables.

    Parameters
    ----------
    
    bounds : 1D numpy.ndarray[float]
        Array of bounds we want to compress.

    categorical : 1D numpy.ndarray[int]
        Array of indices of categorical variables in original space.

    not_categorical : 1D numpy.ndarray[int]
        Array of indices of not categorical variables in original space.

    categorical_expansion : List[(int, float, 1D numpy.ndarray[int])]
        Expansion of original categorical variables into binaries.

    Returns
    -------
    1D numpy.ndarray[float]
        Bounds for points in integer encoding.

    """
    assert(isinstance(bounds, np.ndarray))
    assert(isinstance(categorical, np.ndarray))
    assert(isinstance(not_categorical, np.ndarray))
    assert(len(categorical_expansion)==len(categorical))
    compressed = np.zeros(len(categorical)+len(not_categorical))
    if (len(not_categorical)):
        compressed[not_categorical] = bounds[:len(not_categorical)]
    for index, var_lower, expansion in categorical_expansion:
        # If the bounds are zero in expanded space, then we are
        # working on the lower bounds
        if (bounds[expansion[0]] == 0):
            compressed[index] = var_lower
        else:
            compressed[index] = var_lower + len(expansion) - 1
    return compressed
# -- end function

def compress_categorical_integer_vars(integer_vars, categorical,
                                      not_categorical,
                                      categorical_expansion):
    """Compress integer vars vector in extended space to original space.
    
    Compress the vector of integer vars from unary encoding of
    categorical variables to integer encoding of categorical
    variables.

    Parameters
    ----------
    
    integer_vars : 1D numpy.ndarray[int]
        Array of integer vars to compress.

    categorical : 1D numpy.ndarray[int]
        Array of indices of categorical variables in original space.

    not_categorical : 1D numpy.ndarray[int]
        Array of indices of not categorical variables in original space.

    categorical_expansion : List[(int, float, 1D numpy.ndarray[int])]
        Expansion of original categorical variables into binaries.

    Returns
    -------
    1D numpy.ndarray[int]
        Indices of integer variables in integer encoding.

    """
    assert(isinstance(integer_vars, np.ndarray))
    assert(isinstance(categorical, np.ndarray))
    assert(isinstance(not_categorical, np.ndarray))
    assert(len(categorical_expansion)==len(categorical))
    compressed = np.array(
        [i for i in integer_vars if i < len(not_categorical)],
        dtype=np.int_)
    for index, var_lower, expansion in categorical_expansion:
        # Insert index in the right place, after shifting all larger ones
        compressed[compressed >= index] += 1
        if (len(compressed) == 0 or index > compressed[-1]):
            compressed = np.append(compressed, index)
        else:
            compressed = np.insert(
                compressed, np.where(compressed > index)[0][0], index)
    return compressed
# -- end function

def norm(p):

    """Compute the L2-norm of a vector

    Compute the L2 (Euclidean) norm.

    Parameters
    ----------
    p : 1D numpy.ndarray[float]
        The point whose norm should be computed.

    Returns
    -------
    float
        The norm of the point.
    """
    assert(isinstance(p, np.ndarray))

    return np.sqrt(np.dot(p, p))

# -- end function


def distance(p1, p2):
    """Compute Euclidean distance between two points.

    Compute Euclidean distance between two points.

    Parameters
    ----------
    p1 : 1D numpy.ndarray[float]
        First point.
    p2 : 1D numpy.ndarray[float]
        Second point.

    Returns
    -------
    float
        Euclidean distance.
    """
    assert(isinstance(p1, np.ndarray))
    assert(isinstance(p2, np.ndarray))
    assert(len(p1) == len(p2))

    p = p1 - p2

    return np.sqrt(np.dot(p, p))

# -- end function


def get_min_distance(point, other_points):
    """Compute minimum distance from a set of points.

    Compute the minimum Euclidean distance between a given point and a
    list of points.

    Parameters
    ----------
    point : 1D numpy.ndarray[float]
        The point we compute the distances from.

    other_points : 2D numpy.ndarray[float]
        The list of points we want to compute the distances to.

    Returns
    -------
    float
        Minimum distance between point and the other_points.
    """
    assert(isinstance(point, np.ndarray))
    assert(isinstance(other_points, np.ndarray))
    assert(point.size)
    assert(other_points.size)

    # Create distance matrix
    dist = ss.distance.cdist(np.atleast_2d(point), other_points)
    return np.amin(dist, 1)
# -- end function


def get_min_distance_and_index(point, other_points):
    """Compute the distance and index of the point with minimum distance.

    Compute the distance value and the index of the point in a matrix
    that achieves minimum Euclidean distance to a given point.

    Parameters
    ----------
    point : 1D numpy.ndarray[float]
        The point we compute the distances from.

    other_points : 2D numpy.ndarray[float]
        The list of points we want to compute the distances to.

    Returns
    -------
    (float, int)
        The distance value and the index of the point in other_points
        that achieved minimum distance from point.

    """
    assert (isinstance(point, np.ndarray))
    assert (isinstance(other_points, np.ndarray))
    assert(point.size)
    assert(other_points.size)

    dist = ss.distance.cdist(np.atleast_2d(point), other_points)
    index = np.argmin(dist, 1)[0]
    return (dist[0, index], index)

# -- end function

def get_max_distance(point, other_points):
    """Compute maximum distance from a set of points.

    Compute the maximum Euclidean distance between a given point and a
    list of points.

    Parameters
    ----------
    point : 1D numpy.ndarray[float]
        The point we compute the distances from.

    other_points : 2D numpy.ndarray[float]
        The list of points we want to compute the distances to.

    Returns
    -------
    float
        Maximum distance between point and the other_points.
    """
    assert(isinstance(point, np.ndarray))
    assert(isinstance(other_points, np.ndarray))
    assert(point.size)
    assert(other_points.size)

    # Create distance matrix
    dist = ss.distance.cdist(np.atleast_2d(point), other_points)
    return np.amax(dist, 1)
# -- end function


def get_max_distance_and_index(point, other_points):
    """Compute the distance and index of the point with maximum distance.

    Compute the distance value and the index of the point in a matrix
    that achieves maximum Euclidean distance to a given point.

    Parameters
    ----------
    point : 1D numpy.ndarray[float]
        The point we compute the distances from.

    other_points : 2D numpy.ndarray[float]
        The list of points we want to compute the distances to.

    Returns
    -------
    (float, int)
        The distance value and the index of the point in other_points
        that achieved maximum distance from point.

    """
    assert (isinstance(point, np.ndarray))
    assert (isinstance(other_points, np.ndarray))
    assert(point.size)
    assert(other_points.size)

    dist = ss.distance.cdist(np.atleast_2d(point), other_points)
    index = np.argmax(dist, 1)[0]
    return (dist[0, index], index)

# -- end function


def bulk_get_min_distance(points, other_points):
    """Get the minimum distances between two sets of points.

    Compute the minimum distance of each point in the first set to the
    points in the second set. This is faster than using
    get_min_distance repeatedly, for large sets of points.

    Parameters
    ----------
    points : 2D numpy.ndarray[float]
        The points in R^n that we compute the distances from.

    other_points : 2D numpy.ndarray[float]
        The list of points we want to compute the distances to.

    Returns
    -------
    1D numpy.ndarray[float]
        Minimum distance between each point in points and the
        other_points.

    See also
    --------
    get_min_distance()
    """
    assert(isinstance(points, np.ndarray))
    assert(isinstance(other_points, np.ndarray))
    assert(points.size)
    assert(other_points.size)
    assert(len(points[0]) == len(other_points[0]))

    # Create distance matrix
    dist = ss.distance.cdist(points, other_points)
    return np.amin(dist, 1)

# -- end function


def get_rbf_matrix(settings, n, k, node_pos):
    """Compute the matrix for the RBF system.

    Compute the matrix A = [Phi P; P^T 0] of equation (3) in the paper
    by Costa and Nannicini.

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

    Returns
    -------
    numpy.ndarray
        The matrix A = [Phi P; P^T 0].

    Raises
    ------
    ValueError
        If the type of RBF function is not supported.
    """
    assert(isinstance(node_pos, np.ndarray))
    assert(len(node_pos) == k)
    assert(isinstance(settings, RbfoptSettings))

    rbf = get_rbf_function(settings)
    p = get_size_P_matrix(settings, n)
    # Create matrix P.
    if (p == n + 1):
        # Keep the node coordinates and append a 1.
        # P is ((k) x (n+1)), PTr is its transpose
        P = np.insert(node_pos, n, 1, axis=1)
        PTr = P.T
    elif (p == 1):
        # P is an all-one vector of size ((k) x (1))
        P = np.ones([k, 1])
        PTr = P.T
    elif (p == 0):
        pass
    else:
        raise ValueError('Rbf "' + settings.rbf + '" not implemented yet')

    # Now create matrix Phi. Phi is ((k) x (k))
    dist = ss.distance.cdist(node_pos, node_pos)
    Phi = rbf(dist)

    # Put together to obtain [Phi P; P^T 0].
    if (p > 0):
        A = np.vstack((np.hstack((Phi, P)),
                       np.hstack((PTr, np.zeros((p, p))))))
    else:
        A = Phi
    Amat = np.array(A)

    # Zero out tiny elements
    Amat[np.abs(Amat) < settings.eps_zero] = 0

    return Amat
# -- end function


def get_matrix_inverse(settings, n, k, Amat, categorical_info=None):
    """Compute the inverse of an RBF matrix.

    Compute the inverse of a given RBF matrix, zeroing out small
    coefficients to improve sparsity. If there are categorical
    variables and this is the RBF matrix, the matrix may be singular
    therefore we only invert the nonsingular part.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    Amat : numpy.ndarray
        The matrix to invert.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).


    Returns
    -------
    numpy.ndarray
        The matrix Amat^{-1}.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix cannot be inverted for numerical reasons.

    """
    assert(isinstance(settings, RbfoptSettings))
    assert(isinstance(Amat, np.ndarray))
    p = get_size_P_matrix(settings, n)
    assert(Amat.shape == (k+p, k+p))

    try:
        if (p == n + 1 and categorical_info is not None and
            categorical_info[2]):
            # In this case we know some columns for categorical
            # variables are linearly dependent. Eliminate them from
            # the system.
            columns = np.array([True] * Amat.shape[1])
            for index, lower, expansion in categorical_info[2]:
                columns[k + expansion[-1]] = False
            Amat = Amat[:, columns][columns, :]

        Amatinv = np.linalg.inv(Amat)

        if (p == n + 1 and categorical_info is not None and
            categorical_info[2]):
            new_Amatinv = np.zeros(shape=(len(columns),len(columns)))
            new_Amatinv[np.outer(columns, columns)] = np.ravel(Amatinv)
            Amatinv = new_Amatinv
    except np.linalg.LinAlgError as e:
        if (settings.debug):
            print('Exception raised trying to invert the RBF matrix',
                  file=sys.stderr)
            print(e, file=sys.stderr)
        raise e

    # Zero out tiny elements of the inverse -- this is potentially
    # dangerous as the product between Amat and Amatinv may not be the
    # identity, but if the zero tolerance is chosen not too large,
    # this should help the optimization process
    Amatinv[np.abs(Amatinv) < settings.eps_zero] = 0

    return Amatinv

# -- end function


def get_rbf_coefficients(settings, n, k, Amat, node_val,
                         categorical_info=None):
    """Compute the coefficients of the RBF interpolant.

    Solve a linear system to compute the coefficients of the RBF
    interpolant.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`.
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    Amat : 2D numpy.ndarray[float]
        Matrix [Phi P; P^T 0] defining the linear system. Must be a
        square matrix of appropriate size.

    node_val : 1D numpy.ndarray[float]
        List of values of the function at the nodes.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    Returns
    -------
    (1D numpy.ndarray[float], 1D numpy.ndarray[float])
        Lambda coefficients (for the radial basis functions), and h
        coefficients (for the polynomial).
    """
    assert(len(np.atleast_1d(node_val)) == k)
    assert(isinstance(settings, RbfoptSettings))
    assert(isinstance(Amat, np.ndarray))
    assert(isinstance(node_val, np.ndarray))
    p = get_size_P_matrix(settings, n)
    assert(Amat.shape == (k+p, k+p))
    rhs = np.append(node_val, np.zeros(p))
    try:
        if (p == n + 1 and categorical_info is not None and
            categorical_info[2]):
            # In this case we know some columns for categorical
            # variables are linearly dependent. Eliminate them from
            # the system.
            columns = np.array([True] * Amat.shape[1])
            for index, lower, expansion in categorical_info[2]:
                columns[k + expansion[-1]] = False
            Amat = Amat[:, columns][columns, :]
            rhs = rhs[columns]
            
        # Get solution
        solution = np.linalg.solve(Amat, rhs)
        # Postprocess
        if (p == n + 1 and categorical_info is not None and
            categorical_info[2]):
            new_solution = np.zeros(len(columns))
            new_solution[columns] = solution
            solution = new_solution
    except np.linalg.LinAlgError as e:
        if (settings.debug):
            print('Exception raised in the solution of the RBF linear system',
                  file=sys.stderr)
            print('Exception details:', file=sys.stderr)
            print(e, file=sys.stderr)
        raise e

    return (solution[0:k], solution[k:])

# -- end function

def get_rbf_coefficients_underdet(settings, n, k, Amat, node_val,
                                  categorical_info=None):
    """Compute coefficients of RBF interpolant (underdetermined).

    Solve an underdetermined linear system to compute the coefficients
    of the RBF interpolant.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`.
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    Amat : 2D numpy.ndarray[float]
        Matrix [Phi P; P^T 0] defining the linear system. Must be a
        square matrix of appropriate size.

    node_val : 1D numpy.ndarray[float]
        List of values of the function at the nodes.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    Returns
    -------
    (1D numpy.ndarray[float], 1D numpy.ndarray[float])
        Lambda coefficients (for the radial basis functions), and h
        coefficients (for the polynomial).

    """
    assert(len(np.atleast_1d(node_val)) == k)
    assert(isinstance(settings, RbfoptSettings))
    assert(isinstance(Amat, np.ndarray))
    assert(isinstance(node_val, np.ndarray))
    p = get_size_P_matrix(settings, n)
    assert(Amat.shape == (k+p, k+p))
    rhs = np.append(node_val, np.zeros(p))
    try:
        if (p == n + 1 and categorical_info is not None and
            categorical_info[2]):
            # In this case we know some columns for categorical
            # variables are linearly dependent. Eliminate them from
            # the system.
            columns = np.array([True] * Amat.shape[1])
            for index, lower, expansion in categorical_info[2]:
                columns[k + expansion[-1]] = False
            Amat = Amat[:, columns][columns, :]
            rhs = rhs[columns]
        solution, res, rank, svd = np.linalg.lstsq(Amat, rhs, rcond=-1)
        # Postprocess
        if (p == n + 1 and categorical_info is not None and
            categorical_info[2]):
            new_solution = np.zeros(len(columns))
            new_solution[columns] = solution
            solution = new_solution
    except np.linalg.LinAlgError as e:
        if (settings.debug):
            print('Exception raised in the solution of the ' +
                  'underdetermined RBF linear system',
                  file=sys.stderr)
            print('Exception details:', file=sys.stderr)
            print(e, file=sys.stderr)
        raise e

    return (solution[0:k], solution[k:])
# -- end function


def evaluate_rbf(settings, point, n, k, node_pos, rbf_lambda, rbf_h):
    """Evaluate the RBF interpolant at a given point.

    Evaluate the RBF interpolant at a given point.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`.
        Global and algorithmic settings.

    point : 1D numpy.ndarray[float]
        The point in R^n where we want to evaluate the interpolant.

    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the interpolation points.

    rbf_lambda : 1D numpy.ndarray[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : 1D numpy.ndarray[float]
        The h coefficients of the RBF interpolant, corresponding to he
        polynomial. List of dimension given by get_size_P_matrix().

    Returns
    -------
    float
        Value of the RBF interpolant at the given point.
    """
    assert(isinstance(point, np.ndarray))
    assert(isinstance(node_pos, np.ndarray))
    assert(isinstance(rbf_lambda, np.ndarray))
    assert(isinstance(rbf_h, np.ndarray))
    assert(len(point) == n)
    assert(len(rbf_lambda) == k)
    assert(len(node_pos) == k)
    assert(isinstance(settings, RbfoptSettings))
    p = get_size_P_matrix(settings, n)
    assert(len(rbf_h) == p)

    rbf_function = get_rbf_function(settings)

    # Formula:
    # \sum_{i=1}^k \lambda_i \phi(\|x - x_i\|) + h^T (x 1)
    part1 = math.fsum(rbf_lambda[i] *
                      rbf_function(distance(point, node_pos[i]))
                      for i in range(k))
    part2 = math.fsum(rbf_h[i]*point[i] for i in range(p-1))
    return math.fsum([part1, part2, rbf_h[-1] if (p > 0) else 0.0])

# -- end function

def bulk_evaluate_rbf(settings, points, n, k, node_pos, rbf_lambda, rbf_h,
                      return_distances='no'):
    """Evaluate the RBF interpolant at all points in a given list.

    Evaluate the RBF interpolant at all points in a given list. This
    version uses numpy and should be faster than individually
    evaluating the RBF at each single point, provided that the list of
    points is large enough. It also computes the distance or the
    minimum distance of each point from the interpolation nodes, if
    requested, since this comes almost for free.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`.
        Global and algorithmic settings.

    points : 2D numpy.ndarray[float]
        The list of points in R^n where we want to evaluate the
        interpolant.

    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the interpolation points.

    rbf_lambda : 1D numpy.ndarray[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : 1D numpy.ndarray[float]
        The h coefficients of the RBF interpolant, corresponding to he
        polynomial. List of dimension given by get_size_P_matrix().

    return_distances : string
        If 'no', do nothing. If 'min', return the minimum distance of
        each point to interpolation nodes. If 'all', return the full
        distance matrix to the interpolation nodes.

    Returns
    -------
    1D numpy.ndarray[float] or (1D numpy.ndarray[float], 1D numpy.ndarray[float])
        Value of the RBF interpolant at each point; if
        compute_min_dist is True, additionally returns the minimum
        distance of each point from the interpolation nodes.

    """
    assert(isinstance(points, np.ndarray))
    assert(isinstance(node_pos, np.ndarray))
    assert(isinstance(rbf_lambda, np.ndarray))
    assert(isinstance(rbf_h, np.ndarray))
    assert(points.size)
    assert(len(rbf_lambda) == k)
    assert(len(node_pos) == k)
    assert(isinstance(settings, RbfoptSettings))
    p = get_size_P_matrix(settings, n)
    assert(len(rbf_h) == p)

    rbf_function = get_rbf_function(settings)
    # Formula:
    # \sum_{i=1}^k \lambda_i \phi(\|x - x_i\|) + h^T (x 1)

    # Create distance matrix
    dist_mat = ss.distance.cdist(points, node_pos)
    # Evaluate radial basis function on each distance
    part1 = np.dot(rbf_function(dist_mat), rbf_lambda)
    if (get_degree_polynomial(settings) == 1):
        part2 = np.dot(points, rbf_h[:-1])
    else:
        part2 = np.zeros(len(points))
    part3 = rbf_h[-1] if (p > 0) else 0.0
    if (return_distances == 'min'):
        return ((part1 + part2 + part3), (np.amin(dist_mat, 1)))
    elif (return_distances == 'all'):
        return ((part1 + part2 + part3), dist_mat)
    else:
        return (part1 + part2 + part3)

# -- end function

def bulk_compute_and_evaluate_rbf(settings, points, n, k, node_pos,
                                  node_val, fmin, fmax, node_err_bounds,
                                  categorical_info):
    """Compute interpolant and estimate the value of given points.

    Evaluate a set of points using an RBF interpolated computed from a
    different set of given points. This function is different from
    other (similar) functions because it first computes the
    interpolant from a set of points, then uses the interpolant to
    evaluate at a different set of points, and discards the
    interpolant.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`.
        Global and algorithmic settings.

    points : 2D numpy.ndarray[float]
        The list of points in R^n where we want to evaluate the
        interpolant.

    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the interpolation points.

    node_val : 1D numpy.ndarray[float]
       List of function values at the interpolation nodes.

    fmin : float
       Minimum function value found so far.

    fmax : float
       Maximum function value found so far.

    node_err_bounds : 2D numpy.ndarray[float]
        The lower and upper variation of the function value for the
        nodes in node_pos. The variation is assumed 0 for nodes
        evaluated in accurate mode.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    Returns
    -------
    1D numpy.ndarray[float]
        Value of the RBF interpolant at each point.

    """
    assert(isinstance(points, np.ndarray))
    assert(isinstance(node_pos, np.ndarray))
    assert(len(node_pos) == k)
    assert(isinstance(node_val, np.ndarray))
    assert(len(node_val) == k)
    assert(isinstance(node_err_bounds, np.ndarray))
    assert(len(node_err_bounds) == k)
    assert(isinstance(settings, RbfoptSettings))
    assert(categorical_info is None or len(categorical_info)==3)

    if (len(points) == 0):
        return np.array([])

    tfv = transform_function_values(settings, node_val, fmin, fmax,
                                    node_err_bounds)
    (scaled_node_val, scaled_fmin, scaled_fmax, 
     scaled_err_bounds, rescale_function) = tfv
    
    Amat = get_rbf_matrix(settings, n, k, node_pos)

    # Compute RBF interpolant at current stage
    if (k < n+1):
        # Underdetermined RBF
        rbf_l, rbf_h = get_rbf_coefficients_underdet(
            settings, n, k, Amat, scaled_node_val,
            categorical_info)
    else:
        # Fully accurate RBF
        rbf_l, rbf_h = get_rbf_coefficients(
            settings, n, k, Amat, scaled_node_val,
            categorical_info)
    return bulk_evaluate_rbf(settings, points, n, k, node_pos,
                             rbf_l, rbf_h)
# -- end function

def compute_gap(settings, fmin):
    """Compute the optimality gap w.r.t. the target value.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
       Global and algorithmic settings.

    fmin : float
       Best known function value discovered so far. Note that this
       value should already take into account possible noise at the
       best point.

    Returns
    -------
    float
        The current optimality gap, i.e. relative distance from target
        value.

    """
    assert(isinstance(settings, RbfoptSettings))
    # Denominator of errormin
    gap_den = (abs(settings.target_objval)
               if (abs(settings.target_objval) >= settings.eps_zero)
               else 1.0)
    # Compute current minimum distance from the optimum
    gap = ((fmin - settings.target_objval) / gap_den)
    return gap
# -- end function


def transform_function_values(settings, node_val, fmin, fmax,
                              node_err_bounds):
    """Rescale function values.

    Rescale and adjust function values according to the chosen
    strategy and to the occurrence of large fluctuations (high
    dynamism). May not rescale at all if rescaling is off.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfoptSettings`
       Global and algorithmic settings.

    node_val : 1D numpy.ndarray[float]
       List of function values at the interpolation nodes.

    fmin : float
       Minimum function value found so far.

    fmax : float
       Maximum function value found so far.

    node_err_bounds : 2D numpy.ndarray[float]
        The lower and upper variation of the function value for the
        nodes in node_pos. The variation is assumed 0 for nodes
        evaluated in accurate mode.

    Returns
    -------
    (1D numpy.ndarray[float], float, float, 2D numpy.ndarray[float], 
     Callable[float])
        A tuple (scaled_function_values, scaled_fmin, scaled_fmax,
        scaled_error_bounds, rescale_function) containing a list of
        rescaled function values, the rescaled minimum, the rescaled
        maximum, the rescaled error bounds (one node per row), and a
        callable function to apply the same scaling to further
        function values if needed.

    Raises
    ------
    ValueError
        If the function scaling strategy requested is not implemented.

    """
    assert(isinstance(node_val, np.ndarray))
    assert(isinstance(node_err_bounds, np.ndarray))
    assert(isinstance(settings, RbfoptSettings))
    # Check dynamism: if too high, replace large function values with
    # the median or clip at maximum dynamism
    if (settings.dynamism_clipping != 'off' and
        ((abs(fmin) > settings.eps_zero and
          abs(fmax)/abs(fmin) > settings.dynamism_threshold) or
         (abs(fmin) <= settings.eps_zero and
          abs(fmax) > settings.dynamism_threshold))):
        if (settings.dynamism_clipping == 'median'):
            med = np.median(node_val)
            clip_val = np.clip(node_val, None, med)
            fmax = med
        elif (settings.dynamism_clipping == 'clip_at_dyn'):
            # We should not multiply by abs(fmin) if it is too small
            mult = abs(fmin) if (abs(fmin) > settings.eps_zero) else 1.0
            clip_val = np.clip(node_val, None, 
                               settings.dynamism_threshold*mult)
            fmax = settings.dynamism_threshold*mult
    else:
        clip_val = node_val

    if (settings.function_scaling == 'off'):
        # We make a copy because the caller may assume that
        return (clip_val, fmin, fmax, node_err_bounds, lambda x : x)
    elif (settings.function_scaling == 'affine'):
        # Compute denominator separately to make sure that it is not
        # zero. This may happen if the surface is "flat" after median
        # clipping.
        denom = (fmax - fmin) if (fmax - fmin > settings.eps_zero) else 1.0
        return ((clip_val - fmin)/denom, 0.0,
                1.0 if (fmax - fmin > settings.eps_zero) else 0.0,
                node_err_bounds/denom, lambda x : (x - fmin)/denom)
    elif (settings.function_scaling == 'log'):
        # Compute by how much we should translate to make all points >= 1
        shift = max(0.0, 1 - np.amin(node_val + node_err_bounds[:, 0]))
        if (shift > 0 and np.amin(node_val) / shift < settings.eps_zero):
            # If the node values are so small that they could be
            # absorbed by the shift and therefore become zero,
            # increase the shift.
            shift += 1/settings.eps_zero
        # Get the lower bound and the upper bound of the transformed
        # error bounds
        scaled_err_b = np.log((node_err_bounds.T + clip_val + shift) /
                              (clip_val + shift)).T
        return (np.log(clip_val + shift), np.log(fmin + shift), 
                np.log(fmax + shift), scaled_err_b,
                lambda x : np.log(x + shift) if (x + shift > 0) else x)
    else:
        raise ValueError('Function scaling "' + settings.function_scaling +
                         '" not implemented')

# -- end function


def transform_domain(settings, var_lower, var_upper, integer_vars,
                     point, reverse=False):
    """Rescale the domain.

    Rescale the function domain according to the chosen strategy.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    var_lower : 1D numpy.ndarray[float]
        List of lower bounds of the variables.

    var_upper : 1D numpy.ndarray[float]
        List of upper bounds of the variables.

    integer_vars : 1D numpy.ndarray[int]
        A List containing the indices of the integrality constrained
        variables. If empty, all variables are assumed to be
        continuous.

    point : 1D numpy.ndarray[float]
        Point in the domain to be rescaled.

    reverse : bool
        False if we transform from the original domain to the
        transformed space, True if we want to apply the reverse.

    Returns
    -------
    1D numpy.ndarray[float]
        Rescaled point.

    Raises
    ------
    ValueError
        If the requested rescaling strategy is not implemented.
    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(isinstance(point, np.ndarray))
    assert(isinstance(settings, RbfoptSettings))
    assert(len(var_lower) == len(var_upper))
    assert(len(var_lower) == len(point))

    if (settings.domain_scaling == 'off' or len(integer_vars)):
        # Make a copy because the caller may assume so
        return point.copy()
    elif (settings.domain_scaling == 'affine'):
        # Make an affine transformation to the unit hypercube
        if (reverse):
            return point * (var_upper - var_lower) + var_lower
        else:
            var_diff = var_upper-var_lower
            var_diff[var_diff == 0] = 1.0
            return (point - var_lower) / var_diff

    else:
        raise ValueError('Domain scaling "' + settings.domain_scaling +
                         '" not implemented')

# -- end function


def bulk_transform_domain(settings, var_lower, var_upper, integer_vars,
                          points, reverse=False):
    """Rescale the domain.

    Rescale the function domain according to the chosen strategy.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    var_lower : 1D numpy.ndarray[float]
        List of lower bounds of the variables.

    var_upper : 1D numpy.ndarray[float]
        List of upper bounds of the variables.

    integer_vars : 1D numpy.ndarray[int]
        A List containing the indices of the integrality constrained
        variables. If empty, all variables are assumed to be
        continuous.

    points : 2D numpy.ndarray[float]
        Point in the domain to be rescaled.

    reverse : bool
        False if we transform from the original domain to the
        transformed space, True if we want to apply the reverse.

    Returns
    -------
    2D numpy.ndarray[float]
        Rescaled points.

    Raises
    ------
    ValueError
        If the requested rescaling strategy is not implemented.
    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(isinstance(points, np.ndarray))
    assert(isinstance(settings, RbfoptSettings))
    assert(len(var_lower) == len(var_upper))
    assert(len(var_lower) == len(points[0]))

    if (settings.domain_scaling == 'off' or len(integer_vars)):
        # Make a copy because the caller may assume so
        return points.copy()
    elif (settings.domain_scaling == 'affine'):
        # Make an affine transformation to the unit hypercube
        if (reverse):
            return points * (var_upper - var_lower) + var_lower
        else:
            var_diff = var_upper - var_lower
            var_diff[var_diff == 0] = 1
            return (points - var_lower)/var_diff
    else:
        raise ValueError('Domain scaling "' + settings.domain_scaling +
                         '" not implemented')

# -- end function


def transform_domain_bounds(settings, var_lower, var_upper):
    """Rescale the variable bounds.

    Rescale the bounds of the function domain according to the chosen
    strategy.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.
    var_lower : 1D numpy.ndarray[float]
        List of lower bounds of the variables.
    var_upper : 1D numpy.ndarray[float]
        List of upper bounds of the variables.

    Returns
    -------
    (1D numpy.ndarray[float], 1D numpy.ndarray[float])
        Rescaled bounds as (lower, upper).

    Raises
    ------
    ValueError
        If the requested rescaling strategy is not implemented.
    """
    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))
    assert(isinstance(settings, RbfoptSettings))
    assert(len(var_lower) == len(var_upper))

    if (settings.domain_scaling == 'off'):
        # Make a copy because the caller may assume so
        return (var_lower.copy(), var_upper.copy())
    elif (settings.domain_scaling == 'affine'):
        # Make an affine transformation to the unit hypercube
        return (np.zeros(len(var_lower)), np.ones(len(var_upper)))
    else:
        raise ValueError('Domain scaling "' + settings.domain_scaling +
                         '" not implemented')

# -- end function


def get_sigma_n(k, current_step, num_global_searches, num_initial_points):
    """Compute sigma_n.

    Compute the index :math: `sigma_n`, where :math: `sigma_n` is a
    function described in the paper by Gutmann (2001). The same
    function is called :math: `alpha_n` in a paper of Regis &
    Shoemaker (2007).

    Parameters
    ----------
    k : int
        Number of nodes, i.e. interpolation points.
    current_step : int
        The current step in the cyclic search strategy.
    num_global_searches : int
        The number of global searches in a cycle.
    num_initial_points : int
        Number of points for the initialization phase.

    Returns
    -------
    int
        The value of sigma_n.
    """
    assert(current_step >= 1)
    assert(num_global_searches >= 0)
    if (current_step == 1):
        return k - 1
    return (get_sigma_n(k, current_step - 1, num_global_searches,
                        num_initial_points) -
            int(np.floor((k - num_initial_points)/num_global_searches)))

# -- end function


def get_fmax_current_iter(settings, n, k, current_step, node_val):
    """Compute the largest function value for target value computation.

    Compute the largest function value used to determine the target
    value. This is given by the sorted value in position :math:
    `sigma_n`.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.
    n : int
        Dimension of the problem, i.e. the space where the point lives.
    k : int
        Number of nodes, i.e. interpolation points.
    current_step : int
        The current step in the cyclic search strategy.
    node_val : 1D numpy.ndarray[float]
        List of function values.

    Returns
    -------
    float
        The value that should be used to determine the range of the
        function values when computing the target value.

    See also
    --------
    get_sigma_n
    """
    assert (isinstance(node_val, np.ndarray))
    assert(isinstance(settings, RbfoptSettings))
    assert(k == len(node_val))
    assert(k >= 1)
    assert(current_step >= 1)
    num_initial_points = (2**n if settings.init_strategy == 'all_corners'
                          else round((n + 1)*settings.init_sample_fraction))
    sorted_node_val = np.sort(node_val)
    s_n = get_sigma_n(k, current_step, settings.num_global_searches,
                      num_initial_points)
    return sorted_node_val[s_n]

# -- end function

def get_model_quality_estimate(settings, n, k, node_pos, node_val,
                               num_nodes_to_check):
    """Compute an estimate of model quality.

    Computes an estimate of model quality, performing
    cross-validation. It only checks the best num_nodes_to_check nodes.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the space where the point lives.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : 2D numpy.ndarray[float]
        Location of current interpolation nodes (one on each row).

    node_val : 1D numpy.ndarray[float]
        List of values of the function at the nodes.

    num_nodes_to_check : int
        Number of nodes on which quality should be tested.
    
    Returns
    -------
    float
        An estimate of the leave-one-out cross-validation error, which
        can be interpreted as a measure of model quality.

    Raises
    ------
    ValueError
        If the RBF type is not implemented.
    """
    assert(isinstance(node_pos, np.ndarray))
    assert(isinstance(node_val, np.ndarray))
    assert(isinstance(settings, RbfoptSettings))
    assert(len(node_val) == k)
    assert(len(node_pos) == k)
    assert(num_nodes_to_check <= k)
    # We cannot find a nontrivial leave-one-out interpolant if the
    # following condition is not met.
    assert(k > n + 2)

    # Get size of polynomial part of the matrix (p) and sign of obj
    # function (sign)
    if (get_degree_polynomial(settings) == 1):
        p = n + 1
    elif (get_degree_polynomial(settings) == 0):
        p = 1
    elif (get_degree_polynomial(settings) == -1):
        p = 0
    else:
        raise ValueError('RBF type ' + settings.rbf + ' not supported')

    # Sort interpolation nodes by increasing objective function value
    sorted_idx = node_val.argsort()

    # Initialize the arrays used for the cross-validation
    cv_node_pos = node_pos[sorted_idx]
    cv_node_val = node_val[sorted_idx]

    Amat = get_rbf_matrix(settings, n, k, cv_node_pos)
    lu, piv = la.lu_factor(Amat, check_finite=False)
    rhs = np.zeros(k + p)
    rhs[:k] = cv_node_val
    base_sol = la.lu_solve((lu, piv), rhs)
    # Estimate of the model error
    loo_error = 0.0

    for i in range(num_nodes_to_check):
        # Compute the RBF interpolant with one node left out
        if (abs(base_sol[i]) <= settings.eps_zero):
            # Lambda_i is 0 so we can just take the RBF interpolant as
            # is: it does not involve node i.
            predicted_val = evaluate_rbf(settings, cv_node_pos[i], n, k,
                                         cv_node_pos, base_sol[:k], 
                                         base_sol[k:])
        else:
            # Create basis vector e_i and solve for it
            e_i = np.zeros(k + p)
            e_i[i] = 1
            adj = la.lu_solve((lu, piv), e_i)
            # Adjust the solution of the linear system
            new_sol = base_sol - adj*base_sol[i]/adj[i]
            predicted_val = evaluate_rbf(settings, cv_node_pos[i], n, k,
                                         cv_node_pos, new_sol[:k], 
                                         new_sol[k:])

        # Update leave-one-out error
        loc = np.searchsorted(cv_node_val, predicted_val)
        loo_error += abs(loc - i)

    return loo_error

# -- end function

def get_best_rbf_model(settings, n, k, node_pos, node_val, 
                       num_nodes_to_check):
    """Compute which type of RBF yields the best model.

    Compute which RBF interpolant yields the best surrogate model,
    using cross validation to determine the lowest leave-one-out
    error.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the space where the point lives.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : 2D numpy.ndarray[float]
        Location of current interpolation nodes (one on each row.

    node_val : 1D numpy.ndarray[float]
        List of values of the function at the nodes.

    num_nodes_to_check : int
        Number of nodes on which quality should be tested.

    Returns
    -------
    str
        The type of RBF that currently yields the best surrogate
        model, based on leave-one-out error. This will be one of the
        supported types of RBF.
    """
    assert(isinstance(node_pos, np.ndarray))
    assert(isinstance(node_val, np.ndarray))
    assert(isinstance(settings, RbfoptSettings))
    assert(len(node_val) == k)
    assert(len(node_pos) == k)
    assert(num_nodes_to_check <= k)
    # We cannot find a nontrivial leave-one-out interpolant if the
    # following condition is not met.
    assert(k > n + 2)

    best_loo_error = np.inf
    best_model = settings.rbf
    best_gamma = settings.rbf_shape_parameter
    original_rbf_type = settings.rbf
    original_gamma = settings.rbf_shape_parameter
    rbf_list = ['cubic', 'thin_plate_spline', 'multiquadric', 'linear',
                'gaussian']
    gamma_list = [[0.1], [0.1], [0.1, 1.0], [0.1], [0.001, 0.01]]
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        for (i, rbf_type) in enumerate(rbf_list):
            for gamma in gamma_list[i]:
                settings.rbf = rbf_type
                settings.rbf_shape_parameter = gamma
                try:
                    loo_error = get_model_quality_estimate(
                        settings, n, k, node_pos, node_val, num_nodes_to_check)
                except:
                    if (i >= 2):
                        # If we tested at least some possibilities,
                        # return the best one found so far
                        return best_model, best_gamma
                    else:
                        # Else, return the original one
                        return original_rbf_type, original_gamma
                if (loo_error < best_loo_error):
                    best_loo_error = loo_error
                    best_model = rbf_type
                    best_gamma = gamma

    settings.rbf = original_rbf_type
    settings.rbf_shape_parameter = original_gamma
    return best_model, best_gamma
# -- end function

def get_most_common_element(array, to_exclude=[]):
    """Get most common element in a list.

    Parameters
    ----------
    array : List[any]
        The list whose most common element is sought.

    to_exclude : List[any]
        A list of elements to exclude from the count.
    
    Returns
    -------
    Any
        The most common element, None if all elements were excluded.
    """
    counter = collections.Counter(
        val for val in array if val not in to_exclude)
    if (counter.most_common(1)):
        return counter.most_common(1)[0][0]
    return None
# -- end function

def results_ready(results):
    """Check if some asynchronous results completed.

    Given a list containing results of asynchronous computations
    dispatched to a worker pool, verify if some of them are ready for
    processing.

    Parameters
    ----------
    results : List[(multiprocessing.pool.AsyncResult, any)]
        A list of tasks, where each task is a list and the first
        element is the output of a call to apply_async. The other
        elements of the list will never be scanned by this function,
        and they could be anything.

    Returns
    -------
    bool
        True if at least one result has completed.
    """
    for res in results:
        if res[0].ready():
            return True
    return False
# -- end function

def get_one_ready_index(results):
    """Get index of a single async computation result that is ready.

    Given a list containing results of asynchronous computations
    dispatched to a worker pool, obtain the index of the last
    computation that has concluded. (Last is better to use the pop()
    function in the list.)

    Parameters
    ----------
    results : List[(multiprocessing.pool.AsyncResult, any)]
        A list of tasks, where each task is a list and the first
        element is the output of a call to apply_async. The other
        elements of the list will never be scanned by this function,
        and they could be anything.

    Returns
    -------
    int
        Index of last computation that completed, or len(results) if
        no computation is ready.

    """
    for i in reversed(range(len(results))):
        if results[i][0].ready():
            return i
    return len(results)
# -- end function

def init_environment(settings):
    """Initialize the random seed and disable Pyomo output.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.
    """
    assert(isinstance(settings, RbfoptSettings))
    # Numpy's random seed
    np.random.seed(settings.rand_seed)
    # Disable Pyomo's logger if necessary
    if (not settings.debug):
        logging.getLogger('pyomo.core').setLevel(logging.CRITICAL)
        logging.getLogger('pyomo.opt').setLevel(logging.CRITICAL)
        logging.getLogger('pyomo.solvers').setLevel(logging.CRITICAL)
# -- end function

def init_points_cleanup(settings, points):
    """Clean up set of points to be used as initialization.

    Eliminate points that are too close to each other.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    points : 2D numpy.ndarray[float]
        The list of points to be cleaned.

    Returns
    -------
    numpy.ndarray[int]
        Indices of points that can be used to initialize an
        RbfoptAlgorithm object.
    """
    assert(isinstance(settings, RbfoptSettings))
    assert(isinstance(points, np.ndarray))
    assert(points.size)
    dist = ss.distance.cdist(points, points)
    # Indices of upper triangular matrix (without the diagonal)
    dist = np.triu(dist, 1)
    dist[np.tril_indices(len(points))] = float('+inf')
    min_dist = np.amin(dist, 1)
    points_to_take = np.array([True] * len(points))
    bad_pair_indices = np.arange(len(points))[min_dist <= settings.min_dist]
    for i in bad_pair_indices:
        points_to_take[np.arange(i+1, len(points))[
            dist[i, (i+1):] <= settings.min_dist]] = False
    return np.arange(len(points))[points_to_take]
# -- end function
