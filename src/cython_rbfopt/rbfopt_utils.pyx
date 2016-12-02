"""Utility functions.

This module contains a number of subroutines that are used by the
other modules.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
Research partially supported by SUTD-MIT International Design Center.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import sys
import math
import itertools
import numpy as np
import scipy.spatial as ss
import rbfopt_config as config
from rbfopt_settings import RbfSettings

cimport numpy as np
from libc.math cimport log, sqrt, floor, ceil

DTYPE = np.float64

cpdef get_rbf_function(settings):
    """Return a radial basis function.

    Return the radial basis function appropriate function as indicated
    by the settings.
    
    Parameters
    ----------
    
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.
    
    Returns
    ---
    Callable[float]
        A callable radial basis function.
    """
    assert(isinstance(settings, RbfSettings))
    if (settings.rbf == 'cubic'):
        return _cubic
    elif (settings.rbf == 'thin_plate_spline'):
        return _thin_plate_spline
    elif (settings.rbf == 'linear'):
        return _linear
    elif (settings.rbf == 'multiquadric'):
        return _multiquadric

# -- List of radial basis functions
cpdef double _cubic(double r):
    """Cubic RBF: :math: `f(x) = x^3`"""
    assert(r >= 0)
    return r*r*r

cpdef double _thin_plate_spline(double r):
    """Thin plate spline RBF: :math: `f(x) = x^2 \log x`"""
    assert(r >= 0)
    if (r == 0.0):
        return 0.0
    return log(r)*r*r

cpdef double _linear(double r):
    """Linear RBF: :math: `f(x) = x`"""
    assert(r >= 0)
    return r

cpdef double _multiquadric(double r):
    """Multiquadric RBF: :math: `f(x) = \sqrt{x^2 + \gamma^2}`"""
    assert(r >= 0)
    return sqrt(r*r + config.GAMMA*config.GAMMA)
# -- end list of radial basis functions

def get_degree_polynomial(settings):
    """Compute the degree of the polynomial for the interpolant.

    Return the degree of the polynomial that should be used in the RBF
    expression to ensure unisolvence and convergence of the
    optimization method.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    Returns
    -------
    int
        Degree of the polynomial
    """
    assert(isinstance(settings, RbfSettings))
    if (settings.rbf == 'cubic' or settings.rbf == 'thin_plate_spline'):
        return 1
    elif (settings.rbf == 'linear' or settings.rbf == 'multiquadric'):
        return 0
    else:
        return -1

# -- end function

def get_size_P_matrix(settings, n):
    """Compute size of the P part of the RBF matrix.

    Return the number of columns in the P part of the matrix [\Phi P;
    P^T 0] that is used through the algorithm.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. number of variables.

    Returns
    -------
    int
        Number of columns in the matrix
    """
    assert(isinstance(settings, RbfSettings))
    if (settings.rbf == 'cubic' or settings.rbf == 'thin_plate_spline'):
        return n+1
    elif (settings.rbf == 'linear' or settings.rbf == 'multiquadric'):
        return 1
    else:
        return 0

# -- end function

def get_all_corners(var_lower, var_upper):
    """Compute all corner points of a box.

    Compute and return all the corner points of the given box. Note
    that this number is exponential in the dimension of the problem.

    Parameters
    ----------
    var_lower : List[float]
        List of lower bounds of the variables.

    var_upper : List[float]
        List of upper bounds of the variables.

    Returns
    -------
    List[List[float]]
        All the corner points.
    """
    assert(len(var_lower)==len(var_upper))
    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))

    n = len(var_lower)
    node_pos = np.empty([2 ** n, n], DTYPE)
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
    var_lower : List[float]
        List of lower bounds of the variables.

    var_upper : List[float]
        List of upper bounds of the variables.

    Returns
    -------
    List[List[float]]
        The lower corner points.
    """
    assert(len(var_lower)==len(var_upper))
    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))

    n = len(var_lower)

    # Make sure we copy the lists instead of copying just a reference
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
    var_lower : List[float]
        List of lower bounds of the variables.

    var_upper : List[float]
        List of upper bounds of the variables.

    Returns
    -------
    List[List[float]]
        A list of random corner points.
    """
    assert(len(var_lower)==len(var_upper))
    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))

    n = len(var_lower)
    node_pos = list()
    while (len(node_pos) < n+1):
        point = [var_lower[i] if (np.random.rand() <= 0.5) else var_upper[i]
                 for i in range(n)]
        if (not node_pos or get_min_distance(np.array(point), np.array(node_pos)) > 0):
            node_pos.append(point)

    return np.array(node_pos, DTYPE)
    
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
    List[List[float]]
        A list of n-dimensional points in the unit hypercube.
    """
    assert(n >= 0)
    assert(num_samples >= 0)
    # Generate integer LH in [0, num_samples]
    int_lh = np.array([np.random.permutation(num_samples) for i in range(n)], DTYPE)
    int_lh = int_lh.T
    # Map integer LH back to unit hypercube, and perturb points so that
    # they are uniformly distributed in the corresponding intervals
    lhs = np.array([[np.random.uniform(i/num_samples, (i + 1)/num_samples)
            for i in point] for point in int_lh], DTYPE)
    return lhs

# -- end function

def get_lhd_maximin_points(var_lower, var_upper, num_trials = 50):
    """Compute a latin hypercube design with maximin distance.

    Compute a list of (n+1) points in the given box, where n is the
    dimension of the space. The selected points are picked according
    to a random latin hypercube design with maximin distance
    criterion. 

    Parameters
    ----------
    var_lower : List[float]
        List of lower bounds of the variables.

    var_upper : List[float]
        List of upper bounds of the variables.

    num_trials : int
        Maximum number of generated LHs to choose from.

    Returns
    -------
    List[List[float]]
        List of points in the latin hypercube design.
    """
    assert(len(var_lower)==len(var_upper))
    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))

    n = len(var_lower)
    if (n == 1):
        # For unidimensional problems, simply take the two endpoints
        # of the interval as starting points
        return np.vstack((var_lower, var_upper))
    # Otherwise, generate a bunch of Latin Hypercubes, and rank them
    lhs = [get_uniform_lhs(n, n + 1) for i in range(num_trials)]
    # Indices of upper triangular matrix (without the diagonal)
    indices = np.triu_indices(n + 1, 1)
    # Compute distance matrix of points to themselves, get upper
    # triangular part of the matrix, and get minimum
    dist_values = [np.amin(ss.distance.cdist(mat, mat)[indices])
                   for mat in lhs]
    lhd = lhs[dist_values.index(max(dist_values))]
    node_pos = lhd * (var_upper-var_lower) + var_lower

    return node_pos

# -- end function

def get_lhd_corr_points(var_lower, var_upper, num_trials = 50):

    """Compute a latin hypercube design with min correlation.

    Compute a list of (n+1) points in the given box, where n is the
    dimension of the space. The selected points are picked according
    to a random latin hypercube design with minimum correlation
    criterion. This function relies on the library pyDOE.

    Parameters
    ----------
    var_lower : List[float]
        List of lower bounds of the variables.

    var_upper : List[float]
        List of upper bounds of the variables.

    num_trials : int
        Maximum number of generated LHs to choose from.

    Returns
    -------
    List[List[float]]
        List of points in the latin hypercube design.
    """
    assert(len(var_lower)==len(var_upper))
    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))

    n = len(var_lower)
    if (n == 1):
        # For unidimensional problems, simply take the two endpoints
        # of the interval as starting points
        return np.vstack((var_lower, var_upper))
    # Otherwise, generate a bunch of Latin Hypercubes, and rank them
    lhs = [get_uniform_lhs(n, n + 1) for i in range(num_trials)]
    # Indices of upper triangular matrix (without the diagonal)
    indices = np.triu_indices(n, 1)
    # Compute correlation matrix of points to themselves, get upper
    # triangular part of the matrix, and get minimum
    corr_values = [abs(np.amax(np.corrcoef(mat, rowvar = 0)[indices]))
                   for mat in lhs]
    lhd = lhs[corr_values.index(min(corr_values))]
    node_pos = lhd * (var_upper-var_lower) + var_lower

    return node_pos

# -- end function

def initialize_nodes(settings, var_lower, var_upper, integer_vars):
    """Compute the initial sample points.

    Compute an initial list of nodes using the initialization strategy
    indicated in the algorithmic settings.
    
    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    var_lower : List[float]
        List of lower bounds of the variables.

    var_upper : List[float]
        List of upper bounds of the variables.

    integer_vars : List[int]
        A list containing the indices of the integrality constrained
        variables. If empty list, all variables are assumed to be
        continuous.

    Returns
    -------
    List[List[float]]
        List of at least n+1 corner points, where n is the dimension
        of the space. The number and position of points depends on the
        chosen strategy.

    Raises
    ------
    RuntimeError
        If a set of feasible and linearly independent sample points
        cannot be computed within the prescribed number of iterations.
    """
    assert(len(var_lower)==len(var_upper))
    assert(isinstance(settings, RbfSettings))
    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))
    assert (isinstance(integer_vars, np.ndarray))

    # We must make sure points are linearly independent; if they are
    # not, we perform a given number of iterations
    dependent = True
    itercount = 0
    while (dependent and itercount < config.MAX_RANDOM_INIT):
        itercount += 1
        if (settings.init_strategy == 'all_corners'):
            nodes = get_all_corners(var_lower, var_upper)
        elif (settings.init_strategy == 'lower_corners'):
            nodes = get_lower_corners(var_lower, var_upper)
        elif (settings.init_strategy == 'rand_corners'):
            nodes = get_random_corners(var_lower, var_upper)
        elif (settings.init_strategy == 'lhd_maximin'):
            nodes = get_lhd_maximin_points(var_lower, var_upper)
        elif (settings.init_strategy == 'lhd_corr'):
            nodes = get_lhd_corr_points(var_lower, var_upper)

        if (integer_vars.any()):
                for i in integer_vars:
                    np.around(nodes[:,i],out=nodes[:,i])

        U, s, V = np.linalg.svd(nodes)
        if (min(s) > settings.eps_zero):
            dependent = False

    if (itercount == config.MAX_RANDOM_INIT):
        raise RuntimeError('Exceeded number of random initializations')

    return nodes

# -- end function

def round_integer_vars(point, integer_vars):
    """Round a point to the closest integer.

    Round the values of the integer-constrained variables to the
    closest integer value. The values are rounded in-place.

    Parameters
    ----------
    point : List[float]
        The point to be rounded.
    integer_vars : List[int]
        A list of indices of integer variables.
    """
    assert (isinstance(point, np.ndarray))
    assert (isinstance(integer_vars, np.ndarray))
    if (integer_vars.any()):
        assert(np.amax(integer_vars)<len(point))
        for i in integer_vars:
            point[i] = round(point[i])

# -- end function

def round_integer_bounds(var_lower, var_upper, integer_vars):
    """Round the variable bounds to integer values.

    Round the values of the integer-constrained variable bounds, in
    the usual way: lower bounds are rounded up, upper bounds are
    rounded down.

    Parameters
    ----------
    var_lower : List[float]
        List of lower bounds of the variables.

    var_upper : List[float]
        List of upper bounds of the variables.

    integer_vars : List[int]
        A list containing the indices of the integrality constrained
        variables. If empty list, all variables are assumed to be
        continuous.
    """
    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))
    assert (isinstance(integer_vars, np.ndarray))
    if (integer_vars.any()):
        assert(len(var_lower)==len(var_upper))
        assert(max(integer_vars)<len(var_lower))
        for i in integer_vars:
            var_lower[i] = floor(var_lower[i])
            var_upper[i] = ceil(var_upper[i])
            if (var_upper[i] < var_lower[i]):
                # Swap the two bounds
                var_lower[i], var_upper[i] = var_upper[i], var_lower[i]

# -- end function

def norm(p):
    """Compute the L2-norm of a vector

    Compute the L2 (Euclidean) norm.

    Parameters
    ----------
    p : List[float]
        The point whose norm should be computed.

    Returns
    -------
    float
        The norm of the point.
    """
    assert(isinstance(p, np.ndarray))

    return sqrt(np.dot(p, p))

# -- end function

def distance(p1, p2):
    """Compute Euclidean distance between two points.

    Compute Euclidean distance between two points.

    Parameters
    ----------
    p1 : List[float]
        First point.
    p2 : List[float]
        Second point.

    Returns
    -------
    float
        Euclidean distance.
    """
    assert (isinstance(p1, np.ndarray))
    assert (isinstance(p2, np.ndarray))
    assert(len(p1) == len(p2))

    return sqrt(np.dot(p1 - p2, p1 - p2))

# -- end function

def get_min_distance(point, other_points):
    """Compute minimum distance from a set of points.

    Compute the minimum Euclidean distance between a given point and a
    list of points.

    Parameters
    ----------
    point : List[float]
        The point we compute the distances from.

    other_points : List[List[float]]
        The list of points we want to compute the distances to.

    Returns
    -------
    float
        Minimum distance between point and the other_points.
    """
    assert (isinstance(point, np.ndarray))
    assert (isinstance(other_points, np.ndarray))
    assert(point is not None and point.any())
    assert(other_points is not None and other_points.any())

    distances = map(lambda x : distance(x, point), other_points)
    return min(distances)

# -- end function

def get_min_distance_index(point, other_points):
    """Compute the index of the point with minimum distance.

    Compute the index of the point in a list that achieves minimum
    Euclidean distance to a given point.

    Parameters
    ----------
    point : List[float]
        The point we compute the distances from.

    other_points : List[List[float]]
        The list of points we want to compute the distances to.

    Returns
    -------
    int
        The index of the point in other_points that achieved minimum
        distance from point.
    """
    assert (isinstance(point, np.ndarray))
    assert (isinstance(other_points, np.ndarray))
    assert(point is not None and point.any())
    assert(other_points is not None and other_points.any())

    distances = map(lambda x : distance(np.array(x), np.array(point)), other_points)
    return distances.index(min(distances))

# -- end function

def bulk_get_min_distance(points, other_points):
    """Get the minimum distance between two sets of points.

    Compute the minimum distance of each point in the first set to the
    points in the second set. This is faster than using
    get_min_distance repeatedly, for large sets of points.

    Parameters
    ----------
    point : List[List[float]]
        The points in R^n that we compute the distances from.

    other_points : List[List[float]]
        The list of points we want to compute the distances to.

    Returns
    -------
    List[float]
        Minimum distance between each point in points and the
        other_points.

    See also
    --------
    get_min_distance()
    """
    assert(isinstance(points, np.ndarray))
    assert(isinstance(other_points, np.ndarray))
    assert(points.any())
    assert(other_points.any())
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
    settings : :class:`rbfopt_settings.RbfSettings`.
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    Returns
    -------
    numpy.matrix
        The matrix A = [Phi P; P^T 0].

    Raises
    ------
    ValueError
        If the type of RBF function is not supported.
    """
    assert(len(node_pos)==k)
    assert(isinstance(settings, RbfSettings))

    #### OLD VERSION ####
    # rbf = get_rbf_function(settings)
    # p = get_size_P_matrix(settings, n)
    # # Create matrix P.
    # if (p == n + 1):
    #     # Keep the node coordinates and append a 1.
    #     # P is ((k) x (n+1)), PTr is its transpose
    #     P = [ point + [1.0] for point in node_pos ]
    #     PTr = [ [point[i] for point in node_pos]
    #             for i in range(n) ] + [ [1.0 for i in range(k)] ]
    # elif (p == 1):
    #     # P is an all-one vector of size ((k) x (1))
    #     P = [ [1.0] for i in range(k) ]
    #     PTr = [ [ 1.0 for i in range(k) ] ]
    # else:
    #     raise ValueError('Rbf "' + settings.rbf + '" not implemented yet')
    #
    #
    # # Now create matrix Phi. Phi is ((k) x (k))
    # Phi = [ [rbf(distance(p1, p2)) for p2 in node_pos]
    #         for p1 in node_pos ]
    #
    # # Put together to obtain [Phi P; P^T 0].
    # A = ([ Phi[i] + P[i] for i in range(k) ] +
    #      [ PTr[i] + [0 for j in range(p)] for i in range(p)])

    node_pos_arr = np.array(node_pos, DTYPE)

    rbf = get_rbf_function(settings)
    p = get_size_P_matrix(settings, n)
    # Create matrix P.
    if (p == n + 1):
        # Keep the node coordinates and append a 1.
        # P is ((k) x (n+1)), PTr is its transpose
        P = np.insert(node_pos_arr, n, 1, axis=1)
        PTr = P.T
    elif (p == 1):
        # P is an all-one vector of size ((k) x (1))
        P = np.ones([k, 1])
        PTr = P.T
    else:
        raise ValueError('Rbf "' + settings.rbf + '" not implemented yet')

    # Now create matrix Phi. Phi is ((k) x (k))
    Phi = ss.distance.cdist(node_pos_arr, node_pos_arr)
    Phi = Phi.reshape(-1)
    for i, v in enumerate(Phi):
        Phi[i] = rbf(v)
    Phi = Phi.reshape(np.shape(node_pos_arr)[0],-1)

    # Put together to obtain [Phi P; P^T 0].
    A = np.vstack((np.hstack((Phi, P)), np.hstack((PTr, np.zeros((p, p))))))

    Amat = np.matrix(A)

    # Zero out tiny elements
    Amat[np.abs(Amat) < settings.eps_zero] = 0

    return Amat

# -- end function

def get_matrix_inverse(settings, Amat):
    """Compute the inverse of a matrix.

    Compute the inverse of a given matrix, zeroing out small
    coefficients to improve sparsity.
    
    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.
    Amat : numpy.matrix
        The matrix to invert.

    Returns
    -------
    numpy.matrix
        The matrix Amat^{-1}.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix cannot be inverted for numerical reasons.
    """
    assert(isinstance(settings, RbfSettings))
    assert(isinstance(Amat, np.matrix))

    try:
        Amatinv = Amat.getI()
    except np.linalg.LinAlgError as e:
        print('Exception raised trying to invert the RBF matrix',
              file = sys.stderr)
        print(e, file = sys.stderr)
        raise e

    # Zero out tiny elements of the inverse -- this is potentially
    # dangerous as the product between Amat and Amatinv may not be the
    # identity, but if the zero tolerance is chosen not too large,
    # this should help the optimization process
    Amatinv[np.abs(Amatinv) < settings.eps_zero] = 0

    return Amatinv

# -- end function
    
def get_rbf_coefficients(settings, n, k, Amat, node_val):
    """Compute the coefficients of the RBF interpolant.

    Solve a linear system to compute the coefficients of the RBF
    interpolant.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`.
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    Amat : numpy.matrix
        Matrix [Phi P; P^T 0] defining the linear system. Must be a
        square matrix of appropriate size.

    node_val : 1D numpy.ndarray[float]
        List of values of the function at the nodes.

    Returns
    -------
    (1D numpy.ndarray[float], 1D numpy.ndarray[float])
        Lambda coefficients (for the radial basis functions), and h
        coefficients (for the polynomial).
    """    
    assert(len(np.atleast_1d(node_val))==k)
    assert(isinstance(settings, RbfSettings))
    assert(isinstance(Amat, np.matrix))
    assert(isinstance(node_val, np.ndarray))
    p = get_size_P_matrix(settings, n)
    assert(Amat.shape==(k+p, k+p))
    rhs = np.append(node_val, np.zeros(p))
    try:
        solution = np.linalg.solve(Amat, rhs)
    except np.linalg.LinAlgError as e:
        print('Exception raised in the solution of the RBF linear system',
              file = sys.stderr)
        print('Exception details:', file = sys.stderr)
        print(e, file = sys.stderr)
        exit()
        
    return (solution[0:k], solution[k:])

# -- end function

def evaluate_rbf(settings, point, n, k, node_pos, rbf_lambda, rbf_h):
    """Evaluate the RBF interpolant at a given point.

    Evaluate the RBF interpolant at a given point.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`.
        Global and algorithmic settings.

    point : List[float]
        The point in R^n where we want to evaluate the interpolant.
    
    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    node_pos : List[List[float]]
        List of coordinates of the interpolation points. 

    rbf_lambda : List[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : List[float]
        The h coefficients of the RBF interpolant, corresponding to he
        polynomial. List of dimension given by get_size_P_matrix().

    Returns
    -------
    float
        Value of the RBF interpolant at the given point.
    """
    assert(len(point)==n)
    assert(len(rbf_lambda)==k)
    assert(len(node_pos)==k)
    assert(isinstance(settings, RbfSettings))
    p = get_size_P_matrix(settings, n)
    assert(len(rbf_h)==p)

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
                      return_distances = 'no'):
    """Evaluate the RBF interpolant at all points in a given list.

    Evaluate the RBF interpolant at all points in a given list. This
    version uses numpy and should be faster than individually
    evaluating the RBF at each single point, provided that the list of
    points is large enough. It also computes the distance or the
    minimum distance of each point from the interpolation nodes, if
    requested, since this comes almost for free.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`.
        Global and algorithmic settings.

    points : List[List[float]]
        The list of points in R^n where we want to evaluate the
        interpolant.
    
    n : int
        Dimension of the problem, i.e. the size of the space.

    k : int
        Number of interpolation nodes.

    node_pos : List[List[float]]
        List of coordinates of the interpolation points. 

    rbf_lambda : List[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : List[float]
        The h coefficients of the RBF interpolant, corresponding to he
        polynomial. List of dimension given by get_size_P_matrix().

    return_distances : string
        If 'no', do nothing. If 'min', return the minimum distance of
        each point to interpolation nodes. If 'all', return the full
        distance matrix to the interpolation nodes.

    Returns
    -------
    List[float] or (List[float], List[float])
        Value of the RBF interpolant at each point; if
        compute_min_dist is True, additionally returns the minimum
        distance of each point from the interpolation nodes.

    """
    assert (isinstance(points, np.ndarray))
    assert (isinstance(node_pos, np.ndarray))
    assert(points.any())
    assert(len(rbf_lambda)==k)
    assert(len(node_pos)==k)
    assert(isinstance(settings, RbfSettings))
    p = get_size_P_matrix(settings, n)
    assert(len(rbf_h)==p)

    rbf_function = get_rbf_function(settings)
    # Formula:
    # \sum_{i=1}^k \lambda_i \phi(\|x - x_i\|) + h^T (x 1)

    # Create distance matrix
    dist_mat = ss.distance.cdist(points, node_pos)
    # Evaluate radial basis function on each distance
    rbf_vec = map(rbf_function, dist_mat.ravel())
    rbf_vec_mat = np.reshape(np.array(rbf_vec, DTYPE), (len(points), -1))
    part1 = np.dot(rbf_vec_mat, rbf_lambda)
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

def get_fast_error_bounds(settings, value):
    """Compute error bounds for fast interpolation nodes.

    Compute the interval that contains the true value of a fast
    function evaluation, according to the specified relative and
    absolute error tolerances.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.
    value : float
        The value for which the error interval should be computed.

    Returns
    -------
    (float, float)
        A tuple (lower_variation, upper_variation) indicating the
        possible deviation to the left (given as a negative number)
        and to the right of the current value.
    """
    return (-abs(value)*settings.fast_objfun_rel_error - 
            settings.fast_objfun_abs_error,
            abs(value)*settings.fast_objfun_rel_error +
            settings.fast_objfun_abs_error)

# -- end function

def compute_gap(settings, fmin, is_best_fast):
    """Compute the optimality gap w.r.t. the target value.

    Returns
    -------
    float
        The current optimality gap, i.e. relative distance from target
        value.
    """
    assert(isinstance(settings, RbfSettings))    
    # Denominator of errormin
    gap_den = (abs(settings.target_objval) 
               if (abs(settings.target_objval) >= settings.eps_zero)
               else 1.0)
    # Shift due to fast function evaluation
    gap_shift = (get_fast_error_bounds(settings, fmin)[1]
                 if is_best_fast else 0.0)
    # Compute current minimum distance from the optimum
    gap = ((fmin + gap_shift - settings.target_objval) /
           gap_den)
    return gap
# -- end function

def transform_function_values(settings, node_val, fmin, fmax,
                              fast_node_index = list()):
    """Rescale function values.
    
    Rescale and adjust function values according to the chosen
    strategy and to the occurrence of large fluctuations (high
    dynamism). May not rescale at all if rescaling is off.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfSettings`
       Global and algorithmic settings.

    node_val : List[float]
       List of function values at the interpolation nodes.

    fmin : float
       Minimum function value found so far.

    fmax : float
       Maximum function value found so far.
    
    fast_node_index : List[int] or None
       Index of function evaluations in 'fast' mode. If this is empty,
       then error bounds will not be returned.

    Returns
    -------
    (List[float], float, float, List[(float, float)])
        A quadruple (scaled_function_values, scaled_fmin, scaled_fmax,
        fast_error_bounds) containing a list of rescaled function
        values, the rescaled minimum, the rescaled maximum, the
        rescaled error bounds for function evaluations in 'fast' mode.

    Raises
    ------
    ValueError
        If the function scaling strategy requested is not implemented.
    """
    assert(isinstance(node_val, np.ndarray))
    assert(isinstance(settings, RbfSettings))
    # Check dynamism: if too high, replace large function values with
    # the median or clip at maximum dynamism
    if (settings.dynamism_clipping != 'off' and
        ((abs(fmin) > settings.eps_zero and
          abs(fmax)/abs(fmin) > settings.dynamism_threshold) or
         (abs(fmin) <= settings.eps_zero and
          abs(fmax) > settings.dynamism_threshold))):
        if (settings.dynamism_clipping == 'median'):
            sorted_values = np.sort(node_val)
            median = sorted_values[len(sorted_values)//2]
            clip_val = [min(val, median) for val in node_val]
            fmax = median
        elif (settings.dynamism_clipping == 'clip_at_dyn'):
            # We should not multiply by abs(fmin) if it is too small
            mult = abs(fmin) if (abs(fmin) > settings.eps_zero) else 1.0
            clip_val = [min(val, settings.dynamism_threshold*mult)
                        for val in node_val]
            fmax = settings.dynamism_threshold*mult
    else:
        clip_val = node_val

    if (settings.function_scaling == 'off'):
        # We make a copy because the caller may assume that
        return (list(clip_val), fmin, fmax, 
                [get_fast_error_bounds(settings, clip_val[i])
                 for i in fast_node_index])
    elif (settings.function_scaling == 'affine'):
        # Compute denominator separately to make sure that it is not
        # zero. This may happen if the surface is "flat" after median
        # clipping.
        denom = (fmax - fmin) if (fmax - fmin > settings.eps_zero) else 1.0
        return ([(val - fmin)/denom for val in clip_val], 0.0, 
                1.0 if (fmax - fmin > settings.eps_zero) else 0.0,
                [tuple([val/denom for val in 
                        get_fast_error_bounds(settings, clip_val[i])])
                 for i in fast_node_index])
    elif (settings.function_scaling == 'log'):
        # Compute by how much we should translate to make all points >= 1
        shift = (max(0.0, 1.0 - fmin) if not fast_node_index
                 else max(0.0, 1.0 - fmin -
                          get_fast_error_bounds(settings, fmin)[0]))
        return ([math.log(val + shift) for val in clip_val], 
                math.log(fmin + shift), math.log(fmax + shift),
                [tuple([math.log((clip_val[i] + shift + val) / 
                                 (clip_val[i] + shift))
                        for val in get_fast_error_bounds(settings,
                                                         clip_val[i])])
                 for i in fast_node_index])
    else:
        raise ValueError('Function scaling "' + settings.function_scaling + 
                         '" not implemented')

# -- end function

def transform_domain(settings, var_lower, var_upper, point, reverse = False):
    """Rescale the domain.

    Rescale the function domain according to the chosen strategy.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    var_lower : List[float]
        List of lower bounds of the variables.

    var_upper : List[float]
        List of upper bounds of the variables.

    point : List[float]
        Point in the domain to be rescaled.

    reverse : bool
        False if we transform from the original domain to the
        transformed space, True if we want to apply the reverse.

    Returns
    -------
    List[float]
        Rescaled point.
    
    Raises
    ------
    ValueError
        If the requested rescaling strategy is not implemented.
    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(isinstance(point, np.ndarray))
    assert(isinstance(settings, RbfSettings))
    assert(len(var_lower)==len(var_upper))
    assert(len(var_lower)==len(point))

    if (settings.domain_scaling == 'off'):
        # Make a copy because the caller may assume so
        return point.copy()
    elif (settings.domain_scaling == 'affine'):
        # Make an affine transformation to the unit hypercube
        if (reverse):
            return point * (var_upper - var_lower) + var_lower
        else:
            return np.array([(point[i] - var_lower[i]) /
                    ((var_upper[i] - var_lower[i]) 
                     if (var_upper[i] > var_lower[i]) else 1.0)
                    for i in range(len(point))], DTYPE)
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
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.
    var_lower : List[float]
        List of lower bounds of the variables.
    var_upper : List[float]
        List of upper bounds of the variables.

    Returns
    -------
    (List[float], List[float])
        Rescaled bounds as (lower, upper).
    
    Raises
    ------
    ValueError
        If the requested rescaling strategy is not implemented.
    """
    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))
    assert(isinstance(settings, RbfSettings))
    assert(len(var_lower)==len(var_upper))

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
            int(math.floor((k - num_initial_points)/num_global_searches)))

# -- end function    

def get_fmax_current_iter(settings, n, k, current_step, node_val):
    """Compute the largest function value for target value computation.
    
    Compute the largest function value used to determine the target
    value. This is given by the sorted value in position :math:
    `sigma_n`.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.
    n : int
        Dimension of the problem, i.e. the space where the point lives.
    k : int
        Number of nodes, i.e. interpolation points.
    current_step : int
        The current step in the cyclic search strategy.
    node_val : List[float]
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
    assert(isinstance(settings, RbfSettings))
    assert(k == len(node_val))
    assert(k >= 1)
    assert(current_step >= 1)
    num_initial_points = (2**n if settings.init_strategy == 'all_corners'
                           else n + 1)
    assert(k >= num_initial_points)
    sorted_node_val = np.sort(node_val)
    s_n = get_sigma_n(k, current_step, settings.num_global_searches,
                      num_initial_points)
    return sorted_node_val[s_n]

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
# -- end if

def get_ready_indices(results):
    """Get indices of results of async computations that are ready.
    
    Given a list containing results of asynchronous computations
    dispatched to a worker pool, obtain the index of computations that
    have concluded.

    Parameters
    ----------
    results : List[(multiprocessing.pool.AsyncResult, any)]
        A list of tasks, where each task is a list and the first
        element is the output of a call to apply_async. The other
        elements of the list will never be scanned by this function,
        and they could be anything.

    Returns
    -------
    List[int]
        List of indices of computations that completed, from the
        largest to the smallest.

    """
    ready = list()
    for i in reversed(range(len(results))):
        if results[i][0].ready():
            ready.append(i)
    return ready
# -- end if

def init_rand_seed(seed):
    """Initialize the random seed.

    Parameters
    ----------
    seed : any
        A hashable object that can be used to initialize numpy's
        internal random number generator.
    """
    np.random.seed(seed)
# -- end if
