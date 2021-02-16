"""Auxiliary problems for the optimization process.

This module is responsible for constructing and solving all the
auxiliary problems encountered during the optimization, such as the
minimization of the surrogate model, of the bumpiness. The module acts
as an interface between the high-level routines, the low-level PyOmo
modules, and the search algorithms.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
(C) Copyright International Business Machines Corporation 2017.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import math
import numpy as np
import scipy.spatial as ss
import pyomo.environ
import pyomo.opt
import logging
import rbfopt.rbfopt_utils as ru
import rbfopt.rbfopt_degree1_models as rbfopt_degree1_models
import rbfopt.rbfopt_degree0_models as rbfopt_degree0_models
import rbfopt.rbfopt_degreem1_models as rbfopt_degreem1_models
from rbfopt.rbfopt_settings import RbfoptSettings


def pure_global_search(settings, n, k, var_lower, var_upper,
                       integer_vars, categorical_info, node_pos, mat):
    """Pure global search that disregards objective function.

    If using Gutmann's RBF method, Construct a PyOmo model to maximize
    :math: `1/\mu`. If using the Metric SRM, select a point purely
    based on distance.

    See paper by Costa and Nannicini, equation (7) pag 4, and the
    references therein.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    var_lower : 1D numpy.ndarray[float]
        Vector of variable lower bounds.

    var_upper : 1D numpy.ndarray[float]
        Vector of variable upper bounds.

    integer_vars : 1D numpy.ndarray[int]
        A list containing the indices of the integrality constrained
        variables. If empty list, all variables are assumed to be
        continuous.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes.

    mat : 2D numpy.ndarray[float] or None
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0], see paper as cited above. Must
        be a square 2D numpy.ndarray[float] of appropriate dimension if
        given. Can be None when using the MSRSM algorithm.

    Returns
    -------
    List[float]
        A maximizer. It is difficult to do global optimization so
        typically this method returns a local maximum.

    Raises
    ------
    ValueError
        If some parameters are not supported.
    RuntimeError
        If the solver cannot be found.

    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(isinstance(integer_vars, np.ndarray))
    assert(isinstance(node_pos, np.ndarray))

    assert(len(var_lower) == n)
    assert(len(var_upper) == n)
    assert(len(node_pos) == k)
    assert(isinstance(settings, RbfoptSettings))

    # Determine the size of the P matrix
    p = ru.get_size_P_matrix(settings, n)
    assert((mat is None and settings.algorithm.upper() == 'MSRSM')
           or (isinstance(mat, np.ndarray) and mat.shape == (k + p, k + p)))

    # Instantiate model
    if (ru.get_degree_polynomial(settings) == 1):
        model = rbfopt_degree1_models
    elif (ru.get_degree_polynomial(settings) == 0):
        model = rbfopt_degree0_models
    elif (ru.get_degree_polynomial(settings) == -1):
        model = rbfopt_degreem1_models
    else:
        raise ValueError('RBF type ' + settings.rbf + ' not supported')

    if (settings.global_search_method == 'genetic'):
        # Use a genetic algorithm to optimize
        if (settings.algorithm.upper() == 'GUTMANN'):
            fitness = GutmannMukObj(settings, n, k, node_pos, mat)
        elif (settings.algorithm.upper() == 'MSRSM'):
            fitness = MaximinDistanceObj(settings, n, k, node_pos)
        else:
            raise ValueError('Algorithm ' + settings.algorithm + 
                             ' not supported')
        point = ga_optimize(settings, n, var_lower, var_upper, integer_vars,
                            categorical_info, fitness.evaluate)
    elif (settings.global_search_method == 'sampling'):
        # Sample random points, and rank according to fitness
        if (settings.algorithm.upper() == 'GUTMANN'):
            fitness = GutmannMukObj(settings, n, k, node_pos, mat)
        elif (settings.algorithm.upper() == 'MSRSM'):
            fitness = MaximinDistanceObj(settings, n, k, node_pos)
        else:
            raise ValueError('Algorithm ' + settings.algorithm + 
                             ' not supported')
        num_samples = n * settings.num_samples_aux_problems
        samples = generate_sample_points(settings, n, var_lower, var_upper,
                                         integer_vars, categorical_info,
                                         num_samples)
        scores = fitness.evaluate(samples)
        point = samples[scores.argmin()]
    elif (settings.global_search_method == 'solver'):
        # Optimize using Pyomo
        if (settings.algorithm.upper() == 'GUTMANN'):
            instance = model.create_max_one_over_mu_model(
                settings, n, k, var_lower, var_upper,
                integer_vars, categorical_info, node_pos, mat)
            # Initialize variables for local search
            initialize_instance_variables(settings, instance)
        elif (settings.algorithm.upper() == 'MSRSM'):
            instance = model.create_maximin_dist_model(
                settings, n, k, var_lower, var_upper, integer_vars,
                categorical_info, node_pos)
            # Initialize variables for local search
            initialize_instance_variables(settings, instance)
        else:
            raise ValueError('Algorithm ' + settings.algorithm + 
                             ' not supported')
        # Instantiate optimizer
        opt = pyomo.opt.SolverFactory(
            'bonmin', executable=settings.minlp_solver_path, solver_io='nl')
        if (not opt.available()):
            raise RuntimeError('Solver ' + 'bonmin' + ' not found')
        set_minlp_solver_options(opt)

        # Solve and load results
        try:
            results = opt.solve(instance, keepfiles=False,
                                tee=settings.print_solver_output)
            if ((results.solver.status == pyomo.opt.SolverStatus.ok) and
                (results.solver.termination_condition ==
                 pyomo.opt.TerminationCondition.optimal)):
                # this is feasible and optimal
                instance.solutions.load_from(results)
                point = np.array([instance.x[i].value for i in instance.N])
                ru.round_integer_vars(point, integer_vars)
            else:
                point = None
        except:
            point = None
    else:
        raise ValueError('Global search method ' + settings.algorithm +
                         ' not supported')

    return point

# -- end function


def minimize_rbf(settings, n, k, var_lower, var_upper, integer_vars,
                 categorical_info, node_pos, rbf_lambda, rbf_h,
                 best_node_pos):
    """Compute the minimum of the RBF interpolant.

    Compute the minimum of the RBF interpolant with a PyOmo model.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    var_lower : 1D numpy.ndarray[float]
        Vector of variable lower bounds.

    var_upper : 1D numpy.ndarray[float]
        Vector of variable upper bounds.

    integer_vars : 1D numpy.ndarray[int]
        A list containing the indices of the integrality constrained
        variables. If empty list, all variables are assumed to be
        continuous.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes.

    rbf_lambda : 1D numpy.ndarray[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : 1D numpy.ndarray[float]
        The h coefficients of the RBF interpolant, corresponding to
        the polynomial. List of dimension n+1.

    best_node_pos : 1D numpy.ndarray[float]
        Coordinates of the best interpolation point.

    Returns
    -------
    1D numpy.ndarray[float]
        A minimizer. It is difficult to do global optimization so
        typically this method returns a local minimum.

    Raises
    ------
    ValueError
        If some parameters are not supported.
    RuntimeError
        If the solver cannot be found.
    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(isinstance(integer_vars, np.ndarray))
    assert(isinstance(node_pos, np.ndarray))
    assert(isinstance(rbf_lambda, np.ndarray))
    assert(isinstance(rbf_h, np.ndarray))

    assert(len(var_lower) == n)
    assert(len(var_upper) == n)
    assert(len(rbf_lambda) == k)
    assert(len(node_pos) == k)
    assert(isinstance(settings, RbfoptSettings))

    # Determine the size of the P matrix
    p = ru.get_size_P_matrix(settings, n)
    assert(len(rbf_h) == p)

    # Instantiate model
    if (ru.get_degree_polynomial(settings) == 1):
        model = rbfopt_degree1_models
    elif (ru.get_degree_polynomial(settings) == 0):
        model = rbfopt_degree0_models
    elif (ru.get_degree_polynomial(settings) == -1):
        model = rbfopt_degreem1_models
    else:
        raise ValueError('RBF type ' + settings.rbf + ' not supported')

    instance = model.create_min_rbf_model(
        settings, n, k, var_lower, var_upper, integer_vars,
        categorical_info, node_pos, rbf_lambda, rbf_h)
    # Initialize variables for local search
    initialize_instance_variables(settings, instance,
                                  start_point=best_node_pos)
    # Instantiate optimizer
    opt = pyomo.opt.SolverFactory(
        'bonmin', executable=settings.minlp_solver_path, solver_io='nl')
    if (not opt.available()):
        raise RuntimeError('Solver ' + 'bonmin' + 'not found')
    set_minlp_solver_options(opt)

    # Solve and load results
    try:
        results = opt.solve(instance, keepfiles=False,
                            tee=settings.print_solver_output)
        if ((results.solver.status == pyomo.opt.SolverStatus.ok) and
            (results.solver.termination_condition ==
             pyomo.opt.TerminationCondition.optimal)):
            # this is feasible and optimal
            instance.solutions.load_from(results)
            point = np.array([instance.x[i].value for i in instance.N])
            ru.round_integer_vars(point, integer_vars)
        else:
            point = None
    except:
        point = None

    return point

# -- end function


def global_search(settings, n, k, var_lower, var_upper, integer_vars,
                  categorical_info, node_pos, rbf_lambda, rbf_h, mat,
                  target_val, dist_weight, fmin, fmax):
    """Global search that tries to balance exploration/exploitation.

    If using Gutmann's RBF method, compute the maximum of the h_k
    function, see equation (8) in the paper by Costa and
    Nannicini. If using the Metric SRSM, select a point based on a
    combination of distance and objective function value.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    var_lower : 1D numpy.ndarray[float]
        Vector of variable lower bounds.

    var_upper : 1D numpy.ndarray[float]
        Vector of variable upper bounds.

    integer_vars : 1D numpy.ndarray[int]
        A list containing the indices of the integrality constrained
        variables. If empty list, all variables are assumed to be
        continuous.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes.

    rbf_lambda : 1D numpy.ndarray[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : 1D numpy.ndarray[float]
        The h coefficients of the RBF interpolant, corresponding to
        the polynomial. List of dimension n+1.

    mat : 2D numpy.ndarray[float] or None
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0], see paper as cited above. Must
        be a square 2D numpy.ndarray[float] of appropriate dimension, or None if
        using the MSRSM algorithm.

    target_val : float
        Value f* that we want to find in the unknown objective
        function. Used by Gutmann's RBF method only.

    dist_weight : float
        Relative weight of the distance and objective function value,
        when selecting the next point with a sampling strategy. A
        weight of 1.0 corresponds to using solely distance, 0.0 to
        objective function. Used by Metric SRSM only.

    fmin : float
        Minimum value among the interpolation nodes.

    fmax : float
        Maximum value among the interpolation nodes.

    Returns
    -------
    1D numpy.ndarray[float]
        A local optimum. It is difficult to do global optimization so
        typically this method returns a local optimum.

    Raises
    ------
    ValueError
        If some parameters are not supported.
    RuntimeError
        If the solver cannot be found.

    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(isinstance(integer_vars, np.ndarray))
    assert(isinstance(node_pos, np.ndarray))
    assert(isinstance(rbf_lambda, np.ndarray))
    assert(isinstance(rbf_h, np.ndarray))
    assert(len(var_lower) == n)
    assert(len(var_upper) == n)
    assert(len(rbf_lambda) == k)
    assert(len(node_pos) == k)
    assert(0 <= dist_weight <= 1)
    assert(fmin <= fmax)
    assert(isinstance(settings, RbfoptSettings))

    # Determine the size of the P matrix
    p = ru.get_size_P_matrix(settings, n)
    assert((mat is None and settings.algorithm.upper() == 'MSRSM' )
           or (isinstance(mat, np.ndarray) and mat.shape == (k + p, k + p)))
    assert(len(rbf_h) == p)

    # Instantiate model
    if (ru.get_degree_polynomial(settings) == 1):
        model = rbfopt_degree1_models
    elif (ru.get_degree_polynomial(settings) == 0):
        model = rbfopt_degree0_models
    elif (ru.get_degree_polynomial(settings) == -1):
        model = rbfopt_degreem1_models
    else:
        raise ValueError('RBF type ' + settings.rbf + ' not supported')

    if (settings.global_search_method == 'genetic'):
        # Use a genetic algorithm to optimize
        if (settings.algorithm.upper() == 'GUTMANN'):
            fitness = GutmannHkObj(settings, n, k, node_pos, rbf_lambda,
                                   rbf_h, mat, target_val)
        elif (settings.algorithm.upper() == 'MSRSM'):
            fitness = MetricSRSMObj(settings, n, k, node_pos, rbf_lambda,
                                    rbf_h, dist_weight)
        else:
            raise ValueError('Algorithm ' + settings.algorithm + 
                             ' not supported')
        point = ga_optimize(settings, n, var_lower, var_upper, integer_vars,
                            categorical_info, fitness.evaluate)
    elif (settings.global_search_method == 'sampling'):
        # Sample random points, and rank according to fitness
        if (settings.algorithm.upper() == 'GUTMANN'):
            fitness = GutmannHkObj(settings, n, k, node_pos, rbf_lambda,
                                   rbf_h, mat, target_val)
        elif (settings.algorithm.upper() == 'MSRSM'):
            fitness = MetricSRSMObj(settings, n, k, node_pos, rbf_lambda,
                                    rbf_h, dist_weight)
        else:
            raise ValueError('Algorithm ' + settings.algorithm + 
                             ' not supported')
        num_samples = n * settings.num_samples_aux_problems
        samples = generate_sample_points(
            settings, n, var_lower, var_upper, integer_vars,
            categorical_info, num_samples)
        scores = fitness.evaluate(samples)
        point = samples[scores.argmin()]
    elif (settings.global_search_method == 'solver'):
        # Optimize using Pyomo
        if (settings.algorithm.upper() == 'GUTMANN'):
            instance = model.create_max_h_k_model(
                settings, n, k, var_lower, var_upper, integer_vars,
                categorical_info, node_pos, rbf_lambda, rbf_h, mat,
                target_val)
            initialize_instance_variables(settings, instance)
        elif (settings.algorithm.upper() == 'MSRSM'):
            # Compute minimum and maximum distances between
            # points. This computation could be avoided if
            # RbfoptAlgorithm keeps track of them, but in the grand
            # scheme of things the computation is rarely performed and
            # is not as expensive as the subsequent optimization.
            dist_mat = ss.distance.cdist(node_pos, node_pos)
            dist_min = np.min(dist_mat[np.triu_indices(k, 1)])
            dist_max = np.max(dist_mat[np.triu_indices(k, 1)])
            instance = model.create_min_msrsm_model(
                settings, n, k, var_lower, var_upper, integer_vars,
                categorical_info, node_pos, rbf_lambda, rbf_h,
                dist_weight, dist_min, dist_max, fmin, fmax)
            initialize_instance_variables(settings, instance)
            initialize_msrsm_aux_variables(settings, instance)
        else:
            raise ValueError('Algorithm ' + settings.algorithm + 
                             ' not supported')
        # Instantiate optimizer
        opt = pyomo.opt.SolverFactory(
            'bonmin', executable=settings.minlp_solver_path, solver_io='nl')
        if (not opt.available()):
            raise RuntimeError('Solver ' + 'bonmin' + ' not found')
        set_minlp_solver_options(opt)

        # Solve and load results
        try:
            results = opt.solve(instance, keepfiles=False,
                                tee=settings.print_solver_output)
            if ((results.solver.status == pyomo.opt.SolverStatus.ok) and
                (results.solver.termination_condition ==
                 pyomo.opt.TerminationCondition.optimal)):
                # this is feasible and optimal
                instance.solutions.load_from(results)
                point = np.array([instance.x[i].value for i in instance.N])
                ru.round_integer_vars(point, integer_vars)
            else:
                point = None
        except:
            point = None
    else:
        raise ValueError('Global search method ' + settings.algorithm +
                         ' not supported')
    if point is not None:
        point = np.array(point)

    return point

# -- end function


def initialize_instance_variables(settings, instance, start_point=None):
    """Initialize the variables of a problem instance.

    Initialize the x variables of a problem instance, and set the
    corresponding values for the vectors :math:`(u,\pi)`. This helps
    the local search by starting at a feasible point.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    instance : pyomo.ConcreteModel
        A concrete instance of mathematical optimization model.

    start_point : 1D numpy.ndarray[float] or None
        The starting point for the local search, or None if it should
        be randomly generated.

    """
    assert(isinstance(settings, RbfoptSettings))

    # Initialize variables for local search
    if (start_point is not None):
        assert(len(instance.N) == len(start_point))
        for i in instance.N:
            instance.x[i] = start_point[i]
    else:
        for i in instance.N:
            instance.x[i] = np.random.uniform(instance.var_lower[i],
                                              instance.var_upper[i])
# -- end function


def initialize_msrsm_aux_variables(settings, instance):
    """Initialize auxiliary variables for the MSRSM model.

    Initialize the rbfval and mindist variables of a problem
    instance, using the values for for x and u_pi already given. This
    helps the local search by starting at a feasible point.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    instance : pyomo.ConcreteModel
        A concrete instance of mathematical optimization model.
    """
    assert(isinstance(settings, RbfoptSettings))

    dist = min(sum((instance.x[j].value - instance.node[i, j])**2
                   for j in instance.N) for i in instance.K)
    instance.mindistsq = dist

# -- end function


def get_noisy_rbf_coefficients(settings, n, k, Phimat, Pmat, node_val,
                               node_err_bounds, init_rbf_lambda=None,
                               init_rbf_h=None):
    """Obtain coefficients for the noisy RBF interpolant.

    Solve a quadratic problem to compute the coefficients of the RBF
    interpolant that minimizes bumpiness and lets all points with
    deviate by a given amount from their value.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    Phimat : 2D numpy.ndarray[float]
        Matrix Phi, i.e. top left part of the standard RBF matrix.

    Pmat : 2D numpy.ndarray[float]
        Matrix P, i.e. top right part of the standard RBF matrix.

    node_val : 1D numpy.ndarray[float]
        List of values of the function at the nodes.

    node_err_bounds : 2D numpy.ndarray[float]
        Allowed deviation from node values for nodes affected by
        error. This is a matrix with rows (lower_deviation, 
        upper_deviation).

    init_rbf_lambda : 1D numpy.ndarray[float] or None
        Initial values that should be used for the lambda coefficients
        of the RBF. Can be None.

    init_rbf_h : 1D numpy.ndarray[float] or None
        Initial values that should be used for the h coefficients of
        the RBF. Can be None.

    Returns
    ---
    (1D numpy.ndarray[float], 1D numpy.ndarray[float])
        Two vectors: lambda coefficients (for the radial basis
        functions), and h coefficients (for the polynomial). If
        initialization information was provided and was valid, then
        some values will always be returned. Otherwise, it will be
        None.

    Raises
    ------
    ValueError
        If some parameters are not supported.
    RuntimeError
        If the solver cannot be found.
    """
    assert(isinstance(settings, RbfoptSettings))
    assert(isinstance(node_val, np.ndarray))
    assert(isinstance(node_err_bounds, np.ndarray))
    assert(len(node_val) == k)
    assert(isinstance(Phimat, np.ndarray))
    assert(isinstance(Pmat, np.ndarray))
    assert(len(node_val) == len(node_err_bounds))
    assert(init_rbf_lambda is None or (isinstance(init_rbf_lambda, 
                                                  np.ndarray) and
                                       len(init_rbf_lambda) == k))
    assert(init_rbf_h is None or (isinstance(init_rbf_h, np.ndarray) and
                                  len(init_rbf_h) == Pmat.shape[1]))
    
    # Instantiate model
    if (ru.get_degree_polynomial(settings) == 1):
        model = rbfopt_degree1_models
    elif (ru.get_degree_polynomial(settings) == 0):
        model = rbfopt_degree0_models
    elif (ru.get_degree_polynomial(settings) == -1):
        model = rbfopt_degreem1_models
    else:
        raise ValueError('RBF type ' + settings.rbf + ' not supported')

    instance = model.create_min_bump_model(settings, n, k, Phimat, Pmat,
                                           node_val, node_err_bounds)

    # Instantiate optimizer
    opt = pyomo.opt.SolverFactory(
        'ipopt', executable=settings.nlp_solver_path, solver_io='nl')
    if (not opt.available()):
        raise RuntimeError('Solver ' + 'ipopt' + ' not found')
    set_nlp_solver_options(opt)

    # Initialize instance variables with the solution provided (if
    # available). This should avoid any infeasibility.
    if (init_rbf_lambda is not None and init_rbf_h is not None):
        for i in range(len(init_rbf_lambda)):
            instance.rbf_lambda[i] = init_rbf_lambda[i]
        for i in range(len(init_rbf_h)):
            instance.rbf_h[i] = init_rbf_h[i]

    # Solve and load results
    try:
        results = opt.solve(instance, keepfiles=False,
                            tee=settings.print_solver_output)
        if ((results.solver.status == pyomo.opt.SolverStatus.ok) and
            (results.solver.termination_condition ==
             pyomo.opt.TerminationCondition.optimal)):
            # this is feasible and optimal
            instance.solutions.load_from(results)
            rbf_lambda = np.array([instance.rbf_lambda[i].value 
                                   for i in instance.K])
            if (ru.get_size_P_matrix(settings, n) > 0):
                rbf_h = np.array([instance.rbf_h[i].value for i in instance.P])
            else:
                rbf_h = np.array([])
        else:
            # If we have initialization information, return it. It is
            # a feasible solution. Otherwise, this will be None.
            rbf_lambda = init_rbf_lambda
            rbf_h = init_rbf_h
    except:
        # If we have initialization information, return it. It is
        # a feasible solution. Otherwise, this will be None.
        rbf_lambda = init_rbf_lambda
        rbf_h = init_rbf_h

    return (rbf_lambda, rbf_h)

# -- end function


def get_min_bump_node(settings, n, k, Amat, node_val, node_err_bounds,
                      target_val):
    """Compute the bumpiness obtained by moving an interpolation point.

    Compute the bumpiness of the interpolant obtained by moving a
    single node (the one that yields minimum bumpiness, which is
    determined by this function) within target_val plus or minus
    error, to target_val.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the space where the point lives.

    k : int
        Number of nodes, i.e. interpolation points.

    Amat : 2D numpy.ndarray[float]
        The matrix A = [Phi P; P^T 0] of equation (3) in the paper by
        Costa and Nannicini.

    node_val : 1D numpy.ndarray[float]
        List of values of the function at the nodes.

    node_err_bounds : 2D numpy.ndarray[float]
        Allowed deviation from node values for nodes affected by
        error. This is a matrix with rows (lower_deviation, 
        upper_deviation).

    target_val : float
        Target function value at which we want to move the node.

    Returns
    -------
    (int, float)
        The index of the node and corresponding bumpiness value
        indicating the sought node in the list node_pos.
    """
    assert(isinstance(settings, RbfoptSettings))
    assert(isinstance(node_val, np.ndarray))
    assert(isinstance(node_err_bounds, np.ndarray))
    assert(len(node_val) == k)
    assert(isinstance(Amat, np.ndarray))
    assert(len(node_val) == len(node_err_bounds))

    # Extract the matrices Phi and P from
    Phimat = Amat[:k, :k]
    Pmat = Amat[:k, k:]

    min_bump_index, min_bump = None, float('Inf')
    # We only look at nodes that have some allowed variation, and for
    # which target_val falls within the specified range.
    nodes_to_process = np.where(
        (node_err_bounds[:,1] - node_err_bounds[:,0] > settings.eps_zero) *
        (node_val + node_err_bounds[:, 0] <= target_val) *
        (node_val + node_err_bounds[:, 0] <= target_val))[0]

    for i in nodes_to_process:
        # Compute bumpiness. Save original data.
        orig_node_val = node_val[i]
        orig_node_err_bounds = node_err_bounds[i]
        # Fix this node at the target value.
        node_val[i] = target_val
        node_err_bounds[i, :] = 0.0
        # Compute RBF interpolant.
        # Get coefficients for the exact RBF first
        rbf_l, rbf_h = ru.get_rbf_coefficients(settings, n, k,
                                               Amat, node_val)
        # And now the noisy version
        rbf_l, rbf_h = get_noisy_rbf_coefficients(
            settings, n, k, Phimat, Pmat, node_val, node_err_bounds,
            rbf_l, rbf_h)
        # Restore original values
        node_val[i] = orig_node_val
        node_err_bounds[i] = orig_node_err_bounds
        # Compute bumpiness using the formula \lambda^T \Phi \lambda
        bump = np.dot(np.dot(rbf_l, Phimat), rbf_l)
        if (bump < min_bump):
            min_bump_index, min_bump = i, bump

    return (min_bump_index, min_bump)

# -- end function


def get_bump_new_node(settings, n, k, node_pos, node_val, new_node,
                      node_err_bounds, target_val):
    """Compute the bumpiness with a new interpolation point.

    Computes the bumpiness of the interpolant obtained by setting a
    new node in a specified location, at value target_val.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfoptSettings`
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

    node_err_bounds : 2D numpy.ndarray[float]
        Allowed deviation from node values for nodes affected by
        error. This is a matrix with rows (lower_deviation, 
        upper_deviation).

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
    assert(isinstance(node_err_bounds, np.ndarray))
    assert(isinstance(settings, RbfoptSettings))
    assert(len(node_val) == k)
    assert(len(node_pos) == k)
    assert(len(node_val) == len(node_err_bounds))
    assert(new_node is not None)

    # Add the new node to existing ones
    n_node_pos = np.vstack((node_pos, new_node))
    n_node_val = np.append(node_val, target_val)
    n_node_err_bounds = np.vstack((node_err_bounds, np.array([[0, 0]])))

    # Compute the matrices necessary for the algorithm
    Amat = ru.get_rbf_matrix(settings, n, k + 1, n_node_pos)

    # Get coefficients for the exact RBF
    rbf_l, rbf_h = ru.get_rbf_coefficients(settings, n, k + 1, Amat,
                                             n_node_val)
    # Get RBF coefficients for noisy interpolant
    rbf_l, rbf_h = get_noisy_rbf_coefficients(
        settings, n, k + 1, Amat[:(k + 1), :(k + 1)],
        Amat[:(k + 1), (k + 1):], n_node_val,
        n_node_err_bounds, rbf_l, rbf_h)

    bumpiness = np.dot(np.dot(rbf_l, Amat[:(k+1), :(k+1)]), rbf_l)

    return bumpiness

# -- end function


def set_minlp_solver_options(solver):
    """Set MINLP solver options.

    Set the options of the MINLP solver.

    Parameters
    ----------
    solver: pyomo.opt.SolverFactory
        The solver interface.
    """
    minlp_solver_options = [('bonmin.num_resolve_at_root', 10),
                            ('bonmin.num_retry_unsolved_random_point', 5),
                            ('bonmin.num_resolve_at_infeasibles', 5),
                            ('bonmin.algorithm', 'B-BB'),
                            ('bonmin.time_limit', 45),
                            ('acceptable_tol', 1.0e-3),
                            ('max_cpu_time', 20),
                            ('max_iter', 1000),
                            ('bound_relax_factor', 0.0)]
    minlp_solver_rand_seed_option = 'bonmin.random_generator_seed'
    minlp_solver_max_seed = 2**31 - 2

    for (opt_name, opt_value) in minlp_solver_options:
        solver.options[opt_name] = opt_value
    if (minlp_solver_rand_seed_option is not None):
        solver.options[minlp_solver_rand_seed_option] = np.random.randint(minlp_solver_max_seed)

# -- end function


def set_nlp_solver_options(solver):
    """Set NLP solver options.

    Set the options of the NLP solver.

    Parameters
    ----------
    solver: pyomo.opt.SolverFactory
        The solver interface.
    """

    nlp_solver_options = [('acceptable_tol', 1.0e-3),
                          ('honor_original_bounds', 'yes'),
                          ('max_cpu_time', 20),
                          ('max_iter', 1000),
                          ('bound_relax_factor', 0.0)]
    nlp_solver_rand_seed_option = None
    nlp_solver_max_seed = 2**31 - 2

    for (opt_name, opt_value) in nlp_solver_options:
        solver.options[opt_name] = opt_value
    if (nlp_solver_rand_seed_option is not None):
        solver.options[nlp_solver_rand_seed_option] = np.random.randint(nlp_solver_max_seed)

# -- end function


def generate_sample_points(settings, n, var_lower, var_upper, integer_vars,
                           categorical_info, num_samples):
    """Generate sample points uniformly at random.

    Generate a given number of points uniformly at random in the
    bounding box, ensuring that integer variables take on integer
    values.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    var_lower : 1D numpy.ndarray[float]
        Vector of variable lower bounds.

    var_upper : 1D numpy.ndarray[float]
        Vector of variable upper bounds.

    integer_vars : 1D numpy.ndarray[int]
        A list containing the indices of the integrality constrained
        variables. If empty list, all variables are assumed to be
        continuous.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    num_samples : int
        Number of samples to generate

    Returns
    -------
    2D numpy.ndarray[float]
        A list of sample points (one for each row).
    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(isinstance(integer_vars, np.ndarray))

    assert(len(var_lower) == n)
    assert(len(var_upper) == n)
    assert(isinstance(settings, RbfoptSettings))

    if (categorical_info is not None and categorical_info[2]):
        # Map bounds and integer variables
        var_lower = ru.compress_categorical_bounds(var_lower,
                                                   *categorical_info)
        var_upper = ru.compress_categorical_bounds(var_upper,
                                                   *categorical_info)
        n = len(var_lower)
        integer_vars = ru.compress_categorical_integer_vars(
            integer_vars, *categorical_info)
        
    # Generate samples
    samples = (np.random.rand(num_samples, n) * (var_upper - var_lower) + 
               var_lower)

    # Round integer vars
    if (len(integer_vars)):
        samples[:, integer_vars] = np.around(samples[:, integer_vars])

    if (categorical_info is not None and categorical_info[2]):
        # Uncompress
        return ru.expand_categorical_vars(samples, *categorical_info)
        
    return samples

# -- end function

def ga_optimize(settings, n, var_lower, var_upper, integer_vars,
                categorical_info, objfun):
    """Compute and optimize a fitness function.

    Use a simple genetic algorithm to quickly find a good solution for
    a minimization subproblem.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    var_lower : 1D numpy.ndarray[float]
        Vector of variable lower bounds.

    var_upper : 1D numpy.ndarray[float]
        Vector of variable upper bounds.

    integer_vars : 1D numpy.ndarray[int]
        A list containing the indices of the integrality constrained
        variables. If empty list, all variables are assumed to be
        continuous.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    objfun : Callable[2D numpy.ndarray[float]]
        The objective function. This must be a callable function that
        can be applied to a list of points, and must return a list
        containing one fitness vale for each point, such that lower
        values are better.

    Returns
    -------
    1D numpy.ndarray[float]
        The best solution found.

    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(isinstance(integer_vars, np.ndarray))
    assert(len(var_lower) == n)
    assert(len(var_upper) == n)
    assert(isinstance(settings, RbfoptSettings))

    # Define parameters here, for now. Will move them to
    # rbfopt_settings later if it seems that the user should be able
    # to change their value.
    population_size = settings.ga_base_population_size + 20 * n//5
    mutation_rate = 0.1

    # Derived parameters. Since the best individual will always remain
    # and mutate, there is a -1 in the count for new individuals.
    num_surviving = population_size//4
    num_new = population_size - 2*num_surviving - 1

    if (categorical_info is not None and categorical_info[2]):
        # We will have to work in both the extended and compressed
        # space. Map bounds and integer variables
        categorical, not_categorical, expansion = categorical_info
        var_lower_comp = ru.compress_categorical_bounds(var_lower,
                                                        *categorical_info)
        var_upper_comp = ru.compress_categorical_bounds(var_upper,
                                                        *categorical_info)
        integer_vars_comp = ru.compress_categorical_integer_vars(
            integer_vars, *categorical_info)
        
        n_comp = len(categorical) + len(not_categorical)
        is_integer = np.zeros(len(var_lower_comp), dtype=bool)
        for index in integer_vars_comp:
            is_integer[index] = True

    else:
        # Generate boolean vector of integer variables for convenience
        is_integer = np.zeros(n, dtype=bool)
        if (len(integer_vars)):
            is_integer[integer_vars] = True

    # Compute initial population
    population = generate_sample_points(
        settings, n, var_lower, var_upper, integer_vars,
        categorical_info, population_size)
    for gen in range(settings.ga_num_generations):
        # Mutation rate and maximum perturbed coordinates for this
        # generation of individuals
        curr_mutation_rate = (mutation_rate *
                              (settings.ga_num_generations - gen) /
                              settings.ga_num_generations)
        # Compute fitness score to determine remaining individuals
        fitness_val = objfun(population)
        rank = np.argsort(fitness_val)
        best_individuals = population[rank[:num_surviving]]
        # Crossover: select how mating is done, then create offspring
        father = np.random.permutation(best_individuals)
        mother = np.random.permutation(best_individuals)
        if (categorical_info is not None and categorical_info[2]):
            father = ru.compress_categorical_vars(father, *categorical_info)
            mother = ru.compress_categorical_vars(mother, *categorical_info)
            offspring = ru.expand_categorical_vars(ga_mate(father, mother),
                                                   *categorical_info)
        else:
            offspring = ga_mate(father, mother)
        # New individuals
        new_individuals = generate_sample_points(
            settings, n, var_lower, var_upper, integer_vars,
            categorical_info, num_new)
        if (categorical_info is not None and categorical_info[2]):            
            # If there are categorical variables, we work in
            # compressed space, then expand again.
            # Compute perturbation.
            max_size_pert = min(n_comp,
                                max(2, int(n_comp * curr_mutation_rate)))
            # Map to compressed space
            best_individuals = ru.compress_categorical_vars(
                best_individuals, *categorical_info)
            # Make a copy of best individual, and mutate it
            best_mutated = best_individuals[0, :].copy()
            ga_mutate(n_comp, var_lower_comp, var_upper_comp, is_integer,
                      best_mutated, max_size_pert)
            best_mutated = ru.expand_categorical_vars(
                np.atleast_2d(best_mutated), *categorical_info)
            # Mutate surviving (except best) if necessary
            for point in best_individuals[1:]:
                if (np.random.uniform() < curr_mutation_rate):
                    ga_mutate(n_comp, var_lower_comp, var_upper_comp,
                              is_integer, point, max_size_pert)
            best_individuals = ru.expand_categorical_vars(
                best_individuals, *categorical_info)
        else:
            # We can work in original space. Compute perturbation.
            max_size_pert = min(n, max(2, int(n * curr_mutation_rate)))
            # Make a copy of best individual, and mutate it
            best_mutated = best_individuals[0, :].copy()
            ga_mutate(n, var_lower, var_upper, is_integer,
                      best_mutated, max_size_pert)
            # Mutate surviving (except best) if necessary
            for point in best_individuals[1:]:
                if (np.random.uniform() < curr_mutation_rate):
                    ga_mutate(n, var_lower, var_upper, is_integer,
                              point, max_size_pert)
        # Generate new population
        population = np.vstack((best_individuals, offspring,
                                new_individuals, best_mutated))
    # Determine ranking of last generation.
    # Compute fitness score to determine remaining individuals
    fitness_val = objfun(population)
    rank = np.argsort(fitness_val)

    # Return best individual
    return population[rank[0]]

# -- end function


def ga_mate(father, mother):
    """Generate offspring for genetic algorithm.

    The offspring will get genes uniformly at random from the mother
    and the father.

    Parameters
    ----------
    father : 2D numpy.ndarray[float]
        First set of individuals for mating.

    mother : 2D numpy.ndarray[float]
        Second set of individuals for mating.

    Returns
    -------
    2D numpy.ndarray(float)
        The offspring. Same size as mother and father.
    """
    assert(isinstance(mother, np.ndarray))
    assert(isinstance(father, np.ndarray))
    assert(father.shape == mother.shape)

    # Take elements from father or mother, depending on coin toss
    return np.where(np.random.uniform(size=father.shape) < 0.5,
                    father, mother)
# -- end function


def ga_mutate(n, var_lower, var_upper, is_integer, individual,
              max_size_pert):
    """Mutate an individual (point) for the genetic algorithm.

    The mutation is performed in place.

    Parameters
    ----------

    n : int
        The dimension of the problem, i.e. size of the space.

    var_lower : 1D numpy.ndarray[float]
        Vector of variable lower bounds.

    var_upper : 1D numpy.ndarray[float]
        Vector of variable upper bounds.

    is_integer : 1D numpy.ndarray[bool]
        List of size n, each element is True if the corresponding
        variable is integer.

    individual : 1D numpy.ndarray[float]
        Point to be mutated.

    max_size_pert : int
        Maximum size of the perturbation for the mutation,
        i.e. maximum number of coordinates that can change.

    """
    assert(max_size_pert <= n)

    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(isinstance(is_integer, np.ndarray))
    assert(isinstance(individual, np.ndarray))

    # Randomly mutate some of the coordinates. First determine how
    # many are mutated, then pick them randomly.
    size_pert = np.random.randint(max_size_pert)
    perturbed = np.random.choice(np.arange(n), size_pert, replace=False)
    new = (var_lower[perturbed] + np.random.rand(size_pert) * 
           (var_upper[perturbed] - var_lower[perturbed]))
    new[is_integer[perturbed]] = np.around(new[is_integer[perturbed]])
    individual[perturbed] = new

# -- end function


class MetricSRSMObj:
    """Objective function for the Metric SRM method.

    This class facilitates the computation of the objective function
    for the Metric SRSM. The objective function combines the distance
    from the closest point, and the response surface (i.e. RBF
    interpolant) value. Lower values are better.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes.

    rbf_lambda : 1D numpy.ndarray[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k. Can be
        None if dist_weight is equal to 1, in which case RBF values
        are not used.

    rbf_h : 1D numpy.ndarray[float]
        The h coefficients of the RBF interpolant, corresponding to
        the polynomial. List of dimension n+1. Can be None if
        dist_weight is equal to 1, in which case RBF values are not
        used.

    dist_weight : float
        Relative weight of the distance and objective function value.
        A weight of 1.0 corresponds to using solely distance, 0.0 to
        objective function.
    """
    def __init__(self, settings, n, k, node_pos, rbf_lambda,
                 rbf_h, dist_weight):
        """Constructor.
        """
        assert(isinstance(node_pos, np.ndarray))
        assert(isinstance(rbf_lambda, np.ndarray))
        assert(isinstance(rbf_h, np.ndarray))
        assert(len(node_pos) == k)
        assert(len(rbf_lambda) == k)
        assert(0 <= dist_weight <= 1)
        assert(isinstance(settings, RbfoptSettings))
        p = ru.get_size_P_matrix(settings, n)
        assert(len(rbf_h) == p)
        self.settings = settings
        self.n = n
        self.k = k
        self.node_pos = node_pos
        self.rbf_lambda = rbf_lambda
        self.rbf_h = rbf_h
        self.dist_weight = dist_weight
        self.obj_weight = (1.0 if settings.modified_msrsm_score
                           else (1 - dist_weight))
    # -- end function

    def evaluate(self, points):
        """Evaluate the objective for Metric SRSM.

        Evaluate the score of a set of points.

        Parameters
        ----------
        points : 2D numpy.ndarray[float]
            Points at which we want to evaluate the objective function
            (one for each row).

        Returns
        -------
        float
            The score for the Metric SRSM algorithm (lower is better).
        """
        assert(isinstance(points, np.ndarray))
        # Determine distance and surrogate model value
        obj, dist = ru.bulk_evaluate_rbf(self.settings, points, self.n,
                                         self.k, self.node_pos,
                                         self.rbf_lambda, self.rbf_h, 'min')
        # Determine scaling factors
        min_dist, max_dist = min(dist), max(dist)
        min_obj, max_obj = min(obj), max(obj)
        dist_denom = (max_dist - min_dist if max_dist > min_dist +
                      self.settings.eps_zero else 1.0)
        obj_denom = (max_obj - min_obj if max_obj > min_obj +
                     self.settings.eps_zero else 1.0)
        res = (self.dist_weight * (max_dist - dist) / dist_denom + 
               self.obj_weight * (obj - min_obj) / obj_denom)
        res[dist <= self.settings.min_dist] = float('+inf')
        return res
    # -- end function

# -- end class MetricSRSMObj


class MaximinDistanceObj:
    """Objective function for the Maximin Distance criterion.

    This class facilitates the computation of the objective function
    for the Maximin Distance criterion. The objective function is the
    minimum distance from the closest point, multiplied by -1 so that
    lower values are better (we always minimize).

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes (one for each row).
    """
    def __init__(self, settings, n, k, node_pos):
        """Constructor.
        """
        assert (isinstance(node_pos, np.ndarray))
        assert(len(node_pos) == k)
        assert(isinstance(settings, RbfoptSettings))
        self.settings = settings
        self.n = n
        self.k = k
        self.node_pos = node_pos
    # -- end function

    def evaluate(self, points):
        """Evaluate the objective for Maximin Distance.

        Evaluate the score of a set of points.

        Parameters
        ----------
        points : 2D numpy.ndarray[float]
            Points at which we want to evaluate the objective function
            (one for each row).

        Returns
        -------
        float
            The score for Maximin Distance algorithm (lower is better).
        """
        assert(isinstance(points, np.ndarray))
        dist = ru.bulk_get_min_distance(points, self.node_pos)
        return -dist
    # -- end function
# -- end class MaximinDistanceObj


class GutmannHkObj:
    """Objective function h_k for the Gutmann method.

    This class computes the value of the h_k objective function for
    the Gutmann method. Lower values are better.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes (one for each row).

    rbf_lambda : 1D numpy.ndarray[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k. Can be
        None if dist_weight is equal to 1, in which case RBF values
        are not used.

    rbf_h : 1D numpy.ndarray[float]
        The h coefficients of the RBF interpolant, corresponding to
        the polynomial. List of dimension n+1. Can be None if
        dist_weight is equal to 1, in which case RBF values are not
        used.

    Amatinv : 2D numpy.ndarray[float]
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0]. Must be a square 2D numpy.ndarray[float] of
        appropriate dimension.

    target_val : float
        Value f* that we want to find in the unknown objective
        function. Used by Gutmann's RBF method only.

    """
    def __init__(self, settings, n, k, node_pos, rbf_lambda,
                 rbf_h, Amatinv, target_val):
        """Constructor.
        """
        assert(isinstance(node_pos, np.ndarray))
        assert(isinstance(rbf_lambda, np.ndarray))
        assert(isinstance(rbf_h, np.ndarray))
        assert(len(rbf_lambda) == k)
        assert(len(node_pos) == k)
        assert(isinstance(settings, RbfoptSettings))
        # Determine the size of the P matrix
        p = ru.get_size_P_matrix(settings, n)
        assert(isinstance(Amatinv, np.ndarray) and
               Amatinv.shape == (k + p, k + p))
        assert(len(rbf_h) == p)

        self.settings = settings
        self.n = n
        self.k = k
        self.node_pos = node_pos
        self.rbf_lambda = rbf_lambda
        self.rbf_h = rbf_h
        self.Amatinv = Amatinv
        self.target_val = target_val
    # -- end function

    def evaluate(self, points):
        """Evaluate the objective for the Gutmann h_k objective.

        Compute -1/(\mu_k(x) [s_k(x) - f^\ast]^2)), where s_k is
        the value of the RBF interpolant, and f^\ast is the target
        value. This is because we want to maximize its negative.

        Parameters
        ----------
        points : 2D numpy.ndarray[float]
            Points at which we want to evaluate the objective function
            (one for each row).

        Returns
        -------
        float
            The score for the h_k criterion (lower is better).

        """
        assert(isinstance(points, np.ndarray))

        rbf_function = ru.get_rbf_function(self.settings)
        p = ru.get_size_P_matrix(self.settings, self.n)
        # Formula:
        # \sum_{i=1}^k \lambda_i \phi(\|x - x_i\|) + h^T (x 1)

        # Create distance matrix
        dist_mat = ss.distance.cdist(points, self.node_pos)
        # Evaluate radial basis function on each distance
        rbf_vec = rbf_function(dist_mat.ravel())
        u_mat = np.reshape(rbf_vec, (len(points), -1))
        # Contributions to the RBF interpolant value s_k: the u part,
        # the pi part, and the nonhomogenous part. At the same time,
        # build the matrix with the vectors u_pi.
        part1 = np.dot(u_mat, self.rbf_lambda)
        if (ru.get_degree_polynomial(self.settings) == 1):
            part2 = np.dot(points, self.rbf_h[:-1])
            u_pi_mat = np.concatenate((u_mat, points,
                                       np.ones((len(points), 1))), axis=1)
        elif (ru.get_degree_polynomial(self.settings) == 0):
            part2 = np.zeros(len(points))
            u_pi_mat = np.concatenate((u_mat, np.ones((len(points), 1))),
                                      axis=1)
        else:
            part2 = np.zeros(len(points))
            u_pi_mat = u_mat
        part3 = self.rbf_h[-1] if (p > 0) else 0.0
        # The vector rbf_value contains the value of the RBF interpolant
        rbf_value = part1 + part2 + part3
        # This is the shift in the computation of \mu_k
        shift = rbf_function(0.0)
        sign = (-1)**ru.get_degree_polynomial(self.settings)
        
        return -((sign * (np.sum(np.dot(u_pi_mat, np.array(self.Amatinv)) *
                                 u_pi_mat, axis=1) - shift)) / 
                 np.maximum((rbf_value - self.target_val)**2,
                            np.ones_like(rbf_value)*self.settings.eps_zero))

        # -- end function
# -- end class GutmannHkObj


class GutmannMukObj:
    """Objective function \mu_k for the Gutmann method.

    This class computes the value of the \mu_k objective function for
    the Gutmann method. Lower values are better.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfoptSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes (one for each row).

    Amatinv : 2D numpy.ndarray[float] or None
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0]. Must be a square 2D numpy.ndarray[float]
        of appropriate dimension.

    """
    def __init__(self, settings, n, k, node_pos, Amatinv):
        """Constructor.
        """
        assert(isinstance(node_pos, np.ndarray))
        assert(len(node_pos) == k)
        assert(isinstance(settings, RbfoptSettings))
        # Determine the size of the P matrix
        p = ru.get_size_P_matrix(settings, n)
        assert(isinstance(Amatinv, np.ndarray) and
               Amatinv.shape == (k + p, k + p))

        self.settings = settings
        self.n = n
        self.k = k
        self.node_pos = node_pos
        self.Amatinv = Amatinv
    # -- end function

    def evaluate(self, points):
        """Evaluate the objective for the Gutmann \mu objective.

        Compute -1/\mu_k(x), which we want to minimize.

        Parameters
        ----------
        points : 2D numpy.ndarray[float]
            Points at which we want to evaluate the objective function
            (one for each row).

        Returns
        -------
        float
            The score for the \mu_k criterion (lower is better).

        """
        assert(isinstance(points, np.ndarray))

        rbf_function = ru.get_rbf_function(self.settings)
        p = ru.get_size_P_matrix(self.settings, self.n)
        # Formula:
        # \sum_{i=1}^k \lambda_i \phi(\|x - x_i\|) + h^T (x 1)

        # Create distance matrix
        dist_mat = ss.distance.cdist(points, self.node_pos)
        # Evaluate radial basis function on each distance
        rbf_vec = rbf_function(dist_mat.ravel())
        u_mat = np.reshape(rbf_vec, (len(points), -1))
        # Build the matrix with the vectors u_pi.
        if (ru.get_degree_polynomial(self.settings) == 1):
            u_pi_mat = np.concatenate((u_mat, points,
                                       np.ones((len(points), 1))), axis=1)
        elif (ru.get_degree_polynomial(self.settings) == 0):
            u_pi_mat = np.concatenate((u_mat, np.ones((len(points), 1))),
                                      axis=1)
        elif (ru.get_degree_polynomial(self.settings) == -1):
            u_pi_mat = u_mat
        # This is the shift in the computation of \mu_k
        shift = rbf_function(0.0)
        sign = (-1)**ru.get_degree_polynomial(self.settings)

        return -(sign * (np.sum(np.dot(u_pi_mat, np.array(self.Amatinv)) * 
                                u_pi_mat, axis=1) +
                         shift))
    # -- end function
# -- end class GutmannMukObj
