"""Auxiliary problems for the optimization process.

This module is responsible for constructing and solving all the
auxiliary problems encountered during the optimization, such as the
minimization of the surrogate model, of the bumpiness. The module acts
as an interface between the high-level routines, the low-level PyOmo
modules, and the search algorithms.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
Research partially supported by SUTD-MIT International Design Center.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import math
import numpy as np
import scipy.spatial as ss
import pyomo.environ
import pyomo.opt
from . import rbfopt_utils as ru
import rbfopt_config as config
import rbfopt_degree1_models
import rbfopt_degree0_models
from rbfopt_settings import RbfSettings

def pure_global_search(settings, n, k, var_lower, var_upper,
                       integer_vars, node_pos, mat):
    """Pure global search that disregards objective function.

    If using Gutmann's RBF method, Construct a PyOmo model to maximize
    :math: `1/\mu`. If using the Metric SRM, select a point purely
    based on distance.

    See paper by Costa and Nannicini, equation (7) pag 4, and the
    references therein.

    Parameters
    ----------
 
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    var_lower : List[float]
        Vector of variable lower bounds.
    
    var_upper : List[float]
        Vector of variable upper bounds.

    integer_vars : List[int]
        A list containing the indices of the integrality constrained
        variables. If empty list, all variables are assumed to be
        continuous.

    node_pos : List[List[float]]
        List of coordinates of the nodes.

    mat : numpy.matrix or None
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0], see paper as cited above. Must
        be a square numpy.matrix of appropriate dimension if
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
    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(len(node_pos)==k)
    assert(isinstance(settings, RbfSettings))

    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))
    assert (isinstance(integer_vars, np.ndarray))

    # Determine the size of the P matrix
    p = ru.get_size_P_matrix(settings, n)
    assert((mat is None and settings.algorithm == 'MSRSM')
           or (isinstance(mat, np.matrix) and mat.shape==(k + p, k + p)))

    # Instantiate model
    if (ru.get_degree_polynomial(settings) == 1):
        model = rbfopt_degree1_models
    elif (ru.get_degree_polynomial(settings) == 0):
        model = rbfopt_degree0_models
    else:
        raise ValueError('RBF type ' + settings.rbf + ' not supported')

    if (settings.global_search_method == 'genetic'):
        # Use a genetic algorithm to optimize
        if (settings.algorithm == 'Gutmann'):
            fitness = GutmannMukObj(settings, n, k, node_pos, mat)
        elif (settings.algorithm == 'MSRSM'):
            fitness = MaximinDistanceObj(settings, n, k, node_pos)
        point = ga_optimize(settings, n, var_lower, var_upper,
                            integer_vars, fitness.bulk_evaluate)
    elif (settings.global_search_method == 'sampling'):
        # Sample random points, and rank according to fitness
        if (settings.algorithm == 'Gutmann'):
            fitness = GutmannMukObj(settings, n, k, node_pos, mat)
        elif (settings.algorithm == 'MSRSM'):
            fitness = MaximinDistanceObj(settings, n, k, node_pos)
        num_samples = n * settings.num_samples_aux_problems
        samples = generate_sample_points(settings, n, var_lower, var_upper,
                                         integer_vars, num_samples)
        scores = fitness.bulk_evaluate(samples)
        point = samples[scores.index(min(scores))]
    elif (settings.global_search_method == 'solver'):
        # Optimize using Pyomo    
        if (settings.algorithm == 'Gutmann'):
            instance = model.create_max_one_over_mu_model(settings, n, k,
                                                          var_lower, 
                                                          var_upper,
                                                          integer_vars,
                                                          node_pos, mat)
            # Initialize variables for local search
            initialize_instance_variables(settings, instance)
        elif (settings.algorithm == 'MSRSM'):
            instance = model.create_maximin_dist_model(settings, n, k, 
                                                       var_lower, var_upper,
                                                       integer_vars, node_pos)
            # Initialize variables for local search
            initialize_instance_variables(settings, instance, False)

        # Instantiate optimizer
        opt = pyomo.opt.SolverFactory(config.MINLP_SOLVER_NAME, 
                                      executable = 
                                      config.MINLP_SOLVER_PATH,
                                      solver_io='nl')
        if opt is None:
            raise RuntimeError('Solver ' + config.MINLP_SOLVER_NAME + 
                               ' not found')
        set_minlp_solver_options(opt)

        # Solve and load results
        try:
            results = opt.solve(instance, keepfiles = False, 
                                tee = settings.print_solver_output)
            if ((results.solver.status == pyomo.opt.SolverStatus.ok) and 
                (results.solver.termination_condition == 
                 pyomo.opt.TerminationCondition.optimal)):
                # this is feasible and optimal
                instance.solutions.load_from(results)
                point = [instance.x[i].value for i in instance.N]
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
                 node_pos, rbf_lambda, rbf_h):
    """Compute the minimum of the RBF interpolant.

    Compute the minimum of the RBF interpolant with a PyOmo model.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    var_lower : List[float]
        Vector of variable lower bounds.

    var_upper : List[float]
        Vector of variable upper bounds.

    integer_vars: List[int]
        A list containing the indices of the integrality constrained
        variables. If empty list, all variables are assumed to be
        continuous.

    node_pos : List[List[float]]
        List of coordinates of the nodes.

    rbf_lambda : List[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : List[float]
        The h coefficients of the RBF interpolant, corresponding to
        the polynomial. List of dimension n+1.

    Returns
    -------
    List[float]
        A minimizer. It is difficult to do global optimization so
        typically this method returns a local minimum.

    Raises
    ------
    ValueError
        If some parameters are not supported.
    RuntimeError
        If the solver cannot be found.
    """    

    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(len(rbf_lambda)==k)
    assert(len(node_pos)==k)
    assert(isinstance(settings, RbfSettings))

    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))
    assert (isinstance(integer_vars, np.ndarray))
    var_lower = var_lower.tolist()
    var_upper = var_upper.tolist()
    integer_vars = integer_vars.tolist()

    # Determine the size of the P matrix
    p = ru.get_size_P_matrix(settings, n)
    assert(len(rbf_h)==(p))

    # Instantiate model
    if (ru.get_degree_polynomial(settings) == 1):
        model = rbfopt_degree1_models
    elif (ru.get_degree_polynomial(settings) == 0):
        model = rbfopt_degree0_models
    else:
        raise ValueError('RBF type ' + settings.rbf + ' not supported')

    instance = model.create_min_rbf_model(settings, n, k, var_lower, 
                                          var_upper, integer_vars, 
                                          node_pos, rbf_lambda, rbf_h)

    # Initialize variables for local search
    initialize_instance_variables(settings, instance)

    # Instantiate optimizer
    opt = pyomo.opt.SolverFactory(config.MINLP_SOLVER_NAME, 
                                  executable = config.MINLP_SOLVER_PATH,
                                  solver_io='nl')
    if opt is None:
        raise RuntimeError('Solver ' + config.MINLP_SOLVER_NAME + 
                           'not found')
    set_minlp_solver_options(opt)

    # Solve and load results
    try:
        results = opt.solve(instance, keepfiles = False,
                            tee = settings.print_solver_output)
        if ((results.solver.status == pyomo.opt.SolverStatus.ok) and 
            (results.solver.termination_condition == 
             pyomo.opt.TerminationCondition.optimal)):
            # this is feasible and optimal
            instance.solutions.load_from(results)
            point = [instance.x[i].value for i in instance.N]
            ru.round_integer_vars(point, integer_vars)
        else:
            point = None
    except:
        point = None

    return point

# -- end function

def global_search(settings, n, k, var_lower, var_upper, integer_vars,
                  node_pos, rbf_lambda, rbf_h, mat, target_val, 
                  dist_weight, fmin, fmax):
    """Global search that tries to balance exploration/exploitation.

    If using Gutmann's RBF method, compute the maximum of the h_k
    function, see equation (8) in the paper by Costa and
    Nannicini. If using the Metric SRSM, select a point based on a
    combination of distance and objective function value.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    var_lower : List[float]
        Vector of variable lower bounds.

    var_upper : List[float]
        Vector of variable upper bounds.

    integer_vars: List[int]
        A list containing the indices of the integrality constrained
        variables. If empty list, all variables are assumed to be
        continuous.

    node_pos : List[List[float]]
        List of coordinates of the nodes.

    rbf_lambda : List[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : List[float]
        The h coefficients of the RBF interpolant, corresponding to
        the polynomial. List of dimension n+1.

    mat : numpy.matrix or None
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0], see paper as cited above. Must
        be a square numpy.matrix of appropriate dimension, or None if
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
    List[float]
        A local optimum. It is difficult to do global optimization so
        typically this method returns a local optimum.

    Raises
    ------
    ValueError
        If some parameters are not supported.
    RuntimeError
        If the solver cannot be found.

    """
    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(len(rbf_lambda)==k)
    assert(len(node_pos)==k)
    assert(0 <= dist_weight <= 1)
    assert(fmin <= fmax)
    assert(isinstance(settings, RbfSettings))

    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))
    assert (isinstance(integer_vars, np.ndarray))

    # Determine the size of the P matrix
    p = ru.get_size_P_matrix(settings, n)
    assert((mat is None and settings.algorithm == 'MSRSM' )
           or (isinstance(mat, np.matrix) and mat.shape==(k + p, k + p)))
    assert(len(rbf_h)==(p))

    # Instantiate model
    if (ru.get_degree_polynomial(settings) == 1):
        model = rbfopt_degree1_models
    elif (ru.get_degree_polynomial(settings) == 0):
        model = rbfopt_degree0_models
    else:
        raise ValueError('RBF type ' + settings.rbf + ' not supported')

    if (settings.global_search_method == 'genetic'):
        # Use a genetic algorithm to optimize
        if (settings.algorithm == 'Gutmann'):
            fitness = GutmannHkObj(settings, n, k, node_pos, rbf_lambda, 
                                   rbf_h, mat, target_val)
        elif (settings.algorithm == 'MSRSM'):
            fitness = MetricSRSMObj(settings, n, k, node_pos, rbf_lambda, 
                                    rbf_h, dist_weight)
        point = ga_optimize(settings, n, var_lower, var_upper,
                            integer_vars, fitness.bulk_evaluate)
    elif (settings.global_search_method == 'sampling'):
        # Sample random points, and rank according to fitness
        if (settings.algorithm == 'Gutmann'):
            fitness = GutmannHkObj(settings, n, k, node_pos, rbf_lambda, 
                                   rbf_h, mat, target_val)
        elif (settings.algorithm == 'MSRSM'):
            fitness = MetricSRSMObj(settings, n, k, node_pos, rbf_lambda, 
                                    rbf_h, dist_weight)
        num_samples = n * settings.num_samples_aux_problems
        samples = generate_sample_points(settings, n, var_lower, var_upper,
                                         integer_vars, num_samples)
        scores = fitness.bulk_evaluate(samples)
        point = samples[scores.index(min(scores))]
    elif (settings.global_search_method == 'solver'):
        # Optimize using Pyomo    
        if (settings.algorithm == 'Gutmann'):
            instance = model.create_max_h_k_model(settings, n, k,
                                                  var_lower, var_upper,
                                                  integer_vars, node_pos, 
                                                  rbf_lambda, rbf_h, mat, 
                                                  target_val)
            initialize_instance_variables(settings, instance)
            initialize_h_k_aux_variables(settings, instance)
        elif (settings.algorithm == 'MSRSM'):
            # Compute minimum and maximum distances between
            # points. This computation could be avoided if
            # OptAlgorithm keeps track of them, but in the grand
            # scheme of things the computation is rarely performed and
            # is not as expensive as the subsequent optimization.
            point_mat = np.array(node_pos)
            dist_mat = ss.distance.cdist(node_pos, node_pos)
            dist_min = np.min(dist_mat[np.triu_indices(k, 1)])
            dist_max = np.max(dist_mat[np.triu_indices(k, 1)])
            instance = model.create_min_msrsm_model(settings, n, k,
                                                    var_lower, var_upper,
                                                    integer_vars, node_pos,
                                                    rbf_lambda, rbf_h,
                                                    dist_weight, dist_min,
                                                    dist_max, fmin, fmax)
            initialize_instance_variables(settings, instance)
            initialize_msrsm_aux_variables(settings, instance)

        # Instantiate optimizer
        opt = pyomo.opt.SolverFactory(config.MINLP_SOLVER_NAME, 
                                      executable = config.MINLP_SOLVER_PATH,
                                      solver_io='nl')
        if opt is None:
            raise RuntimeError('Solver ' + config.MINLP_SOLVER_NAME + 
                               ' not found')
        set_minlp_solver_options(opt)

        # Solve and load results
        try:
            results = opt.solve(instance, keepfiles = False,
                                tee = settings.print_solver_output)
            if ((results.solver.status == pyomo.opt.SolverStatus.ok) and 
                (results.solver.termination_condition == 
                 pyomo.opt.TerminationCondition.optimal)):
                # this is feasible and optimal
                instance.solutions.load_from(results)
                point = [instance.x[i].value for i in instance.N]
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

def initialize_instance_variables(settings, instance, init_u_pi = True):
    """Initialize the variables of a problem instance.

    Initialize the x variables of a problem instance, and set the
    corresponding values for the vectors :math:`(u,\pi)`. This helps
    the local search by starting at a feasible point.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    instance : pyomo.ConcreteModel
        A concrete instance of mathematical optimization model.

    init_u_pi : bool
        Whether or not the u_pi variables should be initialized.
    """
    assert(isinstance(settings, RbfSettings))
    
    # Obtain radial basis function
    rbf = ru.get_rbf_function(settings)

    # Initialize variables for local search
    for i in instance.N:
        instance.x[i] = np.random.uniform(instance.var_lower[i], 
                                          instance.var_upper[i])
    if (init_u_pi):
        for j in instance.K:
            instance.u_pi[j] = rbf(math.sqrt(math.fsum((instance.x[i].value - 
                                                        instance.node[j, i])**2
                                                       for i in instance.N)))
        if (ru.get_degree_polynomial(settings) == 1):
            for j in instance.N:
                instance.u_pi[instance.k + j] = instance.x[j].value
                instance.u_pi[instance.k + instance.n] = 1.0
        elif (ru.get_degree_polynomial(settings) == 0):
            # We use "+ 0" here to convert the symbolic parameter
            # instance.k into a number that can be used for indexing.
            instance.u_pi[instance.k + 0] = 1.0

# -- end function

def initialize_h_k_aux_variables(settings, instance):
    """Initialize auxiliary variables for the h_k model.
    
    Initialize the rbfval and mu_k_inv variables of a problem
    instance, using the values for for x and u_pi already given. This
    helps the local search by starting at a feasible point.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    instance : pyomo.ConcreteModel
        A concrete instance of mathematical optimization model.
    """
    assert(isinstance(settings, RbfSettings))

    instance.rbfval = math.fsum(instance.lambda_h[i] * instance.u_pi[i].value
                                for i in instance.Q)
    instance.mu_k_inv = ((-1)**ru.get_degree_polynomial(settings) *
                         math.fsum(instance.Ainv[i,j] * instance.u_pi[i].value
                                   * instance.u_pi[j].value
                                   for i in instance.Q for j in instance.Q) + 
                         instance.phi_0)

# -- end function

def initialize_msrsm_aux_variables(settings, instance):
    """Initialize auxiliary variables for the MSRSM model.
    
    Initialize the rbfval and mindist variables of a problem
    instance, using the values for for x and u_pi already given. This
    helps the local search by starting at a feasible point.

    Parameters
    ----------
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    instance : pyomo.ConcreteModel
        A concrete instance of mathematical optimization model.
    """
    assert(isinstance(settings, RbfSettings))

    instance.rbfval = math.fsum(instance.lambda_h[i] * instance.u_pi[i].value
                                for i in instance.Q)
    dist = [sum((instance.x[j].value - instance.node[i, j])**2 
                for j in instance.N) for i in instance.K]
    instance.mindistsq = min(min(dist), config.DISTANCE_SHIFT)

# -- end function

def get_noisy_rbf_coefficients(settings, n, k, Phimat, Pmat, node_val,
                               fast_node_index, fast_node_err_bounds,
                               init_rbf_lambda = None, init_rbf_h = None):
    """Obtain coefficients for the noisy RBF interpolant.

    Solve a quadratic problem to compute the coefficients of the RBF
    interpolant that minimizes bumpiness and lets all points with
    index in fast_node_index deviate by a specified percentage from
    their value.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    Phimat : numpy.matrix
        Matrix Phi, i.e. top left part of the standard RBF matrix.

    Pmat : numpy.matrix
        Matrix P, i.e. top right part of the standard RBF matrix.

    node_val : List[float]
        List of values of the function at the nodes.
    
    fast_node_index : List[int]
        List of indices of nodes whose function value should be
        considered variable within the allowed range.
    
    fast_node_err_bounds : List[(float, float)]
        Allowed deviation from node values for nodes affected by
        error. This is a list of pairs (lower, upper) of the same
        length as fast_node_index.

    init_rbf_lambda : List[float] or None
        Initial values that should be used for the lambda coefficients
        of the RBF. Can be None.

    init_rbf_h : List[float] or None
        Initial values that should be used for the h coefficients of
        the RBF. Can be None.

    Returns
    ---
    (List[float], List[float])
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
    assert(isinstance(settings, RbfSettings))
    assert(len(node_val)==k)
    assert(isinstance(Phimat, np.matrix))
    assert(isinstance(Pmat, np.matrix))
    assert(len(fast_node_index)==len(fast_node_err_bounds))
    assert(init_rbf_lambda is None or len(init_rbf_lambda)==k)
    assert(init_rbf_h is None or len(init_rbf_h)==Pmat.shape[1])
    
    # Instantiate model
    if (ru.get_degree_polynomial(settings) == 1):
        model = rbfopt_degree1_models
    elif (ru.get_degree_polynomial(settings) == 0):
        model = rbfopt_degree0_models
    else:
        raise ValueError('RBF type ' + settings.rbf + ' not supported')

    instance = model.create_min_bump_model(settings, n, k, Phimat, Pmat,
                                           node_val, fast_node_index,
                                           fast_node_err_bounds)

    # Instantiate optimizer
    opt = pyomo.opt.SolverFactory(config.NLP_SOLVER_NAME, 
                                  executable = config.NLP_SOLVER_PATH,
                                  solver_io='nl')
    if opt is None:
        raise RuntimeError('Solver ' + config.NLP_SOLVER_NAME + ' not found')
    set_nlp_solver_options(opt)

    # Initialize instance variables with the solution provided (if
    # available). This should avoid any infeasibility.
    if (init_rbf_lambda is not None and init_rbf_h is not None):
        for i in range(len(init_rbf_lambda)):
            instance.rbf_lambda[i] = init_rbf_lambda[i]
            instance.slack[i] = 0.0
        for i in range(len(init_rbf_h)):
            instance.rbf_h[i] = init_rbf_h[i]

    # Solve and load results
    try:
        results = opt.solve(instance, keepfiles = False,
                            tee = settings.print_solver_output)
        if ((results.solver.status == pyomo.opt.SolverStatus.ok) and 
            (results.solver.termination_condition == 
             pyomo.opt.TerminationCondition.optimal)):
            # this is feasible and optimal
            instance.solutions.load_from(results)
            rbf_lambda = [instance.rbf_lambda[i].value for i in instance.K]
            rbf_h = [instance.rbf_h[i].value for i in instance.P]
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

def set_minlp_solver_options(solver):
    """Set MINLP solver options.

    Set the options of the MINLP solver, using the options indicated
    in the `rbfopt_config` module.
   
    Parameters
    ----------
    solver: pyomo.opt.SolverFactory
        The solver interface.
    """
    for (opt_name, opt_value) in config.MINLP_SOLVER_OPTIONS:
        solver.options[opt_name] = opt_value
    if (config.MINLP_SOLVER_RAND_SEED_OPTION is not None):
        solver.options[config.MINLP_SOLVER_RAND_SEED_OPTION] = np.random.randint(config.MINLP_SOLVER_MAX_SEED)

# -- end function

def set_nlp_solver_options(solver):
    """Set NLP solver options.

    Set the options of the NLP solver, using the options indicated in
    the `rbfopt_config` module.
   
    Parameters
    ----------
    solver: pyomo.opt.SolverFactory
        The solver interface.
    """

    for (opt_name, opt_value) in config.NLP_SOLVER_OPTIONS:
        solver.options[opt_name] = opt_value
    if (config.NLP_SOLVER_RAND_SEED_OPTION is not None):
        solver.options[config.NLP_SOLVER_RAND_SEED_OPTION] = np.random.randint(config.NLP_SOLVER_MAX_SEED)

# -- end function

def generate_sample_points(settings, n, var_lower, var_upper,
                           integer_vars, num_samples):
    """Generate sample points uniformly at random.

    Generate a given number of points uniformly at random in the
    bounding box, ensuring that integer variables take on integer
    values.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    var_lower : List[float]
        Vector of variable lower bounds.

    var_upper : List[float]
        Vector of variable upper bounds.

    integer_vars : List[int]
        A list containing the indices of the integrality constrained
        variables. If empty list, all variables are assumed to be
        continuous.

    num_samples : int
        Number of samples to generate

    Returns
    -------
    List[List[float]]
        A list of sample points.
    """

    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(isinstance(settings, RbfSettings))

    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))
    assert (isinstance(integer_vars, np.ndarray))
    var_lower = var_lower.tolist()
    var_upper = var_upper.tolist()
    integer_vars = integer_vars.tolist()

    values_by_var = list()
    for i in range(n):
        low = var_lower[i]
        up = var_upper[i]
        if (integer_vars is None or i not in integer_vars):
            values_by_var.append(np.random.uniform(low, up, (1, num_samples)))
        else:
            values_by_var.append(np.random.randint(low, up + 1,
                                                   (1, num_samples)))
    return np.array([[v[0, i] for v in values_by_var] for i in range(num_samples)])



# -- end function

def ga_optimize(settings, n, var_lower, var_upper, integer_vars, objfun):
    """Compute and optimize a fitness function.

    Use a simple genetic algorithm to quickly find a good solution for
    a minimization subproblem.

    Parameters
    ----------
 
    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    var_lower : List[float]
        Vector of variable lower bounds.
    
    var_upper : List[float]
        Vector of variable upper bounds.

    integer_vars : List[int]
        A list containing the indices of the integrality constrained
        variables. If empty list, all variables are assumed to be
        continuous.

    objfun : Callable[List[List[float]]]
        The objective function. This must be a callable function that
        can be applied to a list of points, and must return a list
        containing one fitness vale for each point, such that lower
        values are better.

    Returns
    -------
    List[float]
        The best solution found.

    """
    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(isinstance(settings, RbfSettings))

    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))
    assert (isinstance(integer_vars, np.ndarray))
    
    # Define parameters here, for now. Will move them to
    # rbfopt_settings later if it seems that the user should be able
    # to change their value.
    population_size = settings.ga_base_population_size + 20 * n//5
    mutation_rate = 0.1

    # Derived parameters. Since the best individual will always remain
    # and mutate, there is a -1 in the count for new individuals.
    num_surviving = population_size//4
    num_new = population_size - 2*num_surviving - 1

    # Generate boolean vector of integer variables for convenience
    is_integer = np.empty(n, dtype=bool)
    if len(integer_vars) > 0:
        is_integer[integer_vars] = True

    # Compute initial population
    population = generate_sample_points(settings, n, var_lower,
                                        var_upper, integer_vars,
                                        population_size)
    for gen in range(settings.ga_num_generations):
        # Mutation rate and maximum perturbed coordinates for this
        # generation of individuals
        curr_mutation_rate = (mutation_rate *
                              (settings.ga_num_generations - gen) /
                              settings.ga_num_generations)
        max_size_pert = min(n, max(2, int(n * curr_mutation_rate)))
        # Compute fitness score to determine remaining individuals
        fitness_val = objfun(population)
        rank = np.argsort(fitness_val)
        best_individuals = population[rank[:num_surviving]]
        # Crossover: select how mating is done, then create offspring
        father = np.random.permutation(best_individuals)
        mother = np.random.permutation(best_individuals)
        offspring = map(ga_mate, father, mother)
        # New individuals
        new_individuals = generate_sample_points(settings, n, var_lower, 
                                                 var_upper, integer_vars,
                                                 num_new)
        # Make a copy of best individual, and mutate it
        best_mutated = best_individuals[0,:].copy()
        ga_mutate(n, var_lower, var_upper, is_integer, 
                  best_mutated, max_size_pert)
        # Mutate surviving (except best) if necessary
        for point in best_individuals[1:]:
            if (np.random.uniform() < curr_mutation_rate):
                ga_mutate(n, var_lower, var_upper, is_integer, 
                          point, max_size_pert)
        # Generate new population
        population = np.vstack((best_individuals, offspring, new_individuals,
                                best_mutated))
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
    father : List[float]
        First individual for mating.

    mother : List[float]
        Second individual for mating.

    Returns
    -------
    List[float]
        The offspring. Same size as mother and father.
    """
    assert(len(father) == len(mother))

    # OLD VERSION
    # prob = np.random.uniform(size = len(father))
    # return [(father[i] if prob[i] < 0.5 else mother[i])
    #         for i in range(len(father))]

    n = len(father)
    offspring = np.empty(n, np.float64)
    prob = np.random.uniform(size=n)
    for i in range(n):
        if prob[i] < 0.5:
            offspring[i] = father[i]
        else:
            offspring[i] = mother[i]
    return offspring

# -- end function

def ga_mutate(n, var_lower, var_upper, is_integer, individual, 
              max_size_pert):
    """Mutate an individual (point) for the genetic algorithm.

    The mutation is performed in place.

    Parameters
    ----------

    n : int
        The dimension of the problem, i.e. size of the space.

    var_lower : List[float]
        Vector of variable lower bounds.
    
    var_upper : List[float]
        Vector of variable upper bounds.

    is_integer : List[bool]
        List of size n, each element is True if the corresponding
        variable is integer.

    individual : List[float]
        Point to be mutated.

    max_size_pert : int
        Maximum size of the perturbation for the mutation,
        i.e. maximum number of coordinates that can change.

    """
    assert(max_size_pert <= n)

    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))

    # Randomly mutate some of the coordinates. First determine how
    # many are mutated, then pick them randomly.
    size_pert = np.random.randint(max_size_pert)
    perturbed = np.random.choice(np.arange(n), size_pert, replace = False)
    for i in perturbed:
        if is_integer[i]:
            individual[i] = np.random.randint(var_lower[i], var_upper[i] + 1)
        else:
            individual[i] = np.random.uniform(var_lower[i], var_upper[i])

# -- end function    

class MetricSRSMObj:
    """Objective function for the Metric SRM method.

    This class facilitates the computation of the objective function
    for the Metric SRSM. The objective function combines the distance
    from the closest point, and the response surface (i.e. RBF
    interpolant) value. Lower values are better.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : List[List[float]]
        List of coordinates of the nodes.

    rbf_lambda : List[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k. Can be
        None if dist_weight is equal to 1, in which case RBF values
        are not used.

    rbf_h : List[float]
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
        assert(len(node_pos)==k)
        assert(len(rbf_lambda)==k)
        assert(0 <= dist_weight <= 1)
        assert(isinstance(settings, RbfSettings))
        p = ru.get_size_P_matrix(settings, n)
        assert(len(rbf_h)==(p))
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
    
    def bulk_evaluate(self, points):
        """Evaluate the objective for Metric SRSM.

        Evaluate the score of a set of points.

        Parameters
        ----------
        points : List[List[float]]
            Points at which we want to evaluate the objective function.
        
        Returns
        -------
        float
            The score for the Metric SRSM algorithm (lower is better).
        """
        # Determine distance and surrogate model value
        obj, dist = ru.bulk_evaluate_rbf(self.settings, points, self.n,
                                         self.k, self.node_pos, 
                                         self.rbf_lambda, self.rbf_h, 'min')
        # Determine scaling factors
        min_dist, max_dist = min(dist), max(dist)
        min_obj, max_obj = min(obj), max(obj)
        self.dist_denom = (max_dist - min_dist if max_dist > min_dist +
                           self.settings.eps_zero else 1.0)
        self.obj_denom = (max_obj - min_obj if max_obj > min_obj +
                          self.settings.eps_zero else 1.0)
        # Store useful parameters
        self.max_dist = max_dist
        self.min_obj = min_obj
        return map(self.evaluate, dist, obj)
    # -- end function

    def evaluate(self, distance, objfun):
        """Evaluate the objective for Metric SRSM.

        Evaluate the score of a single point, given its distance value
        and its RBF interpolant value.

        Parameters
        ----------
        distance : float
            Minimum distance of the point w.r.t. the interpolation
            points.

        objfun : float
            Value of the RBF interpolant at this point.
        
        Returns
        -------
        float
            The score for the Metric SRSM algorithm (lower is better).

        """
        if (distance <= self.settings.min_dist):
            return float('inf')
        dist_score = (self.max_dist - distance)/self.dist_denom
        obj_score = (objfun - self.min_obj)/self.obj_denom
        return (self.obj_weight * obj_score + 
                self.dist_weight * dist_score)
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

    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : List[List[float]]
        List of coordinates of the nodes.
    """
    def __init__(self, settings, n, k, node_pos):
        """Constructor.
        """
        assert(len(node_pos)==k)
        assert(isinstance(settings, RbfSettings))
        self.settings = settings
        self.n = n
        self.k = k
        self.node_pos = node_pos
    # -- end function
    
    def bulk_evaluate(self, points):
        """Evaluate the objective for Maximin Distance.

        Evaluate the score of a set of points.

        Parameters
        ----------
        points : List[List[float]]
            Points at which we want to evaluate the objective function.
        
        Returns
        -------
        float
            The score for Maximin Distance algorithm (lower is better).
        """
        dist = ru.bulk_get_min_distance(points, self.node_pos)
        return [-val for val in dist]
    # -- end function
# -- end class MaximinDistanceObj

class GutmannHkObj:
    """Objective function h_k for the Gutmann method.

    This class computes the value of the h_k objective function for
    the Gutmann method. Lower values are better.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : List[List[float]]
        List of coordinates of the nodes.

    rbf_lambda : List[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k. Can be
        None if dist_weight is equal to 1, in which case RBF values
        are not used.

    rbf_h : List[float]
        The h coefficients of the RBF interpolant, corresponding to
        the polynomial. List of dimension n+1. Can be None if
        dist_weight is equal to 1, in which case RBF values are not
        used.

    Amatinv : numpy.matrix or None
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0]. Must be a square numpy.matrix of
        appropriate dimension.

    target_val : float
        Value f* that we want to find in the unknown objective
        function. Used by Gutmann's RBF method only.

    """
    def __init__(self, settings, n, k, node_pos, rbf_lambda, 
                 rbf_h, Amatinv, target_val):
        """Constructor.
        """
        assert(len(rbf_lambda)==k)
        assert(len(node_pos)==k)
        assert(isinstance(settings, RbfSettings))
        # Determine the size of the P matrix
        p = ru.get_size_P_matrix(settings, n)
        assert(isinstance(Amatinv, np.matrix) and 
               Amatinv.shape==(k + p, k + p))
        assert(len(rbf_h)==(p))

        self.settings = settings
        self.n = n
        self.k = k
        self.node_pos = node_pos
        self.rbf_lambda = rbf_lambda
        self.rbf_h = rbf_h
        self.Amatinv = Amatinv
        self.target_val = target_val
    # -- end function
    
    def bulk_evaluate(self, points):
        """Evaluate the objective for the Gutmann h_k objective.

        Compute -1/(\mu_k(x) [s_k(x) - f^\ast]^2)), where s_k is
        the value of the RBF interpolant, and f^\ast is the target
        value. This is because we want to maximize its negative.

        Parameters
        ----------
        points : List[List[float]]
            Points at which we want to evaluate the objective function.
        
        Returns
        -------
        float
            The score for the h_k criterion (lower is better).

        """
        rbf_function = ru.get_rbf_function(self.settings)
        p = ru.get_size_P_matrix(self.settings, self.n)
        # Formula:
        # \sum_{i=1}^k \lambda_i \phi(\|x - x_i\|) + h^T (x 1)

        # Convert to numpy
        point_mat = np.array(points)
        node_mat = np.array(self.node_pos)
        # Create distance matrix
        dist_mat = ss.distance.cdist(point_mat, node_mat)
        # Evaluate radial basis function on each distance
        rbf_vec = map(rbf_function, dist_mat.ravel())
        u_mat = np.reshape(np.array(rbf_vec), (len(point_mat), -1))
        # Contributions to the RBF interpolant value s_k: the u part,
        # the pi part, and the nonhomogenous part. At the same time,
        # build the matrix with the vectors u_pi.
        part1 = np.dot(u_mat, self.rbf_lambda)
        if (ru.get_degree_polynomial(self.settings) == 1):
            part2 = np.dot(point_mat, self.rbf_h[:-1])
            u_pi_mat = np.concatenate((u_mat, point_mat, 
                                       np.ones((len(point_mat), 1))), axis=1)
        else:
            part2 = np.zeros(len(point_mat))
            u_pi_mat = np.concatenate((u_mat, np.ones((len(point_mat), 1))),
                                      axis=1)
        part3 = self.rbf_h[-1] if (p > 0) else 0.0
        # The vector rbf_value contains the value of the RBF interpolant
        rbf_value = part1 + part2 + part3
        # This is the shift in the computation of \mu_k
        shift = rbf_function(0.0)
        sign = (-1)**ru.get_degree_polynomial(self.settings)
        return [-(sign * (np.dot(np.dot(u_pi_mat[i, ], np.array(self.Amatinv)),
                                 u_pi_mat[i, ]) - shift)) /
                (rbf_value[i] - self.target_val)**2
                for i in range(len(points))]
    # -- end function
# -- end class GutmannHkObj

class GutmannMukObj:
    """Objective function \mu_k for the Gutmann method.

    This class computes the value of the \mu_k objective function for
    the Gutmann method. Lower values are better.

    Parameters
    ----------

    settings : :class:`rbfopt_settings.RbfSettings`
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : List[List[float]]
        List of coordinates of the nodes.

    Amatinv : numpy.matrix or None
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0]. Must be a square numpy.matrix of
        appropriate dimension.

    """
    def __init__(self, settings, n, k, node_pos, Amatinv):
        """Constructor.
        """
        assert(len(node_pos)==k)
        assert(isinstance(settings, RbfSettings))
        # Determine the size of the P matrix
        p = ru.get_size_P_matrix(settings, n)
        assert(isinstance(Amatinv, np.matrix) and 
               Amatinv.shape==(k + p, k + p))

        self.settings = settings
        self.n = n
        self.k = k
        self.node_pos = node_pos
        self.Amatinv = Amatinv
    # -- end function
    
    def bulk_evaluate(self, points):
        """Evaluate the objective for the Gutmann \mu objective.

        Compute -1/\mu_k(x), which we want to minimize.

        Parameters
        ----------
        points : List[List[float]]
            Points at which we want to evaluate the objective function.
        
        Returns
        -------
        float
            The score for the \mu_k criterion (lower is better).

        """
        rbf_function = ru.get_rbf_function(self.settings)
        p = ru.get_size_P_matrix(self.settings, self.n)
        # Formula:
        # \sum_{i=1}^k \lambda_i \phi(\|x - x_i\|) + h^T (x 1)

        # Convert to numpy
        point_mat = np.array(points)
        node_mat = np.array(self.node_pos)
        # Create distance matrix
        dist_mat = ss.distance.cdist(point_mat, node_mat)
        # Evaluate radial basis function on each distance
        rbf_vec = map(rbf_function, dist_mat.ravel())
        u_mat = np.reshape(np.array(rbf_vec), (len(point_mat), -1))
        # Build the matrix with the vectors u_pi.
        if (ru.get_degree_polynomial(self.settings) == 1):
            u_pi_mat = np.concatenate((u_mat, point_mat, 
                                       np.ones((len(point_mat), 1))), axis=1)
        else:
            u_pi_mat = np.concatenate((u_mat, np.ones((len(point_mat), 1))),
                                      axis=1)
        # This is the shift in the computation of \mu_k
        shift = rbf_function(0.0)
        sign = (-1)**ru.get_degree_polynomial(self.settings)
        return [-(sign * (np.dot(np.dot(u_pi_mat[i, ], np.array(self.Amatinv)),
                                 u_pi_mat[i, ]) + shift)) 
                for i in range(len(points))]
    # -- end function
# -- end class GutmannMukObj
