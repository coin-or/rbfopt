"""Auxiliary problems for the optimization process.

This module is responsible for constructing and solving all the
auxiliary problems encountered during the optimization, such as the
minimization of the surrogate model, of the bumpiness. The module acts
as an interface between the high-level routines, the low-level PyOmo
modules, and the sampling schemes.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
Research partially supported by SUTD-MIT International Design Center.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import math
import random
import numpy as np
import pyomo.environ
import pyomo.opt
from pyomo.opt import SolverStatus, TerminationCondition
import rbfopt_utils as ru
import rbfopt_config as config
import rbfopt
import rbfopt_degree1_models
import rbfopt_degree0_models
from rbfopt_settings import RbfSettings
import time

def pure_global_search(settings, n, k, var_lower, var_upper, node_pos,
                       mat, integer_vars):
    """Pure global search that disregards objective function.

    If using Gutmann's RBF method, Construct a PyOmo model to maximize
    :math: `1/\mu`. If using the Metric SRM, select a point purely
    based on distance.

    See paper by Costa and Nannicini, equation (7) pag 4, and the
    references therein.

    Parameters
    ----------
 
    settings : rbfopt_settings.RbfSettings
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    var_lower : List[float]
        Vector of variable lower bounds.
    
    var_upper : List[float]
        Vector of variable upper bounds.

    node_pos : List[List[float]]
        List of coordinates of the nodes

    mat : numpy.matrix
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0], see paper as cited above. Must
        be a square numpy.matrix of appropriate dimension.

    integer_vars : List[int] or None
        A list containing the indices of the integrality constrained
        variables. If None or empty list, all variables are assumed to
        be continuous.

    Returns
    -------
    float
        A maximizer. It is difficult to do global optimization so
        typically this method returns a local maximum.

    Raises
    ------
    ValueError
        If the type of radial basis function is not supported.
    RuntimeError
        If the solver cannot be found.
    """
    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(len(node_pos)==k)
    assert(isinstance(mat, np.matrix))
    assert(isinstance(settings, RbfSettings))

    # Determine the size of the P matrix
    p = ru.get_size_P_matrix(settings, n)
    assert(mat.shape==(k + p, k + p))

    # Instantiate model
    if (ru.get_degree_polynomial(settings) == 1):
        model = rbfopt_degree1_models
    elif (ru.get_degree_polynomial(settings) == 0):
        model = rbfopt_degree0_models
    else:
        raise ValueError('RBF type ' + settings.rbf + ' not supported')

    if (settings.algorithm == 'Gutmann'):
        # Optimize using Pyomo
        instance = model.create_max_one_over_mu_model(settings, n, k,
                                                      var_lower,
                                                      var_upper,
                                                      node_pos, mat,
                                                      integer_vars)

        # Initialize variables for local search
        initialize_instance_variables(settings, instance)

        # Instantiate optimizer
        opt = pyomo.opt.SolverFactory(config.MINLP_SOLVER_EXEC, solver_io='nl')
        if opt is None:
            raise RuntimeError('Solver ' + config.MINLP_SOLVER_EXEC + 
                               ' not found')
        set_minlp_solver_options(opt)

        # Solve and load results
        try:
            results = opt.solve(instance, keepfiles = False, 
                                tee = settings.print_solver_output)
            if ((results.solver.status == pyomo.opt.SolverStatus.ok) and 
                (results.solver.termination_condition == 
                 TerminationCondition.optimal)):
                # this is feasible and optimal
                instance.solutions.load_from(results)
                point = [instance.x[i].value for i in instance.N]
                ru.round_integer_vars(point, integer_vars)
            else:
                point = None
        except:
            point = None
    elif (settings.algorithm == 'MSRSM'):
        samples = generate_sample_points(settings, n, var_lower,
                                         var_upper, integer_vars)
        distance = ru.bulk_get_min_distance(samples, node_pos)
        point = samples[values.index(max(distance))]
    else:
        raise ValueError('Algorithm ' + settings.algorithm + ' not supported')

    return point

# -- end function

def minimize_rbf(settings, n, k, var_lower, var_upper, node_pos,
                 rbf_lambda, rbf_h, integer_vars):
    """Compute the minimum of the RBF interpolant.

    Compute the minimum of the RBF interpolant with a PyOmo model.

    Parameters
    ----------

    settings : rbfopt_settings.RbfSettings
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    var_lower : List[float]
        Vector of variable lower bounds.

    var_upper : List[float]
        Vector of variable upper bounds.

    node_pos : List[List[float]]
        List of coordinates of the nodes.

    rbf_lambda : List[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : List[float]
        The h coefficients of the RBF interpolant, corresponding to
        the polynomial. List of dimension n+1.

    integer_vars: List[int] or None
        A list containing the indices of the integrality constrained
        variables. If None or empty list, all variables are assumed to
        be continuous.

    Returns
    -------
    float
        A minimizer. It is difficult to do global optimization so
        typically this method returns a local minimum.

    Raises
    ------
    ValueError
        If the type of radial basis function is not supported.
    RuntimeError
        If the solver cannot be found.
    """    

    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(len(rbf_lambda)==k)
    assert(len(node_pos)==k)
    assert(isinstance(settings, RbfSettings))

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
                                          var_upper, node_pos, rbf_lambda,
                                          rbf_h, integer_vars)

    # Initialize variables for local search
    initialize_instance_variables(settings, instance)

    # Instantiate optimizer
    opt = pyomo.opt.SolverFactory(config.MINLP_SOLVER_EXEC, solver_io='nl')
    if opt is None:
        raise RuntimeError('Solver ' + config.MINLP_SOLVER_EXEC + 
                           'not found')
    set_minlp_solver_options(opt)

    # Solve and load results
    try:
        results = opt.solve(instance, keepfiles = False,
                            tee = settings.print_solver_output)
        if ((results.solver.status == pyomo.opt.SolverStatus.ok) and 
            (results.solver.termination_condition == 
             TerminationCondition.optimal)):
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

def global_search(settings, n, k, var_lower, var_upper, node_pos, rbf_lambda, 
                  rbf_h, mat, target_val, dist_weight, integer_vars):
    """Global search that tries to balance exploration/exploitation.

    If using Gutmann's RBF method, compute the maximum of the h_k
    function, see equation (8) in the paper by Costa and
    Nannicini. If using the Metric SRSM, select a point based on a
    combination of distance and objective function value.

    Parameters
    ----------

    settings : rbfopt_settings.RbfSettings
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    k : int
        Number of nodes, i.e. interpolation points.

    var_lower : List[float]
        Vector of variable lower bounds.

    var_upper : List[float]
        Vector of variable upper bounds.

    node_pos : List[List[float]]
        List of coordinates of the nodes.

    rbf_lambda : List[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : List[float]
        The h coefficients of the RBF interpolant, corresponding to
        the polynomial. List of dimension n+1.

    mat : numpy.matrix
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0], see paper as cited above. Must
        be a square numpy.matrix of appropriate dimension.

    target_val : float
        Value f* that we want to find in the unknown objective
        function. Used by Gutmann's RBF method only.

    dist_weight : float
        Relative weight of the distance and objective function value,
        when selecting the next point with a sampling strategy. A
        weight of 1.0 corresponds to using solely distance, 0.0 to
        objective function. Used by Metric SRSM only.

    integer_vars: List[int] or None
        A list containing the indices of the integrality constrained
        variables. If None or empty list, all variables are assumed to
        be continuous.

    Returns
    -------
    float
        A maximizer. It is difficult to do global optimization so
        typically this method returns a local maximum.

    Raises
    ------
    ValueError
        If the type of radial basis function is not supported.
    RuntimeError
        If the solver cannot be found.
    """
    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(len(rbf_lambda)==k)
    assert(len(node_pos)==k)
    assert(isinstance(mat, np.matrix))
    assert(isinstance(settings, RbfSettings))

    # Determine the size of the P matrix
    p = ru.get_size_P_matrix(settings, n)
    assert(mat.shape==(k + p, k + p))
    assert(len(rbf_h)==(p))

    # Instantiate model
    if (ru.get_degree_polynomial(settings) == 1):
        model = rbfopt_degree1_models
    elif (ru.get_degree_polynomial(settings) == 0):
        model = rbfopt_degree0_models
    else:
        raise ValueError('RBF type ' + settings.rbf + ' not supported')

    if (settings.algorithm == 'Gutmann'):
        # Optimize using Pyomo    
        instance = model.create_max_h_k_model(settings, n, k,
                                              var_lower, var_upper,
                                              node_pos, rbf_lambda,
                                              rbf_h, mat, target_val,
                                              integer_vars)

        # Initialize variables for local search
        initialize_instance_variables(settings, instance)
        initialize_h_k_aux_variables(settings, instance)

        # Instantiate optimizer
        opt = pyomo.opt.SolverFactory(config.MINLP_SOLVER_EXEC, solver_io='nl')
        if opt is None:
            raise RuntimeError('Solver ' + config.MINLP_SOLVER_EXEC + 
                               ' not found')
        set_minlp_solver_options(opt)

        # Solve and load results
        try:
            results = opt.solve(instance, keepfiles = False,
                                tee = settings.print_solver_output)
            if ((results.solver.status == pyomo.opt.SolverStatus.ok) and 
                (results.solver.termination_condition == 
                 TerminationCondition.optimal)):
                # this is feasible and optimal
                instance.solutions.load_from(results)
                point = [instance.x[i].value for i in instance.N]
                ru.round_integer_vars(point, integer_vars)
            else:
                point = None
        except:
            point = None
    elif (settings.algorithm == 'MSRSM'):
        samples = generate_sample_points(settings, n, var_lower,
                                         var_upper, integer_vars)
        # Compute distance and objective function value
        objfun, distance = ru.bulk_evaluate_rbf(settings, samples, n,
                                                k, node_pos, rbf_lambda, 
                                                rbf_h, True)
        srms_obj = MetricSRSMObj(settings, distance, objfun, dist_weight)
        scores = map(srms_obj.evaluate, distance, objfun)
        point = samples[scores.index(min(scores))]
    else:
        raise ValueError('Algorithm ' + settings.algorithm + ' not supported')

    return point

# -- end function

def initialize_instance_variables(settings, instance):
    """Initialize the variables of a problem instance.

    Initialize the x variables of a problem instance, and set the
    corresponding values for the vectors :math:`(u,\pi)`. This helps
    the local search by starting at a feasible point.

    Parameters
    ----------
    settings : rbfopt_settings.RbfSettings
        Global and algorithmic settings.

    instance : pyomo.ConcreteModel
        A concrete instance of mathematical optimization model.
    """
    assert(isinstance(settings, RbfSettings))
    
    # Obtain radial basis function
    rbf = ru.get_rbf_function(settings)

    # Initialize variables for local search
    for i in instance.N:
        instance.x[i] = random.uniform(instance.var_lower[i], 
                                       instance.var_upper[i])
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
    settings : rbfopt_settings.RbfSettings
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

    settings : rbfopt_settings.RbfSettings
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
        considered variable withing the allowed range.
    
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
        If the type of radial basis function is not supported.
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
    opt = pyomo.opt.SolverFactory(config.NLP_SOLVER_EXEC, solver_io='nl')
    if opt is None:
        raise RuntimeError('Solver ' + config.NLP_SOLVER_EXEC + ' not found')
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
             TerminationCondition.optimal)):
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
        solver.options[config.MINLP_SOLVER_RAND_SEED_OPTION] = random.randint(0, config.MINLP_SOLVER_MAX_SEED)

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
        solver.options[config.NLP_SOLVER_RAND_SEED_OPTION] = random.randint(0, config.NLP_SOLVER_MAX_SEED)

# -- end function

def generate_sample_points(settings, n, var_lower, var_upper,
                           integer_vars = None):
    """Generate sample points uniformly at random.

    Generate a given number of points uniformly at random in the
    bounding box, ensuring that integer variables take on integer
    values.

    Parameters
    ----------

    settings : rbfopt_settings.RbfSettings
        Global and algorithmic settings.

    n : int
        The dimension of the problem, i.e. size of the space.

    var_lower : List[float]
        Vector of variable lower bounds.

    var_upper : List[float]
        Vector of variable upper bounds.

    integer_vars: List[int] or None
        A list containing the indices of the integrality constrained
        variables. If None or empty list, all variables are assumed to
        be continuous.

    Returns
    -------
    List[List[float]]
        A list of sample points.
    """
    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(isinstance(settings, RbfSettings))
    num_samples = n * settings.num_samples_aux_problems
    values_by_var = list()
    for i in range(n):
        low = var_lower[i]
        up = var_upper[i]
        if (integer_vars is None or i not in integer_vars):
            values_by_var.append(np.random.uniform(low, up, (1, num_samples)))
        else:
            values_by_var.append(np.random.randint(low, up + 1, 
                                                   (1, num_samples)))
    return ([[v[0, i] for v in values_by_var] for i in range(num_samples)])

# -- end function

class MetricSRSMObj:
    """Objective functon for the Metric SRM method.

    This class facilitates the computation of the objective function
    for the Metric SRSM. The objective function combines the distance
    from the closest point, and the response surface (i.e. RBF
    interpolant) value.

    Parameters
    ----------
    distance_values : List[float]
        Minimum distance of sample points from interpolation nodes.

    objfun_values : List[float]
        Value of the RBF interpolant at the sample points. This array
        must have the same length as `distance`.

    dist_weight : float
        Relative weight of the distance and objective function value.
        A weight of 1.0 corresponds to using solely distance, 0.0 to
        objective function.
    """
    def __init__(self, settings, distance_values, objfun_values,
                 dist_weight):
        """Constructor.
        """
        assert(len(distance_values) == len(objfun_values))
        assert(0 <= dist_weight <= 1)
        assert(isinstance(settings, RbfSettings))
        # Determine scaling factors
        min_dist, max_dist = min(distance_values), max(distance_values)
        min_obj, max_obj = min(objfun_values), max(objfun_values)
        self.dist_denom = (max_dist - min_dist if max_dist > min_dist
                           + settings.eps_zero else 1.0)
        self.obj_denom = (max_obj - min_obj if max_obj > min_obj +
                          settings.eps_zero else 1.0)
        # Store useful parameters
        self.max_dist = max_dist
        self.min_obj = min_obj
        self.dist_weight = dist_weight
        self.min_dist = settings.min_dist

    # -- end function        
    
    def evaluate(self, distance, objfun):
        """Evaluate the objective for Metric SRSM.

        Evaluate the score of a point, given its distance value and
        its RBF interpolant value.

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
        if (distance <= self.min_dist):
            return float('inf')
        dist_score = (self.max_dist - distance)/self.dist_denom
        obj_score = (objfun - self.min_obj)/self.obj_denom
        return obj_score + self.dist_weight * dist_score

    # -- end function

# -- end Class MetricSRSMObj
