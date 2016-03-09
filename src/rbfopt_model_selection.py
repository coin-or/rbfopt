"""Models and routines for automatic model selection.

This module contains the functions to perform leave-one-out
cross-validation, and the corresponding mathematical models to speed
the cross validation up.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2015.
Research partially supported by SUTD-MIT International Design Center.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import numpy as np
try:
    import cplex as cpx
    cpx_available = True
except ImportError:
    cpx_available = False
try:
    import cylp.cy as clp
    from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray
    clp_available = True
except ImportError:
    clp_available = False
import rbfopt_config as config
import rbfopt_utils as ru
from rbfopt_settings import RbfSettings

def get_model_quality_estimate_full(settings, n, k, node_pos, node_val):
    """Compute an estimate of model quality.

    Computes an estimate of model quality, performing
    cross-validation. This version assumes that all interpolation
    points are used for cross-validation.

    Parameters
    ----------
    settings : rbfopt_settings.RbfSettings
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the space where the point lives.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : List[List[float]]
        Location of current interpolation nodes.

    node_val : List[float]
        List of values of the function at the nodes.
    
    Returns
    -------
    float
        An estimate of the leave-one-out cross-validation error, which
        can be interpreted as a measure of model quality.
    """
    assert(isinstance(settings, RbfSettings))
    assert(len(node_val)==k)
    assert(len(node_pos)==k)
    # We cannot find a leave-one-out interpolant if the following
    # condition is not met.
    assert(k > n + 1)

    sorted_list = sorted([(node_val[i], node_pos[i]) for i in range(k)])

    # Create a copy of the interpolation nodes and values
    cv_node_pos = [node_pos[i] for i in range(k-1)]
    cv_node_val = [node_val[i] for i in range(k-1)]

    # The node that was left out
    rm_node_pos = node_pos[-1]
    rm_node_val = node_val[-1]

    # Estimate of the model error
    loo_error = 0.0
    # Variance
    loo_error_var = 0.0
    
    for i in range(k):
        # Compute the RBF interpolant with one node left out
        Amat = ru.get_rbf_matrix(settings, n, k-1, cv_node_pos)
        (rbf_l, rbf_h) = ru.get_rbf_coefficients(settings, n, k-1, 
                                                 Amat, cv_node_val)

        # Compute value of the interpolant at the removed node
        predicted_val = ru.evaluate_rbf(settings, rm_node_pos, n, k-1,
                                        cv_node_pos, rbf_l, rbf_h)

        # Update leave-one-out error
        loo_error += abs(predicted_val - rm_node_val)
        loo_error_var += (abs(predicted_val - rm_node_val))**2

        # Update the node left out, unless we are at the last iteration
        if (i < k - 1):
            cv_node_pos[k-2-i], rm_node_pos = rm_node_pos, cv_node_pos[k-2-i]
            cv_node_val[k-2-i], rm_node_val = rm_node_val, cv_node_val[k-2-i]

    return loo_error/k

# -- end function

def get_model_quality_estimate(settings, n, k, node_pos, node_val,
                               num_iterations):
    """Compute an estimate of model quality.

    Computes an estimate of model quality, performing
    cross-validation. This version does not use all interpolation
    nodes, but rather only the best num_iterations ones.

    Parameters
    ----------
    settings : rbfopt_settings.RbfSettings
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the space where the point lives.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : List[List[float]]
        Location of current interpolation nodes.

    node_val : List[float]
        List of values of the function at the nodes.

    num_iterations : int
        Number of nodes on which quality should be tested.
    
    Returns
    -------
    float
        An estimate of the leave-one-out cross-validation error, which
        can be interpreted as a measure of model quality.
    """
    assert(isinstance(settings, RbfSettings))
    assert(len(node_val)==k)
    assert(len(node_pos)==k)
    assert(num_iterations <= k)
    # We cannot find a leave-one-out interpolant if the following
    # condition is not met.
    assert(k > n + 1)

    # Sort interpolation nodes by increasing objective function value
    sorted_list = sorted([(node_val[i], node_pos[i]) for i in range(k)])

    # Initialize the arrays used for the cross-validation
    cv_node_pos = [sorted_list[i][1] for i in range(1,k)]
    cv_node_val = [sorted_list[i][0] for i in range(1,k)]

    # The node that was left out
    rm_node_pos = sorted_list[0][1]
    rm_node_val = sorted_list[0][0]

    # Estimate of the model error
    loo_error = 0.0
    
    for i in range(num_iterations):
        # Compute the RBF interpolant with one node left out
        Amat = ru.get_rbf_matrix(settings, n, k-1, cv_node_pos)
        (rbf_l, rbf_h) = ru.get_rbf_coefficients(settings, n, k-1, 
                                                 Amat, cv_node_val)

        # Compute value of the interpolant at the removed node
        predicted_val = ru.evaluate_rbf(settings, rm_node_pos, n, k-1,
                                        cv_node_pos, rbf_l, rbf_h)

        # Update leave-one-out error
        loo_error += abs(predicted_val - rm_node_val)

        # Update the node left out
        if (i < k - 1):
            cv_node_pos[i], rm_node_pos = rm_node_pos, cv_node_pos[i]
            cv_node_val[i], rm_node_val = rm_node_val, cv_node_val[i]

    return loo_error/num_iterations

# -- end function

def get_model_quality_estimate_cpx(settings, n, k, node_pos, node_val,
                                   num_iterations):
    """Compute an estimate of model quality using LPs with Cplex.

    Computes an estimate of model quality, performing
    cross-validation. This version does not use all interpolation
    nodes, but rather only the best num_iterations ones, and it uses a
    sequence of LPs instead of a brute-force calculation. It should be
    much faster than the brute-force calculation, as each LP in the
    sequence requires only two pivots. The solver is IBM-ILOG Cplex.

    Parameters
    ----------
    settings : rbfopt_settings.RbfSettings
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the space where the point lives.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : List[List[float]]
        Location of current interpolation nodes.

    node_val : List[float]
        List of values of the function at the nodes.

    num_iterations : int
        Number of nodes on which quality should be tested.
    
    Returns
    -------
    float
        An estimate of the leave-one-out cross-validation error, which
        can be interpreted as a measure of model quality.

    Raises
    ------
    RuntimeError
        If the LPs cannot be solved.

    ValueError
        If some settings are not supported.
    """
    assert(isinstance(settings, RbfSettings))
    assert(len(node_val)==k)
    assert(len(node_pos)==k)
    assert(num_iterations <= k)
    assert(cpx_available)
    # We cannot find a leave-one-out interpolant if the following
    # condition is not met.
    assert(k > n + 1)

    # Get size of polynomial part of the matrix (p) and sign of obj
    # function (sign)
    if (ru.get_degree_polynomial(settings) == 1):
        p = n + 1
        sign = 1
    elif (ru.get_degree_polynomial(settings) == 0):
        p = 1
        sign = -1
    else:
        raise ValueError('RBF type ' + settings.rbf + ' not supported')

    # Sort interpolation nodes by increasing objective function value
    sorted_list = sorted([(node_val[i], node_pos[i]) for i in range(k)])

    # Initialize the arrays used for the cross-validation
    cv_node_pos = [sorted_list[i][1] for i in range(k)]
    cv_node_val = [sorted_list[i][0] for i in range(k)]

    # Compute the RBF matrix
    Amat = ru.get_rbf_matrix(settings, n, k, cv_node_pos)

    # Instantiate model and suppress output
    lp = cpx.Cplex()
    lp.set_log_stream(None)
    lp.set_error_stream(None)
    lp.set_warning_stream(None)
    lp.set_results_stream(None)
    lp.cleanup(settings.eps_zero)
    lp.parameters.threads.set(1)
    # lp.parameters.simplex.display.set(2)

    # Add variables: lambda, h, slacks
    lp.variables.add(obj = [sign*val for val in node_val],
                     lb = [-cpx.infinity] * k,
                     ub = [cpx.infinity] * k)
    lp.variables.add(lb = [-cpx.infinity] * p,
                     ub = [cpx.infinity] * p)
    lp.variables.add(lb = [0] * k,
                     ub = [0] * k)

    # Add constraints: interpolation conditions, unisolvence
    expr = [cpx.SparsePair(ind = [j for j in range(k + p)] + [k + p + i],
                           val = [Amat[i, j] for j in range(k + p)] + [1.0])
            for i in range(k)]
    lp.linear_constraints.add(lin_expr = expr, senses = ['E'] * k,
                              rhs = cv_node_val)
    expr = [cpx.SparsePair(ind = [j for j in range(k + p)],
                           val = [Amat[i, j] for j in range(k + p)])
            for i in range(k, k + p)]
    lp.linear_constraints.add(lin_expr = expr, senses = ['E'] * p,
                              rhs = [0] * p)

    # Estimate of the model error
    loo_error = 0.0

    for i in range(num_iterations):
        # Fix lambda[i] to zero
        lp.variables.set_lower_bounds(i, 0.0)
        lp.variables.set_upper_bounds(i, 0.0)
        # Set slack[i] to unconstrained
        lp.variables.set_lower_bounds(k + p + i, -cpx.infinity)
        lp.variables.set_upper_bounds(k + p + i, cpx.infinity)

        lp.solve()

        if (not lp.solution.is_primal_feasible()):
            raise RuntimeError('Could not solve LP with Cplex')

        rbf_l = lp.solution.get_values(0, k - 1)
        rbf_h = lp.solution.get_values(k, k + p - 1)

        # Compute value of the interpolant at the removed node
        predicted_val = ru.evaluate_rbf(settings, cv_node_pos[i], n, k,
                                        cv_node_pos, rbf_l, rbf_h)

        # Update leave-one-out error
        loo_error += abs(predicted_val - cv_node_val[i])

        # Fix lambda[i] to zero
        lp.variables.set_lower_bounds(i, -cpx.infinity)
        lp.variables.set_upper_bounds(i, cpx.infinity)
        # Set slack[i] to unconstrained
        lp.variables.set_lower_bounds(k + p + i, 0.0)
        lp.variables.set_upper_bounds(k + p + i, 0.0)

    return loo_error/num_iterations

# -- end function

def get_model_quality_estimate_clp(settings, n, k, node_pos, node_val,
                                   num_iterations):
    """Compute an estimate of model quality using LPs with Clp.

    Computes an estimate of model quality, performing
    cross-validation. This version does not use all interpolation
    nodes, but rather only the best num_iterations ones, and it uses a
    sequence of LPs instead of a brute-force calculation. It should be
    much faster than the brute-force calculation, as each LP in the
    sequence requires only two pivots. The solver is COIN-OR Clp.

    Parameters
    ----------
    settings : rbfopt_settings.RbfSettings
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the space where the point lives.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : List[List[float]]
        Location of current interpolation nodes.

    node_val : List[float]
        List of values of the function at the nodes.

    num_iterations : int
        Number of nodes on which quality should be tested.
    
    Returns
    -------
    float
        An estimate of the leave-one-out cross-validation error, which
        can be interpreted as a measure of model quality.

    Raises
    ------
    RuntimeError
        If the LPs cannot be solved.

    ValueError
        If some settings are not supported.
    """
    assert(isinstance(settings, RbfSettings))
    assert(len(node_val)==k)
    assert(len(node_pos)==k)
    assert(num_iterations <= k)
    assert(cpx_available)
    # We cannot find a leave-one-out interpolant if the following
    # condition is not met.
    assert(k > n + 1)

    # Get size of polynomial part of the matrix (p) and sign of obj
    # function (sign)
    if (ru.get_degree_polynomial(settings) == 1):
        p = n + 1
        sign = 1
    elif (ru.get_degree_polynomial(settings) == 0):
        p = 1
        sign = -1
    else:
        raise ValueError('RBF type ' + settings.rbf + ' not supported')

    # Sort interpolation nodes by increasing objective function value
    sorted_list = sorted([(node_val[i], node_pos[i]) for i in range(k)])

    # Initialize the arrays used for the cross-validation
    cv_node_pos = [sorted_list[i][1] for i in range(k)]
    cv_node_val = [sorted_list[i][0] for i in range(k)]

    # Compute the RBF matrix, and the two submatrices of interest
    Amat = ru.get_rbf_matrix(settings, n, k, cv_node_pos)
    Phimat = Amat[:k, :k]
    Pmat = Amat[:k, k:]
    identity = np.matrix(np.identity(k))

    # Instantiate model
    model = CyLPModel()

    # Add variables: lambda, h, slacks
    rbf_lambda = model.addVariable('rbf_lambda', k)
    rbf_h = model.addVariable('rbf_h', p)
    slack = model.addVariable('slack', k)

    # Add constraints: interpolation conditions, unisolvence
    F = CyLPArray(cv_node_val)
    zeros = CyLPArray([0] * p)
    ones = CyLPArray([1] * p)
    if (p > 1):
        # We can use matrix form
        model += (Phimat * rbf_lambda + Pmat * rbf_h + identity * slack == F)
        model += (Pmat.transpose() * rbf_lambda == zeros)
    else:
        # Workaround for one dimensional variables: one row at a time
        for i in range(k):
            model += Phimat[i, :] * rbf_lambda + rbf_h + slack[i] == F[i]
        model += (Pmat.transpose() * rbf_lambda == zeros)

    # Add objective
    obj = CyLPArray([sign*val for val in node_val])
    model.objective = obj * rbf_lambda
                           
    # Create LP, suppress output, and set bounds
    lp = clp.CyClpSimplex(model)
    lp.logLevel = 0
    infinity = lp.getCoinInfinity()
    for i in range(k + p):
        lp.setColumnLower(i, -infinity)
        lp.setColumnUpper(i, infinity)
    for i in range(k):
        lp.setColumnLower(k + p + i, 0.0)
        lp.setColumnUpper(k + p + i, 0.0)

    # Estimate of the model error
    loo_error = 0.0

    for i in range(num_iterations):
        # Fix lambda[i] to zero
        lp.setColumnLower(i, 0.0)
        lp.setColumnUpper(i, 0.0)
        # Set slack[i] to unconstrained
        lp.setColumnLower(k + p + i, -infinity)
        lp.setColumnUpper(k + p + i, infinity)

        lp.dual(startFinishOptions = 'sx')

        if (not lp.getStatusCode() == 0):
            raise RuntimeError('Could not solve LP with Clp, status ' +
                               lp.getStatusString())

        lambda_sol = lp.primalVariableSolution['rbf_lambda'].tolist()
        h_sol = lp.primalVariableSolution['rbf_h'].tolist()

        # Compute value of the interpolant at the removed node
        predicted_val = ru.evaluate_rbf(settings, cv_node_pos[i], n, k,
                                        cv_node_pos, lambda_sol, h_sol)

        # Update leave-one-out error
        loo_error += abs(predicted_val - cv_node_val[i])

        # Fix lambda[i] to zero
        lp.setColumnLower(i, -infinity)
        lp.setColumnUpper(i, infinity)
        # Set slack[i] to unconstrained
        lp.setColumnLower(k + p + i, 0.0)
        lp.setColumnUpper(k + p + i, 0.0)

    return loo_error/num_iterations

# -- end function

def get_best_rbf_model(settings, n, k, node_pos, node_val, num_iter):
    """Compute which type of RBF yields the best model.

    Compute which RBF interpolant yields the best surrogate model,
    using cross validation to determine the lowest leave-one-out
    error.

    Parameters
    ----------
    settings : rbfopt_settings.RbfSettings
        Global and algorithmic settings.

    n : int
        Dimension of the problem, i.e. the space where the point lives.

    k : int
        Number of nodes, i.e. interpolation points.

    node_pos : List[List[float]]
        Location of current interpolation nodes.

    node_val : List[float]
        List of values of the function at the nodes.

    num_iterations : int
        Number of nodes on which quality should be tested.

    Returns
    -------
    str
        The type of RBF that currently yields the best surrogate
        model, based on leave-one-out error. This will be one of the
        supported types of RBF.
    """

    assert(isinstance(settings, RbfSettings))
    assert(len(node_val)==k)
    assert(len(node_pos)==k)
    assert(num_iter <= k)
    # We cannot find a leave-one-out interpolant if the following
    # condition is not met.
    assert(k > n + 1)

    # Choose model selection method
    if (settings.model_selection_solver == 'cplex' and cpx_available):
        model_quality_function = get_model_quality_estimate_cpx
    elif (settings.model_selection_solver == 'clp' and clp_available):
        model_quality_function = get_model_quality_estimate_clp
    else:
        model_quality_function = get_model_quality_estimate

    best_loo_error = float('inf')
    best_model = None
    original_rbf_type = settings.rbf
    for rbf_type in ['cubic', 'thin_plate_spline', 'multiquadric', 'linear']:
        settings.rbf = rbf_type
        try:
            loo_error = model_quality_function(settings, n, k, node_pos,
                                               node_val, num_iter)
        except:
            return original_rbf_type
        if (loo_error < best_loo_error):
            best_loo_error = loo_error
            best_model = rbf_type

    settings.rbf = original_rbf_type
    return best_model

# -- end function
