"""Pyomo models with zero-degree polynomial for RBFOpt.

This module creates all the auxiliary problems that rely on
zero-degree polynomials. The models are created and instantiated using
Pyomo. This module does not solve the problems.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
(C) Copyright International Business Machines Corporation 2017.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from pyomo.environ import *
import sys
import numpy as np
import rbfopt.rbfopt_utils as ru
from rbfopt.rbfopt_settings import RbfoptSettings

_DISTANCE_SHIFT = 1.0e-12

def create_min_rbf_model(settings, n, k, var_lower, var_upper, integer_vars,
                         categorical_info, node_pos, rbf_lambda, rbf_h):
    """Create the concrete model to minimize the RBF.

    Create the concrete model to minimize the RBF.

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
        List of indices of integer variables.

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

    Returns
    -------
    pyomo.ConcreteModel
        The concrete model describing the problem.
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
    assert(len(rbf_h) == 1)
    assert(len(node_pos) == k)
    assert(isinstance(settings, RbfoptSettings))
    assert(ru.get_degree_polynomial(settings) == 0)

    model = ConcreteModel()
    
    # Dimension of the space
    model.n = Param(initialize=n)
    model.N = RangeSet(0, model.n - 1)
        
    # Number of interpolation nodes
    model.k = Param(initialize=k)
    model.K = RangeSet(0, model.k - 1)

    # Dimension of u_pi
    model.q = Param(initialize=(k+1))
    model.Q = RangeSet(0, model.q - 1)

    # Coefficients of the RBF
    lambda_h_param = {}
    for i in range(k):
        lambda_h_param[i] = float(rbf_lambda[i])
    for i in range(1):
        lambda_h_param[k+i] = float(rbf_h[i])
    model.lambda_h = Param(model.Q, initialize=lambda_h_param)

    # Coordinates of the nodes
    node_param = {}
    for i in range(k):
        for j in range(n):
            node_param[i, j] = float(node_pos[i][j])
    model.node = Param(model.K, model.N, initialize=node_param)

    # Variable bounds
    var_lower_param = {}
    var_upper_param = {}
    for i in range(n):
        var_lower_param[i] = float(var_lower[i])
        var_upper_param[i] = float(var_upper[i])
    model.var_lower = Param(model.N, initialize=var_lower_param)
    model.var_upper = Param(model.N, initialize=var_upper_param)

    # Variable: the point in the space
    model.x = Var(model.N, domain=Reals, bounds=_x_bounds)

    if (settings.rbf == 'linear'):
        model.OBJ = Objective(rule=_min_rbf_obj_expression_linear,
                              sense=minimize)
    elif (settings.rbf == 'multiquadric'):
        model.gamma = Param(initialize=settings.rbf_shape_parameter)
        model.OBJ = Objective(rule=_min_rbf_obj_expression_mq,
                              sense=minimize)

    # Add integer variables if necessary
    if (len(integer_vars)):
        add_integrality_constraints(model, integer_vars)

    # Add categorical variables if necessary
    if (categorical_info is not None and categorical_info[2]):
        add_categorical_constraints(model, *categorical_info)

    return model
# -- end function


def create_max_one_over_mu_model(settings, n, k, var_lower, var_upper, 
                                 integer_vars, categorical_info, node_pos,
                                 mat):
    """Create the concrete model to maximize 1/\mu.

    Create the concrete model to maximize :math: `1/\mu`, also known
    as the InfStep of the RBF method.

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
        List of indices of integer variables.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes (one on each row).

    mat: 2D numpy.ndarray[float]
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0], see paper as cited above. Must
        be a 2D numpy.ndarray[float] of dimension ((k+1) x (k+1))

    Returns
    -------
    pyomo.ConcreteModel
        The concrete model describing the problem.
    """
    assert(isinstance(var_lower, np.ndarray))
    assert(isinstance(var_upper, np.ndarray))
    assert(isinstance(integer_vars, np.ndarray))
    assert(isinstance(node_pos, np.ndarray))
    assert(len(var_lower) == n)
    assert(len(var_upper) == n)
    assert(len(node_pos) == k)
    assert(isinstance(mat, np.ndarray))
    assert(mat.shape == (k+1,k+1))
    assert(isinstance(settings, RbfoptSettings))
    assert(ru.get_degree_polynomial(settings) == 0)

    model = ConcreteModel()

    # Dimension of the space
    model.n = Param(initialize=n)
    model.N = RangeSet(0, model.n - 1)
        
    # Number of interpolation nodes
    model.k = Param(initialize=k)
    model.K = RangeSet(0, model.k - 1)

    # Dimension of the matrix
    model.q = Param(initialize=(k+1))
    model.Q = RangeSet(0, model.q - 1)

    # Coordinates of the nodes
    node_param = {}
    for i in range(k):
        for j in range(n):
            node_param[i, j] = float(node_pos[i][j])
    model.node = Param(model.K, model.N, initialize=node_param)

    # Variable bounds
    var_lower_param = {}
    var_upper_param = {}
    for i in range(n):
        var_lower_param[i] = float(var_lower[i])
        var_upper_param[i] = float(var_upper[i])
    model.var_lower = Param(model.N, initialize=var_lower_param)
    model.var_upper = Param(model.N, initialize=var_upper_param)

    # Inverse of the matrix [Phi P; P^T 0]. Because the matrix is
    # symmetric, we only save the upper right part, while doubling the
    # off-diagonal elements.
    Ainv_param = {}
    for i in range(k+1):
        for j in range(i, k+1):
            if (abs(mat[i, j]) != 0.0):
                if (i == j):
                    Ainv_param[i, j] = float(mat[i, j])
                else:
                    Ainv_param[i, j] = float(2*mat[i, j])
    model.Ainv = Param(model.Q, model.Q, initialize=Ainv_param,
                       default=0.0)

    # Value of phi at zero, necessary for shift
    if (settings.rbf == 'linear'):
        model.phi_0 = Param(initialize=0.0)
    elif (settings.rbf == 'multiquadric'):
        model.phi_0 = Param(initialize=settings.rbf_shape_parameter)
        model.gamma = Param(initialize=settings.rbf_shape_parameter)

    # Variable: the point in the space
    model.x = Var(model.N, domain=Reals, bounds=_x_bounds)

    # Objective function. 
    if (settings.rbf == 'linear'):
        model.OBJ = Objective(rule=_max_one_over_mu_obj_expression_linear,
                              sense=maximize)
    elif (settings.rbf == 'multiquadric'):
        model.OBJ = Objective(rule=_max_one_over_mu_obj_expression_mq,
                              sense=maximize)

    # Add integer variables if necessary
    if (len(integer_vars)):
        add_integrality_constraints(model, integer_vars)

    # Add categorical variables if necessary
    if (categorical_info is not None and categorical_info[2]):
        add_categorical_constraints(model, *categorical_info)

    return model
# -- end function


def create_max_h_k_model(settings, n, k, var_lower, var_upper, integer_vars,
                         categorical_info, node_pos, rbf_lambda, rbf_h,
                         mat, target_val):
    """Create the abstract model to maximize h_k.

    Create the abstract model to maximize h_k, also known as the
    Global Search Step of the RBF method.

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
        List of indices of integer variables.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes (one on each row).

    rbf_lambda : 1D numpy.ndarray[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : 1D numpy.ndarray[float]
        The h coefficients of the RBF interpolant, corresponding to
        the polynomial. List of dimension n+1.

    mat: 2D numpy.ndarray[float]
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0], see paper as cited above. Must
        be a 2D numpy.ndarray[float] of dimension ((k+1) x (k+1))

    target_val : float
        Value f* that we want to find in the unknown objective
        function.

    Returns
    -------
    pyomo.ConcreteModel
        The concrete model describing the problem.
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
    assert(len(rbf_h) == 1)
    assert(len(node_pos) == k)
    assert(isinstance(mat, np.ndarray))
    assert(mat.shape == (k+1, k+1))
    assert(isinstance(settings, RbfoptSettings))
    assert(ru.get_degree_polynomial(settings) == 0)

    model = ConcreteModel()

    # Dimension of the space
    model.n = Param(initialize=n)
    model.N = RangeSet(0, model.n - 1)
        
    # Number of interpolation nodes
    model.k = Param(initialize=k)
    model.K = RangeSet(0, model.k - 1)

    # Dimension of the matrix
    model.q = Param(initialize=(k+1))
    model.Q = RangeSet(0, model.q - 1)

    # Coefficients of the RBF
    lambda_h_param = {}
    for i in range(k):
        lambda_h_param[i] = float(rbf_lambda[i])
    for i in range(1):
        lambda_h_param[k+i] = float(rbf_h[i])
    model.lambda_h = Param(model.Q, initialize=lambda_h_param)

    # Coordinates of the nodes
    node_param = {}
    for i in range(k):
        for j in range(n):
            node_param[i, j] = float(node_pos[i][j])
    model.node = Param(model.K, model.N, initialize=node_param)

    # Variable bounds
    var_lower_param = {}
    var_upper_param = {}
    for i in range(n):
        var_lower_param[i] = float(var_lower[i])
        var_upper_param[i] = float(var_upper[i])
    model.var_lower = Param(model.N, initialize=var_lower_param)
    model.var_upper = Param(model.N, initialize=var_upper_param)

    # Inverse of the matrix [Phi P; P^T 0]. Because the matrix is
    # symmetric, we only save the upper right part, while doubling the
    # off-diagonal elements.
    Ainv_param = {}
    for i in range(k+1):
        for j in range(i, k+1):
            if (abs(mat[i, j]) != 0.0):
                if (i == j):
                    Ainv_param[i, j] = float(mat[i, j])
                else:
                    Ainv_param[i, j] = float(2*mat[i, j])
    model.Ainv = Param(model.Q, model.Q, initialize=Ainv_param,
                       default=0.0)

    # Target value
    model.fstar = Param(initialize=target_val)

    # Value of phi at zero, necessary for shift
    if (settings.rbf == 'linear'):
        model.phi_0 = Param(initialize=0.0)
    elif (settings.rbf == 'multiquadric'):
        model.phi_0 = Param(initialize=settings.rbf_shape_parameter)
        model.gamma = Param(initialize=settings.rbf_shape_parameter)

    # Variable: the point in the space
    model.x = Var(model.N, domain=Reals, bounds=_x_bounds)

    # Objective function. 
    if (settings.rbf == 'linear'):
        model.OBJ = Objective(rule=_max_h_k_obj_expression_linear,
                              sense=maximize)
    elif (settings.rbf == 'multiquadric'):
        model.OBJ = Objective(rule=_max_h_k_obj_expression_mq,
                              sense=maximize)

    # Add integer variables if necessary
    if (len(integer_vars)):
        add_integrality_constraints(model, integer_vars)

    # Add categorical variables if necessary
    if (categorical_info is not None and categorical_info[2]):
        add_categorical_constraints(model, *categorical_info)

    return model

# -- end function


def create_min_bump_model(settings, n, k, Phimat, Pmat, node_val, 
                          node_err_bounds):
    """Create a model to find RBF coefficients with min bumpiness.

    Create a quadratic problem to compute the coefficients of the RBF
    interpolant that minimizes bumpiness and lets all points deviate
    by a specified amount from their value.

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

    Returns
    -------
    pyomo.ConcreteModel
        The concrete model describing the problem.
    """
    assert(isinstance(settings, RbfoptSettings))
    assert(isinstance(node_val, np.ndarray))
    assert(isinstance(node_err_bounds, np.ndarray))
    assert(len(node_val) == k)
    assert(isinstance(Phimat, np.ndarray))
    assert(isinstance(Pmat, np.ndarray))
    assert(Phimat.shape == (k, k))
    assert(Pmat.shape == (k, 1))
    assert(len(node_val) == len(node_err_bounds))
    assert(ru.get_degree_polynomial(settings) == 0)

    model = ConcreteModel()

    # Dimension of the space
    model.n = Param(initialize=n)
    model.N = RangeSet(0, model.n - 1)

    # Dimension of P matrix
    model.p = Param(initialize=1)
    model.P = RangeSet(0, 0)
        
    # Number of interpolation nodes
    model.k = Param(initialize=k)
    model.K = RangeSet(0, model.k - 1)

    # Lower and upper bounds on node values, i.e. bounds for the first
    # set of equations in the constraints
    node_val_lower_param = {}
    node_val_upper_param = {}
    for i in range(k):
        node_val_lower_param[i] = float(node_val[i] + node_err_bounds[i, 0])
        node_val_upper_param[i] = float(node_val[i] + node_err_bounds[i, 1])
    model.node_val_lower = Param(model.K, initialize=node_val_lower_param)
    model.node_val_upper = Param(model.K, initialize=node_val_upper_param)

    # Phi matrix.
    Phi_param = {}
    for i in range(k):
        for j in range(k):
            if (abs(Phimat[i, j]) != 0.0):
                Phi_param[i, j] = float(Phimat[i, j])
    model.Phi = Param(model.K, model.K, initialize=Phi_param,
                      default=0.0)

    # P matrix.
    Pm_param = {}
    for i in range(k):
        for j in range(1):
            if (abs(Pmat[i, j]) != 0.0):
                Pm_param[i, j] = float(Pmat[i, j])
    model.Pm = Param(model.K, model.P, initialize=Pm_param,
                     default=0.0)

    # Variable: the lambda coefficients of the RBF
    model.rbf_lambda = Var(model.K, domain=Reals)

    # Variable: the h coefficients of the RBF
    model.rbf_h = Var(model.P, domain=Reals)

    # Objective function.
    model.OBJ = Objective(rule=_min_bump_obj_expression, sense=minimize)

    # Constraints. See definitions below.
    model.IntrConstraint1 = Constraint(model.K, rule=_intr_constraint_rule_pt1)
    model.IntrConstraint2 = Constraint(model.K, rule=_intr_constraint_rule_pt2)
    model.UnisConstraint = Constraint(model.P, rule=_unis_constraint_rule)    

    return model

# -- end function


def create_maximin_dist_model(settings, n, k, var_lower, var_upper,
                              integer_vars, categorical_info, node_pos):
    """Create the concrete model to maximize the minimum distance.

    Create the concrete model to maximize the minimum distance to the
    interpolation nodes, which is the infstep of the MSRSM method.

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
        List of indices of integer variables.

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes.

    Returns
    -------
    pyomo.ConcreteModel
        The concrete model describing the problem.

    """
    assert (isinstance(var_lower, np.ndarray))
    assert (isinstance(var_upper, np.ndarray))
    assert (isinstance(integer_vars, np.ndarray))
    assert (isinstance(node_pos, np.ndarray))
    assert(len(var_lower) == n)
    assert(len(var_upper) == n)
    assert(len(node_pos) == k)
    assert(isinstance(settings, RbfoptSettings))

    model = ConcreteModel()

    # Dimension of the space
    model.n = Param(initialize=n)
    model.N = RangeSet(0, model.n - 1)
        
    # Number of interpolation nodes
    model.k = Param(initialize=k)
    model.K = RangeSet(0, model.k - 1)

    # Coordinates of the nodes
    node_param = {}
    for i in range(k):
        for j in range(n):
            node_param[i, j] = float(node_pos[i][j])
    model.node = Param(model.K, model.N, initialize=node_param)

    # Variable bounds
    var_lower_param = {}
    var_upper_param = {}
    for i in range(n):
        var_lower_param[i] = float(var_lower[i])
        var_upper_param[i] = float(var_upper[i])
    model.var_lower = Param(model.N, initialize=var_lower_param)
    model.var_upper = Param(model.N, initialize=var_upper_param)

    # Variable: the point in the space
    model.x = Var(model.N, domain=Reals, bounds=_x_bounds)

    # Auxiliary variable: value of the minimum distance to the nodes
    model.mindistsq = Var(domain=NonNegativeReals,
                          bounds=(settings.min_dist,float('inf')))

    # Objective function.
    model.OBJ = Objective(expr=(model.mindistsq), sense=maximize)
    model.MdistdefConstraint = Constraint(model.K,
                                          rule=_mdistdef_constraint_rule)

    # Add integer variables if necessary
    if (len(integer_vars)):
        add_integrality_constraints(model, integer_vars)

    # Add categorical variables if necessary
    if (categorical_info is not None and categorical_info[2]):
        add_categorical_constraints(model, *categorical_info)

    return model

# -- end function


def create_min_msrsm_model(settings, n, k, var_lower, var_upper,
                           integer_vars, categorical_info, node_pos,
                           rbf_lambda, rbf_h, dist_weight, dist_min,
                           dist_max, fmin, fmax):
    """Create the concrete model to optimize the MSRSM objective.

    Create the concreate model to minimize a weighted combination of
    the value of the RBF interpolant and the (negative of the)
    distance from the closes interpolation node. This is the Global
    Search Step of the MSRSM method.

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
        List of indices of integer variables.

    node_pos : 2D numpy.ndarray[float]
        List of coordinates of the nodes (one on each row).

    categorical_info : (1D numpy.ndarray[int], 1D numpy.ndarray[int],
                        List[(int, 1D numpy.ndarray[int])]) or None
        Information on categorical variables: array of indices of
        categorical variables in original space, array of indices of
        noncategorical variables in original space, and expansion of
        each categorical variable, given as a tuple (original index,
        indices of expanded variables).

    rbf_lambda : 1D numpy.ndarray[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : 1D numpy.ndarray[float]
        The h coefficients of the RBF interpolant, corresponding to
        the polynomial. List of dimension n+1.

    dist_weight : float
        The weight paramater for distance and RBF interpolant
        value. Must be between 0 and 1. A weight of 1.0 corresponds to
        using solely distance, 0.0 to objective function.

    dist_min : float
        The minimum distance between two interpolation nodes.

    dist_max : float
        The maximum distance between two interpolation nodes.

    fmin : float
        The minimum value of an interpolation node.

    fmax : float
        The maximum value of an interpolation node.

    Returns
    -------
    pyomo.ConcreteModel
        The concrete model describing the problem.

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
    assert(len(rbf_h) == 1)
    assert(len(node_pos) == k)
    assert(isinstance(settings, RbfoptSettings))
    assert(ru.get_degree_polynomial(settings) == 0)
    assert(0 <= dist_weight <= 1)
    assert(dist_max >= dist_min >= 0)

    model = ConcreteModel()

    # Dimension of the space
    model.n = Param(initialize=n)
    model.N = RangeSet(0, model.n - 1)
        
    # Number of interpolation nodes
    model.k = Param(initialize=k)
    model.K = RangeSet(0, model.k - 1)

    # Dimension of u_pi
    model.q = Param(initialize=(k+1))
    model.Q = RangeSet(0, model.q - 1)

    # Coefficients of the RBF
    lambda_h_param = {}
    for i in range(k):
        lambda_h_param[i] = float(rbf_lambda[i])
    for i in range(1):
        lambda_h_param[k+i] = float(rbf_h[i])
    model.lambda_h = Param(model.Q, initialize=lambda_h_param)

    # Coordinates of the nodes
    node_param = {}
    for i in range(k):
        for j in range(n):
            node_param[i, j] = float(node_pos[i][j])
    model.node = Param(model.K, model.N, initialize=node_param)

    # Variable bounds
    var_lower_param = {}
    var_upper_param = {}
    for i in range(n):
        var_lower_param[i] = float(var_lower[i])
        var_upper_param[i] = float(var_upper[i])
    model.var_lower = Param(model.N, initialize=var_lower_param)
    model.var_upper = Param(model.N, initialize=var_upper_param)

    # Adjust parameters to avoid zero denominators in expressions
    if (fmax <= fmin + settings.eps_zero):
        fmax = fmin + 1
    if (dist_max <= dist_min + settings.eps_zero):
        dist_min = dist_max - 1
    # Minimum and maximum distance
    model.dist_min = Param(initialize=dist_min)
    model.dist_max = Param(initialize=dist_max)
    # Minimum and maximum of function values interpolation nodes
    model.fmin = Param(initialize=fmin)
    model.fmax = Param(initialize=fmax)
    # Weight of the distance and objective criteria
    model.dist_weight = Param(initialize=dist_weight)
    model.obj_weight = Param(initialize=(1.0 if settings.modified_msrsm_score
                                         else 1 - dist_weight))

    # Value of phi at zero, necessary for shift
    if (settings.rbf == 'linear'):
        model.phi_0 = Param(initialize=0.0)
    elif (settings.rbf == 'multiquadric'):
        model.phi_0 = Param(initialize=settings.rbf_shape_parameter)
        model.gamma = Param(initialize=settings.rbf_shape_parameter)

    # Variable: the point in the space
    model.x = Var(model.N, domain=Reals, bounds=_x_bounds)

    # Auxiliary variable: value of the minimum distance to the nodes
    model.mindistsq = Var(domain=NonNegativeReals,
                          bounds=(settings.min_dist,float('inf')))

    # Objective function.
    if (settings.rbf == 'linear'):
        model.OBJ = Objective(rule=_min_msrsm_obj_expression_linear,
                              sense=minimize)
    elif (settings.rbf == 'multiquadric'):
        model.OBJ = Objective(rule=_min_msrsm_obj_expression_mq,
                              sense=minimize)
    model.MdistdefConstraint = Constraint(model.K,
                                          rule=_mdistdef_constraint_rule)

    # Add integer variables if necessary
    if (len(integer_vars)):
        add_integrality_constraints(model, integer_vars)

    # Add categorical variables if necessary
    if (categorical_info is not None and categorical_info[2]):
        add_categorical_constraints(model, *categorical_info)

    return model

# -- end function


def add_integrality_constraints(model, integer_vars):
    """Add integrality constraints to the model.

    Add integrality constraints to the model by introducing extra
    variables.

    Parameters
    ----------

    model : pyomo.ConcreteModel
        The model to which we want to add integrality constraints.

    integer_vars : 1D numpy.ndarray[int]
        List of indices of integer variables.
    """
    assert(isinstance(integer_vars, np.ndarray))
    assert(len(integer_vars))

    ni = len(integer_vars)
    
    # Number of integer variables
    model.ni = Param(initialize=ni)
    model.NI = RangeSet(0, model.ni-1)

    # Parameter: list of indices of integer variables
    int_var_param = {}
    for i in range(ni):
        int_var_param[i] = integer_vars[i]
    model.integer_vars = Param(model.NI, initialize=int_var_param)

    # Variable: the point in the space
    model.y = Var(model.NI, domain=Integers, bounds=_y_bounds)

    # Constraints: every integer variable is simply equal to one
    # of the existing variables
    model.IntConstraint = Constraint(model.NI, rule=_int_constraint_rule)

# -- end function

def add_categorical_constraints(model, categorical, not_categorical,
                                categorical_expansion):
    """Add categorical constraints to the model.

    Add constraints enforcing the assignment constraints for
    categorical variables.

    Parameters
    ----------

    model : pyomo.ConcreteModel
        The model to which we want to add integrality constraints.

    categorical : 1D numpy.ndarray[int]
        Array of indices of categorical variables in original space.

    not_categorical : 1D numpy.ndarray[int]
        Array of indices of not categorical variables in original space.

    categorical_expansion : List[(int, float, 1D numpy.ndarray[int])]
        Expansion of original categorical variables into binaries.

    """
    assert(isinstance(categorical, np.ndarray))
    assert(isinstance(not_categorical, np.ndarray))
    assert(len(categorical_expansion)==len(categorical))

    nc = len(categorical_expansion)
    
    # Number of categorical variables
    model.nc = Param(initialize=nc)
    model.NC = RangeSet(0, model.nc-1)

    # Parameter: list of indices of categorical variables
    cat_var_param = {}
    for i in range(nc):
        for j in categorical_expansion[i][2]:
            cat_var_param[i, j] = 1
    model.cat_vars = Param(model.NC, model.N, initialize=cat_var_param,
                           default=0.0)

    # Constraints: the unary representation of categorical variables
    # adds up to exactly one
    model.CatConstraint = Constraint(model.NC, rule=_cat_constraint_rule)

# -- end function


# Function to return bounds
def _x_bounds(model, i):
    return (model.var_lower[i], model.var_upper[i])

# Constraints: definition of the interpolation conditions with
# slack. Expression: f^L <= Phi lambda + P h
def _intr_constraint_rule_pt1(model, i):
    if (model.node_val_lower[i] >= model.node_val_upper[i]):
        return (sum(model.Phi[i, j]*model.rbf_lambda[j] for j in model.K) +
                sum(model.Pm[i, j]*model.rbf_h[j] for j in model.P) ==
                model.node_val_lower[i])
    else:
        return (model.node_val_lower[i] <=
                sum(model.Phi[i, j]*model.rbf_lambda[j] for j in model.K) +
                sum(model.Pm[i, j]*model.rbf_h[j] for j in model.P))

    # Constraints: definition of the interpolation conditions with
# slack. Expression: Phi lambda + P h <= f^U
def _intr_constraint_rule_pt2(model, i):
    if (not model.node_val_lower[i] >= model.node_val_upper[i]):
        return (sum(model.Phi[i, j]*model.rbf_lambda[j] for j in model.K) +
                sum(model.Pm[i, j]*model.rbf_h[j] for j in model.P)
                <= model.node_val_upper[i])
    else:
        return Constraint.Skip

# Constraints: definition of the unisolvence conditions. Expression:
# P \lambda = 0
def _unis_constraint_rule(model, i):
    return (sum(model.Pm[j, i]*model.rbf_lambda[j] for j in model.K) == 0.0)


# Constraints: definition of the minimum distance constraint.
# for i in K: mindistsq <= dist(x, x^i)^2
def _mdistdef_constraint_rule(model, i):
    return (model.mindistsq <= 
            sum((model.x[j] - model.node[i, j])**2 for j in model.N))

# Objective function for the "minimize rbf" problem. The expression is:
# min sum_{j in K} lambda_j d_j + h_{0}
def _min_rbf_obj_expression_linear(model):
    return (sum(model.lambda_h[i] *
                sqrt(_DISTANCE_SHIFT +
                     sum((model.x[j] - model.node[i, j])**2 for j in model.N))
                for i in model.K) + model.lambda_h[model.k.value])

# Objective function for the "minimize rbf" problem. The expression is:
# min sum_{j in K} lambda_j \sqrt{d_j + gamma^2} + h_{0}
def _min_rbf_obj_expression_mq(model):
    return (sum(model.lambda_h[i] *
                sqrt(model.gamma*model.gamma +
                     sum((model.x[j] - model.node[i, j])**2 for j in model.N))
                for i in model.K) + model.lambda_h[model.k.value])

# Objective function for the "maximize 1/\mu" problem. The expression is:
# max -\sum_{i in Q, j in Q} A^{-1}_{ij} upi_i upi_j;
def _max_one_over_mu_obj_expression_linear(model):
    # We need all products between index sets
    return (# Product 0..k-1 with 0..k-1
        sum(model.Ainv[i,j] *
            sqrt(_DISTANCE_SHIFT +
                 sum((model.x[h] - model.node[i, h])**2 for h in model.N)*
                 sum((model.x[h] - model.node[j, h])**2 for h in model.N))
            for i in model.K for j in model.K) +
        # Product 0..k-1 with k
        sum(model.Ainv[j, model.k] *
            sqrt(_DISTANCE_SHIFT +
                 sum((model.x[h] - model.node[j, h])**2 for h in model.N))
            for j in model.K) +
        # Product k with k
        model.Ainv[model.k.value, model.k.value] - model.phi_0)

def _max_one_over_mu_obj_expression_mq(model):
    # We need all products between index sets
    return (# Product 0..k-1 with 0..k-1
        sum(model.Ainv[i,j] *
            sqrt(model.gamma*model.gamma +
                 sum((model.x[h] - model.node[i, h])**2 for h in model.N))
            *sqrt(model.gamma*model.gamma +
                  sum((model.x[h] - model.node[j, h])**2 for h in model.N))
            for i in model.K for j in model.K) +
        # Product 0..k-1 with k
        sum(model.Ainv[j, model.k] *
            sqrt(model.gamma*model.gamma +
                 sum((model.x[h] - model.node[j, h])**2 for h in model.N))
            for j in model.K) +
        # Product k with k
        model.Ainv[model.k.value, model.k.value] - model.phi_0)

# Objective function for the "maximize h_k" problem. The expression is:
# 1/(\mu_k(x) [s_k(x) - f^\ast]^2)
def _max_h_k_obj_expression_linear(model):
    return ((sum(model.Ainv[i,j] *
                 sqrt(_DISTANCE_SHIFT +
                      sum((model.x[h] - model.node[i, h])**2 for h in model.N)*
                      sum((model.x[h] - model.node[j, h])**2 for h in model.N))
                 for i in model.K for j in model.K) +
             sum(model.Ainv[j, model.k] *
                sqrt(_DISTANCE_SHIFT +
                     sum((model.x[h] - model.node[j, h])**2 for h in model.N))
                for j in model.K) +
            model.Ainv[model.k.value, model.k.value] - model.phi_0) /
            ((sum(model.lambda_h[i] *
                  sqrt(_DISTANCE_SHIFT +
                       sum((model.x[j] - model.node[i, j])**2
                           for j in model.N))
                  for i in model.K) +
              model.lambda_h[model.k.value] - model.fstar)**2))

def _max_h_k_obj_expression_mq(model):
    return ((sum(model.Ainv[i,j] *
                sqrt(model.gamma*model.gamma +
                     sum((model.x[h] - model.node[i, h])**2 for h in model.N))
                *sqrt(model.gamma*model.gamma +
                     sum((model.x[h] - model.node[j, h])**2 for h in model.N))
                for i in model.K for j in model.K) +
             sum(model.Ainv[j, model.k] *
                 sqrt(model.gamma*model.gamma +
                      sum((model.x[h] - model.node[j, h])**2 for h in model.N))
                 for j in model.K) +
             model.Ainv[model.k.value, model.k.value] - model.phi_0)/
            ((sum(model.lambda_h[i] *
                  sqrt(model.gamma*model.gamma +
                       sum((model.x[j] - model.node[i, j])**2
                           for j in model.N))
                  for i in model.K) +
              model.lambda_h[model.k.value] - model.fstar)**2))


# Objective function for the "minimize bumpiness with variable nodes"
# problem. The expression is:
# - lambda^T \Phi lambda.
def _min_bump_obj_expression(model):
    return (-sum(model.Phi[i,j] * model.rbf_lambda[i] * model.rbf_lambda[j]
                 for i in model.K for j in model.K))


# Objective function for the "minimize MSRSM obj" problem. The expression is:
# dist_weight * (dist_max - \min_k distance(x, x^k)) / (dist_max - dist_min)
# + (obj_weight) * (s_k(x) - fmin) / (fmax - fmin)
def _min_msrsm_obj_expression_linear(model):
    return (model.dist_weight * (model.dist_max - sqrt(model.mindistsq)) /
            (model.dist_max - model.dist_min) +
            model.obj_weight *
            (sum(model.lambda_h[i] *
                 sqrt(_DISTANCE_SHIFT +
                      sum((model.x[j] - model.node[i, j])**2 for j in model.N))
                 for i in model.K) + model.lambda_h[model.k.value] -
             model.fmin) / 
            (model.fmax - model.fmin))

def _min_msrsm_obj_expression_mq(model):
    return (model.dist_weight * (model.dist_max - sqrt(model.mindistsq)) /
            (model.dist_max - model.dist_min) +
            model.obj_weight *
            (sum(model.lambda_h[i] *
                 sqrt(model.gamma*model.gamma +
                      sum((model.x[j] - model.node[i, j])**2 for j in model.N))
                 for i in model.K) + model.lambda_h[model.k.value] -
             model.fmin) / 
            (model.fmax - model.fmin))

# Function to return bounds of the y variables
def _y_bounds(model, i):
    return (model.var_lower[model.integer_vars[i]], 
            model.var_upper[model.integer_vars[i]])

# Constraints: definition of the integrality constraints for the
# variables.
def _int_constraint_rule(model, i):
    return (model.x[model.integer_vars[i]] == model.y[i])

# Constraints: assignment of categorical variables.
def _cat_constraint_rule(model, i):
    return sum(model.cat_vars[i, j]*model.x[j] for j in model.N) == 1
