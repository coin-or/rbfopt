"""Pyomo models with degree-one polynomial for RBFOpt.

This module creates all the auxiliary problems that rely on degree-one
polynomials. The models are created and instantiated using Pyomo. This
module does *not* solve the problems.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
Research partially supported by SUTD-MIT International Design Center.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from pyomo.environ import *
try:
    import cython_rbfopt.rbfopt_utils as ru
    print('Imported Cython version of rbfopt_utils')
except ImportError:
    import rbfopt_utils as ru
import numpy as np
import rbfopt_config as config
from rbfopt_settings import RbfSettings

def create_min_rbf_model(settings, n, k, var_lower, var_upper, 
                         integer_vars, node_pos, rbf_lambda, rbf_h):
    """Create the concrete model to minimize the RBF.

    Create the concrete model to minimize the RBF.

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
        List of indices of integer variables.

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
    pyomo.ConcreteModel
        The concrete model describing the problem.
    """    
    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(len(rbf_lambda)==k)
    assert(len(rbf_h)==(n+1))
    assert(len(node_pos)==k)
    assert(isinstance(settings, RbfSettings))
    assert(ru.get_degree_polynomial(settings)==1)

    model = ConcreteModel()
    
    # Dimension of the space
    model.n = Param(initialize=n)
    model.N = RangeSet(0, model.n - 1)
        
    # Number of interpolation nodes
    model.k = Param(initialize=k)
    model.K = RangeSet(0, model.k - 1)

    # Dimension of u_pi
    model.q = Param(initialize=(n+k+1))
    model.Q = RangeSet(0, model.q - 1)
    model.Qlast = RangeSet(model.q-1, model.q-1)

    # Coefficients of the RBF
    lambda_h_param = {}
    for i in range(k):
        lambda_h_param[i] = float(rbf_lambda[i])
    for i in range(n+1):
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

    # Auxiliary variables: the vectors (u, \pi),
    # see equations (6) and (7) in the paper by Costa and Nannicini
    model.u_pi = Var(model.Q, domain=Reals)

    model.OBJ = Objective(rule=_min_rbf_obj_expression, sense=minimize)

    # Constraints. See definitions below.
    if (settings.rbf == 'cubic'):        
        model.UdefConstraint = Constraint(model.K, 
                                          rule=_udef_cubic_constraint_rule)
    elif (settings.rbf == 'thin_plate_spline'):
        model.UdefConstraint = Constraint(model.K, 
                                          rule=_udef_thinplate_constraint_rule)
    model.PidefConstraint = Constraint(model.N, rule=_pidef_constraint_rule)
    model.NonhomoConstraint = Constraint(model.Qlast, 
                                         rule=_nonhomo_constraint_rule)

    # Add integer variables if necessary
    if (len(integer_vars) > 0):
        add_integrality_constraints(model, integer_vars)

    return model
# -- end function

def create_max_one_over_mu_model(settings, n, k, var_lower, var_upper, 
                                 integer_vars, node_pos, mat):

    """Create the concrete model to maximize 1/\mu.

    Create the concrete model to maximize :math: `1/\mu`, also known
    as the InfStep of the RBF method.

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
        List of indices of integer variables.

    node_pos : List[List[float]]
        List of coordinates of the nodes.

    mat: numpy.matrix
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0], see paper as cited above. Must
        be a numpy.matrix of dimension ((k+1) x (k+1))

    Returns
    -------
    pyomo.ConcreteModel
        The concrete model describing the problem.
    """
    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(len(node_pos)==k)
    assert(isinstance(mat, np.matrix))
    assert(mat.shape==(n+k+1,n+k+1))
    assert(isinstance(settings, RbfSettings))
    assert(ru.get_degree_polynomial(settings)==1)

    model = ConcreteModel()

    # Dimension of the space
    model.n = Param(initialize=n)
    model.N = RangeSet(0, model.n - 1)
        
    # Number of interpolation nodes
    model.k = Param(initialize=k)
    model.K = RangeSet(0, model.k - 1)

    # Dimension of the matrix
    model.q = Param(initialize=(n+k+1))
    model.Q = RangeSet(0, model.q - 1)
    model.Qlast = RangeSet(model.q - 1, model.q - 1)

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
    for i in range(n+k+1):
        for j in range(i, n+k+1):
            if (abs(mat[i, j]) != 0.0):
                if (i == j):
                    Ainv_param[i, j] = float(mat[i, j])
                else:
                    Ainv_param[i, j] = float(2*mat[i, j])
    model.Ainv = Param(model.Q, model.Q, initialize=Ainv_param,
                       default=0.0)

    # Value of phi at zero, necessary for shift
    if (settings.rbf == 'cubic' or settings.rbf == 'thin_plate_spline'):
        model.phi_0 = Param(initialize=0.0)

    # Variable: the point in the space
    model.x = Var(model.N, domain=Reals, bounds=_x_bounds)

    # Auxiliary variables: the vectors (u, \pi),
    # see equations (6) and (7) in the paper by Costa and Nannicini
    model.u_pi = Var(model.Q, domain=Reals)

    # Objective function. Remember that there should be a constant
    # term \phi(0) at the beginning, but because this is zero in this
    # case we take it out.
    model.OBJ = Objective(rule=_max_one_over_mu_obj_expression,
                          sense=maximize)

    # Constraints. See definitions below.
    if (settings.rbf == 'cubic'):        
        model.UdefConstraint = Constraint(model.K, 
                                          rule=_udef_cubic_constraint_rule)
    elif (settings.rbf == 'thin_plate_spline'):
        model.UdefConstraint = Constraint(model.K, 
                                          rule=_udef_thinplate_constraint_rule)
    model.PidefConstraint = Constraint(model.N, rule=_pidef_constraint_rule)
    model.NonhomoConstraint = Constraint(model.Qlast, 
                                         rule=_nonhomo_constraint_rule)

    # Add integer variables if necessary
    if (len(integer_vars) > 0):
        add_integrality_constraints(model, integer_vars)

    return model
# -- end function

def create_max_h_k_model(settings, n, k, var_lower, var_upper, integer_vars,
                         node_pos, rbf_lambda, rbf_h, mat, target_val):
    """Create the concrete model to maximize h_k.

    Create the concrete model to maximize h_k, also known as the
    Global Search Step of the RBF method.

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
        List of indices of integer variables.

    node_pos : List[List[float]]
        List of coordinates of the nodes.

    rbf_lambda : List[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : List[float]
        The h coefficients of the RBF interpolant, corresponding to
        the polynomial. List of dimension n+1.

    mat: numpy.matrix
        The matrix necessary for the computation. This is the inverse
        of the matrix [Phi P; P^T 0], see paper as cited above. Must
        be a numpy.matrix of dimension ((k+1) x (k+1))

    target_val : float
        Value f* that we want to find in the unknown objective
        function.

    Returns
    -------
    pyomo.ConcreteModel
        The concrete model describing the problem.
    """
    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(len(rbf_lambda)==k)
    assert(len(rbf_h)==(n+1))
    assert(len(node_pos)==k)
    assert(isinstance(mat, np.matrix))
    assert(mat.shape==(n+k+1,n+k+1))
    assert(isinstance(settings, RbfSettings))
    assert(ru.get_degree_polynomial(settings)==1)

    model = ConcreteModel()

    # Dimension of the space
    model.n = Param(initialize=n)
    model.N = RangeSet(0, model.n - 1)
        
    # Number of interpolation nodes
    model.k = Param(initialize=k)
    model.K = RangeSet(0, model.k - 1)

    # Dimension of the matrix
    model.q = Param(initialize=(n+k+1))
    model.Q = RangeSet(0, model.q - 1)
    model.Qlast = RangeSet(model.q - 1, model.q - 1)

    # Coefficients of the RBF
    lambda_h_param = {}
    for i in range(k):
        lambda_h_param[i] = float(rbf_lambda[i])
    for i in range(n+1):
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
    for i in range(n+k+1):
        for j in range(i, n+k+1):
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
    if (settings.rbf == 'cubic' or settings.rbf == 'thin_plate_spline'):
        model.phi_0 = Param(initialize=0.0)

    # Variable: the point in the space
    model.x = Var(model.N, domain=Reals, bounds=_x_bounds)

    # Auxiliary variables: the vectors (u, \pi),
    # see equations (6) and (7) in the paper by Costa and Nannicini
    model.u_pi = Var(model.Q, domain=Reals)

    # Auxiliary variable: value of the rbf at a given point
    model.rbfval = Var(domain=Reals)

    # Auxiliary variable: value of the inverse of \mu_k at a given point
    model.mu_k_inv = Var(domain=Reals)

    # Objective function.
    model.OBJ = Objective(rule=_max_h_k_obj_expression, sense=maximize)

    # Constraints. See definitions below.
    if (settings.rbf == 'cubic'):        
        model.UdefConstraint = Constraint(model.K, 
                                          rule=_udef_cubic_constraint_rule)
    elif (settings.rbf == 'thin_plate_spline'):
        model.UdefConstraint = Constraint(model.K, 
                                          rule=_udef_thinplate_constraint_rule)
    model.PidefConstraint = Constraint(model.N, rule=_pidef_constraint_rule)
    model.NonhomoConstraint = Constraint(model.Qlast, 
                                         rule=_nonhomo_constraint_rule)
    model.RbfdefConstraint = Constraint(rule=_rbfdef_constraint_rule)
    model.MukdefConstraint = Constraint(rule=_mukdef_constraint_rule)

    # Add integer variables if necessary
    if (len(integer_vars) > 0):
        add_integrality_constraints(model, integer_vars)

    return model

# -- end function

def create_min_bump_model(settings, n, k, Phimat, Pmat, node_val, 
                          fast_node_index, fast_node_err_bounds):
    """Create a model to find RBF coefficients with min bumpiness.

    Create a quadratic problem to compute the coefficients of the RBF
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
        considered variable withing the allowed range.
    
    fast_node_err_bounds : List[(float, float)]
        Allowed deviation from node values for nodes affected by
        error. This is a list of pairs (lower, upper) of the same
        length as fast_node_index.

    Returns
    -------
    pyomo.ConcreteModel
        The concrete model describing the problem.
    """
    assert(isinstance(settings, RbfSettings))
    assert(len(node_val)==k)
    assert(isinstance(Phimat, np.matrix))
    assert(isinstance(Pmat, np.matrix))
    assert(Phimat.shape==(k,k))
    assert(Pmat.shape==(k,n+1))
    assert(len(fast_node_index)==len(fast_node_err_bounds))
    assert(ru.get_degree_polynomial(settings)==1)

    model = ConcreteModel()

    # Dimension of the space
    model.n = Param(initialize=n)
    model.N = RangeSet(0, model.n - 1)

    # Dimension of P matrix
    model.p = Param(initialize=n+1)
    model.P = RangeSet(0, model.n)
        
    # Number of interpolation nodes
    model.k = Param(initialize=k)
    model.K = RangeSet(0, model.k - 1)

    # Node values, i.e. right hand sides of the first set of equations
    # in the constraints
    node_val_param = {}
    for i in range(k):
        node_val_param[i] = float(node_val[i])
    model.node_val = Param(model.K, initialize=node_val_param)

    # Slack variable bounds
    slack_lower_param = {}
    slack_upper_param = {}
    for (pos, var_index) in enumerate(fast_node_index):
        slack_lower_param[var_index] = float(fast_node_err_bounds[pos][0])
        slack_upper_param[var_index] = float(fast_node_err_bounds[pos][1])
    model.slack_lower = Param(model.K, initialize=slack_lower_param,
                              default=0.0)
    model.slack_upper = Param(model.K, initialize=slack_upper_param,
                              default=0.0)

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
        for j in range(n+1):
            if (abs(Pmat[i, j]) != 0.0):
                Pm_param[i, j] = float(Pmat[i, j])
    model.Pm = Param(model.K, model.P, initialize=Pm_param,
                     default=0.0)

    # Variable: the lambda coefficients of the RBF
    model.rbf_lambda = Var(model.K, domain=Reals)

    # Variable: the h coefficients of the RBF
    model.rbf_h = Var(model.P, domain=Reals)

    # Variable: the slacks for the equality constraints
    model.slack = Var(model.K, domain=Reals, bounds=_slack_bounds)

    # Objective function.
    model.OBJ = Objective(rule=_min_bump_obj_expression, sense=minimize)

    # Constraints. See definitions below.
    model.IntrConstraint = Constraint(model.K, rule=_intr_constraint_rule)
    model.UnisConstraint = Constraint(model.P, rule=_unis_constraint_rule)    

    return model

# -- end function

def create_maximin_dist_model(settings, n, k, var_lower, var_upper,
                              integer_vars, node_pos):
    """Create the concrete model to maximize the minimum distance.

    Create the concrete model to maximize the minimum distance to the
    interpolation nodes, which is the infstep of the MSRSM method.

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
        List of indices of integer variables.

    node_pos : List[List[float]]
        List of coordinates of the nodes.

    Returns
    -------
    pyomo.ConcreteModel
        The concrete model describing the problem.

    """
    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(len(node_pos)==k)
    assert(isinstance(settings, RbfSettings))

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

    # Auxiliary variable: value of the inverse of \mu_k at a given point
    model.mindistsq = Var(domain=NonNegativeReals)

    # Objective function.
    model.OBJ = Objective(expr=(model.mindistsq), sense=maximize)
    model.MdistdefConstraint = Constraint(model.K,
                                          rule=_mdistdef_constraint_rule)

    # Add integer variables if necessary
    if (len(integer_vars) > 0):
        add_integrality_constraints(model, integer_vars)

    return model

# -- end function

def create_min_msrsm_model(settings, n, k, var_lower, var_upper,
                           integer_vars, node_pos, rbf_lambda, rbf_h, 
                           dist_weight, dist_min, dist_max, fmin, fmax):
    """Create the concrete model to optimize the MSRSM objective.

    Create the concreate model to minimize a weighted combination of
    the value of the RBF interpolant and the (negative of the)
    distance from the closes interpolation node. This is the Global
    Search Step of the MSRSM method.

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
        List of indices of integer variables.

    node_pos : List[List[float]]
        List of coordinates of the nodes.

    rbf_lambda : List[float]
        The lambda coefficients of the RBF interpolant, corresponding
        to the radial basis functions. List of dimension k.

    rbf_h : List[float]
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
    assert(len(var_lower)==n)
    assert(len(var_upper)==n)
    assert(len(rbf_lambda)==k)
    assert(len(rbf_h)==(n+1))
    assert(len(node_pos)==k)
    assert(isinstance(settings, RbfSettings))
    assert(ru.get_degree_polynomial(settings)==1)
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
    model.q = Param(initialize=(n+k+1))
    model.Q = RangeSet(0, model.q - 1)
    model.Qlast = RangeSet(model.q - 1, model.q - 1)

    # Coefficients of the RBF
    lambda_h_param = {}
    for i in range(k):
        lambda_h_param[i] = float(rbf_lambda[i])
    for i in range(n+1):
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
    if (settings.rbf == 'cubic' or settings.rbf == 'thin_plate_spline'):
        model.phi_0 = Param(initialize=0.0)

    # Variable: the point in the space
    model.x = Var(model.N, domain=Reals, bounds=_x_bounds)

    # Auxiliary variables: the vectors (u, \pi),
    # see equations (6) and (7) in the paper by Costa and Nannicini
    model.u_pi = Var(model.Q, domain=Reals)

    # Auxiliary variable: value of the rbf at a given point
    model.rbfval = Var(domain=Reals)

    # Auxiliary variable: value of the inverse of \mu_k at a given point
    model.mindistsq = Var(domain=NonNegativeReals)

    # Objective function.
    model.OBJ = Objective(rule=_min_msrsm_obj_expression, sense=minimize)

    # Constraints. See definitions below.
    if (settings.rbf == 'cubic'):        
        model.UdefConstraint = Constraint(model.K, 
                                          rule=_udef_cubic_constraint_rule)
    elif (settings.rbf == 'thin_plate_spline'):
        model.UdefConstraint = Constraint(model.K, 
                                          rule=_udef_thinplate_constraint_rule)
    model.PidefConstraint = Constraint(model.N, rule=_pidef_constraint_rule)
    model.NonhomoConstraint = Constraint(model.Qlast, 
                                         rule=_nonhomo_constraint_rule)
    model.RbfdefConstraint = Constraint(rule=_rbfdef_constraint_rule)
    model.MdistdefConstraint = Constraint(model.K,
                                          rule=_mdistdef_constraint_rule)

    # Add integer variables if necessary
    if (len(integer_vars) > 0):
        add_integrality_constraints(model, integer_vars)

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

    integer_vars : List[int]
        List of indices of integer variables.
    """
    assert(len(integer_vars) > 0)

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

# Function to return bounds
def _x_bounds(model, i):
    return (model.var_lower[i], model.var_upper[i])

# Constraints: definition of the u components of u_pi for cubic RBF. 
# The expression is:
# for i in K: upi_i = \sqrt(sum_{j in N} (x_j - node_{i, j})^2)^3;
def _udef_cubic_constraint_rule(model, i):
    return (model.u_pi[i] == 
            sqrt(config.DISTANCE_SHIFT + 
                 sum((model.x[j] - model.node[i, j])**2 for j in model.N))**3)

# Constraints: definition of the u components of u_pi for thin plate spline
# RBF. The expression is:
# for i in K: upi_i = \log(\sqrt(sum_{j in N} (x_j - node_{i, j})^2)) *
#                     (sum_{j in N} (x_j - node_{i, j})^2)
def _udef_thinplate_constraint_rule(model, i):
    return (model.u_pi[i] == 
            (sum((model.x[j] - model.node[i, j])**2 for j in model.N)) *
            log(sqrt(config.DISTANCE_SHIFT + 
                     sum((model.x[j] - model.node[i, j])**2
                         for j in model.N))))


# Constraints: definition of the pi component of u_pi. The expression is:
# for i in N: upi_{k+i} = x_i
def _pidef_constraint_rule(model, i):
    return (model.u_pi[i + model.k] == model.x[i])

# Constraint: definition of the nonhomogeneous term of the polynomial. 
# The expression is: upi_q = 1.0
def _nonhomo_constraint_rule(model, i):
    return (model.u_pi[i] == 1.0)

# Constraints: definition of the value of the RBF. Expression:
# sum_{j in K} lambda_j u_j + sum_{j in N} h_j \pi_j + h_{n+1} \pi_{n+1}
# Because of the definition of u_pi, we can just write:
# min sum_{j in Q} lambda_h_j u_pi_j
def _rbfdef_constraint_rule(model):
    return (model.rbfval == summation(model.lambda_h, model.u_pi))

# Constraints: Definition of \mu_k. There should be a constant term
# \phi(0). Removed because it is zero in this case. Expression:
# -\sum_{i in Q, j in Q} A^{-1}_{ij} upi_i upi_j
def _mukdef_constraint_rule(model):
    return (-1.0*sum(model.Ainv[i,j] * model.u_pi[i] * model.u_pi[j] 
                     for i in model.Q for j in model.Q) + 
            model.phi_0 == model.mu_k_inv)

# Constraints: definition of the interpolation conditions. Expression:
# Phi lambda + P h + slack = F
def _intr_constraint_rule(model, i):
    return (sum(model.Phi[i, j]*model.rbf_lambda[j] for j in model.K) +
            sum(model.Pm[i, j]*model.rbf_h[j] for j in model.P) +
            model.slack[i] == model.node_val[i])

# Constraints: definition of the unisolvence conditions. Expression:
# P \lambda = 0
def _unis_constraint_rule(model, i):
    return (sum(model.Pm[j, i]*model.rbf_lambda[j] for j in model.K) == 0.0)

# Constraints: definition of the minimum distance constraint.
# for i in K: mindistsq <= dist(x, x^i)^2
def _mdistdef_constraint_rule(model, i):
    return (model.mindistsq <= config.DISTANCE_SHIFT + 
            sum((model.x[j] - model.node[i, j])**2 for j in model.N))

# Objective function for the "minimize rbf" problem. The expression is:
# min sum_{j in K} lambda_j d_j^3 + sum_{j in N} h_j x_j + h_{n+1}
# Because of the definition of u_pi, we can just write:
# min sum_{j in Q} lambda_h_j u_pi_j
def _min_rbf_obj_expression(model):
    return (summation(model.lambda_h, model.u_pi))

# Objective function for the "maximize 1/\mu" problem. The expression is:
# max -\sum_{i in Q, j in Q} A^{-1}_{ij} upi_i upi_j;
def _max_one_over_mu_obj_expression(model):
    return (-1.0*sum(model.Ainv[i,j] * model.u_pi[i] * model.u_pi[j] 
                     for i in model.Q for j in model.Q) + model.phi_0)

# Objective function for the "maximize h_k" problem. The expression is:
# 1/(\mu_k(x) [s_k(x) - f^\ast]^2)
def _max_h_k_obj_expression(model):
    return (model.mu_k_inv/((model.rbfval - model.fstar)**2))

# Objective function for the "minimize bumpiness with variable nodes"
# problem. The expression is:
# lambda^T \Phi lambda.
def _min_bump_obj_expression(model):
    return (sum(model.Phi[i,j] * model.rbf_lambda[i] * model.rbf_lambda[j]
                for i in model.K for j in model.K))

# Objective function for the "minimize MSRSM obj" problem. The expression is:
# dist_weight * (dist_max - \min_k distance(x, x^k)) / (dist_max - dist_min)
# + (obj_weight) * (s_k(x) - fmin) / (fmax - fmin)
def _min_msrsm_obj_expression(model):
    return (model.dist_weight * (model.dist_max - sqrt(model.mindistsq)) /
            (model.dist_max - model.dist_min) +
            model.obj_weight * (model.rbfval - model.fmin) / 
            (model.fmax - model.fmin))

# Function to return bounds on the slack variables
def _slack_bounds(model, i):
    return (model.slack_lower[i], model.slack_upper[i])

# Function to return bounds of the y variables
def _y_bounds(model, i):
    return (model.var_lower[model.integer_vars[i]], 
            model.var_upper[model.integer_vars[i]])

# Constraints: definition of the integrality constraints for the
# variables.
def _int_constraint_rule(model, i):
    return (model.x[model.integer_vars[i]] == model.y[i])
