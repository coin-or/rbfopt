"""Optimize a black-box function using the RBF method.

This module contains the main optimization algorithm and its settings.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
Research partially supported by SUTD-MIT International Design Center.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import math
import random
import time
import copy
import itertools as it
import numpy as np
import rbfopt_utils as ru
import rbfopt_aux_problems as aux
import rbfopt_config as config

class RbfSettings:
    """Global and algorithmic settings for RBF method.

    Class containing algorithmic settings for the enhanced RBF method,
    as well as global settings such as tolerances, limits and so on.

    Parameters
    ----------

    rbf : str
        Radial basis function used by the method. Choice of 'cubic',
        'thin_plate_spline', 'linear', 'multiquadric', 'auto'. Default
        'cubic'.

    max_iterations : int
        Maximum number of iterations. Default 150.

    max_evaluations : int
        Maximum number of function evaluations in accurate mode. This
        includes the evaluations to initialize the algorithm. Default
        250.

    max_fast_evaluations : int
        Maximum number of function evaluations in fast mode.
        Default 150.

    max_clock_time : float
        Maximum wall clock time in seconds. Default 1.0e30.

    target_objval : float
        The objective function value we want to reach, i.e. the value
        of the unknown optimum. It can be set to an acceptable value,
        if the optimum is unknown. Default -1.0e10.

    eps_opt : float
        Optimality threshold. Any solution within this relative
        distance from the target_objval is considered optimal.
        Default 1.0e-2.

    eps_zero : float
        Tolerance for zeroing out small coefficients in the
        calculations. Any value smaller than this will be considered
        zero. Default 1.0e-15.

    eps_impr : float
        Tolerance for improvement of the objective function. Any
        improvement in the objective function by less than this amount
        in absolute and relative terms, will be ignored.  Default
        1.0e-3.

    min_dist : float
        Minimum Euclidean distance between nodes. A new point will be
        discarded if it is closer than this value from existing
        nodes. This prevents the RBF pairwise distance matrix from
        becoming singular. Default 1.0e-5.

    do_infstep : bool
        Should we perform the InfStep? Default False.

    skip_targetval_clipping : bool
        Should we skip the method to clip target value selection based
        on periodically eliminating some of the largest function
        values, as proposed by Gutmann (2001) and later Regis and
        Shoemaker (2007)? Default False.

    num_global_searches : int
        Number of steps in the global search phase. Default 5.

    max_consecutive_local_searches : int
        Maximum number of consecutive local searches during the
        optimization phase. Default 2.

    init_strategy : str
        Strategy to select initial points. Choice of 'all_corners',
        'lower_corners', 'rand_corners', 'lhd_maximin',
        'lhd_corr'. Default 'lhd_maximin'.

    function_scaling : str
        Rescaling method for the function values. Choice of 'off',
        'affine', 'log', 'auto'. Default 'auto'.

    domain_scaling : str
        Rescaling method for the domain. Choice of 'off', 'affine',
        'auto'. Default 'auto'.

    dynamism_clipping : str
        Dynamism clipping strategy. Choice of 'off', 'median',
        'clip_at_dyn', 'auto'. Default 'auto'.

    dynamism_threshold : float
        Minimum value of the ratio between the largest and the
        smallest absolute function values before the dynamism clipping
        strategy is applied. Default 1.0e3.

    local_search_box_scaling : float
        Rescaling factor for the hyperbox used for local search. See
        parameter nu of Regis and Shoemaker (2007). Default 0.5.

    max_stalled_cycles : int
        Maximum number of consecutive optimization cycles without
        improvement before we perform a full restart. Default 6.

    max_stalled_objfun_impr : float
        Maximum relative objective function improvement between
        consecutive optimization cycles to be considered
        "stalling". Default 0.05.
 
    fast_objfun_rel_error : float
        An estimate of the relative error by which the fast version of
        the objective function is affected. Default 0.1.
        
    fast_objfun_abs_error : float
        An estimate of the absolute error by which the fast
        version of the objective function is affected. Default
        0.0.
        
    max_fast_restarts : int
        Maximum number of restarts in fast mode before we switch
        to accurate mode. Default 2.
    
    max_fast_iterations : int
        Maximum number of iterations in fast mode before switching
        to accurate mode. Default 100.
    
    print_solver_output : bool
        Print the output of the solvers to screen? Note that this
        cannot be redirected to file so it will go to
        stdout. Default False.

    rand_seed : int
        Seed for the random number generator. Default 937627691.

    Attributes
    ----------

    _allowed_rbf : Dict[str]
        Allowed types of RBF functions.
    _allowed_init_strategy : Dict[str]
        Allowed initialization strategies.
    _allowed_function_scaling : Dict[str]
        Allowed function scaling strategies.
    _allowed_domain_scaling : Dict[str]
        Allowed domain scaling strategies.
    _allowed_dynamism_clipping : Dict[str]
        Allowed dynamism clipping strategies.
    """

    # Allowed values for multiple choice options
    _allowed_rbf = {'auto', 'cubic', 'thin_plate_spline', 'linear',
                   'multiquadric'}
    _allowed_init_strategy = {'all_corners', 'lower_corners', 'rand_corners',
                             'lhd_maximin', 'lhd_corr'}
    _allowed_function_scaling = {'off', 'affine', 'log', 'auto'}
    _allowed_domain_scaling = {'off', 'affine', 'auto'}
    _allowed_dynamism_clipping = {'off', 'median', 'clip_at_dyn', 'auto'}

    def __init__(self,
                 rbf = 'cubic',
                 max_iterations = 150,
                 max_evaluations = 250,
                 max_fast_evaluations = 150,
                 max_clock_time = 1.0e30,
                 target_objval = -1.0e10,
                 eps_opt = 1.0e-2,
                 eps_zero = 1.0e-15,
                 eps_impr = 1.0e-3,
                 min_dist = 1.0e-5,
                 do_infstep = False,
                 skip_targetval_clipping = False,
                 num_global_searches = 5,
                 max_consecutive_local_searches = 2,
                 init_strategy = 'lhd_maximin',
                 function_scaling = 'auto',
                 domain_scaling = 'off',
                 dynamism_clipping = 'auto',
                 dynamism_threshold = 1.0e3,
                 local_search_box_scaling = 0.5,
                 max_stalled_cycles = 6,
                 max_stalled_objfun_impr = 0.05,
                 fast_objfun_rel_error = 0.1,
                 fast_objfun_abs_error = 0.0,
                 max_fast_restarts = 2,
                 max_fast_iterations = 100,
                 print_solver_output = False,
                 rand_seed = 937627691):
        """Class constructor with default values. 
        """
        self.rbf = rbf
        self.max_iterations = max_iterations
        self.max_evaluations = max_evaluations
        self.max_fast_evaluations = max_fast_evaluations
        self.max_clock_time = max_clock_time
        self.target_objval = target_objval
        self.eps_opt = eps_opt
        self.eps_zero = eps_zero
        self.eps_impr = eps_impr
        self.min_dist = min_dist
        self.do_infstep = do_infstep
        self.skip_targetval_clipping = skip_targetval_clipping
        self.num_global_searches = num_global_searches
        self.max_consecutive_local_searches = max_consecutive_local_searches
        self.init_strategy = init_strategy
        self.function_scaling = function_scaling
        self.domain_scaling = domain_scaling
        self.dynamism_clipping = dynamism_clipping
        self.dynamism_threshold = dynamism_threshold
        self.local_search_box_scaling = local_search_box_scaling
        self.max_stalled_cycles = max_stalled_cycles
        self.max_stalled_objfun_impr = max_stalled_objfun_impr
        self.fast_objfun_rel_error = fast_objfun_rel_error
        self.fast_objfun_abs_error = fast_objfun_abs_error
        self.max_fast_restarts = max_fast_restarts
        self.max_fast_iterations = max_fast_iterations
        self.print_solver_output = print_solver_output
        self.rand_seed = rand_seed

        if (self.rbf not in RbfSettings._allowed_rbf):
            raise ValueError('settings.rbf = ' + 
                             str(self.rbf) + ' not supported')
        if (self.init_strategy not in RbfSettings._allowed_init_strategy):
            raise ValueError('settings.init_strategy = ' + 
                             str(self.init_strategy) + ' not supported')
        if (self.function_scaling not in 
            RbfSettings._allowed_function_scaling):
            raise ValueError('settings.function_scaling = ' + 
                             str(self.function_scaling) + ' not supported')
        if (self.domain_scaling not in RbfSettings._allowed_domain_scaling):
            raise ValueError('settings.domain_scaling = ' + 
                             str(self.domain_scaling) + ' not supported')
        if (self.dynamism_clipping not in 
            RbfSettings._allowed_dynamism_clipping):
            raise ValueError('settings.dynamism_clipping = ' + 
                             str(self.dynamism_clipping) + ' not supported')

    def set_auto_parameters(self, dimension, var_lower, var_upper,
                            integer_vars):
        """Determine the value for 'auto' parameters.

        Create a copy of the settings and assign 'auto' parameters. The
        original settings are untouched.

        Parameters
        ----------

        dimension : int
            The dimension of the problem, i.e. size of the space.

        var_lower : List[float]
            Vector of variable lower bounds.

        var_upper : List[float]
            Vector of variable upper bounds.

        integer_vars : List[int] or None
            A list containing the indices of the integrality
            constrained variables. If None or empty list, all
            variables are assumed to be continuous.

        Returns
        -------
        RbfSettings
            A copy of the settings, without any 'auto' parameter values.
        """
        assert(dimension==len(var_lower))
        assert(dimension==len(var_upper))
        assert((integer_vars is None) or (len(integer_vars) == 0) or
               (max(integer_vars) < dimension))

        l_settings = copy.deepcopy(self)

        if (l_settings.rbf == 'auto'):
            l_settings.rbf = 'cubic'

        if (l_settings.function_scaling == 'auto'):
            if (l_settings.rbf == 'linear'):
                l_settings.function_scaling = 'affine'
            else:
                l_settings.function_scaling = 'off'
            
        if (l_settings.dynamism_clipping == 'auto'):
            l_settings.dynamism_clipping = 'median'
                
        if (l_settings.domain_scaling == 'auto'):
            if (integer_vars is not None):
                l_settings.domain_scaling = 'off'
            else:
                # Compute the length of the domain of each variable
                size = [var_upper[i]-var_lower[i] for i in range(dimension)]
                size.sort()
                # If the problem is badly scaled, i.e. a variable has
                # a domain 5 times as large as anoether, rescale.
                if (size[-1] >= 5*size[0]):
                    l_settings.domain_scaling = 'affine'
                else:
                    l_settings.domain_scaling = 'off'

        return l_settings
        
    def print(self, output_stream):
        """Print the value of all settings.

        Prints the settings to the output stream, on a very long line.

        Parameters
        ----------

        output_stream : file
            The stream on which messages are printed.
        """
        print('RbfSettings:', file = output_stream)
        attrs = vars(self)
        print(', '.join('{:s}: {:s}'.format(str(item[0]), str(item[1])) 
                        for item in sorted(attrs.items())),
              file = output_stream)
        print(file = output_stream)
        output_stream.flush()

# -- end of class RbfSettings

def rbf_optimize(settings, dimension, var_lower, var_upper, objfun,
                 objfun_fast = None, integer_vars = None, 
                 init_node_pos = None, init_node_val = None,
                 output_stream = sys.stdout):
    """Optimize a black-box function.

    Optimize an unknown function over a box using the enhanced RBF
    method.

    Parameters
    ----------

    settings : RbfSettings
        Global and algorithmic settings.

    dimension : int
        The dimension of the problem, i.e. size of the space.

    var_lower : List[float]
        Vector of variable lower bounds.

    var_upper : List[float]
        Vector of variable upper bounds.

    objfun : Callable[List[float]]
        The unknown function we want to optimize.

    objfun_fast : Callable[List[float]]
        A faster, lower quality version of the unknown function we
        want to optimize. If None, it is assumed that such a version
        of the function is not available.

    integer_vars : List[int] or None
        A list containing the indices of the integrality constrained
        variables. If None or empty list, all variables are assumed to
        be continuous.

    init_node_pos : List[List[float]] or None
        Coordinates of points at which the function value is known. If
        None, the initial points will be generated by the
        algorithm. This must be of length at least dimension + 1, if
        provided.

    init_node_val: List[float] or None
        Function values corresponding to the points given in
        init_node_pos. Should be None if the previous argument is
        None.

    output_stream : file or None 
        A stream object that will be used to print output. By default,
        this will be the standard output stream.

    Returns
    ---
    (float, List[float], int, int, int)
        A quintuple (value, point, itercount, evalcount,
        fast_evalcount) containing the objective function value of the
        best solution found, the corresponding value of the decision
        variables, the number of iterations of the algorithm, the
        total number of function evaluations, and the number of these
        evaluations that were performed in 'fast' mode.
    """

    assert(len(var_lower) == dimension)
    assert(len(var_upper) == dimension)
    assert((integer_vars is None) or (len(integer_vars) == 0) or
           (max(integer_vars) < dimension))
    assert(init_node_pos is None or 
           (len(init_node_pos) == len(init_node_val) and
            len(init_node_pos) >= dimension + 1))
    assert(isinstance(settings, RbfSettings))

    # Start timing
    start_time = time.time()

    # Set the value of 'auto' parameters if necessary
    l_settings = settings.set_auto_parameters(dimension, var_lower,
                                              var_upper, integer_vars)

    # Local and global RBF models are usually the same
    best_local_rbf, best_global_rbf = l_settings.rbf, l_settings.rbf
    
    # We use n to denote the dimension of the problem, same notation
    # of the paper. This is redundant but it simplifies our life.
    n = dimension

    # Set random seed. Some of the (external) libraries use numpy's
    # random generator, we use python's internal generator, so we have
    # to seed both for consistency.
    random.seed(l_settings.rand_seed)
    np.random.seed(l_settings.rand_seed)

    # Iteration number
    itercount = 0

    # Total number of function evaluations in accurate mode
    evalcount = 0
    # Total number of fast function evaluation
    fast_evalcount = 0

    # Identifier of the current step within the cyclic optimization
    # strategy counter. This typically increases at every iteration,
    # but sometimes we may decide to repeat a step.
    current_step = 0

    # Current number of consecutive local searches
    num_cons_ls = 0

    # Number of consecutive cycles without improvement
    num_stalled_cycles = 0

    # Number of restarts in fast mode
    num_fast_restarts = 0
    
    # Initialize identifiers of the search steps
    inf_step = 0
    local_search_step = (l_settings.num_global_searches + 1)
    cycle_length = (l_settings.num_global_searches + 2)

    # Initialize settings for two-phase optimization.
    # two_phase_optimization indicates if the fast buy noisy objective
    # function is available.
    # is_best_fast indicates if the best known objective function
    # value was evaluated in fast mode or in accurate mode.
    # current_mode indicates the evaluation mode for the objective
    # function at a given stage.
    if (objfun_fast is not None):
        two_phase_optimization = True
        is_best_fast = True
        current_mode = 'fast'
    else:
        two_phase_optimization = False
        is_best_fast = False
        current_mode = 'accurate'

    # Round variable bounds to integer if necessary
    ru.round_integer_bounds(var_lower, var_upper, integer_vars)

    # List of node coordinates is node_pos, list of node values is in
    # node_val. We keep a current list and a global list; they can be
    # different in case of restarts.

    # We must choose the initial interpolation points. If they are not
    # given, generate them using the chosen strategy.
    if (init_node_pos is None):
        node_pos = ru.initialize_nodes(l_settings, var_lower, var_upper, 
                                       integer_vars)
        if (current_mode == 'accurate'):
            node_val = [objfun(point) for point in node_pos]
            evalcount += len(node_val)
        else:
            node_val = [objfun_fast(point) for point in node_pos]
            fast_evalcount += len(node_val)
        node_is_fast = [current_mode == 'fast' for val in node_val]
    else:
        node_pos = init_node_pos
        node_val = init_node_val
        # We assume that initial points provided by the user are
        # 'accurate'.
        node_is_fast = [False for val in node_val]

    # Make a copy, in the original space
    all_node_pos = [point for point in node_pos]
    all_node_val = [val for val in node_val]
    # Store if each function evaluation is fast or accurate
    all_node_is_fast = [val for val in node_is_fast]
    # We need to remember the index of the first node in all_node_pos
    # after every restart
    all_node_pos_size_at_restart = 0

    # Rescale the domain of the function
    node_pos = [ru.transform_domain(l_settings, var_lower, var_upper, point)
                for point in node_pos]

    (l_lower,
     l_upper) = ru.transform_domain_bounds(l_settings, var_lower, var_upper)

    # Current minimum value among the nodes, and its index
    fmin_index = node_val.index(min(node_val))
    fmin = node_val[fmin_index]
    # Current maximum value among the nodes
    fmax = max(all_node_val)
    # Denominator of errormin
    gap_den = (abs(l_settings.target_objval) 
               if (abs(l_settings.target_objval) >= l_settings.eps_zero)
               else 1.0)
    # Shift due to fast function evaluation
    gap_shift = (ru.get_fast_error_bounds(l_settings, fmin)[1]
                 if is_best_fast else 0.0)
    # Current minimum distance from the optimum
    gap = ((fmin + gap_shift - l_settings.target_objval)/gap_den)

    # Best value function at the beginning of an optimization cycle
    fmin_cycle_start = fmin

    # Print the initialization points
    for (i, val) in enumerate(node_val):
        min_dist = ru.get_min_distance(node_pos[i], node_pos[:i] + 
                                       node_pos[(i+1):])
        print('Iteration {:3d}'.format(itercount) + 
              ' {:16s}'.format('Initialization') +
              ': objval{:s}'.format('~' if node_is_fast[i] else ' ') +
              ' {:16.6f}'.format(val) +
              ' min_dist {:9.4f}'.format(min_dist) +
              ' gap {:8.2f}'.format(gap*100),
              file = output_stream)

    # Main loop
    while (itercount < l_settings.max_iterations and
           evalcount < l_settings.max_evaluations and
           time.time() - start_time < l_settings.max_clock_time and
           gap > l_settings.eps_opt):
        # Number of nodes at current iteration
        k = len(node_pos)

        # Compute indices of fast node evaluations (sparse format)
        fast_node_index = ([i for (i, val) in enumerate(node_is_fast) if val]
                           if two_phase_optimization else list())
        
        # Rescale nodes if necessary
        (scaled_node_val, scaled_fmin, scaled_fmax,
         node_err_bounds) = ru.transform_function_values(l_settings, node_val,
                                                         fmin, fmax,
                                                         fast_node_index)

        # If selection is automatic, at the beginning of each cycle
        # check if a different RBF yields a better model
        if (settings.rbf == 'auto' and k > n+1 and current_step == inf_step):
            best_local_rbf = ru.get_best_rbf_model(l_settings, n, k, node_pos,
                                                   scaled_node_val,
                                                   int(math.ceil(k*0.1)))
            best_global_rbf = ru.get_best_rbf_model(l_settings, n, k, node_pos,
                                                    scaled_node_val,
                                                    int(math.ceil(k*0.7)))
        # If we are in local search or just before local search, use a
        # local model.
        if (current_step >= (local_search_step - 1)):
            l_settings.rbf = best_local_rbf
        # Otherwise, global.
        else:
            l_settings.rbf = best_global_rbf

        # Compute the matrices necessary for the algorithm
        Amat = ru.get_rbf_matrix(l_settings, n, k, node_pos)
        Amatinv = ru.get_matrix_inverse(l_settings, Amat)
        # Compute RBF interpolant at current stage
        if (fast_node_index):
            # Get coefficients for the exact RBF
            (rbf_l, rbf_h) = ru.get_rbf_coefficients(l_settings, n, k, 
                                                     Amat, scaled_node_val)
            # RBF with some fast function evaluations
            (rbf_l, rbf_h) = aux.get_noisy_rbf_coefficients(l_settings, n, k, 
                                                            Amat[:k, :k],
                                                            Amat[:k, k:],
                                                            scaled_node_val,
                                                            fast_node_index,
                                                            node_err_bounds,
                                                            rbf_l, rbf_h)
        else:
            # Fully accurate RBF
            (rbf_l, rbf_h) = ru.get_rbf_coefficients(l_settings, n, k, 
                                                     Amat, scaled_node_val)

        # For displaying purposes, record what type of iteration we
        # are performing
        iteration_id = ''
        
        # Initialize the new point to None
        next_p = None

        if (current_step == inf_step):
            # If the user wants to skip inf_step as in the original
            # paper of Gutmann (2001), we proceed to the next
            # iteration.
            if (not l_settings.do_infstep):
                current_step = (current_step+1) % cycle_length
                continue
            # Infstep: explore the parameter space
            next_p = aux.maximize_one_over_mu(l_settings, n, k, l_lower,
                                              l_upper, node_pos, Amatinv,
                                              integer_vars)
            iteration_id = 'InfStep'
            
        elif (current_step == local_search_step):
            # Local search: compute the minimum of the RBF.
            min_rbf = aux.minimize_rbf(l_settings, n, k, l_lower,
                                       l_upper, node_pos,
                                       rbf_l, rbf_h, integer_vars)
            if (min_rbf is not None):
                min_rbf_val = ru.evaluate_rbf(l_settings, min_rbf, n, k, 
                                              node_pos, rbf_l, rbf_h)
            # If the RBF cannot me minimized, or if the minimum is
            # larger than the node with smallest value, just take the
            # node with the smallest value.
            if (min_rbf is None or 
                (min_rbf_val >= scaled_fmin + l_settings.eps_zero)):
                min_rbf = node_pos[fmin_index]
                min_rbf_val = scaled_fmin
            # Check if point can be accepted: is there an improvement?
            if (min_rbf_val <= (scaled_fmin - l_settings.eps_impr * 
                                max(1.0, abs(scaled_fmin)))):
                target_val = min_rbf_val
                next_p = min_rbf
                iteration_id = 'LocalStep'                    
            else:
                # If the point is not improving, we solve a global
                # search problem, rescaling the search box to enforce
                # some sort of local search
                target_val = scaled_fmin - 0.01*max(1.0, abs(scaled_fmin))
                local_varl = [max(l_lower[i], min_rbf[i] -
                                  l_settings.local_search_box_scaling * 
                                  0.33 * (l_upper[i] - l_lower[i]))
                              for i in range(n)]
                local_varu = [min(l_upper[i], min_rbf[i] +
                                  l_settings.local_search_box_scaling * 
                                  0.33 * (l_upper[i] - l_lower[i]))
                              for i in range(n)]
                ru.round_integer_bounds(local_varl, local_varu,
                                        integer_vars)
                next_p  = aux.maximize_h_k(l_settings, n, k, local_varl,
                                           local_varu, node_pos, rbf_l, 
                                           rbf_h, Amatinv, target_val,
                                           integer_vars)
                iteration_id = 'AdjLocalStep'

        else:
            # Global search: compromise between finding a good value
            # of the objective function, and improving the model.
            # Choose target value for the objective function. To do
            # so, we need the minimum of the RBF interpolant.
            min_rbf = aux.minimize_rbf(l_settings, n, k, l_lower,
                                       l_upper, node_pos,
                                       rbf_l, rbf_h, integer_vars)
            if (min_rbf is not None):
                min_rbf_val = ru.evaluate_rbf(l_settings, min_rbf, n, k, 
                                              node_pos, rbf_l, rbf_h)
            # If the RBF cannot me minimized, or if the minimum is
            # larger than the node with smallest value, just take the
            # node with the smallest value.
            if (min_rbf is None or 
                min_rbf_val >= scaled_fmin + l_settings.eps_zero):
                min_rbf = node_pos[fmin_index]
                min_rbf_val = scaled_fmin
            # The scaling factor is 1 - h/kappa, where h goes from
            # 0 to kappa-1 over the course of one global search
            # cycle, and kappa is the number of global searches.
            scaling = (1 - ((current_step - 1) /
                            l_settings.num_global_searches))**2
            # Compute the function value used to determine the
            # target value. This is given by the sorted value in
            # position sigma_n, where sigma_n is a function
            # described in the paper by Gutmann (2001). Unless we
            # want to skip this approach.
            if (l_settings.skip_targetval_clipping):
                local_fmax = fmax
            else:
                local_fmax = ru.get_fmax_current_iter(l_settings, n, k, 
                                                      current_step, 
                                                      scaled_node_val)
            target_val = (min_rbf_val - 
                          scaling * (local_fmax - min_rbf_val))

            # If the global search is almost a local search, we
            # restrict the search to a box following Regis and
            # Shoemaker (2007)
            if (scaling <= config.LOCAL_SEARCH_THRESHOLD):
                local_varl = [max(l_lower[i], min_rbf[i] -
                                  l_settings.local_search_box_scaling * 
                                  math.sqrt(scaling) * 
                                  (l_upper[i] - l_lower[i]))
                              for i in range(n)]
                local_varu = [min(l_upper[i], min_rbf[i] +
                                  l_settings.local_search_box_scaling * 
                                  math.sqrt(scaling) * 
                                  (l_upper[i] - l_lower[i]))
                              for i in range(n)]
                ru.round_integer_bounds(local_varl, local_varu,
                                        integer_vars)
            # Otherwise, use original bounds
            else:
                local_varl = l_lower
                local_varu = l_upper

            next_p  = aux.maximize_h_k(l_settings, n, k, local_varl,
                                       local_varu, node_pos, rbf_l,
                                       rbf_h, Amatinv, target_val,
                                       integer_vars)
            iteration_id = 'GlobalStep'

        # -- end if

        # If previous points were evaluated in low quality and we are
        # now in high-quality local search mode, then we should verify
        # if it is better to evaluate a brand new point or re-evaluate
        # a previously known point.
        if ((two_phase_optimization == True) and
            (current_step == local_search_step) and
            (current_mode == 'accurate')):
            (ind, bump) = ru.get_min_bump_node(l_settings, n, k, Amat, 
                                               scaled_node_val,
                                               fast_node_index, 
                                               node_err_bounds, target_val)
            
            if (ind is not None and next_p is not None):
                # Check if the newly proposed point is very close to
                # an existing one.
                if (ru.get_min_distance(next_p, node_pos) >
                    l_settings.min_dist):
                    # If not, compute bumpiness of the newly proposed point.
                    n_bump = ru.get_bump_new_node(l_settings, n, k, node_pos, 
                                                  scaled_node_val, next_p,
                                                  fast_node_index,
                                                  node_err_bounds, target_val)
                else:
                    # If yes, we will simply reevaluate the existing
                    # point (if it can be reevaluated).
                    ind = ru.get_min_distance_index(next_p, node_pos)
                    n_bump = (float('inf') if node_is_fast[ind] 
                              else float('-inf'))
                if (n_bump > bump):
                    # In this case we want to put the new point at the
                    # same location as one of the old points.  Remove
                    # the noisy function evaluation from the data
                    # structures, so that the point can be evaluated
                    # in accurate mode.
                    next_p = node_pos.pop(ind)
                    node_val.pop(ind)
                    node_is_fast.pop(ind)
                    all_node_pos.pop(all_node_pos_size_at_restart + ind)
                    all_node_val.pop(all_node_pos_size_at_restart + ind)
                    all_node_is_fast.pop(all_node_pos_size_at_restart + ind)
                    # We must update k here to make sure it is consistent
                    # until the start of the next iteration.
                    k = len(node_pos)
                                                     
        # If the optimization failed or the point is too close to
        # current nodes, discard it. Otherwise, add it to the list.
        if ((next_p is None) or 
            (ru.get_min_distance(next_p, node_pos) <= l_settings.min_dist)):
            current_step = (current_step+1) % cycle_length
            num_cons_ls = 0
            print('Iteration {:3d}'.format(itercount) + ' Discarded',
                  file = output_stream)
            output_stream.flush()
        else:
            min_dist = ru.get_min_distance(next_p, node_pos)
            # Transform back to original space if necessary
            next_p_orig = ru.transform_domain(l_settings, var_lower,
                                              var_upper, next_p, True)
            # Evaluate the new point, in accurate mode or fast mode
            if (current_mode == 'accurate'):
                next_val = objfun(next_p_orig)
                evalcount += 1
                node_is_fast.append(False)
            else: 
                next_val = objfun_fast(next_p_orig)
                fast_evalcount += 1
                # Check if the point could be optimal in accurate
                # mode. In that case, perform an accurate evaluation
                # immediately. Otherwise, add to the list of fast
                # evaluations.
                if ((next_val + 
                     ru.get_fast_error_bounds(l_settings, next_val)[0]) <=
                    (l_settings.target_objval + 
                     l_settings.eps_opt*abs(l_settings.target_objval))):
                    next_val = objfun(next_p_orig)
                    evalcount += 1
                    node_is_fast.append(False)
                else:
                    node_is_fast.append(True)

            # Add to the lists
            node_pos.append(next_p)
            node_val.append(next_val)
            all_node_pos.append(next_p_orig)
            all_node_val.append(next_val)
            all_node_is_fast.append(node_is_fast[-1])

            if ((current_step == local_search_step) and
                (next_val <= fmin - l_settings.eps_impr*max(1.0,abs(fmin))) and
                (num_cons_ls < l_settings.max_consecutive_local_searches - 1)):
                # Keep doing local search
                num_cons_ls += 1
            else:
                current_step = (current_step+1) % cycle_length
                num_cons_ls = 0
                        
            # Update fmin
            if (next_val < fmin):
                fmin_index = k
                fmin = next_val
                is_best_fast = node_is_fast[-1]
            fmax = max(fmax, next_val)
            # Shift due to fast function evaluation
            gap_shift = (ru.get_fast_error_bounds(l_settings, next_val)[1]
                         if is_best_fast else 0.0)
            gap = min(gap, (next_val + gap_shift - l_settings.target_objval) /
                      gap_den)

            print('Iteration {:3d}'.format(itercount) + 
                  ' {:16s}'.format(iteration_id) +
                  ': objval{:s}'.format('~' if node_is_fast[-1] else ' ') +
                  ' {:16.6f}'.format(next_val) +
                  ' min_dist {:9.4f}'.format(min_dist) +
                  ' gap {:8.2f}'.format(gap*100),
                  file = output_stream)
            output_stream.flush()

        # Update iteration number
        itercount += 1

        # At the beginning of each loop of the cyclic optimization
        # strategy, check if the main loop is stalling
        if (current_step == inf_step or
            ((not l_settings.do_infstep) and current_step == inf_step + 1)):
            if (fmin <= (fmin_cycle_start- l_settings.max_stalled_objfun_impr *
                         max(1.0, abs(fmin_cycle_start)))):
                num_stalled_cycles = 0
                fmin_cycle_start = fmin
            else:
                num_stalled_cycles += 1

        # Check if we should restart. We only restart if the initial
        # sampling strategy is random, otherwise it makes little sense.
        if (num_stalled_cycles >= l_settings.max_stalled_cycles and
            evalcount + n + 1 < l_settings.max_evaluations and
            l_settings.init_strategy != 'all_corners' and
            l_settings.init_strategy != 'lower_corners'):
            print('Restart at iteration {:3d}'.format(itercount),
                  file = output_stream)
            output_stream.flush()
            # We update the number of fast restarts here, so that if
            # we hit the limit on fast restarts, we can evaluate
            # points in accurate mode after restarting (even if
            # current_mode is updated in a subsequent block of code)
            num_fast_restarts += (1 if current_mode == 'fast' else 0.0)
            # Store the current number of nodes
            all_node_pos_size_at_restart = len(all_node_pos)
            # Compute a new set of starting points
            node_pos = ru.initialize_nodes(l_settings, var_lower, var_upper, 
                                           integer_vars)
            if (current_mode == 'accurate' or
                num_fast_restarts > l_settings.max_fast_restarts or
                fast_evalcount + n + 1 >= l_settings.max_fast_evaluations):
                node_val = [objfun(point) for point in node_pos]
                evalcount += len(node_val)
            else:
                node_val = [objfun_fast(point) for point in node_pos]
                fast_evalcount += len(node_val)
            node_is_fast = [current_mode == 'fast' for val in node_val]
            # Print the initialization points
            for (i, val) in enumerate(node_val):
                min_dist = ru.get_min_distance(node_pos[i], node_pos[:i] + 
                                               node_pos[(i+1):])
                print('Iteration {:3d}'.format(itercount) + 
                      ' {:16s}'.format('Initialization') +
                      ': objval{:s}'.format('~' if node_is_fast[i] else ' ') +
                      ' {:16.6f}'.format(val) +
                      ' min_dist {:9.4f}'.format(min_dist) +
                      ' gap {:8.2f}'.format(gap*100),
                      file = output_stream)
            all_node_pos.extend(node_pos)
            all_node_val.extend(node_val)
            all_node_is_fast.extend(node_is_fast)
            # Rescale the domain of the function
            node_pos = [ru.transform_domain(l_settings, var_lower,
                                            var_upper, point)
                        for point in node_pos]            
            (l_lower, l_upper) = ru.transform_domain_bounds(l_settings,
                                                            var_lower,
                                                            var_upper)
            # Update all counters and values to restart properly
            fmin_index = node_val.index(min(node_val))
            fmin = node_val[fmin_index]
            fmax = max(node_val)
            fmin_cycle_start = fmin
            num_stalled_cycles = 0

        # Check if we should switch to the second phase of two-phase
        # optimization. The conditions for switching are:
        # 1) Optimization in fast mode restarted too many times.
        # 2) We reached the limit of fast mode iterations.
        if ((two_phase_optimization == True) and (current_mode == 'fast') and
            ((num_fast_restarts > l_settings.max_fast_restarts) or
             (itercount >= l_settings.max_fast_iterations) or
             (fast_evalcount >= l_settings.max_fast_evaluations))):
            print('Switching to accurate mode ' +
                  'at iteration {:3d}'.format(itercount),
                  file = output_stream)
            output_stream.flush()
            current_mode = 'accurate'            
            
    # -- end while

    # Find best point and return it
    i = all_node_val.index(min(all_node_val))
    fmin = all_node_val[i]
    gap_shift = (ru.get_fast_error_bounds(l_settings, fmin)[1]
                   if all_node_is_fast[i] else 0.0)
    gap = ((fmin + gap_shift - l_settings.target_objval) / gap_den)

    print('Summary: iterations {:3d}'.format(itercount) + 
          ' evaluations {:3d}'.format(evalcount) + 
          ' fast_evals {:3d}'.format(fast_evalcount) + 
          ' clock time {:7.2f}'.format(time.time() - start_time) + 
          ' objval{:s}'.format('~' if (all_node_is_fast[i]) else ' ') +
          ' {:15.6f}'.format(fmin) + 
          ' gap {:6.2f}'.format(100*gap),
          file = output_stream)
    output_stream.flush()

    return (all_node_val[i], all_node_pos[i],
            itercount, evalcount, fast_evalcount)
