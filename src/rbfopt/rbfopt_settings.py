"""Settings for RBFOpt.

This module contains the settings of the main optimization algorithm.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2015.
(C) Copyright International Business Machines Corporation 2016.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import copy
import math
import numpy as np

class RbfoptSettings:
    """Global and algorithmic settings for RBF method.

    Class containing algorithmic settings for the enhanced RBF method,
    as well as global settings such as tolerances, limits and so on.

    *NOTICE*: The docstring for the parameters below is used to build
    the help in the command-line interface. It is therefore very
    important that it is kept tidy in the correct numpy docstring
    format.

    Parameters
    ----------

    max_iterations : int
        Maximum number of iterations. Default 1000.

    max_evaluations : int
        Maximum number of function evaluations in accurate mode. 
        Default 300.

    max_noisy_evaluations : int
        Maximum number of function evaluations in noisy mode.
        Default 200.

    max_cycles : int
        Maximum number of optimization cycles. Default 1000.

    max_clock_time : float
        Maximum wall clock time in seconds. Default 1.0e30.

    algorithm : string
        Optimization algorithm used. Choice of 'Gutmann' and 'MSRSM',
        see References Gutmann (2001) and Regis and Shoemaker
        (2007). Default 'MSRSM'.

    num_cpus : int
        Number of CPUs used. Default 1.

    parallel_wakeup_time : float
        Time (in seconds) after which the main optimization engine
        checks the arrival of results from workers busy with function
        evaluations or other computations. This parameter is only used
        by the parallel optimizer. Default 0.1.

    rbf : string
        Radial basis function used by the method. Choice of 'cubic',
        'thin_plate_spline', 'linear', 'multiquadric', 'gaussian',
        'auto'. In case of 'auto', the type of rbf and the shape
        parameter will be dynamically selected by the
        algorithm. Default 'auto'.

    rbf_shape_parameter : float
        Shape parameter for the radial basis function. Used only by
        the gaussian and multiquadric RBF, this is also known as the
        gamma parameter. If the rbf is 'auto', this will be
        automatically selected from a finite set. Default 0.1.

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
        1.0e-4.

    eps_linear_dependence : float
        Tolerance to determine if a set of columns/rows is linearly
        dependent. Default 1.0e-6.

    min_dist : float
        Minimum Euclidean distance between nodes. A new point will be
        discarded if it is closer than this value from existing
        nodes. This prevents the RBF matrix, which depends on pairwise
        distances, from becoming singular. Default 1.0e-5.

    do_infstep : bool
        Perform a pure global search in every optimization cycle.
        Default False.

    do_local_search : bool
        Perform a pure local search in every optimization cycle. 
        Default True.

    num_global_searches : int
        Number of steps in the global search phase. Default 5.

    init_strategy : string
        Strategy to select initial points. Choice of 'all_corners',
        'lower_corners', 'rand_corners', 'lhd_maximin',
        'lhd_corr'. Default 'lhd_maximin'.

    init_include_midpoint : bool
        Include midpoint of the box among the initialization points. 
        Default True.

    init_sample_fraction : float
        The initial sample size is set to n + 1 times this number,
        with some adjustment for parallel optimization based on
        init_sample_increase_parallel. If set to -1 (or any negative
        number), the size of the initial sample set will be determined
        automatically. Default -1.

    init_sample_increase_parallel : float
        Fraction of increase of the number of initial sample points in
        order to reduce synchronization efforst in asynchronous
        parallel evaluation. The number of total initialization points
        is increase by a factor (1 + num_cpus *
        init_sample_increase_parallel), and optimization starts when
        the originally targetd number of samples is reached. Default
        0.05.

    max_random_init : int
        Maximum number of trials for the random initialization
        strategies, in case they generate a linearly dependent set of
        samples. After this number of trials, the initialization
        algorithm will bail out. Default 50.

    function_scaling : string
        Rescaling method for the function values. Choice of 'off',
        'affine', 'log', 'auto'. Default 'auto'.

    log_scaling_threshold : float
        Minimum value for the difference between median and minimum
        function value before a log scaling of the function values is
        applied in the 'auto' setting. Default 1.0e6.

    domain_scaling : string
        Rescaling method for the domain. Choice of 'off', 'affine',
        'auto'. Default 'auto'.

    dynamism_clipping : string
        Dynamism clipping strategy. Choice of 'off', 'median',
        'clip_at_dyn', 'auto'. Default 'auto'.

    dynamism_threshold : float
        Minimum value of the ratio between the largest and the
        smallest absolute function values before the dynamism clipping
        strategy is applied. Default 1.0e3.

    local_search_threshold : float
        Threshold used to determines what is a local search. If the
        scaling factor used in the computation of f_n^* is less than
        this value, it is assumed that the search is a local search.
        Default 0.25.

    local_search_box_scaling : float
        Rescaling factor for the hyperbox used for local search. See
        parameter nu of Regis and Shoemaker (2007). Default 0.5.

    max_stalled_iterations : int
        Maximum number of iterations without improvement before we
        perform a full restart. Default 100.

    discarded_window_size : int
        Number of consecutive iterations that are considered to
        determine if a restart should be triggered, based on too many
        discarded points. This number is multiplied by the number of
        cpus to determine the actual rolling window size. Default 30.

    max_fraction_discarded : float
        Maximum fraction of discarded points within the last
        discarded_window_size*num_cpus iterations before a restart is
        triggered. Default 0.5.

    max_consecutive_restoration : int
        Maximum number of consecutive nonsingularity restoration
        phases before the algorithm fails. Default 15.

    max_cross_validations : int
        Maximum number of cross validations before we trust our
        previous results. Default 50.
        
    max_noisy_restarts : int
        Maximum number of restarts in noisy mode before we switch
        to accurate mode. Default 2.
    
    max_noisy_iterations : int
        Maximum number of iterations in noisy mode before switching
        to accurate mode. Default 200.
    
    targetval_clipping : bool
        Clip target value selection based on periodically eliminating
        some of the largest function values, as proposed by Gutmann
        (2001) and later Regis and Shoemaker (2007). Used by Gutmann
        RBF method only. Default True.

    global_search_method : string
        The methodology to be used in the solution of global search
        problems, i.e. the infstep and the global step. The options
        are'genetic', 'sampling' and 'solver'. If 'genetic', a
        heuristic based on a genetic algorithm is used. If 'sampling',
        random sampling is used. If 'solver', the available solvers
        are used to try to solve mathematical programming
        models. Default 'genetic'.

    ga_base_population_size : int
        Minimum population size for the genetic algorithm used to
        optimize the global search step or infstep, when the genetic
        global search method is chosen. The final population is
        computed as the minimum population + n/5, where n is the
        number of decision variables. Default 400.

    ga_num_generations : int
        Number of generations for the genetic algorithm used to
        optimize the global search step or infstep, when the genetic
        global search method is chosen. Default 20.

    num_samples_aux_problems : int
        Multiplier for the dimension of the problem to determine the
        number of samples used by the Metric SRSM traditional
        algorithm at every iteration. Default 1000.

    modified_msrsm_score : bool
        Use the modified MSRSM score function in which the objective
        function value contribution always has a weight of 1, instead
        of 1 - distance_weight. This setting is more aggressive in
        improving the objective function value, compared to the
        original MSRSM score function. Default True.

    max_consecutive_refinement : int
        Maximum number of consecutive refinement steps. Default 5.

    thresh_unlimited_refinement : float
        Lower threshold for the amount of search budget depleted,
        after which the maximum limit on consecutive refinement is
        ignored. The search budget here is in terms of number of
        iterations, number of evaluations, wall clock time. Default
        0.9.

    thresh_unlimited_refinement_stalled : float
        Lower threshold for the percentage of stalled iterations,
        relative to the maximum number of stalled iterations that will
        trigger a restart, after which unlimited consecutive
        refinement are allowed. Default 0.9.

    refinement_frequency : int
        In serial search mode, this indicates the number of full
        global search cycles after which the refinement step can be
        performed (in case a better solution has been found in the
        meantime). In parallel mode, this determines the maximum
        acceptable ratio between other search steps and refinement 
        steps. Default 3.

    ref_num_integer_candidates : int
        Number of integer candidates per dimension of the problem that
        are considered when rounding the (fractional) point computed
        during the refinement step. Default 10.

    ref_acceptable_decrease_shrink : float
        Maximum ratio between real decrease and refinement model
        decrease for which the radius of the local search gets
        shrunk. Default 0.2.

    ref_acceptable_decrease_enlarge : float
        Minimum ratio between real decrease and refinement model
        decrease for which the radius of the local search gets
        enlarged. Default 0.6.

    ref_acceptable_decrease_move : float
        Minimum ratio between real decrease and refinement model
        decrease for which the new candidate point is accepted as the
        new iterate. Default 0.1.

    ref_min_radius : float
        Minimum radius of the local search for the refinement
        step. Default 1.0e-3.

    ref_init_radius_multiplier : float
        Exponent (with base 2) of the multiplier used to determine the
        minimum initial radius of the local serach for the refinement
        step. Default 2.0.

    ref_min_grad_norm : float
        Minimum norm of the gradient for the local search method in
        the refinement step, before we assume that we converged to a
        stationary point. Default 1.0e-2.

    save_state_interval : int 
        Number of iterations after which the state of the algorithm
        should be dumped to file. The algorithm can be resumed from a
        saved state. It can be useful in case something goes
        wrong. Default 100000.

    save_state_file : string
        Name of the file in which the state of the algorithm will be
        saved at regular intervals, see save_state_interval. Default
        'rbfopt_algorithm_state.dat'.
    
    print_solver_output : bool
        Print the output of the solvers to screen? Note that this
        cannot be redirected to file so it will go to
        stdout. Default False.

    minlp_solver_path : string
        Full path to the MINLP solver executable, i.e., bonmin. If
        only the name solver is specified, it is assumed that the
        solver is part of your system path and can be called from
        anywhere. Default 'bonmin'.

    nlp_solver_path : string
        Full path to the NLP solver executable, i.e., ipopt. If
        only the name solver is specified, it is assumed that the
        solver is part of your system path and can be called from
        anywhere. Default 'ipopt'.

    debug : bool
        Print debug output. Internal error messages are typically
        printed to stderr, Pyomo error messages are determined by its
        logger. If False, all warnings and error messages are
        suppressed. Default False.

    rand_seed : int
        Seed for the random number generator. The maximum number
        supported by numpy on all platforms is 2^32. Default
        937627691.

    Attributes
    ----------

    _allowed_rbf : Dict[string]
        Allowed types of RBF functions.
    _allowed_init_strategy : Dict[string]
        Allowed initialization strategies.
    _allowed_function_scaling : Dict[string]
        Allowed function scaling strategies.
    _allowed_domain_scaling : Dict[string]
        Allowed domain scaling strategies.
    _allowed_dynamism_clipping : Dict[string]
        Allowed dynamism clipping strategies.
    _allowed_algorithm : Dict[string]
        Allowed algorithms.
    _allowed_global_search_method : Dict[string]
        Allowed global search methods.

    """

    # Allowed values for multiple choice options
    _allowed_rbf = {'auto', 'cubic', 'thin_plate_spline', 'linear',
                    'multiquadric', 'gaussian'}
    _allowed_init_strategy = {'all_corners', 'lower_corners', 'rand_corners',
                             'lhd_maximin', 'lhd_corr'}
    _allowed_function_scaling = {'off', 'affine', 'log', 'auto'}
    _allowed_domain_scaling = {'off', 'affine', 'auto'}
    _allowed_dynamism_clipping = {'off', 'median', 'clip_at_dyn', 'auto'}
    _allowed_algorithm = {'Gutmann', 'MSRSM'}
    _allowed_global_search_method = {'genetic', 'sampling', 'solver'}
    # Parameters that are only allowed to be nonnegative
    _nonnegative_parameters = [
        'max_iterations',
        'max_evaluations',
        'max_noisy_evaluations',
        'max_cycles',
        'max_clock_time',
        'parallel_wakeup_time',
        'eps_opt',
        'eps_zero',
        'eps_impr',
        'eps_linear_dependence',
        'min_dist',
        'num_global_searches',
        'init_sample_increase_parallel',
        'max_random_init',
        'log_scaling_threshold',
        'dynamism_threshold',
        'local_search_threshold',
        'local_search_box_scaling',
        'max_stalled_iterations',
        'discarded_window_size',
        'max_fraction_discarded',
        'max_consecutive_restoration',
        'max_cross_validations',
        'max_noisy_restarts',
        'max_noisy_iterations',
        'ga_base_population_size',
        'ga_num_generations',
        'num_samples_aux_problems',
        'max_consecutive_refinement',
        'thresh_unlimited_refinement',
        'thresh_unlimited_refinement_stalled',
        'refinement_frequency',
        'ref_num_integer_candidates',
        'ref_acceptable_decrease_shrink',
        'ref_acceptable_decrease_enlarge',
        'ref_acceptable_decrease_move',
        'ref_min_radius',
        'ref_init_radius_multiplier',
        'ref_min_grad_norm',
        'save_state_interval'
    ]

    def __init__(self,
                 max_iterations=1000,
                 max_evaluations=300,
                 max_noisy_evaluations=200,
                 max_cycles=1000,
                 max_clock_time=1.0e30,
                 num_cpus=1,
                 parallel_wakeup_time=0.1,
                 algorithm='MSRSM',
                 rbf='auto',
                 rbf_shape_parameter=0.1,
                 target_objval=-1.0e10,
                 eps_opt=1.0e-2,
                 eps_zero=1.0e-15,
                 eps_impr=1.0e-4,
                 eps_linear_dependence=1.0e-6,
                 min_dist=1.0e-5,
                 do_infstep=False,
                 do_local_search=True,
                 num_global_searches=5,
                 init_strategy='lhd_maximin',
                 init_include_midpoint=True,
                 init_sample_fraction=-1.0,
                 init_sample_increase_parallel=0.05,
                 max_random_init=50,
                 function_scaling='auto',
                 log_scaling_threshold=1.0e6,
                 domain_scaling='auto',
                 dynamism_clipping='auto',
                 dynamism_threshold=1.0e3,
                 local_search_threshold=0.25,
                 local_search_box_scaling=0.5,
                 max_stalled_iterations=100,
                 discarded_window_size=30,
                 max_fraction_discarded=0.5,
                 max_consecutive_restoration=15,
                 max_cross_validations=50,
                 max_noisy_restarts=2,
                 max_noisy_iterations=200,
                 targetval_clipping=True,
                 global_search_method='genetic',
                 ga_base_population_size=400,
                 ga_num_generations=20,
                 num_samples_aux_problems=1000,
                 modified_msrsm_score=True,
                 max_consecutive_refinement=5,
                 thresh_unlimited_refinement=0.9,
                 thresh_unlimited_refinement_stalled=0.9,
                 refinement_frequency=3,
                 ref_num_integer_candidates=10,
                 ref_acceptable_decrease_shrink=0.2,
                 ref_acceptable_decrease_enlarge=0.6,
                 ref_acceptable_decrease_move=0.1,
                 ref_min_radius=1.0e-3,
                 ref_init_radius_multiplier=2.0,
                 ref_min_grad_norm=1.0e-2,
                 print_solver_output=False,
                 save_state_interval=100000,
                 save_state_file='rbfopt_algorithm_state.dat',
                 minlp_solver_path='bonmin',
                 nlp_solver_path='ipopt',
                 debug=False,
                 rand_seed=937627691):
        """Class constructor with default values. 
        """
        self.max_iterations = max_iterations
        self.max_evaluations = max_evaluations
        self.max_noisy_evaluations = max_noisy_evaluations
        self.max_cycles = max_cycles
        self.max_clock_time = max_clock_time
        self.num_cpus = num_cpus
        self.parallel_wakeup_time = parallel_wakeup_time
        self.rbf = rbf
        self.rbf_shape_parameter = rbf_shape_parameter
        self.target_objval = target_objval
        self.eps_opt = eps_opt
        self.eps_zero = eps_zero
        self.eps_impr = eps_impr
        self.eps_linear_dependence = eps_linear_dependence
        self.min_dist = min_dist
        self.do_infstep = do_infstep
        self.do_local_search = do_local_search
        self.num_global_searches = num_global_searches
        self.init_strategy = init_strategy
        self.init_include_midpoint = init_include_midpoint
        self.init_sample_fraction = init_sample_fraction
        self.init_sample_increase_parallel = init_sample_increase_parallel
        self.max_random_init = max_random_init
        self.function_scaling = function_scaling
        self.log_scaling_threshold = log_scaling_threshold
        self.domain_scaling = domain_scaling
        self.dynamism_clipping = dynamism_clipping
        self.dynamism_threshold = dynamism_threshold
        self.local_search_threshold = local_search_threshold
        self.local_search_box_scaling = local_search_box_scaling
        self.max_stalled_iterations = max_stalled_iterations
        self.discarded_window_size = discarded_window_size
        self.max_fraction_discarded = max_fraction_discarded
        self.max_consecutive_restoration = max_consecutive_restoration
        self.max_cross_validations = max_cross_validations
        self.max_noisy_restarts = max_noisy_restarts
        self.max_noisy_iterations = max_noisy_iterations
        self.algorithm = algorithm
        self.targetval_clipping = targetval_clipping
        self.global_search_method = global_search_method
        self.ga_base_population_size = ga_base_population_size
        self.ga_num_generations = ga_num_generations
        self.num_samples_aux_problems = num_samples_aux_problems
        self.modified_msrsm_score = modified_msrsm_score
        self.max_consecutive_refinement = max_consecutive_refinement
        self.thresh_unlimited_refinement = thresh_unlimited_refinement
        self.thresh_unlimited_refinement_stalled = thresh_unlimited_refinement_stalled
        self.refinement_frequency = refinement_frequency
        self.ref_num_integer_candidates = ref_num_integer_candidates
        self.ref_acceptable_decrease_shrink = ref_acceptable_decrease_shrink
        self.ref_acceptable_decrease_enlarge = ref_acceptable_decrease_enlarge
        self.ref_acceptable_decrease_move = ref_acceptable_decrease_move
        self.ref_min_radius = ref_min_radius
        self.ref_init_radius_multiplier = ref_init_radius_multiplier
        self.ref_min_grad_norm = ref_min_grad_norm
        self.print_solver_output = print_solver_output
        self.save_state_interval = save_state_interval
        self.save_state_file = save_state_file
        self.minlp_solver_path = minlp_solver_path
        self.nlp_solver_path = nlp_solver_path
        self.debug = debug
        self.rand_seed = rand_seed

        if (self.rbf not in RbfoptSettings._allowed_rbf):
            raise ValueError('settings.rbf = ' + 
                             str(self.rbf) + ' not supported')
        if (self.init_strategy not in RbfoptSettings._allowed_init_strategy):
            raise ValueError('settings.init_strategy = ' + 
                             str(self.init_strategy) + ' not supported')
        if (self.function_scaling not in 
            RbfoptSettings._allowed_function_scaling):
            raise ValueError('settings.function_scaling = ' + 
                             str(self.function_scaling) + ' not supported')
        if (self.domain_scaling not in RbfoptSettings._allowed_domain_scaling):
            raise ValueError('settings.domain_scaling = ' + 
                             str(self.domain_scaling) + ' not supported')
        if (self.dynamism_clipping not in 
            RbfoptSettings._allowed_dynamism_clipping):
            raise ValueError('settings.dynamism_clipping = ' + 
                             str(self.dynamism_clipping) + ' not supported')
        if (self.algorithm not in RbfoptSettings._allowed_algorithm):
            raise ValueError('settings.algorithm = ' + 
                             str(self.algorithm) + ' not supported')
        if (self.global_search_method not in 
            RbfoptSettings._allowed_global_search_method):
            raise ValueError('settings.global_search_method = ' + 
                             str(self.global_search_method) + 
                             ' not supported')
        attrs = vars(self)
        for param in RbfoptSettings._nonnegative_parameters:
            if (attrs[param] < 0):
                raise ValueError('settings.' + param + ' = ' +
                                 str(attrs[param]) + ' not supported')        
    # -- end function

    @classmethod
    def from_dictionary(cls, args):
        """Construct settings from dictionary containing parameter values.
    
        Construct an instance of RbfoptSettings by looking up the value
        of the parameters from a given dictionary. The dictionary must
        contain only parameter values in the form args['name'] =
        value. Anything else present in the dictionary will raise an
        exception.

        Parameters
        ----------

        args : Dict[string]
            A dictionary containing the values of the parameters in a
            format args['name'] = value. 

        Returns
        -------
        RbfoptSettings
            An instance of the object of the class.

        Raises
        ------
        ValueError
            If the dictionary contains invalid parameters.
        """
        valid_params = vars(cls())
        for param in args.keys():
            if param not in valid_params:
                raise ValueError('Parameter name {:s}'.format(param) +
                                 'not recognized')

        return cls(**args)

    def set_auto_parameters(self, dimension, var_lower, var_upper,
                            integer_vars):
        """Determine the value for 'auto' parameters.

        Create a copy of the settings and assign 'auto' parameters. The
        original settings are untouched.

        Parameters
        ----------

        dimension : int
            The dimension of the problem, i.e. size of the space.

        var_lower : 1D numpy.ndarray[float]
            Vector of variable lower bounds.

        var_upper : 1D numpy.ndarray[float]
            Vector of variable upper bounds.

        integer_vars : 1D numpy.ndarray[int]
            A list containing the indices of the integrality constrained 
            variables. If empty list, all variables are assumed to be 
            continuous.

        Returns
        -------
        RbfoptSettings
            A copy of the settings, without any 'auto' parameter values.
        """
        assert(isinstance(var_lower, np.ndarray))
        assert(isinstance(var_upper, np.ndarray))
        assert(isinstance(integer_vars, np.ndarray))
        assert(dimension == len(var_lower))
        assert(dimension == len(var_upper))
        assert((len(integer_vars)==0) or (np.max(integer_vars) < dimension))

        l_settings = copy.deepcopy(self)

        if (l_settings.rbf == 'auto'):
            l_settings.rbf = 'thin_plate_spline'

        if (l_settings.function_scaling == 'auto'):
            l_settings.function_scaling = 'off'
            
        if (l_settings.dynamism_clipping == 'auto'):
            l_settings.dynamism_clipping = 'median'
                
        if (l_settings.domain_scaling == 'auto'):
            if (len(integer_vars)):
                l_settings.domain_scaling = 'off'
            else:
                # Compute the length of the domain of each variable
                size = var_upper - var_lower
                size.sort()
                # If the problem is badly scaled, i.e. a variable has
                # a domain 5 times as large as anoether, rescale.
                if (size[-1] >= 5*size[0]):
                    l_settings.domain_scaling = 'affine'
                else:
                    l_settings.domain_scaling = 'off'

        if (l_settings.init_sample_fraction < 0):
            if (l_settings.num_cpus <= 2):
                l_settings.init_sample_fraction = (0.5 if dimension <= 20
                                                   else 0.4)
            else:
                if (dimension <= 20):
                    l_settings.init_sample_fraction = 1.0
                elif (dimension <= 50):
                    l_settings.init_sample_fraction = 0.75
                else:
                    l_settings.init_sample_fraction = 0.5                    

        return l_settings

    # - end function
        
    def print(self, output_stream):
        """Print the value of all settings.

        Prints the settings to the output stream, on a very long line.

        Parameters
        ----------

        output_stream : file
            The stream on which messages are printed.
        """
        print('RbfoptSettings:', file = output_stream)
        attrs = vars(self)
        print(', '.join('{:s}: {:s}'.format(str(item[0]), str(item[1])) 
                        for item in sorted(attrs.items())),
              file = output_stream)
        print(file = output_stream)
        output_stream.flush()

# -- end of class RbfoptSettings
