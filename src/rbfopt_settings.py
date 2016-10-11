"""Settings for the RBF method.

This module contains the settings of the main optimization algorithm.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2015.
(C) Copyright International Business Machines Corporation 2016.
Research partially supported by SUTD-MIT International Design Center.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import copy
import math

class RbfSettings:
    """Global and algorithmic settings for RBF method.

    Class containing algorithmic settings for the enhanced RBF method,
    as well as global settings such as tolerances, limits and so on.

    *NOTICE*: The docstring for the parameters below is used to build
    the help in the command-line interface. It is therefore very
    important that it is kept tidy in the correct numpy docstring
    format.

    Parameters
    ----------

    rbf : str
        Radial basis function used by the method. Choice of 'cubic',
        'thin_plate_spline', 'linear', 'multiquadric', 'auto'. Default
        'thin_plate_spline'.

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

    num_cpus : int
        Number of CPUs used. Default 1.

    parallel_wakeup_time : float
        Time (in seconds) after which the main optimization engine
        checks the arrival of results from workers busy with function
        evaluations or other computations. This parameter is only used
        by the parallel optimizer. Default 0.1.

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
        Perform a pure global search in every optimization
        loop. Default False.

    num_global_searches : int
        Number of steps in the global search phase. Default 5.

    max_consecutive_local_searches : int
        Maximum number of consecutive local searches during the
        optimization phase. This parameter is ignored by the parallel
        optimizer. Default 1.

    init_strategy : str
        Strategy to select initial points. Choice of 'all_corners',
        'lower_corners', 'rand_corners', 'lhd_maximin',
        'lhd_corr'. Default 'lhd_maximin'.

    function_scaling : str
        Rescaling method for the function values. Choice of 'off',
        'affine', 'log', 'auto'. Default 'auto'.

    log_scaling_threshold : float
        Minimum value for the difference between median and minimum
        function value before a log scaling of the function values is
        applied in the 'auto' setting. Default 1.0e6.

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

    max_consecutive_discarded : int
        Maximum number of discarded points before a restart is
        triggered. Default 15.

    max_consecutive_restoration : int
        Maximum number of consecutive nonsingularity restoration
        phases before the algorithm fails. Default 15.

    fast_objfun_rel_error : float
        An estimate of the relative error by which the fast version of
        the objective function is affected. Default 0.0.
        
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
    
    model_selection_solver : string
        Solver to compute leave-one-out errors in cross validation for
        model selection. Choice of 'clp', 'cplex', 'numpy'. Default
        'numpy'.

    algorithm : string
        Optimization algorithm used. Choice of 'Gutmann' and 'MSRSM',
        see References Gutmann (2001) and Regis and Shoemaker
        (2007). Default 'MSRSM'.

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

    save_state_interval : int 
        Number of iterations after which the state of the algorithm
        should be dumped to file. The algorithm can be resumed from a
        saved state. It can be useful in case something goes
        wrong. Default 100000.

    save_state_file : string
        Name of the file in which the state of the algorithm will be
        saved at regular intervals, see save_state_interval. Default
        'optalgorithm_state.dat'.
    
    print_solver_output : bool
        Print the output of the solvers to screen? Note that this
        cannot be redirected to file so it will go to
        stdout. Default False.

    rand_seed : int
        Seed for the random number generator. The maximum number
        supported by numpy on all platforms is 2^32. Default
        937627691.

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
    _allowed_model_selection_solver : Dict[str]
        Allowed model selection method.
    _allowed_algorithm : Dict[str]
        Allowed algorithms.
    _allowed_global_search_method : Dict[str]
        Allowed global search methods.

    """

    # Allowed values for multiple choice options
    _allowed_rbf = {'auto', 'cubic', 'thin_plate_spline', 'linear',
                   'multiquadric'}
    _allowed_init_strategy = {'all_corners', 'lower_corners', 'rand_corners',
                             'lhd_maximin', 'lhd_corr'}
    _allowed_function_scaling = {'off', 'affine', 'log', 'auto'}
    _allowed_domain_scaling = {'off', 'affine', 'auto'}
    _allowed_dynamism_clipping = {'off', 'median', 'clip_at_dyn', 'auto'}
    _allowed_model_selection_solver = {'clp', 'cplex', 'numpy'}
    _allowed_algorithm = {'Gutmann', 'MSRSM'}
    _allowed_global_search_method = {'genetic', 'sampling', 'solver'}

    def __init__(self,
                 rbf = 'thin_plate_spline',
                 max_iterations = 150,
                 max_evaluations = 250,
                 max_fast_evaluations = 150,
                 max_clock_time = 1.0e30,
                 num_cpus = 1,
                 parallel_wakeup_time = 0.1,
                 target_objval = -1.0e10,
                 eps_opt = 1.0e-2,
                 eps_zero = 1.0e-15,
                 eps_impr = 1.0e-3,
                 min_dist = 1.0e-5,
                 do_infstep = False,
                 num_global_searches = 5,
                 max_consecutive_local_searches = 1,
                 init_strategy = 'lhd_maximin',
                 function_scaling = 'auto',
                 log_scaling_threshold = 1.0e6,
                 domain_scaling = 'auto',
                 dynamism_clipping = 'auto',
                 dynamism_threshold = 1.0e3,
                 local_search_box_scaling = 0.5,
                 max_stalled_cycles = 6,
                 max_stalled_objfun_impr = 0.05,
                 max_consecutive_discarded = 15,
                 max_consecutive_restoration = 15,
                 fast_objfun_rel_error = 0.0,
                 fast_objfun_abs_error = 0.0,
                 max_fast_restarts = 2,
                 max_fast_iterations = 100,
                 model_selection_solver = 'numpy',
                 algorithm = 'MSRSM',
                 targetval_clipping = True,
                 global_search_method = 'genetic',
                 ga_base_population_size = 400,
                 ga_num_generations = 20,
                 num_samples_aux_problems = 1000,
                 modified_msrsm_score = True,
                 print_solver_output = False,
                 save_state_interval = 100000,
                 save_state_file = 'optalgorithm_state.dat',
                 rand_seed = 937627691):
        """Class constructor with default values. 
        """
        self.rbf = rbf
        self.max_iterations = max_iterations
        self.max_evaluations = max_evaluations
        self.max_fast_evaluations = max_fast_evaluations
        self.max_clock_time = max_clock_time
        self.num_cpus = num_cpus
        self.parallel_wakeup_time = parallel_wakeup_time
        self.target_objval = target_objval
        self.eps_opt = eps_opt
        self.eps_zero = eps_zero
        self.eps_impr = eps_impr
        self.min_dist = min_dist
        self.do_infstep = do_infstep
        self.num_global_searches = num_global_searches
        self.max_consecutive_local_searches = max_consecutive_local_searches
        self.init_strategy = init_strategy
        self.function_scaling = function_scaling
        self.log_scaling_threshold = log_scaling_threshold
        self.domain_scaling = domain_scaling
        self.dynamism_clipping = dynamism_clipping
        self.dynamism_threshold = dynamism_threshold
        self.local_search_box_scaling = local_search_box_scaling
        self.max_stalled_cycles = max_stalled_cycles
        self.max_stalled_objfun_impr = max_stalled_objfun_impr
        self.max_consecutive_discarded = max_consecutive_discarded
        self.max_consecutive_restoration = max_consecutive_restoration
        self.fast_objfun_rel_error = fast_objfun_rel_error
        self.fast_objfun_abs_error = fast_objfun_abs_error
        self.max_fast_restarts = max_fast_restarts
        self.max_fast_iterations = max_fast_iterations
        self.model_selection_solver = model_selection_solver
        self.algorithm = algorithm
        self.targetval_clipping = targetval_clipping
        self.global_search_method = global_search_method
        self.ga_base_population_size = ga_base_population_size
        self.ga_num_generations = ga_num_generations
        self.num_samples_aux_problems = num_samples_aux_problems
        self.modified_msrsm_score = modified_msrsm_score
        self.print_solver_output = print_solver_output
        self.save_state_interval = save_state_interval
        self.save_state_file = save_state_file
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
        if (self.model_selection_solver not in 
            RbfSettings._allowed_model_selection_solver):
            raise ValueError('settings.model_selection_solver = ' + 
                             str(self.model_selection_solver) + 
                             ' not supported')
        if (self.algorithm not in RbfSettings._allowed_algorithm):
            raise ValueError('settings.algorithm = ' + 
                             str(self.algorithm) + ' not supported')
        if (self.global_search_method not in 
            RbfSettings._allowed_global_search_method):
            raise ValueError('settings.global_search_method = ' + 
                             str(self.global_search_method) + 
                             ' not supported')
    # -- end function

    @classmethod
    def from_dictionary(cls, args):
        """Construct settings from dictionary containing parameter values.
    
        Construct an instance of RbfSettings by looking up the value
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
        RbfSettings
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

        var_lower : List[float]
            Vector of variable lower bounds.

        var_upper : List[float]
            Vector of variable upper bounds.

        integer_vars : List[int]
            A list containing the indices of the integrality
            constrained variables. If empty list, all
            variables are assumed to be continuous.

        Returns
        -------
        RbfSettings
            A copy of the settings, without any 'auto' parameter values.
        """
        assert(dimension==len(var_lower))
        assert(dimension==len(var_upper))
        assert((not integer_vars) or (max(integer_vars) < dimension))

        l_settings = copy.deepcopy(self)

        if (l_settings.rbf == 'auto'):
            l_settings.rbf = 'thin_plate_spline'

        if (l_settings.function_scaling == 'auto'):
            l_settings.function_scaling = 'off'
            
        if (l_settings.dynamism_clipping == 'auto'):
            l_settings.dynamism_clipping = 'median'
                
        if (l_settings.domain_scaling == 'auto'):
            if (integer_vars):
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

    # - end function
        
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
