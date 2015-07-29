"""Settings for the RBF method.

This module contains the settings of the main optimization algorithm.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2015.
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
        optimization phase. Default 1.

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
    
    model_selection_method : string
        Method to compute leave-one-out errors in cross validation for
        model selection. Choice of 'clp', 'cplex', 'numpy'. Default
        'numpy'.
    
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
    _allowed_model_selection_method : Dict[str]
        Allowed model selection method.
    """

    # Allowed values for multiple choice options
    _allowed_rbf = {'auto', 'cubic', 'thin_plate_spline', 'linear',
                   'multiquadric'}
    _allowed_init_strategy = {'all_corners', 'lower_corners', 'rand_corners',
                             'lhd_maximin', 'lhd_corr'}
    _allowed_function_scaling = {'off', 'affine', 'log', 'auto'}
    _allowed_domain_scaling = {'off', 'affine', 'auto'}
    _allowed_dynamism_clipping = {'off', 'median', 'clip_at_dyn', 'auto'}
    _allowed_model_selection_method = {'clp', 'cplex', 'numpy'}

    def __init__(self,
                 rbf = 'thin_plate_spline',
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
                 max_consecutive_local_searches = 1,
                 init_strategy = 'lhd_maximin',
                 function_scaling = 'auto',
                 log_scaling_threshold = 1.0e6,
                 domain_scaling = 'off',
                 dynamism_clipping = 'auto',
                 dynamism_threshold = 1.0e3,
                 local_search_box_scaling = 0.5,
                 max_stalled_cycles = 6,
                 max_stalled_objfun_impr = 0.05,
                 fast_objfun_rel_error = 0.0,
                 fast_objfun_abs_error = 0.0,
                 max_fast_restarts = 2,
                 max_fast_iterations = 100,
                 model_selection_method = 'numpy',
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
        self.log_scaling_threshold = log_scaling_threshold
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
        self.model_selection_method = model_selection_method
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
        if (self.model_selection_method not in 
            RbfSettings._allowed_model_selection_method):
            raise ValueError('settings.model_selection_method = ' + 
                             str(self.model_selection_method) + 
                             ' not supported')
    # -- end function

    @classmethod
    def from_dictionary(cls, args):
        """Construct settings from dictionary containing parameter values.
    
        Construct an instance of RbfSettings by looking up the value
        of all the parameters from a given dictionary. The dictionary
        must contain only parameter values in the form args['name'] =
        value.  anything else present in the dictionary will raise an
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
            l_settings.rbf = 'thin_plate_spline'

        if (l_settings.function_scaling == 'auto'):
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
