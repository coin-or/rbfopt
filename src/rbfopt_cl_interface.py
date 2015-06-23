"""Command-line interface for RBFOpt.

This module provides a basic command-line interface for RBFOpt,
allowing the user to launch the optimization process and set the
algorithmic options using a standard UNIX sintax.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
Research partially supported by SUTD-MIT International Design Center.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import argparse
import rbfopt
import black_box as bb
from rbfopt_settings import RbfSettings


def register_options(parser):
    """Add options to the command line parser.

    Register all the options for the optimization algorithm.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser.

    See also
    --------   
    :class:`rbfopt_settings.RbfSettings` for a detailed description of
    all the command line options.
    """
    # Get default values from here
    default = RbfSettings()
    attrs = vars(default)
    docstring = default.__doc__
    param_docstring = docstring[docstring.find('Parameters'):
                                docstring.find('Attributes')].split(' : ')
    param_name = [val.split(' ')[-1].strip() for val in param_docstring[:-1]]
    param_type = [val.split('\n')[0].strip() for val in param_docstring[1:]]
    param_help = [' '.join(line.strip() for line in val.split('\n')[1:-2])
                  for val in param_docstring[1:]]
    # We extract the default from the docstring in case it is
    # necessary, but we use the actual default from the object above.
    param_default = [val.split(' ')[-1].rstrip('.').strip('\'') 
                     for val in param_help]
    for i in range(len(param_name)):
        if (param_type[i] == 'float'):
            type_fun = float
        elif (param_type[i] == 'int'):
            type_fun = int
        elif (param_type[i] == 'bool'):
            type_fun = bool
        else:
            type_fun = str
        parser.add_argument('--' + param_name[i], action = 'store',
                            dest = param_name[i],
                            type = type_fun,
                            help = param_help[i],
                            default = getattr(default, param_name[i]))
    parser.add_argument('--log', '-o', action = 'store',
                        metavar = 'LOG_FILE_NAME', dest = 'output_stream',
                        help = 'name of log file for output redirection')

# -- end function

def rbfopt_cl_interface(args, black_box):
    """Command-line interface.
    
    Optimize the specified objective function using the algorithmic
    options given on the command line.

    Parameters
    ----------
    args : Any
        A namespace containing the options, as created by ArgumentParser.

    black_box : black_box.BlackBox
        An object containing the function to be optimized and its main
        characteristics. It is possible to pass an object of a
        different class, provided that it as the same public
        attributes.
    """

    assert(hasattr(black_box, 'dimension'))
    assert(hasattr(black_box, 'var_lower'))
    assert(hasattr(black_box, 'var_upper'))
    assert(hasattr(black_box, 'optimum_value'))
    assert(hasattr(black_box, 'integer_vars'))
    assert(hasattr(black_box, 'evaluate'))
    assert(hasattr(black_box, 'evaluate_fast'))

    # Open output stream if necessary
    if (args.output_stream is None):
        output_stream = sys.stdout
    else:
        try:
            output_stream = open(args.output_stream, 'w')
        except IOError as e:
            print('Exception in opening log file', file = sys.stderr)
            print(e, file = sys.stderr)

    settings = RbfSettings(target_objval = black_box.optimum_value,
                           eps_opt = args.eps_opt,
                           max_iterations = args.max_iterations,
                           max_evaluations = args.max_evaluations,
                           max_fast_evaluations = 
                           args.max_fast_evaluations,
                           max_clock_time = args.max_clock_time,
                           do_infstep = args.do_infstep,
                           skip_targetval_clipping = 
                           args.skip_targetval_clipping,
                           num_global_searches = 
                           args.num_global_searches,
                           max_consecutive_local_searches = 
                           args.max_consecutive_local_searches,
                           rand_seed = args.rand_seed,
                           dynamism_clipping = args.dynamism_clipping,
                           function_scaling = args.function_scaling,
                           domain_scaling = args.domain_scaling,
                           local_search_box_scaling =
                           args.local_search_box_scaling,
                           max_stalled_cycles = 
                           args.max_stalled_cycles,
                           rbf = args.rbf,
                           init_strategy = args.init_strategy,
                           fast_objfun_rel_error = 
                           args.fast_objfun_rel_error,
                           fast_objfun_abs_error = 
                           args.fast_objfun_abs_error,
                           print_solver_output = 
                           args.print_solver_output)
    settings.print(output_stream = output_stream)
    (opt, point, itercount, evalcount,
     fast_evalcount) = rbfopt.rbf_optimize(settings,
                                           black_box.dimension, 
                                           black_box.var_lower,
                                           black_box.var_upper,
                                           black_box.evaluate,
                                           integer_vars = 
                                           black_box.integer_vars,
                                           objfun_fast =
                                           black_box.evaluate_fast,
                                           output_stream = output_stream)
    print('rbf_optimize returned function value {:.15f}'.format(opt),
          file = output_stream)
    output_stream.close()

# -- end function

if (__name__ == "__main__"):
    if (sys.version_info[0] >= 3):
        print('Error: Python 3 is currently not supported by PyOmo/Coopr.')
        print('Please use Python 2.7')
        exit()
    # Create command line parsers
    desc = ('Apply the RBF method to the class "BlackBox" in black_box.py.' + 
            '\nSee rbfopt.py for a detailed description of all options.')
    parser = argparse.ArgumentParser(description = desc)
    # Add options to parser and parse arguments
    register_options(parser)
    args = parser.parse_args()
    # Run the interface
    rbfopt_cl_interface(args, bb.BlackBox())
