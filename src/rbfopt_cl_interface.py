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

import rbfopt
import sys
import argparse
import black_box as bb


def register_options(parser):
    """Add options to the command line parser.

    Register all the options for the optimization algorithm.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser.

    See also
    --------
    :class:`rbfopt.RbfSettings` for a detailed description of all the
    command line options.
    """
    parser.add_argument('--rbf', '-f', action = 'store', 
                        dest = 'rbf', default = 'cubic',
                        help = 'type of radial basis function')
    parser.add_argument('--eps-opt', '-opt', action = 'store',
                        dest = 'eps_opt', 
                        default = 1.0e-2, type = float,
                        help = 'optimality threshold (w.r.t. optimum)')
    parser.add_argument('--max-iter', '-n', action = 'store',
                        dest = 'max_iterations', default = 150, type = int,
                        help = 'maximum number of iterations')
    parser.add_argument('--max-eval', '-e', action = 'store',
                        dest = 'max_evaluations', default = 250, type = int,
                        help = 'maximum number of function evaluations')
    parser.add_argument('--max-fast-eval', '-fe', action = 'store',
                        dest = 'max_fast_evaluations', default = 150, 
                        type = int,
                        help = 'maximum number of fast function evaluations')
    parser.add_argument('--max-time', '-t', action = 'store',
                        dest = 'max_clock_time', 
                        default = 1.0e30, type = float,
                        help = 'maximum wall-clock time')
    parser.add_argument('--infstep', '-p', action = 'store_true',
                        dest = 'do_infstep', default = False,
                        help = 'perform infstep')
    parser.add_argument('--skip-tval-clipping', '-a', action = 'store_true',
                        dest = 'skip_targetval_clipping', default = False,
                        help = 'skip the Gutmann method to clip target value')
    parser.add_argument('--num-glob-search', '-k', action = 'store',
                        dest = 'num_global_searches', default = 5, type = int,
                        help = 'number of global searches in each cycle')
    parser.add_argument('--max-cons-loc-search', '-l', action = 'store',
                        dest = 'max_consecutive_local_searches', 
                        default = 2, type = int,
                        help = 'maximum number of consecutive local searches')
    parser.add_argument('--init-strategy', '-i', action = 'store',
                        dest = 'init_strategy', default = 'lhd_maximin',
                        help = 'strategy to choose initial points')
    parser.add_argument('--function-scaling', '-s', action = 'store',
                        dest = 'function_scaling', default = 'auto',
                        help = 'strategy to rescale function values')
    parser.add_argument('--domain-scaling', '-m', action = 'store',
                        dest = 'domain_scaling', default = 'auto',
                        help = 'strategy to rescale domain')
    parser.add_argument('--dyn-clipping', '-c', action = 'store',
                        dest = 'dynamism_clipping', default = 'auto',
                        help = 'strategy to clip large function values')
    parser.add_argument('--ls-box-scaling', '-b', action = 'store',
                        dest = 'local_search_box_scaling',
                        default = 0.5, type = float,
                        help = 'scaling factor for local search box')
    parser.add_argument('--max-stalled', '-d', action = 'store',
                        dest = 'max_stalled_cycles', default = 6, type = int,
                        help = 'maximum number of cycles without ' +
                        'improvement before restart')
    parser.add_argument('--rand-seed', '-r', action = 'store',
                        dest = 'rand_seed', default = 937627691, type = int,
                        help = 'seed of the random seed generator')
    parser.add_argument('--with-rel-noise', '-nr', action = 'store',
                        dest = 'fast_objfun_rel_error', default = 0.0,
                        type = float,
                        help = 'amount of relative noise of the fast oracle')
    parser.add_argument('--with-abs-noise', '-na', action = 'store',
                        dest = 'fast_objfun_abs_error', default = 0.0,
                        type = float,
                        help = 'amount of relative noise of the fast oracle')
    parser.add_argument('--log', '-o', action = 'store',
                        metavar = 'LOG_FILE_NAME', dest = 'output_stream',
                        help = 'name of log file for output redirection')
    parser.add_argument('--print-solver-output', '-pso', action = 'store_true',
                        dest = 'print_solver_output', default = False,
                        help = 'print solver output')

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

    settings = rbfopt.RbfSettings(target_objval = black_box.optimum_value,
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
