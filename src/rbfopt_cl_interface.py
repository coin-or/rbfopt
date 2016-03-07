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
import ast
import rbfopt
import black_box as bb
from rbfopt_settings import RbfSettings
from rbfopt_algorithm import OptAlgorithm


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
            type_fun = ast.literal_eval
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
    parser.add_argument('--load', '-l', action = 'store', dest = 'load_state',
                        help = 'file to read state to resume optimization')
    parser.add_argument('--save', '-s', action = 'store', dest = 'dump_state',
                        help = 'file to save state after optimization')
    parser.add_argument('--pause', '-p', action = 'store', dest = 'pause',
                        default = sys.maxint, type = int,
                        help = 'number of iterations after which ' +
                        'the optimization process should be paused')



# -- end function

def rbfopt_cl_interface(args, black_box):
    """Command-line interface.
    
    Optimize the specified objective function using the algorithmic
    options given on the command line.

    Parameters
    ----------
    args : Dict[string]
        A dictionary containing the values of the parameters in a
        format args['name'] = value. 

    black_box : black_box.BlackBox
        An object containing the function to be optimized and its main
        characteristics. It is possible to pass an object of a
        different class, provided that it as the same public
        attributes.
    """

    assert(hasattr(black_box, 'dimension'))
    assert(hasattr(black_box, 'var_lower'))
    assert(hasattr(black_box, 'var_upper'))
    assert(hasattr(black_box, 'integer_vars'))
    assert(hasattr(black_box, 'evaluate'))
    assert(hasattr(black_box, 'evaluate_fast'))

    # Open output stream if necessary
    if (args['output_stream'] is None):
        output_stream = sys.stdout
    else:
        try:
            output_stream = open(args['output_stream'], 'w')
        except IOError as e:
            print('Error while opening log file', file = sys.stderr)
            print(e, file = sys.stderr)

    # Make a copy of parameters and adjust them, deleting keys
    # that are not recognized as valid by RbfSettings.
    local_args = args.copy()
    del local_args['output_stream']
    del local_args['load_state']
    del local_args['dump_state']
    del local_args['pause']

    settings = RbfSettings.from_dictionary(local_args)
    settings.print(output_stream = output_stream)
    if (args['load_state'] is not None):
        alg = OptAlgorithm.load_from_file(args['load_state'],
                                          black_box.evaluate,
                                          black_box.evaluate_fast)
    else:
        alg = OptAlgorithm(settings = settings,
                           dimension = black_box.dimension, 
                           var_lower = black_box.var_lower,
                           var_upper = black_box.var_upper,
                           objfun = black_box.evaluate,
                           objfun_fast = black_box.evaluate_fast,
                           integer_vars = black_box.integer_vars)
    alg.set_output_stream(output_stream)
    result = alg.optimize(args['pause'])
    print('OptAlgorithm.optimize() returned ' + 
          'function value {:.15f}'.format(result[0]),
          file = output_stream)
    if (args['dump_state'] is not None):
        alg.save_to_file(args['dump_state'])
        print('Dumped state to file {:s}'.format(args['dump_state']),
              file = output_stream)
    output_stream.close()

# -- end function

if (__name__ == "__main__"):
    if (sys.version_info[0] >= 3):
        print('Error: Python 3 is currently not tested.')
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
    rbfopt_cl_interface(vars(args), bb.BlackBox())
