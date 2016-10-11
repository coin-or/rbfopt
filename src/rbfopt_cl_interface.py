"""Command-line interface for RBFOpt.

This module provides a basic command-line interface for RBFOpt,
allowing the user to launch the optimization process and set the
algorithmic options using a standard UNIX sintax.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
(C) Copyright International Business Machines Corporation 2016.
Research partially supported by SUTD-MIT International Design Center.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import argparse
import ast
import importlib
from rbfopt_settings import RbfSettings
from rbfopt_black_box import BlackBox
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
    # Algorithmic settings
    algset = parser.add_argument_group('Algorithmic settings')
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
        algset.add_argument('--' + param_name[i], action = 'store',
                            dest = param_name[i],
                            type = type_fun,
                            help = param_help[i],
                            default = getattr(default, param_name[i]))
    intset = parser.add_argument_group('Execution settings')
    intset.add_argument('--black_box_module', '-m', action = 'store',
                        metavar = 'MODULE_NAME', dest = 'black_box_module',
                        type = str, default = 'rbfopt_black_box_example',
                        help = 'Name of module containing black box function'
                        + ' and the description of its characteristics. ' +
                        'This module should implement a BlackBox class ' +
                        'derived from rbfopt_black_box.BlackBox.')
    intset.add_argument('--load', '-l', action = 'store', dest = 'load_state',
                        help = 'File to read state to resume optimization')
    intset.add_argument('--log', '-o', action = 'store',
                        metavar = 'LOG_FILE_NAME', dest = 'output_stream',
                        help = 'Name of log file for output redirection')
    intset.add_argument('--pause', '-p', action = 'store', dest = 'pause',
                        default = sys.maxint, type = int,
                        help = 'Number of iterations after which ' +
                        'the optimization process should be paused')
    intset.add_argument('--points_from_file', '-f', action = 'store',
                        metavar = 'POINTS_FILE_NAME', dest = 'points_file',
                        type = str, default = None,
                        help = 'Name of a file containing coordinates of ' +
                        'points that have already been evaluated or must ' +
                        'be evaluated. The points must be given one per ' +
                        'line with each value separated by space. Each ' +
                        'row can optionally terminate with the objective ' +
                        'function value at the point (if such value is ' +
                        'not given, it will be evaluated by the algorithm).')
    intset.add_argument('--print_solution', '-ps', action = 'store', 
                        dest = 'print_solution',
                        default = True, type = ast.literal_eval,
                        help = 'Print solution at the end of the ' +
                        'optimization (or after a pause).')
    intset.add_argument('--save', '-s', action = 'store', dest = 'dump_state',
                        help = 'File to save state after optimization. ' +
                        'Note that this is different from the options ' +
                        'save_state_interval and save_state_file because ' +
                        'here the state is only saved at the end of ' +
                        'the optimization (or after a pause).')



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

    black_box : :class:`rbfopt_black_box.BlackBox`
        The black box to be optimized.
    """
    if (not isinstance(black_box, BlackBox)):
        raise ValueError('The specified module does not contain a ' +
                         'valid BlackBox instance')

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
    del local_args['black_box_module']
    del local_args['output_stream']
    del local_args['load_state']
    del local_args['dump_state']
    del local_args['points_file']
    del local_args['pause']
    del local_args['print_solution']

    settings = RbfSettings.from_dictionary(local_args)
    settings.print(output_stream = output_stream)
    if (args['load_state'] is not None):
        alg = OptAlgorithm.load_from_file(args['load_state'])
    elif (args['points_file'] is not None):
        try:
            init_node_pos = list()
            points_file = open(args['points_file'], 'r')
            for line in points_file:
                init_node_pos.append([float(val) for val in line.split()])
            if (len(init_node_pos[0]) == black_box.get_dimension() + 1):
                # In this case the file contains function values as well,
                # as the last column
                init_node_val = [val[-1] for val in init_node_pos]
                init_node_pos = [val[:-1] for val in init_node_pos]
            else:
                init_node_val = None
        except Exception as e:
            print('Exception raised reading file with initialization points',
                  file = output_stream)
            print(type(e), file = output_stream)
            print(e, file = output_stream)
            output_stream.close()
            exit()
        alg = OptAlgorithm(settings = settings, black_box = black_box,
                           init_node_pos = init_node_pos,
                           init_node_val = init_node_val)
    else:
        alg = OptAlgorithm(settings = settings, black_box = black_box)
    alg.set_output_stream(output_stream)
    result = alg.optimize(args['pause'])
    print('OptAlgorithm.optimize() returned ' + 
          'function value {:.15f}'.format(result[0]),
          file = output_stream)
    if (args['print_solution']):
        for (i, val) in enumerate(result[1]):
            print('x{:<4d}: {:16.6f}'.format(i, val), file = output_stream)
    if (args['dump_state'] is not None):
        alg.save_to_file(args['dump_state'])
        print('Dumped state to file {:s}'.format(args['dump_state']),
              file = output_stream)
    #output_stream.close()

# -- end function

if (__name__ == "__main__"):
    if (sys.version_info[0] >= 3):
        print('Error: Python 3 is currently not tested.')
        print('Please use Python 2.7')
        exit()
    # Create command line parsers
    desc = ('Apply the RBF method to an object of class "BlackBox".')
    parser = argparse.ArgumentParser(description = desc)
    # Add options to parser and parse arguments
    register_options(parser)
    args = parser.parse_args()
    bb = importlib.import_module(args.black_box_module)
    # Run the interface
    rbfopt_cl_interface(vars(args), bb.BlackBox())
