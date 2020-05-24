#!/usr/bin/env python3
"""Command-line interface for RBFOpt.

This module provides a basic command-line interface for RBFOpt,
allowing the user to launch the optimization process and set the
algorithmic options using a standard UNIX sintax.

Licensed under Revised BSD license, see LICENSE.
(C) Copyright Singapore University of Technology and Design 2014.
(C) Copyright International Business Machines Corporation 2016.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import importlib
# We must set the threading options before numpy is loaded, otherwise
# there might be issues when running several processes in parallel.
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import ast
import numpy as np
from rbfopt import RbfoptSettings
from rbfopt import RbfoptBlackBox
from rbfopt import RbfoptAlgorithm


def register_options(parser):
    """Add options to the command line parser.

    Register all the options for the optimization algorithm.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser.

    See also
    --------   
    :class:`rbfopt_settings.RbfoptSettings` for a detailed description of
    all the command line options.
    """
    # Algorithmic settings
    algset = parser.add_argument_group('Algorithmic settings')
    # Get default values from here
    default = RbfoptSettings()
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
        algset.add_argument('--' + param_name[i], action='store',
                            dest=param_name[i],
                            type=type_fun,
                            help=param_help[i],
                            default=getattr(default, param_name[i]))
    intset = parser.add_argument_group('Execution settings')
    intset.add_argument('black_box_file', action='store',
                        metavar='FILE_NAME', type=str, 
                        help='Name of python file containing black box ' +
                        'function and the description of its ' +
                        'characteristics. This file should implement a ' +
                        'BlackBox class derived from ' +
                        'rbfopt_black_box.BlackBox, with name ' +
                        'RbfoptBlackBox. Note: the directory containing ' +
                        'it will be added to the Python path.')
    intset.add_argument('--load', '-l', action='store', dest='load_state',
                        help='File to read state to resume optimization')
    intset.add_argument('--log', '-o', action='store',
                        metavar='LOG_FILE_NAME', dest='output_stream',
                        help='Name of log file for output redirection')
    intset.add_argument('--pause', '-p', action='store', dest='pause',
                        default=sys.maxsize, type=int,
                        help='Number of iterations after which ' +
                        'the optimization process should be paused')
    intset.add_argument('--points_from_file', '-f', action='store',
                        metavar='POINTS_FILE_NAME', dest='points_file',
                        type=str, default=None,
                        help='Name of a file containing coordinates of ' +
                        'points that have already been evaluated or must ' +
                        'be evaluated. The points must be given one per ' +
                        'line with each value separated by space. Each ' +
                        'row can optionally terminate with the objective ' +
                        'function value at the point (if such value is ' +
                        'not given, it will be evaluated by the algorithm).')
    intset.add_argument('--print_solution', '-ps', action='store', 
                        dest='print_solution',
                        default=True, type=ast.literal_eval,
                        help='Print solution at the end of the ' +
                        'optimization (or after a pause).')
    intset.add_argument('--save', '-s', action='store', dest='dump_state',
                        help='File to save state after optimization. ' +
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

    black_box : :class:`rbfopt_black_box.RbfoptBlackBox`
        The black box to be optimized.
    """
    if (not isinstance(black_box, RbfoptBlackBox)):
        raise ValueError('The specified module does not contain a ' +
                         'valid BlackBox instance')

    # Open output stream if necessary
    if (args['output_stream'] is None):
        output_stream = sys.stdout
    else:
        try:
            output_stream = open(args['output_stream'], 'w')
        except IOError as e:
            print('Error while opening log file', file=sys.stderr)
            print(e, file=sys.stderr)

    # Make a copy of parameters and adjust them, deleting keys
    # that are not recognized as valid by RbfoptSettings.
    local_args = args.copy()
    del local_args['black_box_file']
    del local_args['output_stream']
    del local_args['load_state']
    del local_args['dump_state']
    del local_args['points_file']
    del local_args['pause']
    del local_args['print_solution']

    settings = RbfoptSettings.from_dictionary(local_args)
    settings.print(output_stream=output_stream)
    if (args['load_state'] is not None):
        alg = RbfoptAlgorithm.load_from_file(args['load_state'])
    elif (args['points_file'] is not None):
        try:
            init_node_pos = list()
            points_file = open(args['points_file'], 'r')
            for line in points_file:
                init_node_pos.append([float(val) for val in line.split()])
            if (len(init_node_pos[0]) == black_box.get_dimension() + 1):
                # In this case the file contains function values as well,
                # as the last column
                init_node_val = np.array([val[-1] for val in init_node_pos])
                init_node_pos = np.array([val[:-1] for val in init_node_pos])
            else:
                init_node_pos = np.array(init_node_pos)
                init_node_val = None
        except Exception as e:
            print('Exception raised reading file with initialization points',
                  file=output_stream)
            print(e, file=output_stream)
            output_stream.close()
            raise
        alg = RbfoptAlgorithm(settings=settings, black_box=black_box,
                              init_node_pos=init_node_pos,
                              init_node_val=init_node_val)
    else:
        alg = RbfoptAlgorithm(settings=settings, black_box=black_box)
    alg.set_output_stream(output_stream)
    result = alg.optimize(args['pause'])
    print('RbfoptAlgorithm.optimize() returned ' + 
          'function value {:.15f}'.format(result[0]),
          file=output_stream)
    if (args['print_solution']):
        for (i, val) in enumerate(result[1]):
            print('x{:<4d}: {:16.6f}'.format(i, val), file=output_stream)
    if (args['dump_state'] is not None):
        alg.save_to_file(args['dump_state'])
        print('Dumped state to file {:s}'.format(args['dump_state']),
              file=output_stream)
    output_stream.close()

# -- end function

if (__name__ == "__main__"):
    if (sys.version_info[0] <= 2 and sys.version_info[1] < 7):
        print('Error: this software requires Python 2.7 or later')
        exit()
    # Create command line parsers
    desc = ('Apply the RBF method to an object of class "BlackBox".')
    parser = argparse.ArgumentParser(description=desc)
    # Add options to parser and parse arguments
    register_options(parser)
    args = parser.parse_args()
    if (not os.path.isfile(args.black_box_file)):
        raise ValueError('The file {:s} '.format(args.black_box_file) +
                         'supposed to provide the implementation of ' +
                         'RbfoptBlackBox does not exist.')

    try:
        rel_dir = os.path.dirname(args.black_box_file)
        if (rel_dir == ''):
            rel_dir = '.'
        abs_dir = os.path.abspath(rel_dir)
        # Add directory to path
        sys.path.append(abs_dir)
        # Import module (after removing trailing .py, if any)
        module_name = os.path.basename(args.black_box_file)
        if (module_name.endswith('.py')):
            module_name = module_name[:-3]
        bb = importlib.import_module(module_name)
    except Exception as e:
        print('Error while opening module with user black box',
              file=sys.stderr)
        print(e, file=sys.stderr)
        raise
    # Run the interface
    rbfopt_cl_interface(vars(args), bb.RbfoptBlackBox())
