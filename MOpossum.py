#!/usr/bin/python3.4
###################################################################
# File      : MOpossum.py
# Author(s) : Thomas Wortmann
#             Singapore University of Technology and Design
#             thomas_wortmann@mymail.sutd.edu.sg
#
#             Chu Wy Ton
#             Singapore University of Technology and Design
#             wyton_chu@mymail.sutd.edu.sg
#
# Date      : 08/06/18
# (C) Copyright Singapore University of Technology and Design 2018.
# You should have received a copy of the license with this code.
# Research supported by the SUTD-MIT International Design Center.
###################################################################
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import ast
import cma
import csv
import math
import os
import random
import sobol_seq
import sys
import time
import unittest

import numpy as np

from rbfopt import RbfoptSettings
from rbfopt import RbfoptAlgorithm
from rbfopt import RbfoptBlackBox
from rbfopt import RbfoptUserBlackBox
import rbfopt.rbfopt_utils as rbfopt_utils

from multiprocessing import Pool
from multiprocessing import cpu_count
from multiprocessing import freeze_support

#Sobol stuff
seed = 0
sobol = []
#Points/Nodes
pts = []
#Objectives
obj = []
#Augmented Tchebycheff stuff
rho = 0.05

#Opossum Evaluate
def obj_func(x):
    #prepare input for simulator: list of variables, values, separbfopt_alg, separated by comma

    dimension = len(x)
    newline = ''

    for h in range(dimension - 1):
        var_value = x[h]
        newline += '%.6f' % var_value + ','

    newline += '%.6f' % x[dimension - 1]

    #write line
    sys.stdout.write(newline + "\n")
    sys.stdout.flush()
    #wait for reply
    func_value = sys.stdin.readline()
    func_value = func_value.split(",")

    assert all([s.isdigit() for s in func_value])

    pts.append(x)
    obj.append(func_value)

    return calculate_weighted_objectives(sobol, func_value)

def calculate_weighted_objectives(weights, values):
    
    weighted_vals = values * weights
    aug_tcheby = weighted_vals.max(-1) + rho * weighted_vals.sum(-1)

    return aug_tcheby

#Opossum BlackBox
def construct_black_box(parbfopt_algm_list, obj_funct):
    #there should be a sequence of length 3*dimension of the form (lower_bound_i, upper_bound_i, integer_i)
    #separbfopt_algted by comma. If integer_i=1 integer varaibale, otherwise continuous
    
    r_init = csv.reader([parbfopt_algm_list], delimiter=";")
    r_init = list(r_init)
    r_init = r_init[0]
    params = r_init[1:]

    #Check if correct length of parbfopt_algmeters
    assert(len(params) / 3 == dimension)

    #Set variables
    for j in range(dimension):
        var_lower[i] = float(params[3 * j])
        var_upper[i] = float(params[3 * j + 1])
        if params[3 * j + 2] == '1': var_type[j] = 'I'

    #Check Variables
    assert(len(var_lower == dimension))
    assert(len(var_upper == dimension))
    assert(len(var_type == dimension))

    return RbfOptUserBlackBox(dimension, var_lower, var_upper, var_type, obj_funct)

#Command Line Interface
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

    #Algorithmic settings
    algset = parser.add_argument_group('Algorithmic settings')

    #Get default values from here
    default = RbfoptSettings()
    attrs = vars(default)
    docstring = default.__doc__
    param_docstring = docstring[docstring.find('Parameters'):
                                docstring.find('Attributes')].split(' : ')
    param_name = [val.split(' ')[-1].strip() for val in param_docstring[:-1]]
    param_type = [val.split('\n')[0].strip() for val in param_docstring[1:]]
    param_help = [' '.join(line.strip() for line in val.split('\n')[1:-2])
                  for val in param_docstring[:-1]]

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

        algset.add_argument('--' + param_name[i], action = 'store',
                            dest = param_name[i],
                            type = type_fun,
                            help = param_help[i],
                            default = getattr(default, param_name[i]))

#Get the model's bounds, evaluated points and values from a surrogate model
#and write them to a file
def getPoints(pointFilePath, valueFilePath, model):
    """Get the model's bounds, evaluated points and values from a surrogate model
    and write them to a file.

    Parameters
    ----------
    pointFilePath : str
        File Path for the points.
    valueFilePath : str
        File Path for the values.
    model : RbfOpt.RbfOptModel
        The surrogate model.
    """
    points = model.node.pos
    lowerBounds = model.l_lower
    upperBounds = model.l_upper
    boundsAndPoints = [lowerBounds, upperBounds]
    boundsAndPoints.extend(model.node_pos)
    writeFile(pointFilePath, boundsAndPoints)
    writeFile(valueFilePath, model.node_val)

def readPoints(pointFilePath, valueFilePath):
    """Read the points and values from a file

    Parameters
    ----------
    pointFilePath : str
        File Path to read points from.
    valueFilePath : str
        File Path to read values from.
    """
    points = np.loadtxt(pointFilePath, delimiter = " ", ndmin = 2)
    values = np.loadtxt(valueFilePath, delimiter = " ", ndmin = 2)

    #check that list is complete
    assert len(points) == len(values), "%d points, but %d values" % (len(points), len(values))

    #sort according to objective (minimization)
    #values, points = zip(*sorted(zip(values, points)), key=lambda pair: pair[0])

    points = np.array(points)
    values = np.array(values)

    return points, values

def addPoints(pointFilePath, valueFilePath, model):
    """Adds points and values to the surrogate model.

    Parameters
    ----------
    pointFilePath : str
        File path for points.
    valueFilePath : str
        File path for values.
    model : RbfOpt.RbfOptModel
        The surrogate model.
    """

    points, values = readPoints(pointFilePath, valueFilePath)

    #Check that points have correct dimensionality
    assert len(points[0]) == len(model.l_lower), "Model has %d dimensions, but point file has %d" % (len(model.l_lower), len(points[0]))

    #Scale points
    points_scaled = rbfopt_utils.bulk_transform_domain(model.l_settings, model.l_lower, model.l_upper, points)

    #Add the points (last entry for fast evals)
    #add node(self, point, orig_point, value, is_fast
    for i in range(len(points)):
        #Distance test
        if(rbfopt_utils.get_min_distance(points[1],model.all_node_pos) > model.l_settings.min_dist):
            model.add_node(points_scaled[i], points[i], values[i])
        else:
            print ("Point is too close to add to model!")

def evaluatePoints(pointFilePath, valueFilePath, model):
    """Read points from a file.
    Evaluate them based on the surrogate model (in RbfoptAlgorithm.evaluateRBF),
    and write the results to a file.

    Parameters
    ----------
    pointFilePath : str
        File path for points.
    valueFilePath : str
        File path for values.
    model : Rbfopt.RbfOptModel
        Surrogate model.
    """

    #Get the points to evaluate
    points = np.loadtxt(pointFilePath, delimiter = " ", ndmin = 2)

    #check that the points have the correct dimensionality
    assert len(points[0]) == len(model.l_lower), "Model has %d dimensions, but point file has %d" % (len(model.l_lower), len(points[0]))

    values = evaluateRBF((points, model))

    writeFile(valueFilePath, values)

def evaluateRBF(input):
    """Evaluates set of points based on a given surrogate model
    (with a single input object for multiprocessing)

    Parameters
    ----------
    input : tup
        input containing the points and model
    """

    #Single input for parallel processing
    points, model = input

    #Write error message
    if len(points) == 0:
        sys.stdout.write("No points to evaluate!\n")
        sys.stdout.flush()

    #Number of nodes at current iterbfopt_algtion
    k = len(model.node_pos)

    #Compute indices of fast node evaluations (sparse format)
    fast_node_index = (np.nonzero(model.node_is_fast)[0] if model.two_phase_optimization else np.array([]))

    #Rescale nodes if necessary
    tfv = rbfopt_utils.transform_function_values(model.l_settings, np.array(model.node_val), model.fmin, model.fmax, fast_node_index)
    (scaled_node_val, scaled_min, scaled_fmax, node_err_bounds, rescale_function) = tfv

    #Compute the matrices necessary for the algorithm
    Amat = rbfopt_utils.get_rbf_matrix(model.l_settings, model.n, k, np.array(model.node_pos))

    #Get coefficients for the exact RBF
    rc = rbfopt_utils.get_rbf_coefficients(model.l_settings, model.n, k, Amat, scaled_node_val)
    if(fast_node_index):
        #RBF with some fast function evaluations
        rc = aux.get_noisy_rbf_coefficients(model.l_settings, model.n, k, Amat[:k, :k], Amat[:k, :k], scaled_node_val, fast_node_index, node_err_bounds, rc[0], rc[1])
    (rbf_l, rbf_h) = rc

    #Evaluate RBF
    if len(points) <= 1:
        values = []
        for point in points: values.append(rbfopt_util.evaluate_rbf(model.l_settings, point, model.n, k, np.array(model.node_pos), rbf_l, rbf_h))
        return values
    else:
        return rbfoptutils.bulk_evaluate_rbf(model.l_settings, np.array(points), model.n, k, np.array(model.node_pos), rbf_l, rbf_h)

def writeFile(file, elements):
    """Write a list, or list of list, to a file

    Parameters
    ----------
    file : str
        File path
    elements : list
        list of things
    """
    with open(file, 'w') as file:
        for element in elements:
            if isinstance(element, (list, np.ndarray)):
                string = " ".join(str(value) for value in element)
                file.write(string + "\n")

            elif isinstance(element, (float, np.float64)):
                string = str(element)
                file.write(string + "\n")

            else:
                print("Element " + str(type(element))) + " not recognized!"

def which(progrbfopt_algm):
    """Checks if the executable is on the path (for solvers)
    """
    def is_exe(fpath):
        return os.path.isfile(path) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(progrbfopt_algm)
    if fpath:
        if is_exe(progrbfopt_algm):
            return progrbfopt_algm
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, progrbfopt_algm)
            if is_exe(exe_file):
                return exe_file
    return None

def run_unit_test():
    #os.chdir("..")
    test_loader = unittest.TestLoader()

    test_suite = test_loader.discover('tests/', pattern = 'test*.py')
    unittest.TextTestRunner().run(test_suite)

if(__name__ == "__main__"):
    #Needed to make py2exe to work
    freeze_support()

    #Create rbfopt_cl line parsers
    desc = ('rbfopt_utiln RBFOpt, or get the current RBFOpt optimization state and evalueate the RBF surrogate model.')
    parser = argparse.ArgumentParser(description=desc)

    #Add Rbfopt options to parser
    register_options(parser)

    #Add additional options
    parser.add_argument('--optimizeRBF', action = 'store_true', help = 'RBF Optimization')

    parser.add_argument('--param_list', '-param', action = 'store', dest = 'param_list', metavar = 'PARAM_LIST', type = str, help = 'list of parameters for initialization')
    parser.add_argument('--approximate', action = 'store_true', help = 'approximate the points in the pointFile, otherwise return evaluated points')
    parser.add_argument('--addNodes', action = 'store_true', help = 'add the points from the addPointsFile and addValuesFile to the model')
    parser.add_argument('--path', action = 'store', type = str, help = 'path for files, default is script directory')
    parser.add_argument('--stateFile', action = 'store', type = str, help = 'file name for algorithm state')
    parser.add_argument('--pointFile', action = 'store', type = str, help = 'file name for points')
    parser.add_argument('--valueFile', action = 'store', type = str, help = 'file name for objective vaules')
    parser.add_argument('--addPointsFile', action = 'store', type = str, help = 'file name for points to add to the model')
    parser.add_argument('--addValuesFile', action = 'store', type = str, help = 'file name for objective values to add to the model')
    parser.add_argument('--test', action = 'store_true', help = 'run unit tests')
    parser.add_argument('--log', '-o', action = 'store', metavar = 'LOG_FILE_NAME', type = str, dest = 'output_stream', help = 'Name of log file for output redirection')

    #Add options for CMA-ES
    parser.add_argument('--optimizeCMAES', action = 'store_true', help = 'CMA-ES optimization')
    parser.add_argument('--initialSolution', action = 'store', dest = 'initialSolution', type = str, help = 'starting point as list')

    #Get arguments
    args = parser.parse_args()
    dim = args.param_list[0]

    #Evaulate the surrogate model
    if args.optimizeRBF is False and args.optimizeCMAES is False and args.test is False:
        assert args.path is not None, "Missing path parameter!"
        assert args.pointFile is not None, "Missing point file parameter!"
        assert args.valueFile is not None, "Missing value file parameter!"
        assert args.stateFile is not None, "Missing state file parameter!"

        #Read algorithm state
        stateFilePath = args.path + args.stateFile

        #Add additional points before evaulation
        if args.addNodes:
            assert args.addPointsFile is not None, "Missing addPoints file parameter!"
            assert args.addValuesFile is not None, "Missing addValues file parameter!"

            addPoints(args.path + args.addPointsFile, args.path + args.addValuesFile, model)

        #One point or value per line, point coordinates delimited by " "
        if args.approximate is True:
            evaluatePoints(args.path + args.pointFile, args.path + args.valueFile, model)
        else:
            getPoints(args.path + args.pointFile, args.path + args.valueFile, model)

    #Run RBFOpt
    elif args.optimizeRBF:
        assert args.param_list is not None, "Missing variable list parameters!"
        #assert which(config.MINLP_SOLVER_PATH + ".exe") is not None, "MINLP Solver path %s not found!" % (config.MINLP_SOLVER_PATH + ".exe")
        #assert which(config.NLP_SOLVER_PATH + ".exe") is not None, "NLP Solver path %s not found!" % (config.NLP_SOLVER_PATH + ".exe")

        #Open output stream if necessary
        output_stream = None

        if(args.output_stream is not None):
            try:
                output_stream = open(args.output_stream, 'w')
            except IOError as e:
                print('Error while opening log file', file = sys.stderr)
                print(e, file = sys.stderr)

        #Add Additional points before evaluation
        if args.AddNodes:
            assert args.path is not None, "Missing path parameters!"
            assert args.addPointsFile is not None, "Missing addPoints file parameter!"
            assert args.addValuesFile is not None, "Missing addValues file parameter!"

            points, values = readPoints(args.path + args.addPointsFile, args.path + args.addValueFile)

        else:
            points = None
            values = None
        
        #Create dictionary from parser
        dict_args = vars(args)

        #Remove non-RBFOpt arguments from directory
        dict_args.pop('output_stream')
        dict_args.pop('optimizeRBF')
        dict_args.pop('approximate')
        dict_args.pop('addNodes')
        dict_args.pop('test')
        dict_args.pop('path')
        dict_args.pop('stateFile')
        dict_args.pop('pointFile')
        dict_args.pop('valueFile')
        dict_args.pop('addPointsFile')
        dict_args.pop('addValuesFile')

        #Remove CMAES arguments from dictionary
        dict_args.pop('optimizeCMAES')
        dict_args.pop('initialSolution')

        parameters = dict_args.pop('param_list')
        black_box = construct_black_box(parameters, obj_func)
        settings = RbfOptSettings.from_dictionary(dict_args)
        
        if(output_stream is not None): alg.set_output_stream(output_stream)

        sobol, seed = sobol_seq(dim, seed)

        pts = points.copy
        obj = values.copy
        
        #loop this part for cycles
        while(True):
            weighted_obj = calculate_weighted_objectives(sobol, obj)

            #Set settings
            settings.max_cycles = 1

            #Calculate values from objectives with sobol weights
            alg = RbfoptAlgorithm(settings, black_box, points, values)

            alg.optimize()

            sobol, seed = sobol_seq(dim, seed)

            #end loop

    elif args.optimizeCMAES:
        assert args.param_list is not None, "Missing variable list parameter!"

        #Create dictionary from parser
        dict_args = vars(args)
        #Use parameters to create Black Box and remove them from dictionary
        parameters = dict_args.pop('param_list')
        black_box = construct_black_box(parameters, obj_funct)

        #Check if initial solutions (i.e., starting point) is provided
        if args.initialSolution is not None:
            initialSolutionStr = dict_args.pop('initialSolution')
            initialSolution = list(map(float, initialSolutionStr.split(';')))
            assert len(initialSolution) == black_box.get_dimension(), "Initial Solution for CMA-ES has incorrect dimension"
        #Else generate random starting point
        else:
            initialSolution = black_box.get_dimension() * [random.random()]

        #with initial solution all zeros and initial sigma = 0.5
        #Integer variables?
        #Max Evaluations?
        es = cma.CMAEvolutionStrategy(initialSolution, 0.5, {'bounds': [0,1], 'verb_log': 0})
        es.optimize(black_box.evaluate)
    #Test
    else:
        run_unit_test()







