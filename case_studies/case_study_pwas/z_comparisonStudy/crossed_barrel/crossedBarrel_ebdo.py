from olympus import Emulator
emulator = Emulator(dataset='crossed_barrel', model='BayesNeuralNet')

from edbo.utils import Data
from edbo.plot_utils import average_convergence, plot_avg_convergence
import pandas as pd
import numpy as np

from numpy.random import seed
import time # for tic-toc

# plotting libraries
from numpy import arange, meshgrid, zeros
import matplotlib.pyplot as plt

from itertools import product

"""
Taken from the demo file in the EDBO (Experimental Design via Bayesian Optimization) package
Reference file: https://github.com/b-shields/edbo/blob/master/experiments/edbo_demo_and_simulations.ipynb by Benjamin Shields

"""

from edbo.bro import BO

def obj_fun(x):## it is because in PWAS, the variables are ordered with [continuous, integer]
    value, _, __ = emulator.run(x)
    return float(value[0])


def crossedBarrel_multiRun(export_path_1 = None,
                    export_path_2 = None,
                    export_path_3 = None,
                    Ntests = 30,
                    initial_sample = 10,
                    batch_size = 1,
                    iterations = 40, # note it does not count for the initial samples
                    plot = False):

    viridis = plt.cm.get_cmap('viridis', Ntests)
    results = []
    cpu_time = []
    total_exp = iterations+initial_sample
    opt_results = np.zeros((Ntests * total_exp,5))

    n_hollow_columns = np.array([6, 7, 8, 9, 10, 11, 12])
    twist_angles = np.linspace(0, 200, 10)
    outer_radius = np.linspace(1.5, 2.5, 5)
    thicknesses = np.linspace(0.7, 1.4, 5)

    # Generate all combinations
    combinations = product(n_hollow_columns, twist_angles, outer_radius, thicknesses)

    # Calculate objective function for each combination
    results_obj = []
    for combination in combinations:
        result = obj_fun(np.array(combination))
        results_obj.append(combination + (result,))

    # Create DataFrame
    columns = ['n_hollow_column', 'twist_angle', 'outer_radius', 'thickness', 'objective_function']
    exindex = pd.DataFrame(results_obj, columns=columns)

    for i in range(0, Ntests):
        # tic = time.perf_counter()
        tic = time.process_time()

        # Instantiate edbo.BO
        bo = BO(exindex=exindex,                       # Experiment index to look up results from
                domain=exindex.drop('objective_function', axis=1),  # Reaction space
                # results=reaction1.data.iloc[start],         # Starting point
                # init_method='external',                     # Default initial scheme: random ('rand')
                initial_sample = initial_sample,
                batch_size=batch_size,                         # Choose 1 experiments on each iteraiton, i.e., Sequential optimization experiment
                acquisition_function='EI',                     # Use expected improvement
                fast_comp=True)                                # Speed up the simulations using gpytorch's fast computation features

        # Run simulation
        bo.simulate(iterations=iterations, seed=seed(i))  # rng default for reproducibility

        # toc = time.perf_counter()
        toc = time.process_time()

        # Append results to record
        results.append(bo.obj.results_input()['objective_function'].values) # contain yield info only
        cpu_time.append(toc-tic)

        opt_results[i*total_exp:(i+1)*total_exp,:] = bo.obj.results_input().values  # contain info of yield and opt. var. combinations

        print("Test # %2d, elapsed time: %5.4f" % (i + 1, toc - tic))

    # Save the results to a CSV file
    results = pd.DataFrame(results)
    opt_results = pd.DataFrame(opt_results)
    cpu_time = pd.DataFrame(cpu_time)
    if export_path_1 != None:
        results.to_csv(export_path_1)

    if export_path_2 != None:
        opt_results.to_csv(export_path_2)

    if export_path_3 != None:
        cpu_time.to_csv(export_path_3)

    # Average performance
    index, mean, std = average_convergence(results, batch_size)

    # Plot
    if plot:
        plot_avg_convergence(results, batch_size)


    if Ntests ==1:
        # Plot convergence
        bo.plot_convergence()

    return results, mean, std



Ntests = 30  # number of tests executed on the same problem
initial_sample = 10
batch_size = 1 # sequential exp.
iterations = 40
export_path_1 = 'C:/Users/Mengjia/Desktop/IMT/z-Research/a_on_going_project/MILP_IC/Rxn opt benchmark/z_olympus_code/olympus/case_studies/case_study_pwas/z_comparisonStudy/crossed_barrel/edbo_toughness_1901_5.csv'
export_path_2 = 'C:/Users/Mengjia/Desktop/IMT/z-Research/a_on_going_project/MILP_IC/Rxn opt benchmark/z_olympus_code/olympus/case_studies/case_study_pwas/z_comparisonStudy/crossed_barrel/ebdo_all_1901_5.csv'
export_path_3 = 'C:/Users/Mengjia/Desktop/IMT/z-Research/a_on_going_project/MILP_IC/Rxn opt benchmark/z_olympus_code/olympus/case_studies/case_study_pwas/z_comparisonStudy/crossed_barrel/ebdo_cpu_1901_5.csv'

results, mean, std = crossedBarrel_multiRun(export_path_1 = export_path_1,
                                        export_path_2 = export_path_2,
                                        export_path_3 = export_path_3,
                                        Ntests = Ntests,
                                        initial_sample = initial_sample,
                                        batch_size = batch_size,
                                        iterations = iterations,
                                        plot = True)
