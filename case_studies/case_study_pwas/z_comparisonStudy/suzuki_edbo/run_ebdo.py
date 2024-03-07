# Install the EDBO package, note that slight updates of the code is needed to enable input the number of initial samples
# see the forked version at https://github.com/mjzhu-p/olympus/tree/pwas_comp for the changes needed
# pip install edbo; https://github.com/b-shields/edbo
from data_loader import suzuki
from edbo.utils import Data
from edbo.plot_utils import average_convergence, plot_avg_convergence
import pandas as pd
import numpy as np

from numpy.random import seed
import time # for tic-toc

import sys

# plotting libraries
from numpy import arange, meshgrid, zeros
import matplotlib.pyplot as plt

"""
Taken from the demo file in the EDBO (Experimental Design via Bayesian Optimization) package
Reference file: https://github.com/b-shields/edbo/blob/master/experiments/edbo_demo_and_simulations.ipynb by Benjamin Shields

"""

from edbo.bro import BO

def suzuki_multiRun(export_path_1 = None,
                    export_path_2 = None,
                    Ntests = 30,
                    initial_sample = 10,
                    batch_size = 1,
                    iterations = 40, # number of active learning iterations (does not count for the initial samples)
                    plot = False):

    viridis = plt.cm.get_cmap('viridis', Ntests)
    results = []
    total_exp = iterations+initial_sample
    opt_results = np.zeros((Ntests * total_exp,30))  #note here 30 is hard-coded for the example

    for i in range(0, Ntests):

        tic = time.perf_counter()

        try:
            # Instantiate edbo.BO
            bo = BO(exindex=reaction1.data,                       # Experiment index to look up results from
                    domain=reaction1.data.drop('yield', axis=1),  # Reaction space
                    # results=reaction1.data.iloc[start],         # Starting point
                    # init_method='external',                     # Default initial scheme: random ('rand')
                    initial_sample = initial_sample,
                    batch_size=batch_size,                         # Choose 1 experiments on each iteraiton, i.e., Sequential optimization experiment
                    acquisition_function='EI',                     # Use expected improvement
                    fast_comp=True)                                # Speed up the simulations using gpytorch's fast computation features

        except:
            errstr = "Make sure to update the EDBO package code according to the notes at the begining of this file"
            print(errstr)
            sys.exit(1)

        # Run simulation
        bo.simulate(iterations=iterations, seed=seed(i))  # rng default for reproducibility

        toc = time.perf_counter()

        # Append results to record
        results.append(bo.obj.results_input()['yield'].values) # contain yield info (at each iteration)only

        opt_results[i*total_exp:(i+1)*total_exp,:] = bo.obj.results_input().values  # contain info of optimal yield and opt. var. combinations

        print("Test # %2d, elapsed time: %5.4f" % (i + 1, toc - tic))

    # Save the results to a CSV file
    results = pd.DataFrame(results)
    opt_results = pd.DataFrame(opt_results)
    if export_path_1 != None:
        results.to_csv(export_path_1)

    if export_path_2 != None:
        opt_results.to_csv(export_path_2)

    # Average performance
    index, mean, std = average_convergence(results, batch_size)

    # Plot
    if plot:
        plot_avg_convergence(results, batch_size)


    if Ntests ==1:
        # Plot convergence
        bo.plot_convergence()

    return results, mean, std


# Build search spaces for reactions 1 with DFT encoded components

reaction1 = Data(suzuki(electrophile='ohe',
                        nucleophile='ohe',
                        base='ohe',
                        ligand='ohe',
                        solvent='ohe'))

reaction1.drop(['entry', 'electrophile_SMILES', 'nucleophile_SMILES', 'base_SMILES', 'ligand_SMILES',
                    'solvent_SMILES'])

Ntests = 1  # number of tests executed on the same problem
initial_sample = 10
batch_size = 1 # sequential exp.
iterations = 40
export_path_1 = 'suzuki_ebdo_yield.csv'
export_path_2 = 'suzuki_ebdo_all.csv'

results, mean, std = suzuki_multiRun(export_path_1 = export_path_1,
                                        export_path_2 = export_path_2,
                                        Ntests = Ntests,
                                        initial_sample = initial_sample,
                                        batch_size = batch_size,
                                        iterations = iterations,
                                        plot = True)
