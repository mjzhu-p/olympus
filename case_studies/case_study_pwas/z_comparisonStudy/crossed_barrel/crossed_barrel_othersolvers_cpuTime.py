#!/usr/bin/env python

import pickle
from olympus import Olympus
from olympus.campaigns import Campaign
from olympus.databases import Database
import time # for tic-toc

olymp = Olympus()
planners = [
    # 'RandomSearch',
    # 'Botorch',
    # 'Hyperopt',
    # 'Genetic',
    'Gryffin',
]
database = Database(kind='sqlite')

cpu_time = []

Ntests = 30
for i in range(0, Ntests):
    # tic = time.perf_counter()
    tic = time.process_time()

    olymp.benchmark(
            dataset='crossed_barrel',
            # dataset='vapdiff_crystal',
            # dataset='suzuki_edbo',
            # dataset='suzuki_i',
            planners=planners,
            database=database,
            num_ind_runs=1,
            num_iter=50,
    )
    # toc = time.perf_counter()
    toc = time.process_time()
    cpu_time.append(toc - tic)

import pandas as pd
cpu_time = pd.DataFrame(cpu_time)
# export_path_4 = 'C:/Users/Mengjia/Desktop/IMT/z-Research/a_on_going_project/MILP_IC/Rxn opt benchmark/z_olympus_code/olympus/case_studies/case_study_pwas/z_comparisonStudy/crossed_barrel/gryffin_cpu.csv'
export_path_4 = 'C:/Users/j18951mz/OneDrive - The University of Manchester/Desktop/UoM/z_research/zzz_GitHub repo/olympus/case_studies/case_study_pwas/z_comparisonStudy/crossed_barrel/gryffin_cpu.csv'
cpu_time.to_csv(export_path_4)

observations = [campaign.observations for campaign in database]
pickle.dump(observations, open('results_test.pkl', 'wb'))

from olympus import Plotter
plotter = Plotter()
plotter.plot_from_db(olymp.evaluator.database)