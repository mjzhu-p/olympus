#!/usr/bin/env python

import pickle
from olympus import Olympus
from olympus.campaigns import Campaign
from olympus.databases import Database

olymp = Olympus()
planners = [
    'RandomSearch',
    'Botorch',
    'Hyperopt',
    'Genetic',
    'Gryffin'
]
database = Database(kind='sqlite')

olymp.benchmark(
        dataset='crossed_barrel',
        # dataset='vapdiff_crystal',
        # dataset='suzuki_edbo',
        # dataset='suzuki_i',
        planners=planners,
        database=database,
        num_ind_runs=30,
        num_iter=50,
)


observations = [campaign.observations for campaign in database]
pickle.dump(observations, open('results.pkl', 'wb'))

from olympus import Plotter
plotter = Plotter()
plotter.plot_from_db(olymp.evaluator.database)