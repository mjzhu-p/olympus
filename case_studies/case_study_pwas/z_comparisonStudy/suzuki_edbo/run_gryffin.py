#!/usr/bin/env python

import os, sys
import pickle
import numpy as np

import olympus
from olympus.datasets import Dataset
from olympus.evaluators import Evaluator
from olympus.emulators import Emulator
from olympus.campaigns import Campaign
from olympus.planners import Planner
from olympus.scalarizers import Scalarizer

sys.path.append('../../../')
from utils import save_pkl_file, load_data_from_pkl_and_continue

# --------
# CONFIG
# --------

dataset_name = 'suzuki_edbo'
planner_name = 'Gryffin'

budget = 50
num_repeats = 30

# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(num_repeats)

for num_repeat in range(missing_repeats):

    print(f'\nTESTING {planner_name} ON {dataset_name} REPEAT {num_repeat} ...\n')

    # dataset_name == 'suzuki_edbo':

    # fully categorical, lookup table
    dataset = Dataset(kind=dataset_name)

    planner = Planner(kind=planner_name)
    planner.set_param_space(dataset.param_space)

    campaign = Campaign()
    campaign.set_param_space(dataset.param_space)
    campaign.set_value_space(dataset.value_space)

    evaluator = Evaluator(
        planner=planner,
        emulator=dataset,
        campaign=campaign,
    )


    evaluator.optimize(num_iter=budget)

    data_all_repeats.append(campaign)
    save_pkl_file(data_all_repeats)

    print('Done!')

