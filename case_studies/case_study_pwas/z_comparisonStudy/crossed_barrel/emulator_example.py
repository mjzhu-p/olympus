#%%
# loading one of our emulators:
from olympus import Emulator
emulator = Emulator(dataset='crossed_barrel', model='BayesNeuralNet')
result = emulator.run([[11,89.034,1.542,0.724]])

#%%

# from olympus.emulators import load_emulator
# loaded = load_emulator('test_save')





# emulator.dataset.param_space

# #%%
#
# from olympus import Emulator, Dataset
# from olympus.models import BayesNeuralNet
# from olympus.emulators.emulator import load_emulator
#
# #%%
#
# dataset = Dataset(kind='crossed_barrel', test_frac=0.2, num_folds=5)
# model = BayesNeuralNet(max_epochs=150)
#
# #%%
#
# emulator = Emulator(dataset=dataset, model=model, feature_transform='identity', target_transform='identity')
#
# #%%
#
# # this is where the files related to the emulator model are saved temporarily
# emulator._scratch_dir.name
#
# #%%
#
# #emulator.cross_validate()
# emulator.train(retrain=True)
#
# #%%
#
# emulator.__dict__
#
# #%%
#
# emulator.save('test_save', include_cv=True)
#
# #%%
#
# from olympus.emulators.emulator import load_emulator
# loaded = load_emulator('test_save')
