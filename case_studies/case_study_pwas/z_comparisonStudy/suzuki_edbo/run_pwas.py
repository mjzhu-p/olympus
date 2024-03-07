from pwasopt.main_pwas import PWAS
# from pwas.categorical_encoder import cat_encoder
import numpy as np
import pandas as pd
from exp_data import get_exp_yield
from numpy.random import seed
import time # for tic-toc

# plotting libraries
from numpy import arange, meshgrid, zeros
import matplotlib.pyplot as plt


Ntests = 30 #number of tests executed on the same problem

# solver related parameters
delta_E = 0.05  # trade-off hyperparameter in acquisition function between exploitation of surrogate and exploration of exploration function
acq_stage = 'multi-stage'  # can specify whether to solve the acquisition step in one or multiple stages (as noted in Section 3.4 in the paper [1]. Default: `multi-stage`
feasible_sampling = True  # can specify whether infeasible samples are allowed. Default True
K_init = 10  # number of initial PWA partitions

reaction_name = 'suzuki'
# reaction_name = 'direct_arylation'

# optimization variables

if reaction_name == 'suzuki':
    nc = 0
    nint = 0
    nd = 1+ 1 + 1 +1 +1 # electrophile, nucleophile, base, ligand, solvent
    X_d = [4, 3, 7, 11, 4]

    lb = np.zeros((nd))
    ub = np.array([3, 2, 6, 10, 3])

elif reaction_name == 'direct_arylation':
    nc = 0
    # nint = 0
    # nd = 1 + 1 + 1 + 1 + 1  # base, ligand, solvent, concentration, temperature
    # X_d = [4, 12, 4, 3, 3]

    nint = 2
    nd = 1 + 1 + 1  # base, ligand, solvent, concentration, temperature
    X_d = [4, 12, 4]

    # lb = np.zeros((nd))
    # ub = np.array([3, 11, 3, 2, 2])

    lb = np.zeros((nd+nint))
    ub = np.array([2, 2, 3, 11, 3])

isLin_eqConstrained = False
isLin_ineqConstrained = False

preProcessingCatVar = False

nsamp = 10
maxevals = 100

plt.close('all')
plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(14, 7))


rxn_yield = lambda x: get_exp_yield(x, rxn_name=reaction_name)


viridis = plt.cm.get_cmap('viridis', Ntests)
out_Ntests = []
for i in range(0, Ntests):
    seed(i)  # rng default for reproducibility

    tic = time.perf_counter()

    # initialize the PWAS solver
    optimizer = PWAS(rxn_yield, lb, ub, delta_E, nc, nint, nd, X_d, nsamp, maxevals,  # pass fun to PWAS
                     feasible_sampling=feasible_sampling,
                     isLin_eqConstrained=isLin_eqConstrained,
                     isLin_ineqConstrained=isLin_ineqConstrained,
                     K=K_init, categorical=False,
                     acq_stage=acq_stage, integer_cut=True)
    xopt1, fopt1 = optimizer.solve()

    toc = time.perf_counter()

    out = optimizer.result_output()
    out_Ntests.append(out)
    print("Test # %2d, elapsed time: %5.4f" % (i + 1, toc - tic))

if Ntests == 1:
    xopt1 = out["xopt"]
    fopt1 = out["fopt"]
    X1 = out["X"]
    fbest_seq1 = out["fbest_seq"]

    plt.plot(arange(0, maxevals), fbest_seq1, color=(.6, 0, 0), linewidth=1.0)
    plt.scatter(arange(0, maxevals), fbest_seq1, color=(.6, 0, 0), marker='o', linewidth=1.0)


    plt.xlabel("number of fun. eval.")
    thelegend = ["PWAS"]
    plt.title("Best value of latent function")

    fopt_best = -100
    plt.plot(arange(0, maxevals), fopt_best * np.ones(maxevals), linestyle='--',
             color=(0, 0, .6), linewidth=2.0)

    plt.grid()
    thelegend.append("optimum")
    plt.legend(thelegend)

    plt.show()

else:
    nvars = nc + nint + nd
    nvars_encoded = out_Ntests[0]['self'].prob.nvars_encoded
    xopt_Ntests = zeros((Ntests, nvars))
    F_Ntests = zeros((Ntests, maxevals))
    fopt_Ntests = zeros((Ntests, 1))
    ibest_Ntests = zeros((Ntests, 1)).astype(int)
    ibestseq_Ntests = zeros((Ntests, maxevals, 1)).astype(int)
    isfeas_opt_Ntests = zeros((Ntests, 1)).astype(bool)
    isfeas_seq_Ntests = zeros((Ntests, maxevals, 1))
    X_Ntests = zeros((Ntests, maxevals, nvars))
    a_Ntests = zeros((Ntests, K_init, nvars_encoded, 1))
    b_Ntests = zeros((Ntests, K_init, 1))
    omega_Ntests = zeros((Ntests, K_init, nvars_encoded))
    gamma_Ntests = zeros((Ntests, K_init))
    kf_Ntests = zeros((Ntests, 1)).astype(int)
    minf_Ntests = zeros((Ntests, maxevals))

    for i in range(0, Ntests):
        out_i = out_Ntests[i]

        X_Ntests[i, :, :] = out_i['X']
        kf_Ntests[i, :] = out_i['Kf']
        kf_i = out_i['Kf']
        a_Ntests[i, :kf_i, :, :] = out_i['a']
        b_Ntests[i, :kf_i, :] = out_i['b']
        omega_Ntests[i, :kf_i, :] = out_i['omega']
        gamma_Ntests[i, :kf_i] = out_i['gamma']
        xopt_Ntests[i, :] = out_i['xopt']
        ibest_Ntests[i, :] = out_i['ibest']
        ibestseq_Ntests[i, :, :] = np.array(out_i['ibestseq']).reshape(-1,1)
        isfeas_seq_Ntests[i, :, :] = np.array(out_i['isfeas_seq']).reshape(-1,1)

        X = out_i['X']
        ibestseq = out_i["ibestseq"]

        F_Ntests[i, :] = np.array(out_i['F'])
        fopt_Ntests[i, 0] = out_i["fopt"]
        minf_Ntests[i, :] = np.array(out_i["fbest_seq"])


        res_Ntests = {'xopt_Ntests': xopt_Ntests,
                      'fopt_Ntests': fopt_Ntests,
                      'X_Ntests': X_Ntests,
                      'F_Ntests': F_Ntests,
                      'ibest_Ntests': ibest_Ntests,
                      'ibestseq_Ntests': ibestseq_Ntests,
                      'isfeas_seq_Ntests': isfeas_seq_Ntests,
                      'a_Ntests': a_Ntests,
                      'b_Ntests': b_Ntests,
                      'omega_Ntests': omega_Ntests,
                      'gamma_Ntests': gamma_Ntests,
                      'kf_Ntests': kf_Ntests,
                      'minf_Ntests': minf_Ntests,
                      }

        from scipy.io import savemat

        file_name_s = 'PWAS_' + reaction_name

        savemat("%s.mat" % file_name_s, res_Ntests)

