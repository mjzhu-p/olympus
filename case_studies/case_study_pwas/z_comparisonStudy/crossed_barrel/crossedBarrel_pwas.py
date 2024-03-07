# This file uses EDBO to solve the crossed-barrel structure optimization problem

from pwasopt.main_pwas import PWAS # pip install pwasopt; https://github.com/mjzhu-p/PWAS
import numpy as np

from olympus import Emulator
emulator = Emulator(dataset='crossed_barrel', model='BayesNeuralNet')

from numpy.random import seed
import time # for tic-toc

# plotting libraries
from numpy import arange, meshgrid, zeros
import matplotlib.pyplot as plt


# value, _, __ = emulator.run([[11.        , 89.03488752,  1.54265495,  0.72413367]])
# print(value)

def obj_fun(x):
    x_rearranged = np.roll(x, 1) ## it is because in PWAS, the variables are ordered with [continuous, integer]
    value, _, __ = emulator.run(x_rearranged)
    return -float(value[0])

Ntests = 30 #number of tests executed on the same problem

# solver related parameters
delta_E = 0.05  # trade-off hyperparameter in acquisition function between exploitation of surrogate and exploration of exploration function
acq_stage = 'multi-stage'  # can specify whether to solve the acquisition step in one or multiple stages (as noted in Section 3.4 in the paper [1]. Default: `multi-stage`
feasible_sampling = True  # can specify whether infeasible samples are allowed. Default True
K_init = 10  # number of initial PWA partitions

nc = 3
nint = 1
nd = 0
X_d = []

lb = np.array([0, 1.5, 0.7, 6])
ub = np.array([200, 2.5, 1.4, 12])

isLin_eqConstrained = False
isLin_ineqConstrained = False

preProcessingCatVar = False

nsamp = 10
maxevals = 50

plt.close('all')
plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(14, 7))

viridis = plt.cm.get_cmap('viridis', Ntests)
out_Ntests = []
cpu_time = []
for i in range(0, Ntests):
    seed(i)  # rng default for reproducibility

    # tic = time.perf_counter()
    tic = time.process_time()

    # initialize the PWAS solver
    optimizer = PWAS(obj_fun, lb, ub, delta_E, nc, nint, nd, X_d, nsamp, maxevals,  # pass fun to PWAS
                     feasible_sampling=feasible_sampling,
                     isLin_eqConstrained=isLin_eqConstrained,
                     isLin_ineqConstrained=isLin_ineqConstrained,
                     K=K_init, categorical=False,
                     acq_stage=acq_stage, integer_cut=True)
    xopt1, fopt1 = optimizer.solve()

    # toc = time.perf_counter()
    toc = time.process_time()

    out = optimizer.result_output()
    out_Ntests.append(out)
    cpu_time.append(toc-tic)
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

    plt.grid()
    thelegend.append("optimum")
    plt.legend(thelegend)

    plt.show()

else:
    import pandas as pd
    cpu_time = pd.DataFrame(cpu_time)
    export_path_3 = 'pwas_cpu.csv' #Define the path to save the results
    cpu_time.to_csv(export_path_3)
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

        file_name_s = 'PWAS_' + 'crossedBarrel'

        savemat("%s.mat" % file_name_s, res_Ntests)
