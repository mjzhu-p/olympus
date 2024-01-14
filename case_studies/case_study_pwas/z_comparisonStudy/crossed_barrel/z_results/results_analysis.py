import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_trace_mean(traces, obj_num=None, ax=None, color=None, label=None, use_std_err=True):
    if ax is None:
        fig, ax = plt.subplots()
    mean = np.mean(traces, axis=0)
    if use_std_err is True:
        stde = np.std(traces, axis=0, ddof=1) / np.sqrt(np.shape(traces)[0] - 1)
    else:
        stde = np.nanstd(traces, axis=0, ddof=1)

    x = np.arange(1, len(mean) + 1, 1)

    ax.plot(x, mean, color=color, linewidth=5, zorder=11)
    ax.plot(x, mean, color=color, linewidth=4, label=label, zorder=11)

    ax.fill_between(x, y1=mean - 1.96 * stde, y2=mean + 1.96 * stde, alpha=0.2, color=color, zorder=10)
    ax.plot(x, mean - 1.96 * stde, color=color, linewidth=1, alpha=0.5, zorder=10)
    ax.plot(x, mean + 1.96 * stde, color=color, linewidth=1, alpha=0.5, zorder=10)


def get_raw_traces(data, goal='maximize'):
    traces = []
    for i in range(30):
        traces.append(np.squeeze(np.maximum.accumulate(data[i].values)))

    return np.array(traces)

# %%
matrix_excel_file = 'results_pwas.xlsx'
sheet_name_pwas = 'pwas'
pwas_df = pd.read_excel(matrix_excel_file, sheet_name = sheet_name_pwas).iloc[0:, 1:]

raw_traces_pwas = np.array(pwas_df.cummax(axis=1))

# %%

# load the results
res_other_solvers = pickle.load(open('results_others.pkl', 'rb'))
# Group every 30 entries into a new list
grouped_lists = []
for i in range(0, len(res_other_solvers), 30):
    group = res_other_solvers[i:i + 30]
    grouped_lists.append(group)

res_random = grouped_lists[0]
res_hyperopt = grouped_lists[1]
res_deap = grouped_lists[2]
res_botorch = grouped_lists[3]

raw_traces_random = get_raw_traces(res_random)
raw_traces_deap = get_raw_traces(res_deap)
raw_traces_hyperopt = get_raw_traces(res_hyperopt)
raw_traces_botorch = get_raw_traces(res_botorch)



# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

plot_trace_mean(raw_traces_random, use_std_err=True, label='Random', ax=ax,color='#0e4581')
plot_trace_mean(raw_traces_deap, use_std_err=True, label='Genetic', ax=ax, color='#EB0789')
plot_trace_mean(raw_traces_hyperopt, use_std_err=True, label='Hyperopt', ax=ax, color="#75BBE1")
plot_trace_mean(raw_traces_botorch, use_std_err=True, label='Botorch', ax=ax, color="#F75BB6")
plot_trace_mean(raw_traces_pwas, use_std_err=True, label='PWAS', ax=ax, color="#4CAF50")


ax.axvline(x=10, color='grey', linestyle='--', label='initial samples')
ax.legend(loc='lower right')
ax.set_yticks(range(9, 38, 3))
ax.set_ylim(9, 38)
ax.set_ylabel('best toughness achieved (J)', fontsize=14)
ax.set_xlabel('# evaluations', fontsize=14)

ax.grid(linestyle=":")
plt.tight_layout()
plt.savefig('toughness_trace_mean_crossedBarrel.png', dpi=400)

