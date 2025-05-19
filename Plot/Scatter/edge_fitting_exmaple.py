import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import matplotlib as mpl
plt.ion()
import matplotlib.ticker as ticker
import seaborn as sns

LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)

def cm2inch(value):
    return value/2.54

# Exponential Function
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

popt_path = r"F:\Research\AMFEdge\Edge\Amazon_UndistEdge_Effect_2023.csv"
popt_df = pd.read_csv(popt_path)

dst_var = 'nirv'
dst_id = 776
popt = popt_df.loc[popt_df['Id'] == dst_id, [dst_var+'_para1', dst_var+'_para2', dst_var+'_para3']].values[0]

edge_path = r"F:\Research\AMFEdge\Edge\anoVI_Amazon_UndistEdge_2023.csv"
edge_df = pd.read_csv(edge_path)
edge_df = edge_df.loc[(edge_df['Id'] == dst_id)&(edge_df['Dist'] <= 6000), ['Dist', 'NIRv_mean']].reset_index(drop=True)

# Plot.
fig, ax = plt.subplots(
    nrows=1, ncols=1, 
    )
fig.set_size_inches(cm2inch(8), cm2inch(5))
ax.plot(edge_df['Dist'][1:], edge_df['NIRv_mean'][1:], 'o', color='#3d98c2', markersize=3, label='Observed')
ax.plot(edge_df['Dist'], func(edge_df['Dist'], *popt), '-', color='#ed3e2e', label='Fitted curve')
ax.set_xlabel('Distance (m)', fontsize=LABEL_SIZE)
ax.set_ylabel('$\Delta$NIRv', fontsize=LABEL_SIZE)

fig.subplots_adjust(bottom=0.2, top=0.95, left=0.2, right=0.9, hspace=0.55, wspace=0.22)
plt.legend(loc='best', frameon=False, prop={'size':10}, ncol=1)

outpath = r"E:\Thesis\AMFEdge\Figures\Description\edge_fitting_example.pdf"
fig.savefig(outpath, dpi=600)