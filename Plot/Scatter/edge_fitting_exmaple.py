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
dst_id = 776 # Moderately fragmented example
# dst_id = 1065 # Highly fragmented example
popt = popt_df.loc[popt_df['Id'] == dst_id, [dst_var+'_para1', dst_var+'_para2', dst_var+'_para3']].values[0]

edge_path = r"F:\Research\AMFEdge\Edge\anoVI_Amazon_UndistEdge_2023.csv"
edge_df = pd.read_csv(edge_path)
edge_df = edge_df.loc[(edge_df['Id'] == dst_id)&(edge_df['Dist'] <= 6000), :].reset_index(drop=True)
intact_df = edge_df.loc[(edge_df['Id'] == dst_id)&((edge_df['Dist'] >= 3000))&(edge_df['Dist'] <= 6000), :]
intact_value = (intact_df['NIRv_mean'] * intact_df['NIRv_count']).sum()/intact_df['NIRv_count'].sum()
edge_df['NIRv_mean'] = (edge_df['NIRv_mean'] - intact_value)*-1
edge_df['Dist'] = edge_df['Dist']/1000  # Convert to km

# Plot.
fig, ax = plt.subplots(
    nrows=1, ncols=1, 
    )
fig.set_size_inches(cm2inch(8), cm2inch(5))
scatter1, = ax.plot(edge_df['Dist'][1:], edge_df['NIRv_mean'][1:], 'o', color='#3d98c2', markersize=3, label='$\Delta$NIRv')
ax.plot(edge_df['Dist'], func(edge_df['Dist']*1000, *popt), '-', color='#ed3e2e', )
ax.set_xlabel('Distance (km)', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\nabla$NIRv$_{edge-interior}$ (%)', fontsize=LABEL_SIZE)
fig.subplots_adjust(bottom=0.2, top=0.95, left=0.2, right=0.9, hspace=0.55, wspace=0.22)

# Draw edge and interior forests.
ax.axvspan(0, 0.12, color= "#e5086b8f", alpha=0.3)
ax.axvspan(3, 6, color='#e3b87f', alpha=0.3)

# Add RH fitting curve for comparison.

# popt_path = r"F:\Research\AMFEdge\EdgeRH\Amazon_UndistEdge_Effect_2023.csv"
# popt_df = pd.read_csv(popt_path)

# dst_var = 'rh98'
# popt = popt_df.loc[popt_df['Id'] == dst_id, [dst_var+'_para1', dst_var+'_para2', dst_var+'_para3']].values[0]

# edge_path = r"F:\Research\AMFEdge\EdgeRH\RH_Amazon_UndistEdge_2023.csv"
# edge_df = pd.read_csv(edge_path)
# edge_df = edge_df.loc[(edge_df['Id'] == dst_id)&(edge_df['Dist'] <= 6000), ['Dist', 'rh98_mean']].reset_index(drop=True)
# ax2 = ax.twinx()
# scatter2, = ax2.plot(edge_df['Dist'][1:], edge_df['rh98_mean'][1:], 'x', color='#299d8f', markersize=5, label='RH98')
# ax2.plot(edge_df['Dist'], func(edge_df['Dist'], *popt), '-', color='#ed3e2e', )
# ax2.set_xlabel('Distance (km)', fontsize=LABEL_SIZE)
# ax2.set_ylabel('RH98 (m)', fontsize=LABEL_SIZE)
# fig.subplots_adjust(bottom=0.2, top=0.95, left=0.18, right=0.8, hspace=0.55, wspace=0.22)
# ax.legend(handles=[scatter1, scatter2], loc='center right', frameon=False, prop={'size':LABEL_SIZE}, ncol=1)

outpath = r"E:\Thesis\AMFEdge\Figures\Description\anoNIRv_edge_fitting_example.jpg"
fig.savefig(outpath, dpi=600)