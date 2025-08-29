import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from scipy.stats import linregress

import numpy as np
import pandas as pd

import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.ticker as ticker
import seaborn as sns
import GlobVars as gv

LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)

def cm2inch(value):
    return value/2.54

fig, axes = plt.subplots(
    nrows=1, ncols=1, 
    )
fig.set_size_inches(cm2inch(7.5), cm2inch(6))

path = r"F:\Research\AMFEdge\Model\Amazon_Attribution.csv"
df = pd.read_csv(path)

xcol, ycol, ccol = 'HAND_mean', 'length_mean', 'nirv_magnitude'

dst_df = df.loc[(~np.isnan(df[ccol]))]


axes.scatter(dst_df[xcol], dst_df[ycol], c=dst_df[ccol], cmap='RdBu_r')
# axes.plot([0, 1], [0, 1], transform=axes.transAxes, ls='--', c='.3',zorder=-1)
axes.set_xlabel(xcol, fontsize=LABEL_SIZE)
axes.set_ylabel(ycol, fontsize=LABEL_SIZE)
fig.subplots_adjust(bottom=0.2, top=0.95, left=0.18, right=0.95, hspace=0.55, wspace=0.22)

outpath = r"E:\Thesis\AMFEdge\Figures\Cause\corr_nirv2ndwi_r07.pdf"
fig.savefig(outpath, dpi=600)