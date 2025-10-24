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
import matplotlib.colors as mcolors
import seaborn as sns
import GlobVars as gv

y = 'rh98_scale'

path = r"F:\Research\AMFEdge\Model\Amazon_Edge_Attribution.csv"
df = pd.read_csv(path)
# df = df[~df['Id'].isin(gv.ANDES_IDS)]
df = df[df['rh98_scale'] <= 6000]
df['class'] = df['Id'].isin(gv.NEGRID_IDS).map({True: 1, False: 2})

# Mann–Whitney U Test.
data1 = df.loc[df['class']==1, y].dropna()
data2 = df.loc[df['class']==2, y].dropna()
stat, p = stats.mannwhitneyu(data1, data2)

print(f'Stat={stat}, p={p}')