import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

path = r"F:\Research\AMFEdge\Edge\Main\anoVI_Amazon_Edge_Effect_2023.csv"
df = pd.read_csv(path)
df = df[df['nirv_scale'] <= 6000]

print(df)
