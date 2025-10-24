import pandas as pd
import numpy as np
import geopandas as gpd

edge_path = r"F:\Research\AMFEdge\Edge\Amazon_Edge_Effect_2023.csv"
df = pd.read_csv(edge_path)

pos_grids = df['nirv_magnitude'] > 0
neg_grids = df['nirv_magnitude'] < 0

print(list(df['Id'][pos_grids]))
print(list(df['Id'][neg_grids]))