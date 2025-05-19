import pandas as pd

# 读取 txt 文件为 DataFrame
path = r"E:\CYN\global_surface_temperature.txt"
df = pd.read_fwf(path,skiprows=7)

# 显示前几行内容
print(df.head())
