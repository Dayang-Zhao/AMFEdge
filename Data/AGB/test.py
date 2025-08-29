import tarfile
import os

# 压缩文件路径
tgz_file = r"C:\Users\DELL\Downloads\GlobBiomass_D16_2010_global_AGB_25m_v20180531_South_America_N (1).tgz"

# 解压目录（可以自己改）
extract_path = r"F:\Research\AMFEdge\AGB\GlobBiomass"

# 确保目录存在
os.makedirs(extract_path, exist_ok=True)

# 打开并解压
with tarfile.open(tgz_file, "r:gz") as tar:
    fnames = tar.getmembers()
    # tar.extractall(path=extract_path)

print("解压完成，文件已保存到：", extract_path)