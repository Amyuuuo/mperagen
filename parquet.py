import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet("ragen/env/sokoban/data/sokoban/test.parquet")

# 输出数据
print(df)
