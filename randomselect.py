import pandas as pd
import numpy as np

# 读取输入文件
input_file = "/home/chenzihong/doc/noisy-contrastive/save/20241211-060951/best_model_predictions.csv"
data = pd.read_csv(input_file)

# 确保与原文件行数一致
num_rows = len(data)

# 生成新的第一列和第二列
second_column = np.random.randint(0, 50, size=num_rows)

# 替换原数据
data.iloc[:, 1] = second_column

# 保存到新文件
output_file = "/home/chenzihong/doc/noisy-contrastive/output.csv"
data.to_csv(output_file, index=False)

print(f"已生成文件：{output_file}")
