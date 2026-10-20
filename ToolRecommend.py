
import pandas as pd
import os
import numpy as np
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)

# 显示所有行（可选）
pd.set_option('display.max_rows', None)

# 不限制每列显示宽度
pd.set_option('display.max_colwidth', None)

# 不限制总宽度
pd.set_option('display.width', 10000)

# 不折叠DataFrame
pd.set_option('display.expand_frame_repr', False)
tool_path = r"F:\DL-Study\MachiningParameterRecommend\ToolLibrary\Milling_Cutter.xlsx"

tool_df = pd.read_excel(tool_path)

# tool={
#     "Diameter": 9.06,
#     "FluteLength": 29.37,
#     "ToolLength": 77.74,
#     "LowerRadius": 0.36,
#     "Handlediameter":9.06
# }
##endface tool
# tool={
#     "Diameter": 22.09,
#     "FluteLength": 29.61,
#     "ToolLength": 73.59,
#     "LowerRadius": 0.52,
#     "Handlediameter":22.09
# }
#Hole tool
# tool={
#     "Diameter": 5.24,
#     "FluteLength": 12.56,
#     "ToolLength": 52.12,
#     "LowerRadius":0.36,
#     "Handlediameter":5.24
# }
#pcoket tool
tool={
    "Diameter": 5.58,
    "FluteLength": 14.77,
    "ToolLength": 59.93,
    "LowerRadius":0.42,
    "Handlediameter":5.58
}

# 参与匹配的字段
features = [
    "Diameter",
    "FluteLength",
    "ToolLength",
    "LowerRadius",
    "Handlediameter"
]

# 权重
weights = np.array([
    0.30,
    0.20,
    0.15,
    0.1,
    0.25,
])

# 查询刀具
target = np.array([tool[f] for f in features])

# 刀具库矩阵
X = tool_df[features].to_numpy(dtype=float)

# 每个参数标准差
sigma = tool_df[features].std().to_numpy()

# 防止标准差为0
sigma[sigma == 0] = 1

# 加权欧氏距离
tool_df["Distance"] = np.sqrt(
    np.sum(
        weights * ((X - target) / sigma) ** 2,
        axis=1
    )
)

# 相似度
tool_df["Similarity"] = 1 / (1 + tool_df["Distance"])

# 排序
tool_df = tool_df.sort_values(
    by="Similarity",
    ascending=False
)

# 输出前5把推荐刀具
print(tool_df.head(5))

stats = tool_df[features].agg(['mean', 'std'])

print("刀具库统计信息：")
print(stats)