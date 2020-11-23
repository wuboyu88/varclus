from varclus import VarClus
import pandas as pd
import numpy as np
import math

demo_df = pd.read_csv('data/winequality-red.csv')

# /*原始方法计算变量与第一个主成分的相关系数*/
# 只考虑cluster0中对应的三个变量
cluster0_feat_list = ['fixed acidity', 'density', 'pH']
df = demo_df[cluster0_feat_list]

# 标准化数据
stand_df = (df - df.mean()) / df.std()

# 特征值和特征向量
eigvals, eigvecs, corr_df, _ = VarClus.correig(df, cluster0_feat_list, n_pcs=2)

# 主成分转化过的数据
princomps = np.dot(stand_df.values, eigvecs)

# 变量fixed acidity的原始数据
x = demo_df['fixed acidity'].values

# 第一个主成分
y = princomps[:, 0]

# 变量fixed acidity与第一个主成分直接的相关系数
corr1 = np.corrcoef(x, y)[0][1]

# /*代码中计算变量与第一个主成分的相关系数*/
comb_sigma = math.sqrt(eigvals[0])
comb_cov = np.dot(eigvecs[:, 0], corr_df['fixed acidity'].values.T)
corr2 = comb_cov / comb_sigma

# 代码中的方法数学原理目前尚不知，但是经过与原始方法对比，结果一致
print(corr1)
print(corr2)
