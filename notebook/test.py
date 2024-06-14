# -*- coding: utf-8 -*-

from causalml.metrics.visualize import auuc_score, get_cumlift
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# 随便乱模拟的数据
n = 1000
T = [1, 0] * n
tau = []
for i in [10, 8, 6, 4, 2]:
    tau.extend(np.random.normal(i, 2, 400))
Y = [20, 0]*200 + [4, 0]*200 + [0, 0]*200 + [-2, 0]*200 + [-3, 0] * 200
df = pd.DataFrame([Y, T, tau]).T
df.columns=['y','w','ite']
print(df.head())

# uplift曲线与auuc
lift = get_cumlift(df)
gain = lift.mul(lift.index.values, axis=0)
gain.plot()
gain = gain.div(np.abs(gain.iloc[-1, :]))  #纵坐标归一化缩放
gain.plot()
plt.show()

print(auuc_score(df))