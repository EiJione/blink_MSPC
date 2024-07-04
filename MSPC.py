import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
X = pd.read_csv('wake_drozy30fps_blink_features.csv')
Y = pd.read_csv('wake_drozy_all_blink_features.csv')

# 初始化StandardScaler
scaler = StandardScaler()

# 标准化特征

# 创建足够的图形和轴来展示每个特征，注意更新行数以匹配特征数量
#fig, axes = plt.subplots(nrows=len(features), figsize=(10, 50))  # 调整大小适应特征数量

# 为每个特征绘制变化趋势
# for i, ax in enumerate(axes):
#     ax.plot(X_np[:, i], label=f'Trend of {features[i]}')
#     ax.set_title(features[i])
#     ax.set_xlabel('Sample Index')
#     ax.set_ylabel(features[i])
#     ax.legend()
#
# # 调整布局以防止标签重叠
# plt.tight_layout()
# plt.show()
# 训练归一化器并转换 X
X_normalized = scaler.fit_transform(X)

# 保存归一化器到文件
joblib.dump(scaler, 'standard_scaler.pkl')
Y_nor=scaler.transform(Y)
U, Sigma, VT = np.linalg.svd(X_normalized, full_matrices=False)

# 选择前R个主成分
R = 8
Sigma_R = np.diag(Sigma[:R])
VR = VT[:R, :]

    # 计算投影矩阵和单位矩阵
projection_matrix = VR.T.dot(VR)
I = np.eye(X_normalized.shape[1])

    # 计算T2统计量
T2 = np.array([x.dot(VR.T).dot(np.linalg.inv(Sigma_R)).dot(VR).dot(x.T) for x in X_normalized])

    # 计算Q统计量
Q = np.array([x.dot(I - projection_matrix).dot(x.T) for x in X_normalized])
T2_Y = np.array([y.dot(VR.T).dot(np.linalg.inv(Sigma_R)).dot(VR).dot(y.T) for y in Y_nor])

# 计算Y中各个特征的Q统计量
Q_Y = np.array([y.dot(I - projection_matrix).dot(y.T) for y in Y_nor])

import matplotlib.pyplot as plt

# 假设T2, Q, T2_Y, Q_Y都已经计算好并赋值
# 下面的代码将这些值绘制到同一个图形中以便比较

plt.figure(figsize=(12, 5))

# 绘制T2统计量的对比
plt.subplot(1, 2, 1)  # 1行2列的第1个位置
plt.plot(T2, label='T2 Statistic for wake', color='blue')  # X数据
plt.plot(T2_Y, label='T2 Statistic for fatigue', color='red', linestyle='--')  # Y数据
plt.title('T2 Statistic Comparison')
plt.xlabel('Sample Index')
plt.ylabel('T2 Value')
plt.legend()

# 绘制Q统计量的对比
plt.subplot(1, 2, 2)  # 1行2列的第2个位置
plt.plot(Q, label='Q Statistic for wake', color='blue')  # X数据
plt.plot(Q_Y, label='Q Statistic for fatigue', color='red', linestyle='--')  # Y数据
plt.title('Q Statistic Comparison')
plt.xlabel('Sample Index')
plt.ylabel('Q Value')
plt.legend()

plt.tight_layout()
plt.show()

# import seaborn as sns
# # 绘制T2和Q的核密度估计图
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# sns.kdeplot(T2, label='T2 KDE', fill=True)
# plt.title('Kernel Density Estimate of T2')
# plt.xlabel('T2 Value')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# sns.kdeplot(Q, label='Q KDE', fill=True)
# plt.title('Kernel Density Estimate of Q')
# plt.xlabel('Q Value')
# plt.legend()
# plt.tight_layout()
# plt.show()

n = X_normalized.shape[0]
p = R
alpha = 99  # 95% 置信度

# 计算 T^2 和 Q 的控制限
T2_limit = np.percentile(T2, alpha)
Q_limit = np.percentile(Q, alpha)

plt.figure(figsize=(12, 5))

# 绘制T2统计量和控制限
plt.subplot(1, 2, 1)
plt.plot(T2_Y, label='T2 Statistic for fatigue', color='red')
plt.axhline(y=T2_limit, color='green', linestyle='--', label=f'T2 Control Limit (α={alpha})')
plt.title('T2 Statistic for fatigue with Control Limit')
plt.xlabel('Sample Index')
plt.ylabel('T2 Value')
plt.legend()

# 绘制Q统计量和控制限
plt.subplot(1, 2, 2)
plt.plot(Q_Y, label='Q Statistic for fatigue', color='red')
plt.axhline(y=Q_limit, color='green', linestyle='--', label='Q Control Limit (95th Percentile)')
plt.title('Q Statistic for fatigue with Control Limit')
plt.xlabel('Sample Index')
plt.ylabel('Q Value')
plt.legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

# 绘制X数据集的点
plt.scatter(T2, Q, color='blue', label='X data (wake)')

# 绘制Y数据集的点
plt.scatter(T2_Y, Q_Y, color='red', label='Y data (fatigue)')

# 添加图例
plt.legend()

# 添加轴标签
plt.xlabel('T2 Statistic')
plt.ylabel('Q Statistic')

# 添加标题
plt.title('Scatter Plot of T2 and Q Statistics for X and Y Data')

# 显示图形
plt.show()
