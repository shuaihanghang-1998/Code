import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt  # 导入图形展示库
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

base = pd.read_excel('D:/data/math/selected_feature.xlsx')
X = base.iloc[1:, 2:]
y = base.iloc[1:, 1]
y = y.values.ravel()
fn = base.columns[1:]
# 归一化
X = minmax_scale(X, feature_range=[-1, 1])
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)
# 设定参数
svr = GridSearchCV(SVR(), param_grid={"C": np.logspace(-3, 3, 5),
                                      "gamma": np.logspace(-3, 3, 5)},
                   n_jobs=-1)
# 训练
svr.fit(X_train, y_train)
# 预性测试
pre_y_list = svr.predict(X_test)

# mean_squared_error
print("训练集合上RMSE = {:.3f}".format(np.sqrt(mean_squared_error(y_test, pre_y_list))))

# median_absolute_error

print("训练集合上median_absolute_error = {:.3f}".format(median_absolute_error(y_test, pre_y_list)))

# mean_absolute_error

print("训练集合上mean_absolute_error = {:.3f}".format(mean_absolute_error(y_test, pre_y_list)))

# mean_squared_log_error

print("训练集合上mean_squared_log_error = {:.3f}".format(mean_squared_log_error(y_test, pre_y_list)))

# explained_variance_score

print("训练集合上explained_variance_score = {:.3f}".format(explained_variance_score(y_test, pre_y_list)))

# 画图
pre_y_list = svr.predict(X)

plt.plot(pre_y_list, y, 'o', markersize=1.0)  # 画出每条预测结果线

#plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real value')  # y轴标题
plt.xlabel('predicted value')
plt.show()  # 展示图像
