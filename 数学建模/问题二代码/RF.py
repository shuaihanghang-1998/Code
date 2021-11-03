import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import GridSearchCV

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
# 模型建立和训练
regr = RandomForestRegressor(n_estimators=200, random_state=0)
regr.fit(X_train, y_train)
# 性能测试
y_pred = regr.predict(X_test)

# mean_squared_error
print("训练集合上RMSE = {:.3f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))

# median_absolute_error

print("训练集合上median_absolute_error = {:.3f}".format(median_absolute_error(y_test, y_pred)))

# mean_absolute_error

print("训练集合上mean_absolute_error = {:.3f}".format(mean_absolute_error(y_test, y_pred)))

# mean_squared_log_error

print("训练集合上mean_squared_log_error = {:.3f}".format(mean_squared_log_error(y_test, y_pred)))

# explained_variance_score

print("训练集合上explained_variance_score = {:.3f}".format(explained_variance_score(y_test, y_pred)))

print("训练集合上R^2 = {:.3f}".format(r2_score(y_test, y_pred)))


y_pred = regr.predict(X_test)
# 画图
plt.plot(y_pred, y_test, 'o', markersize=1.0)  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
