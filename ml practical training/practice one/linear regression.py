#线性回归预测房价
import pandas as pd
from io import StringIO
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

#房屋面积与价格的历史数据
csv_data =  'square_feet,price\n150,6450\n200,7450\n250,8450\n300,9450\n350,11450\n400,15450\n600,18450\n'
#读入dataframe
df = pd.read_csv(StringIO(csv_data))
#values.rehape(-1, 1)把数组转换成一列
x = df['square_feet'].values.reshape(-1, 1)
y = df['price']

#建立线性回归模型
regr = linear_model.LinearRegression()
#拟合
regr.fit(x, y)
a, b = regr.coef_, regr.intercept_
#给出待预测的面积
area = 238.5
#方式1：根据直线方程计算价格,并输出
y_pred=a*area+b
print("预测结果:", y_pred)
#画图
#1.真实的点
plt.scatter(x, y, color='blue', label='real price')
#2.拟合的直线
plt.plot(x, regr.predict(x), color='red', linewidth=4, label='predict price')
plt.xlabel('area')
plt.ylabel('price')
plt.legend(loc='lower right')
plt.show()