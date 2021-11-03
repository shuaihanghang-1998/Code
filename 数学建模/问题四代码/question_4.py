
from __future__ import division
import pandas as pd
import random
import math
from xgboost import XGBRegressor

# 超参数设置
population_size = 500  # 种群数量  500
chrom_length = 50  # 染色体长度
pc = 0.6  # 交配概率
pm = 0.01  # 变异概率
results = []  # 存储每一代的最优解，N个三元组（auc最高值, n_estimators, max_depth）
fit_value = []  # 个体适应度
fit_mean = []  # 平均适应度


random_seed = 50
def xgboostModel(model, MDEC, C1SP2, nC, LipoaffinityIndex, minsOH, MLogP, minHsOH, maxsOH, maxHsOH,
                 minsssN):
    data = [[MDEC, C1SP2, nC, LipoaffinityIndex, minsOH, MLogP, minHsOH, maxsOH, maxHsOH, minsssN]]
    df = pd.DataFrame(data,
                      columns=['MDEC-23', 'C1SP2', 'nC', 'LipoaffinityIndex', 'minsOH', 'MLogP', 'minHsOH', 'maxsOH',
                               'maxHsOH',
                               'minsssN'])
    pic = model.predict(df)
    return pic


def loadFile(filePath):
    fileData = pd.read_csv(filePath)
    return fileData


# Step 1 : 初始化种群基因
def geneEncoding(population_size, chrom_length):
    population = [[]]
    for i in range(population_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        population.append(temp)
    return population[1:]


# Step 2 : 计算个体的目标函数值
def cal_obj_value(model, population):
    value = []
    variable = decodechrom(population)
    for i in range(len(variable)):
        tempVar = variable[i]
        # 设置要优化的值的大致数据范围
        MDEC = tempVar[0] + 10
        C1SP2 = tempVar[1]
        nC = tempVar[2] + 10
        LipoaffinityIndex = tempVar[3] + 2
        minsOH = 0.1 * tempVar[4] + 8
        MLogP = 0.1 * tempVar[5] + 2
        minHsOH = 0.01 * tempVar[6]
        maxsOH = 0.1 * tempVar[7] + 8.7
        maxHsOH = 0.01 * tempVar[8]
        minsssN = 0.1 * tempVar[9]

        score = xgboostModel(model, MDEC, C1SP2, nC, LipoaffinityIndex, minsOH, MLogP, minHsOH, maxsOH, maxHsOH,
                                minsssN)
        value.append(score)
    return value


# 解码
def decodechrom(population):
    variable = []
    for i in range(len(population)):
        res = []

        # 计算第1个变量值，即 0101->10(逆转)
        temp1 = population[i][0:5]
        v1 = 0
        for i1 in range(5):
            v1 += temp1[i1] * (math.pow(2, i1))
        res.append(int(v1))

        # 计算第2个变量值
        temp2 = population[i][5:7]
        v2 = 0
        for i2 in range(2):
            v2 += temp2[i2] * (math.pow(2, i2))
        res.append(int(v2))

        # 计算第3个变量值
        temp3 = population[i][7:12]
        v3 = 0
        for i3 in range(5):
            v3 += temp3[i3] * (math.pow(2, i3))
        res.append(int(v3))

        # 计算第4个变量值
        temp4 = population[i][12:16]
        v4 = 0
        for i4 in range(4):
            v4 += temp4[i4] * (math.pow(2, i4))
        res.append(int(v4))

        # 计算第5个变量值
        temp5 = population[i][16:21]
        v5 = 0
        for i5 in range(5):
            v5 += temp5[i5] * (math.pow(2, i5))
        res.append(int(v5))

        # 计算第6个变量值
        temp6 = population[i][21:26]
        v6 = 0
        for i6 in range(5):
            v6 += temp6[i6] * (math.pow(2, i6))
        res.append(int(v6))

        # 计算第7个变量值
        temp7 = population[i][26:33]
        v7 = 0
        for i7 in range(7):
            v7 += temp7[i7] * (math.pow(2, i7))
        res.append(int(v7))

        # 计算第8个变量值
        temp8 = population[i][33:38]
        v8 = 0
        for i8 in range(5):
            v8 += temp8[i8] * (math.pow(2, i8))
        res.append(int(v8))

        # 计算第9个变量值
        temp9 = population[i][38:45]
        v9 = 0
        for i9 in range(7):
            v9 += temp9[i9] * (math.pow(2, i9))
        res.append(int(v9))

        # 计算第10个变量值
        temp10 = population[i][45:50]
        v10 = 0
        for i10 in range(5):
            v10 += temp10[i4] * (math.pow(2, i10))
        res.append(int(v10))

        variable.append(res)
    return variable


# Step 3: 计算个体的适应值
def calfitvalue(obj_value):
    fit_value = []
    temp = 0.0
    Cmin = 0
    for i in range(len(obj_value)):
        if (obj_value[i] + Cmin > 0):
            temp = Cmin + obj_value[i]
        else:
            temp = 0.0
        fit_value.append(temp)
    return fit_value


# Step 4: 找出最优的个体
def best(population, fit_value):
    best_individual = population[0]
    best_fit = fit_value[0]
    for i in range(1, len(population)):
        if (fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = population[i]
    return [best_individual, best_fit]


# Step 5: 记录最优个体数据
def b2d(best_individual):
    # 计算第1个变量值，即二进制转十进制
    temp1 = best_individual[0:5]
    v1 = 0
    for i1 in range(5):
        v1 += temp1[i1] * (math.pow(2, i1))
    v1 = v1 + 10

    # 计算第2个变量值
    temp2 = best_individual[5:7]
    v2 = 0
    for i2 in range(2):
        v2 += temp2[i2] * (math.pow(2, i2))
    v2 = v2

    # 计算第3个变量值
    temp3 = best_individual[7:12]
    v3 = 0
    for i3 in range(5):
        v3 += temp3[i3] * (math.pow(2, i3))
    v3 = v3 + 10

    # 计算第4个变量值
    temp4 = best_individual[12:16]
    v4 = 0
    for i4 in range(4):
        v4 += temp4[i4] * (math.pow(2, i4))
    v4 = v4 + 2

    # 计算第5个变量值
    temp5 = best_individual[16:21]
    v5 = 0
    for i5 in range(5):
        v5 += temp5[i5] * (math.pow(2, i5))
    v5 = v5 * 0.1 + 8

    # 计算第6个变量值
    temp6 = best_individual[21:26]
    v6 = 0
    for i6 in range(5):
        v6 += temp6[i6] * (math.pow(2, i6))
    v6 = v6 * 0.1 + 2

    # 计算第7个变量值
    temp7 = best_individual[26:33]
    v7 = 0
    for i7 in range(7):
        v7 += temp7[i7] * (math.pow(2, i7))
    v7 = v7 * 0.01

    # 计算第8个变量值
    temp8 = best_individual[33:38]
    v8 = 0
    for i8 in range(5):
        v8 += temp8[i8] * (math.pow(2, i8))
    v8 = v8 * 0.1 + 8.7

    # 计算第9个变量值
    temp9 = best_individual[38:45]
    v9 = 0
    for i9 in range(7):
        v9 += temp9[i9] * (math.pow(2, i9))
    v9 = v9 * 0.01

    # 计算第10个变量值
    temp10 = best_individual[45:50]
    v10 = 0
    for i10 in range(5):
        v10 += temp10[i4] * (math.pow(2, i10))
    v10 = v10 * 0.1

    return int(v1), int(v2), int(v3), int(v4), float(v5), float(v6), float(v7), float(v8), float(v9), float(v10)


# Step 6: 自然选择
def selection(population, fit_value):
    # 计算每个适应值的概率
    new_fit_value = []
    total_fit = sum(fit_value)
    for i in range(len(fit_value)):
        new_fit_value.append(fit_value[i] / total_fit)
    # 计算每个适应值的累积概率
    cumsum(new_fit_value)
    # 生成随机浮点数序列
    ms = []
    population_len = len(population)
    for i in range(population_len):
        ms.append(random.random())
    # 对生成的随机浮点数序列进行排序
    ms.sort()
    # 轮盘赌算法
    fitin = 0
    newin = 0
    newpopulation = population
    while newin < population_len:
        if (ms[newin] < new_fit_value[fitin]):
            newpopulation[newin] = population[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    population = newpopulation


# 求适应值的总和
def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total


# 计算累积概率
def cumsum(fit_value):
    temp = []
    for i in range(len(fit_value)):
        t = 0
        j = 0
        while (j <= i):
            t += fit_value[j]
            j = j + 1
        temp.append(t)
    for i in range(len(fit_value)):
        fit_value[i] = temp[i]


# Step 7: 繁殖
def crossover(population, pc):  # 个体间交配，实现基因交换
    populationlen = len(population)
    for i in range(populationlen - 1):
        if (random.random() < pc):
            cpoint = random.randint(0, len(population[0]))
            temp1 = []
            temp2 = []
            temp1.extend(population[i][0: cpoint])
            temp1.extend(population[i + 1][cpoint: len(population[i])])
            temp2.extend(population[i + 1][0: cpoint])
            temp2.extend(population[i][cpoint: len(population[i])])
            population[i] = temp1
            population[i + 1] = temp2


# Step 8: 基因突变
def mutation(population, pm):
    x = len(population)
    y = len(population[0])
    for i in range(x):
        if (random.random() < pm):
            mpoint = random.randint(0, y - 1)
            if (population[i][mpoint] == 1):
                population[i][mpoint] = 0
            else:
                population[i][mpoint] = 1

def generAlgo(model, generations):
    population = geneEncoding(population_size, chrom_length)
    for i in range(generations):
        value = cal_obj_value(model, population)  # 计算目标函数值
        fit_value = calfitvalue(value)  # 计算个体的适应值
        [best_individual, best_fit] = best(population, fit_value)  # 选出最好的个体和最好的函数值
        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 = b2d(best_individual)
        results.append([best_fit, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])  # 每次繁殖，将最好的结果记录下来
        selection(population, fit_value)  # 自然选择，淘汰掉一部分适应性低的个体
        crossover(population, pc)  # 交叉繁殖
        mutation(population, pc)  # 基因突变
    #输出最优个体信息
    results.sort()
    print(results[-1])
#主函数
if __name__ == '__main__':
    # 设定迭代次数
    gen = [10, 50, 100, 200, 300, 400, 500]
    dataset = pd.read_excel('D:/data/math/selected_feature.xlsx', index_col='SMILES')
    # 划分数据，使用前十个特征进行训练
    X = dataset.iloc[:, 1:11].values
    y = dataset.iloc[:, 0].values
    regr = XGBRegressor(max_depth=3, n_estimators=500, random_state=400, learning_rate=0.1, gamma=0.2,
                        min_child_weight=6,
                        reg_alpha=1, reg_lambda=1)
    regr.fit(X, y)
    for g in gen:
        generAlgo(regr, int(g))

