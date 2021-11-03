# -*- coding: utf-8 -*-
# @Time    : 2021/10/14 15:49
# @Author  : Liu
# @File    : feature_select.py
# @Software: PyCharm
# @Comment :


import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_selection import mutual_info_regression as mic
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import matplotlib.pyplot as plt
import matplotlib


train_x_path = './data/Molecular_Descriptor.xlsx'
train_y_path = './data/ERα_activity.xlsx'
selected_feature_path = './data/selected_feature2.xlsx'


def data_loader():
    base = pd.read_excel(train_x_path, sheet_name='training')
    SMILES = base[['SMILES']]
    # print(SMILES)
    base.drop(columns='SMILES', inplace=True)

    #去掉方差为0的
    selector = VarianceThreshold()
    selector.fit(base)
    base = base[base.columns[selector.get_support(indices=True)]]

    label = pd.read_excel(train_y_path, sheet_name='training')
    label.drop(columns='SMILES', inplace=True)
    data = label.join(base)
    data.drop(columns='IC50_nM', inplace=True)
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    f_name = list(base.columns)
    return SMILES, X, y, f_name


def pearson_feature_selector(x, y):
    data = pd.concat([y, x], axis=1)
    score = data.corr(method='pearson').abs().iloc[1:, 0]
    score = score.fillna(value=0)
    score = score.values
    score = normalize(score[:, np.newaxis], axis=0, norm='max').ravel()
    np.set_printoptions(suppress=False, precision=5)
    print('-------------------------------per-------------------------------')
    print(len(score), type(score))
    print(score)
    return score


def mic_feature_selector(x, y):
    stand = StandardScaler()
    x = stand.fit_transform(x)
    score = mic(x, y)
    score = normalize(score[:, np.newaxis], axis=0, norm='max').ravel()
    # np.set_printoptions(suppress=False, precision=5)
    print('-------------------------------mic-------------------------------')
    print(len(score), type(score))
    print(score)
    return score


def rdf_feature_selector(x, y):
    rfr = RandomForestRegressor()
    rfr_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('standardize', StandardScaler()), ('rf', rfr)])
    rfr.fit(x, y)
    rf = rfr_pipe.__getitem__('rf')
    score = rf.feature_importances_
    score = normalize(score[:, np.newaxis], axis=0, norm='max').ravel()
    # np.set_printoptions(suppress=False, precision=5)
    print('-------------------------------rdf-------------------------------')
    print(len(score), type(score))
    print(score)
    return score


def xgboost_feature_selector(x, y):
    model = XGBRegressor()
    model.fit(x, y)
    score = model.feature_importances_
    score = normalize(score[:, np.newaxis], axis=0, norm='max').ravel()
    # np.set_printoptions(suppress=False, precision=5)
    print('-------------------------------xgb-------------------------------')
    print(len(score), type(score))
    print(score)
    return score


def data_slice(smiles, feature_idx, x, y):
    df = pd.concat([smiles, y], axis=1)
    for idx in feature_idx:
        df = pd.concat([df, x.iloc[:, idx]], axis=1)
    wb = Workbook()
    ws = wb.active
    ws.title = 'Top20 feature'
    for x in dataframe_to_rows(df):
        ws.append(x)
    ws.delete_cols(1)
    ws.delete_rows(2)
    wb.save(selected_feature_path)
    print(ws)


if __name__ == '__main__':
    # 导入数据
    smiles, x, y, fn = data_loader()
    selector = VarianceThreshold()
    selector.fit(x)
    x = x[x.columns[selector.get_support(indices=True)]]
    #print(x)
    # 分别计算得分
    pearson_score = pearson_feature_selector(x, y)
    mic_score = mic_feature_selector(x, y)
    rdf_score = rdf_feature_selector(x, y)
    xgboost_score = xgboost_feature_selector(x, y)
    # 加和选取Top20特征
    score = pearson_score + mic_score + rdf_score + xgboost_score

    selected_feature_index = np.argsort(-score)[:20]
    # 保存excel
    data_slice(smiles, selected_feature_index, x, y)
    # 输出、绘图
    label_list = []
    score_mat = []
    for i in reversed(range(len(selected_feature_index))):
    #for i in range(len(selected_feature_index)):
        label_list.append(fn[selected_feature_index[i]])
        score_line = []
        score_line.append(pearson_score[selected_feature_index[i]])
        score_line.append(mic_score[selected_feature_index[i]])
        score_line.append(rdf_score[selected_feature_index[i]])
        score_line.append(xgboost_score[selected_feature_index[i]])
        score_mat.append(score_line)
        #print("%2d. %-*s %f" % (i + 1, 30, label_list[i], score[selected_feature_index[i]]))

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    score_mat = np.array(score_mat)  # size 20*4
    y = range(len(score_mat[:, 0]))
    rects1 = plt.barh(y=y, width=score_mat[:, 0], height=0.45, alpha=0.8, color='red', label="pearson")
    #rects1 = plt.barh(y=y, width=score_mat[:, 0], height=0.45, alpha=0.8, color='blue', label="XGBoost")
    rects2 = plt.barh(y=y, width=score_mat[:, 1], height=0.45, color='orange', label="MIC", left=score_mat[:, 0])
    rects3 = plt.barh(y=y, width=score_mat[:, 2], height=0.45, color='green', label="RDF", left=score_mat[:, 0]+score_mat[:, 1])
    rects4 = plt.barh(y=y, width=score_mat[:, 3], height=0.45, color='blue', label="XGBoost", left=score_mat[:, 0]+score_mat[:, 1]+score_mat[:, 2])

    plt.xlim(0, 4)
    #plt.xlim(0, 1)
    plt.xlabel("score")
    plt.yticks(y, label_list, fontsize=12)
    plt.ylabel("feature")
    plt.title("Top20 feature")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("save.png", dpi=300, bbox_inches="tight")

