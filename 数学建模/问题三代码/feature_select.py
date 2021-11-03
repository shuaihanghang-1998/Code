# -*- coding: utf-8 -*-
# @Time    : 2021/10/14 15:49
# @Author  : Liu
# @File    : feature_select.py
# @Software: PyCharm
# @Comment :


import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_selection import mutual_info_regression as mic_reg
from sklearn.feature_selection import mutual_info_classif as mic_cla
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import matplotlib.pyplot as plt
import matplotlib


train_x_path_reg = './data/Molecular_Descriptor.xlsx'
train_x_path_cla = './data/Molecular_Descriptor_dev.xlsx'
train_y_path_reg = './data/ERα_activity.xlsx'
train_y_path_cla = './data/ADMET.xlsx'
# selected_feature_path_reg = './data/Top_20_features.xlsx'
selected_feature_path_reg = ['./data/Test_50_features_1.xlsx',
                             './data/Test_50_features_2.xlsx',
                             './data/Test_50_features_3.xlsx',
                             './data/Test_50_features_4.xlsx',
                             './data/Test_50_features_5.xlsx',
                             ]
selected_feature_path_cal = ['./data/Top_50_features_1.xlsx',
                         './data/Top_50_features_2.xlsx',
                         './data/Top_50_features_3.xlsx',
                         './data/Top_50_features_4.xlsx',
                         './data/Top_50_features_5.xlsx',
                         ]
IS_REG = True   # True表示用于回归问题 False表示用于分类问题
Top_N_Features = 30  # 选取的特征个数
ADEMT_ID = 5  # ADMET特征编号  1： Caco-2  2：CYP3A4  3：hERG  4：HOB  5：MN


def data_loader():
    if IS_REG:
        # X_df = pd.read_excel(train_x_path_reg, sheet_name='training').iloc[:, 1:]
        X_df = pd.read_excel(train_x_path_reg, sheet_name='test').iloc[:, 1:]
        y_df = pd.read_excel(train_y_path_reg, sheet_name='training').iloc[:, 2]
    else:
        X_df = pd.read_excel(train_x_path_cla, sheet_name='training').iloc[:, 1:]
        y_df = pd.read_excel(train_y_path_cla, sheet_name='training').iloc[:, ADEMT_ID]
    X = X_df  # DataFrame [1974 rows x 729 columns]
    y = y_df  # Series Length: 1974
    return X, y


def pearson_feature_selector(x, y):
    data = pd.concat([y, x], axis=1)
    if IS_REG:
        score = data.corr(method='pearson').abs().iloc[1:, 0].fillna(value=0).values
    else:
        score = data.corr(method='pearson').abs().iloc[1:, 0].fillna(value=0).values
    score = normalize(score[:, np.newaxis], axis=0, norm='max').ravel()
    # np.set_printoptions(suppress=False, precision=5)
    print('-------------------------------per-------------------------------')
    print(len(score), type(score))
    return score


def mic_feature_selector(x, y):
    stand = StandardScaler()
    x = stand.fit_transform(x)
    if IS_REG:
        score = mic_reg(x, y)
    else:
        score = mic_cla(x, y)
    score = normalize(score[:, np.newaxis], axis=0, norm='max').ravel()
    print('-------------------------------mic-------------------------------')
    print(len(score), type(score))
    return score


def rdf_feature_selector(x, y):
    if IS_REG:
        rf = RandomForestRegressor()
    else:
        rf = RandomForestClassifier()
    rf_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('standardize', StandardScaler()), ('rf', rf)])
    rf.fit(x, y)
    rf = rf_pipe.__getitem__('rf')
    score = rf.feature_importances_
    score = normalize(score[:, np.newaxis], axis=0, norm='max').ravel()
    print('-------------------------------rdf-------------------------------')
    print(len(score), type(score))
    return score


def xgboost_feature_selector(x, y):
    if IS_REG:
        model = XGBRegressor()
    else:
        model = XGBClassifier(use_label_encoder=False)
    model.fit(x, y)
    score = model.feature_importances_
    score = normalize(score[:, np.newaxis], axis=0, norm='max').ravel()
    # np.set_printoptions(suppress=False, precision=5)
    print('-------------------------------xgb-------------------------------')
    print(len(score), type(score))
    return score


def save_excel(feature_name, x, n):
    df = x[feature_name]
    wb = Workbook()
    ws = wb.active
    # ws.title = 'training'
    ws.title = 'testing'
    for x in dataframe_to_rows(df):
        ws.append(x)
    if IS_REG:
        # wb.save(selected_feature_path_reg)
        wb.save(selected_feature_path_reg[n])
    else:
        wb.save(selected_feature_path_cal[ADEMT_ID - 1])
    # print(ws)


if __name__ == '__main__':
    # 导入数据
    x, y = data_loader()

    feature_name_20 = ['MDEC-23','C1SP2','nC','LipoaffinityIndex','minsOH','minHsOH','maxHsOH','maxsOH','SsOH',
                    'BCUTc-1l','MLogP','SHsOH','SaaCH','BCUTp-1h','MLFER_A','maxssO','BCUTc-1h','hmin','SwHBa','minsssN']
    feature_name_1 =[['WPATH','ETA_Eta_R_L', 'VP-0', 'ATSm2','WTPT-1',    'SP-1',    'ECCEN',
                      'ETA_Alpha',    'SP-2',    'SP-3',    'ATSm3',    'nHeavyAtom',    'MLFER_L',    'nBonds',
                      'MW',    'Zagreb',    'CrippenMR',    'AMR',    'SP-0',    'VAdjMat',    'ETA_Beta_s',
                      'VABC',    'ETA_Eta_F_L',    'SP-4',    'SP-5',    'ETA_Beta',    'sumI',    'apol',
                      'McGowan_Volume',
                      'Kier2'],
                     ['SP-4',    'SP-6',    'VP-7',    'ETA_Eta_L',    'SP-3',    'VP-4',    'VP-1',
                      'VP-3',    'Zagreb',    'SP-1',    'VP-0',    'ETA_Alpha',    'VP-2',    'WTPT-1',    'ETA_Eta',
                      'apol',    'VAdjMat',    'ETA_Beta_s',    'SP-2',    'ETA_Eta_R_L',    'nBonds2',    'nBonds',
                      'SP-0',    'ATSp3',    'SP-7',    'SP-5',    'ATSp1',    'CrippenMR',    'McGowan_Volume',
                      'ETA_Eta_R'],
                     ['VP-0',    'ECCEN',    'McGowan_Volume',    'bpol',    'CrippenMR',    'SP-0',
                      'VP-1',    'SP-1',    'ETA_Alpha',    'VAdjMat',    'WTPT-1',    'apol',    'ETA_Eta_R_L',    'AMR',
                      'Kier2',    'Kier1',    'MW',    'fragC',    'nAtom',    'ETA_Beta_s',    'VABC',    'nBondsS',
                      'nH',    'ATSm2',    'minsssN',    'nBonds',    'nBonds2',    'nHeavyAtom',    'ETA_Eta_L',
                      'SP-3'],
                     ['nHsOH',    'nsOH',    'WTPT-5',    'VP-5',    'WTPT-4',    'SHBd',    'maxHBd',
                      'BCUTp-1l',    'MDEC-22',    'WTPT-1',    'fragC',    'minaasC',    'VCH-6',    'SC-5',    'SP-2',
                      'VABC',    'ETA_dEpsilon_D',    'SCH-7',    'MDEC-33',    'ETA_Epsilon_5',    'nHBAcc',    'minssO',
                      'SP-5',    'maxHBint10',    'VP-4',    'SHBint10',    'ETA_Epsilon_4',    'maxdO',    'minHBint10',
                      'apol'],
                     [
                         'WTPT-3',    'ETA_BetaP_s',    'WTPT-5',    'ETA_Epsilon_1',    'TopoPSA',    'ETA_Epsilon_4',    'nHBAcc_Lipinski',
                         'ETA_dEpsilon_C',    'MDEC-24',    'ETA_dEpsilon_A',    'ETA_Epsilon_2',    'FMF',    'MLFER_S',    'ETA_EtaP_L',
                         'ETA_BetaP',    'MLFER_E',    'nBondsD',    'minHBa',    'ETA_dEpsilon_B',    'ETA_EtaP_F_L',    'ETA_dPsi_A',
                         'ETA_Psi_1',    'ETA_Eta_F_L',    'ETA_EtaP_F',    'XLogP',    'ETA_Epsilon_5',    'ETA_BetaP_ns',  'maxHother', 'ETA_Shape_Y',
                         'hmax']
                     ]
    # 保存excel
    # save_excel(feature_name, x)
    for n, fn in enumerate(feature_name_1):
        save_excel(fn + feature_name_20, x, n)
    # 输出、绘图
    # label_list = []
    # score_mat = []
    # for i in range(Top_N_Features):
    #     label_list.append(feature_name[i])
    #     score_line = []
    #     score_line.append(pearson_score[selected_feature_index[i]])
    #     score_line.append(mic_score[selected_feature_index[i]])
    #     score_line.append(rdf_score[selected_feature_index[i]])
    #     score_line.append(xgboost_score[selected_feature_index[i]])
    #     score_mat.append(score_line)
    #     print("%2d. %-*s %f" % (i + 1, 30, label_list[i], score[selected_feature_index[i]]))
    #
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['axes.unicode_minus'] = False
    # score_mat = np.array(score_mat)  # size 20*4
    # y = range(len(score_mat[:, 0]))
    # rects1 = plt.barh(y=y, width=score_mat[:, 0], height=0.45, alpha=0.8, color='yellow', label="pearson")
    # rects2 = plt.barh(y=y, width=score_mat[:, 1], height=0.45, color='green', label="MIC", left=score_mat[:, 0])
    # rects3 = plt.barh(y=y, width=score_mat[:, 2], height=0.45, color='blue', label="RDF", left=score_mat[:, 0]+score_mat[:, 1])
    # rects4 = plt.barh(y=y, width=score_mat[:, 3], height=0.45, color='red', label="XGBoost", left=score_mat[:, 0]+score_mat[:, 1]+score_mat[:, 2])
    #
    # plt.xlim(0, 4)
    # plt.xlabel("score")
    # plt.yticks(y, label_list, fontsize=12)
    # plt.ylabel("feature")
    # plt.title("Top{} feature".format(Top_N_Features))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

