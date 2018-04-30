#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from sklearn import preprocessing

warnings.filterwarnings("ignore")

data = pd.read_csv('../data/data_keep_50000.csv', low_memory=False)
train_df = pd.read_csv('../data/meinian_round1_train_20180408.csv')
test_df = pd.read_csv('../data/meinian_round1_test_a_20180409.csv')

# 通过 describe 查看特征的类型，标准差，修正类型和异常数据
train_df['收缩压'] = pd.to_numeric(train_df['收缩压'], errors='coerce')
train_df['舒张压'] = pd.to_numeric(train_df['舒张压'], errors='coerce')
train_df.loc[22712, '血清甘油三酯'] = 7.75
train_df['血清甘油三酯'] = pd.to_numeric(train_df['血清甘油三酯'], errors='coerce')
# train_df['舒张压'].sort_values(ascending=False)[:5]
train_df.loc[22357, '舒张压'] = np.nan
train_df.loc[29394, '舒张压'] = np.nan
train_df.loc[29394, '收缩压'] = np.nan
train_df['血清低密度脂蛋白'][train_df['血清低密度脂蛋白']<0] = 0

# 合并数据
merged_train_df = pd.merge(train_df, data, on='vid', sort=False)
merged_test_df = pd.merge(test_df, data, on='vid', sort=False)
combine = [merged_train_df, merged_test_df]


# 提取数值特征
def get_num_prop(data_col):
    num_counts = data_col.astype(
        str).str.match(r'^(-?\d+)(\.\d+)?$').sum()
    na_counts = data_col.isna().sum()

    return num_counts / (data_col.shape[0] - na_counts)


numerical_feature = []

for col in merged_train_df.columns.values:
    if get_num_prop(merged_train_df[col]) > 0.5:
        numerical_feature.append(col)

numerical_feature = numerical_feature[5:]
print('numerical feature count: %s' % len(numerical_feature))
start = time.time()
print('dealing numerical features...')
# print(numerical_feature)

# # 打印出所有数值特征中混合数据
# def search_non_numeric(data):
#     if not re.search(r'^(-?\d+)(\.\d+)?$', data) and data != 'nan':
#         non_numeric.append(data)

# non_numeric = []
# # applymap 会有问题，第一列会操作两次

# for col in numerical_feature:
#     non_numeric.append('----' + col + '----')
#     temp = merged_train_df[col].astype('str').apply(search_non_numeric)

# for col in numerical_feature:
#     non_numeric.append('----' + col + '----')
#     temp = merged_test_df[col].astype('str').apply(search_non_numeric)
# with open('mix_in_numeric.txt', 'w') as f:
#     for t in non_numeric:
#         f.write(t + '\n')


# 处理混合数据类型
def convert_mixed_num(data):
    data = data.strip()
    special_cases = ['未见', '阴性']
    try:
        ret = float(data)
        return ret if data >=0 else np.nan    # 保证没有负数
    except:
        if data in special_cases:
            return 0
        all_match = re.findall(r'\d+\.?\d*', data)  # 注意：不带负号

        if all_match:
            all_list = [float(i) for i in all_match]

            return sum(all_list) / len(all_list)    # 取均值
        else:
            return np.nan


def print_non_num(feature_series):
    print(feature_series[feature_series.str.match(
        r'^(-?\d+)(\.\d+)?$') == False])


# test = merged_train_df['300017'].astype(
#         'str').apply(convert_mixed_num).dropna()
# sns.distplot(test)
# plt.show()
# print_non_num(merged_test_df['1840'])
# 特殊情况
merged_train_df.loc[32230, '1850'] = 3.89    # 有个句号
merged_train_df.loc[[2527, 3027], '192'] = 16.07, 12.01
merged_train_df.loc[3163, '193'] = np.nan
merged_train_df.loc[6055, '2333'] = 5.0    # 多了小数点
merged_train_df.loc[5085, '269013']    # 未见，映射成0
merged_train_df.loc[[8551, 8840, 9072, 9309], '3193'] = '>=1.030'
merged_test_df.loc[2327, '3193'] = '>=1.030'
merged_test_df.loc[2327, '1840'] = '<=5.0'

merged_train_df.loc[21196,'2405'] = np.nan    # 异常大
merged_train_df.loc[33729,'0424'] = np.nan    # 异常小

# RF 得到的特征重要性
low_importance = ['269024', '979013', '979018', '1325', '979014', '1326']
print('Drop %s features' %(len(low_importance)))

for df in combine:
    df[numerical_feature] = df[numerical_feature].astype(
    'str').applymap(convert_mixed_num)
    # to_fill = df[numerical_feature].median()
    # df[numerical_feature] = preprocessing.robust_scale(df[numerical_feature].fillna(to_fill))
    df.drop(columns=low_importance, inplace=True)    # 去掉不重要的特征

need_log1p = ['100007', '1117', '1127', '1814', '1815', '183']
for col in need_log1p:
    for df in combine:
        df[col] = np.log1p(df[col])
# for df in combine:
#     df[numerical_feature[5:]] = df[numerical_feature[5:]].apply(
#         lambda x: pd.to_numeric(x, downcast='float', errors='coerce'))
print('done!time used: %s s' %(time.time()-start))

# 导出数据
merged_train_df.to_pickle('../data/data_train_num.pkl')
merged_test_df.to_pickle('../data/data_test_num.pkl')

# sns.heatmap(merged_train_df[label+numerical_feature].corr())
# plt.show()