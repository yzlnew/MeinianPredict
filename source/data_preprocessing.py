#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
print(numerical_feature)

# 打印出所有数值特征中混合数据


def search_non_numeric(data):
    if not re.search(r'^(-?\d+)(\.\d+)?$', data) and data != 'nan':
        non_numeric.append(data)


non_numeric = []
# applymap 会有问题，第一列会操作两次

for col in numerical_feature:
    non_numeric.append('----' + col + '----')
    temp = merged_train_df[col].astype('str').apply(search_non_numeric)

for col in numerical_feature:
    non_numeric.append('----' + col + '----')
    temp = merged_test_df[col].astype('str').apply(search_non_numeric)
with open('mix_in_numeric.txt', 'w') as f:
    for t in non_numeric:
        f.write(t + '\n')


# 处理混合数据类型
def convert_mixed_num(data):
    data = data.strip()
    special_cases = ['未见', '阴性']
    try:
        ret = float(data)

        return ret
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


for df in combine:
    df[numerical_feature] = df[numerical_feature].astype(
        'str').applymap(convert_mixed_num)
# for df in combine:
#     df[numerical_feature[5:]] = df[numerical_feature[5:]].apply(
#         lambda x: pd.to_numeric(x, downcast='float', errors='coerce'))


# 导出数据
merged_train_df.to_csv('../data/data_train.csv')
merged_test_df.to_csv('../data/data_test.csv')

sns.heatmap(merged_train_df[numerical_feature].corr())
plt.show()