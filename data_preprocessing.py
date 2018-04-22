#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

data = pd.read_csv('data_keep_50000.csv', low_memory=False)
train_df = pd.read_csv('meinian_round1_train_20180408.csv')
test_df = pd.read_csv('meinian_round1_test_a_20180409.csv')

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

# 提取数值特征并转化成 float 类型
numerical_feature = []
train_data_counts = merged_train_df.shape[0]

for col in merged_train_df.columns.values:
    num_counts = merged_train_df[col].astype(
        str).str.match(r'^(-?\d+)(\.\d+)?$').sum()
    na_counts = merged_train_df[col].isna().sum()

    if num_counts / (train_data_counts - na_counts) > 0.8:
        numerical_feature.append(col)

for df in combine:
    df[numerical_feature[5:]] = df[numerical_feature[5:]].apply(
        lambda x: pd.to_numeric(x, downcast='float', errors='coerce'))

merged_train_df.loc[21234, '10004'] = np.nan
merged_train_df.loc[21196, '2403'] = np.nan
merged_train_df.loc[21196, '2405'] = np.nan

# 导出数据
merged_train_df.to_csv('data_train.csv')
merged_test_df.to_csv('data_test.csv')
