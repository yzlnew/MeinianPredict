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

merged_train_df = pd.read_pickle('../data/data_train_num.pkl')
merged_test_df = pd.read_pickle('../data/data_test_num.pkl')
combine = [merged_train_df, merged_test_df]

numerical_feature = merged_train_df.describe().columns.values
non_numerical_feature = merged_train_df.describe(include='O').columns.values[1:]

print('non numerical feature count: %s' % len(non_numerical_feature))
start = time.time()
print('dealing non numerical features...')

# 去掉前后空白
for col in non_numerical_feature:
    merged_train_df.loc[:, col] = merged_train_df.loc[:, col].str.strip()
    merged_test_df.loc[:, col] = merged_test_df.loc[:, col].str.strip()

# 二分类转换器：0,1
def converter(pat):
    def convert(data):
        if data==data:
            if re.search(pat, data):
                return 0
            else:
                return 1
        return data
    return convert

def convert_0421(data):
    if data == data:
        normal = ['整齐','齐','正常','整','整齐;整齐','齐;齐','未见异常']
        if data in normal:
            return 0
        elif re.search(r'早搏',data):
            return 1
        elif re.search(r'(不齐|过|窦性)',data):
            return 2
        elif re.search(r'房颤',data):
            return 3
        elif re.search(r'齐',data):
            return 0
    return np.nan

def convert_0423(data):
    if data == data:
        if re.search(r'(正常|未见)',data):
            return 0
        elif re.search(r'粗',data):
            return 1
        elif re.search(r'(清|鸣)',data):
            return 2
        elif re.search(r'弱',data):
            return 3
        elif re.search(r'齐',data):
            return 0
    return np.nan

for df in combine:
    df['0405'] = df['0405'].apply(converter(r'(无|未)')).astype('category')
    df['0406'] = df['0406'].apply(converter(r'(未|正常)')).astype('category')
    df['0407'] = df['0407'].apply(converter(r'(未|弃)')).astype('category')
    df['0420'] = df['0420'].apply(converter(r'(未|正常)')).astype('category')
    df['0421'] = df['0421'].apply(convert_0421).astype('category')
    df['0423'] = df['0423'].apply(convert_0423).astype('category')

print('done!time used: %s s' %(time.time()-start))

merged_train_df.to_pickle('../data/data_train.pkl')
merged_test_df.to_pickle('../data/data_test.pkl') 