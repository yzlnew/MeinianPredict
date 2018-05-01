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
        return np.nan
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

def convert_0440(data):
    if data == data:
        if data.isdigit():
            return float(data) - 86
        elif re.search(r'(无|未)',data):
            return 0
        elif re.search(r'有',data):
            return 1
    return np.nan

def convert_0429(data):
    if data == data:
        if re.search(r'(无|未见)',data):
            return 0
        elif re.search(r'减',data):
            return 1
        elif re.search(r'(粗糙|鸣|干)',data):
            return 2
    return np.nan

def convert_0435(data):
    if data == data:
        if re.search(r'(未见|软|正常)',data):
            return 0
        elif re.search(r'肠鸣',data):
            return 1
        elif re.search(r'不满意',data):
            return 2
        elif re.search(r'痛',data):
            return 3
    return np.nan

def convert_0436(data):
    if data == data:
        if re.search(r'(未|无)',data):
            return 0
        elif re.search(r'青霉素',data):
            return 1
        elif re.search(r'磺胺',data):
            return 2
        elif re.search(r'(cm|CM)',data):
            return 3
        elif re.search(r'过敏',data):
            return 4
    return np.nan

def convert_0216(data):
    if data == data:
        if re.search(r'(正常|未见)',data):
            return 0
        elif re.search(r'悬雍垂肥大',data):
            return 2
        elif re.search(r'(充血|水肿)',data):
            return 1
        elif re.search(r'过长',data):
            return 3
        elif re.search(r'切除',data):
            return 4
    return np.nan

def convert_0124(data):
    if data == data:
        if re.search(r'\d',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0901(data):
    if data == data:
        if re.search(r'白癜风',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0973(data):
    if data == data:
        if re.search(r'(无|未|弃)',data):
            return 0
        elif re.search(r'已手术',data):
            return 1
        elif re.search(r'疝',data):
            return 2
    return np.nan

def convert_0974(data):
    if data == data:
        if re.search(r'(无|弃|未)',data):
            return 0
        elif re.search(r'疹',data):
            return 1
        elif re.search(r'癣',data):
            return 2
        elif re.search(r'皮炎',data):
            return 3    
    return np.nan

for df in combine:
    type = 'float'
    df['0124'] = df['0124'].apply(convert_0124).astype(type)
    df['0216'] = df['0216'].apply(convert_0216).astype(type)
    df['0405'] = df['0405'].apply(converter(r'(无|未)')).astype(type)
    df['0406'] = df['0406'].apply(converter(r'(未|正常)')).astype(type)
    df['0407'] = df['0407'].apply(converter(r'(未|弃)')).astype(type)
    df['0420'] = df['0420'].apply(converter(r'(未|正常)')).astype(type)
    df['0421'] = df['0421'].apply(convert_0421).astype(type)
    df['0423'] = df['0423'].apply(convert_0423).astype(type)
    df['0426'] = df['0426'].apply(converter(r'(未|正常|无)')).astype(type)
    df['0429'] = df['0429'].apply(convert_0429).astype(type)
    df['0430'] = df['0430'].apply(converter(r'(未|正常)')).astype(type)
    df['0431'] = df['0431'].apply(converter(r'(未|无)')).astype(type)
    df['0435'] = df['0435'].apply(convert_0435).astype(type)
    df['0436'] = df['0436'].apply(convert_0436).astype(type)
    df['0440'] = df['0440'].apply(convert_0423).astype(type)
    df['0707'] = df['0707'].apply(converter(r'未见')).astype(type)
    df['0901'] = df['0901'].apply(convert_0901).astype(type)
    df['0973'] = df['0973'].apply(convert_0973).astype(type)
    df['0974'] = df['0974'].apply(convert_0974).astype(type)
    df['0976'] = df['0976'].apply(converter(r'(无|弃查)')).astype(type)

print('done!time used: %s s' %(time.time()-start))

merged_train_df.to_pickle('../data/data_train.pkl')
merged_test_df.to_pickle('../data/data_test.pkl') 