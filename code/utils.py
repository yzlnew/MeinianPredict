#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
import warnings
import lightgbm as lgb
import chardet

warnings.filterwarnings("ignore")

def get_low_importance(filename):
    bst = lgb.Booster(model_file=filename)
    importance_df = pd.DataFrame()
    importance_df['feature'] = bst.feature_name()
    importance_df['importance'] = bst.feature_importance()

    low_list = importance_df[importance_df['importance']==0]['feature'].tolist()

    return low_list

def convert_mixed_num(data):
    try:
        ret = float(data)
        return ret if data >=0 else np.nan    # 保证没有负数
    except:
        all_match = re.findall(r'\d+\.?\d*', data)  # 注意：不带负号

        if all_match:
            all_list = [float(i) for i in all_match]

            return sum(all_list) / len(all_list)    # 取均值
        else:
            return np.nan

def get_encoding(file):
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']

def get_data():
    # 读取数据
    train_df = pd.read_pickle('../data/data_train.pkl')
    test_df = pd.read_pickle('../data/data_test.pkl')
    # low_importance = get_low_importance('../model/gbdt_model2018-05-03_1853_4_0.029917.txt')
    # train_df.drop(columns=low_importance, inplace=True)
    # test_df.drop(columns=low_importance, inplace=True)
    # 获取特征列表，并填充 NaN
    num_feature = train_df.describe().columns.values.tolist()[5:]
    label = train_df.describe().columns.values.tolist()[0:5]
    # cate_feature = train_df.describe(include='category').columns.values.tolist()
    cate_feature = {}
    most_num = train_df[num_feature+label].mean()
    # most_cate ={}
    # for col in cate_feature:
        # most_cate[col] = train_df[col].value_counts().index[0]

    X = train_df.loc[:, num_feature].fillna(most_num)
    y = train_df.loc[:, label].fillna(most_num)
    # X, y = shuffle(X, y, random_state=0)
    X_test = test_df.loc[:, num_feature].fillna(most_num)
    
    # X[cate_feature] = train_df.loc[:, cate_feature].fillna(most_cate)
    # X_test[cate_feature] = test_df.loc[:, cate_feature].fillna(most_cate)
    test_vid = test_df['vid']
    return X, y, X_test, num_feature, cate_feature, label, test_vid

if __name__ == '__main__':
    # print(get_low_importance('../model/gbdt_model2018-05-03_1853_4_0.029917.txt'))
    # file = '../data/meinian_round1_train_20180408.csv'
    # print(get_encoding(file))
    pass