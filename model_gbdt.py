#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss
from sklearn.model_selection import (ShuffleSplit, cross_val_score,
                                     train_test_split)

warnings.filterwarnings("ignore")

# 读取数据
train_df = pd.read_csv('data_train.csv', low_memory=False, index_col=0)
test_df = pd.read_csv('data_test.csv', low_memory=False, index_col=0)

# 获取数值特征列表，并填充 NaN
feature = train_df.describe().columns.values.tolist()[5:]
label = train_df.describe().columns.values.tolist()[0:5]
to_fill = train_df.median()
X_train = train_df.loc[:, feature].fillna(to_fill)
Y_train = train_df.loc[:, label].fillna(to_fill)
X_test = test_df.loc[:, feature].fillna(to_fill)

# GBDT 参数
params = {
    'learning_rate': 0.02,
    'boosting_type': 'gbdt',
    'objective': 'rmse',
    'metric': 'rmse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}


