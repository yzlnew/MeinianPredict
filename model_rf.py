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

# 交叉验证设定
cv = ShuffleSplit(n_splits=2, test_size=0.3, random_state=0)

# 随机森林
start_time = time.time()
print('随机森林开始运行')
random_forest = RandomForestRegressor(n_estimators=200)
random_forest.fit(X_train, Y_train)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
end_time = time.time()
print('随机森林结束运行，模型评分：')
print(acc_random_forest)
print('交叉验证结果：')
print(cross_val_score(random_forest, X_train, Y_train, cv=cv,
                      scoring='neg_mean_squared_log_error'))
