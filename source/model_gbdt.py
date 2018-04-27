#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss, mean_squared_log_error
from sklearn.model_selection import (ShuffleSplit, cross_val_score,
                                     train_test_split)
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

warnings.filterwarnings("ignore")


def lgbm(x, y, params):
    """generate gbdt

    :x: train feature
    :y: train label
    :params: train parameters
    :returns: gbm

    """
    lgb_train = lgb.Dataset(x, y)
    print('Start training...')
    start = time.time()
    gbm = lgb.train(params, lgb_train, num_boost_round=1000)
    print('Finished. %s s used' % round(time.time() - start, 2))

    return gbm


def get_data():
    # 读取数据
    train_df = pd.read_pickle('../data/data_train.pkl')
    test_df = pd.read_pickle('../data/data_test.pkl')

    # 获取数值特征列表，并填充 NaN
    feature = train_df.describe().columns.values.tolist()[5:]
    label = train_df.describe().columns.values.tolist()[0:5]
    to_fill = train_df.median()
    X = train_df.loc[:, feature].fillna(to_fill)
    y = train_df.loc[:, label].fillna(to_fill)
    X_test = test_df.loc[:, feature].fillna(to_fill)
    test_vid = test_df['vid']

    return X, y, X_test, feature, label, test_vid


def set_params_space():
    space = [
        Real(0.5, 0.7, name='sub_feature'),
        Integer(60, 70, name='num_leaves'),
        Integer(110, 130, name='min_data')
    ]
    return space


def objective(values):
    # GBDT 参数
    params = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective': 'rmse',
        'metric': 'rmse',
        'sub_feature': values[0],
        'num_leaves': values[1],
        'min_data': values[2],
        'min_hessian': 1,
        'verbose': -1,
    }
    print('params: ')
    print(params)

    rmse = []
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=0)
    for i in range(5):
        # X_train, y_train = X, y
        gbm = lgbm(X_train, y_train.iloc[:, i], params)
        y_pred = gbm.predict(X_eval, num_iteration=gbm.best_iteration)
        rmse.append(mean_squared_log_error(
            y_eval.iloc[:, i], np.ndarray.round(y_pred, 3)))
        # print('rmse: ')
        # print(sum(rmse) / len(rmse))
    print('RMSE..... %s' % (sum(rmse) / len(rmse)))
    gc.collect()

    return sum(rmse) / len(rmse)


def find_best_params(space):
    res_gp = gp_minimize(objective, space, n_calls=20,
                         random_state=0)

    print('Best score=%.4f' % res_gp.fun)
    print(res_gp.x)
    # [0.5163199844704593, 60, 110]

def get_best_params():
    params = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective': 'rmse',
        'metric': 'rmse',
        'sub_feature': 0.52,
        'num_leaves': 60,
        'min_data': 110,
        'min_hessian': 1,
        'verbose': -1,
        'bagging_fraction': 0.85,
        'bagging_freq': 50
    }
    return params

def main():
    X, y, X_test, feature, label, test_vid = get_data()
    params = get_best_params()
    Y_pred_df = pd.DataFrame()
    print('Total Feature: %s' %(len(feature)))
    for i in range(len(label)):
        gbm = lgbm(X, y.iloc[:, i], params)
        y_pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        Y_pred_df[label[i]] = y_pred_test

    Y_pred_df['vid'] = test_vid
    Y_pred_gbdt_df = Y_pred_df.loc[:, ['vid'] + label]
    Y_pred_gbdt_df = Y_pred_gbdt_df.round(3)
    Y_pred_gbdt_df.to_csv('../data/gbdt_output_tuned.csv', index=False, header=False)

def cvtest():
    values = [0.52, 60, 110]
    objective(values)

def log1p_test():
    X, y, X_test, feature, label, test_vid = get_data()
    params = get_best_params()
    Y_pred_df = pd.DataFrame()
    y.iloc[:,1:5] = np.log1p(y.iloc[:,1:5])
    print('Total Feature: %s' %(len(feature)))

    rmse = []
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=80)
    gbm_store = []
    for i in [0]:
        gbm = lgbm(X_train, y_train.iloc[:, i], params)
        gbm_store.append(gbm)
        y_pred = gbm.predict(X_eval, num_iteration=gbm.best_iteration)
        y_pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        Y_pred_df[label[i]] = y_pred_test

        rmse.append(mean_squared_log_error(
            y_eval.iloc[:, i], y_pred))

    for i in [1,2,3,4]:
        gbm = lgbm(X_train, y_train.iloc[:, i], params)
        gbm_store.append(gbm)
        y_pred = gbm.predict(X_eval, num_iteration=gbm.best_iteration)
        y_pred = np.expm1(y_pred)
        y_pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        Y_pred_df[label[i]] = np.expm1(y_pred_test)

        rmse.append(mean_squared_log_error(
            np.expm1(y_eval.iloc[:, i]), y_pred))

    # for i in [4]:
    #     gbm = lgbm(X_train, y_train.iloc[:, i], params)
    #     y_pred = gbm.predict(X_eval, num_iteration=gbm.best_iteration)
    #     y_pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    #     Y_pred_df[label[i]] = y_pred_test

    #     rmse.append(mean_squared_log_error(
    #         y_eval.iloc[:, i], y_pred))

    print(rmse)
    score = (sum(rmse) / len(rmse))
    print('RMSE..... %s' % score)
    score = round(score, 6)
    for i, gbm in enumerate(gbm_store):
        gbm.save_model('../model/gbdt_model'+str(i)+'_'+str(score)+'.txt')
    Y_pred_df['vid'] = test_vid
    Y_pred_gbdt_df = Y_pred_df.loc[:, ['vid'] + label]
    # Y_pred_gbdt_df = Y_pred_gbdt_df.round(3)
    Y_pred_gbdt_df.to_csv('../data/gbdt/gbdt_output_log1p_'+str(score)+'.csv', index=False, header=False)

if __name__ == '__main__':
    # X, y, X_test, feature, label, test_vid = get_data()
    # cvtest()
    # main()
    log1p_test()
