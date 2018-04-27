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


def lgbm(x, y, params, num_boost_round=1000):
    """generate gbdt

    :x: train feature
    :y: train label
    :params: train parameters
    :returns: gbm

    """
    lgb_train = lgb.Dataset(x, y)
    print('Start training...')
    start = time.time()
    gbm = lgb.train(params, lgb_train, num_boost_round=num_boost_round)
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


def universal_objective(values):
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


def find_best_universal_params(space):
    res_gp = gp_minimize(universal_objective, space, n_calls=20,
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
        'sub_feature': 0.5,
        'num_leaves': 60,
        'min_data': 110,
        'min_hessian': 1,
        'verbose': -1,
        'bagging_fraction': 0.85,
        'bagging_freq': 50
    }
    return params


def universal_model():
    X, y, X_test, feature, label, test_vid = get_data()
    params = get_best_params()
    y_pred_df = pd.DataFrame()
    print('Total Feature: %s' %(len(feature)))
    for i in range(len(label)):
        gbm = lgbm(X, y.iloc[:, i], params)
        y_pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        y_pred_df[label[i]] = y_pred_test

    y_pred_df['vid'] = test_vid
    Y_pred_gbdt_df = y_pred_df.loc[:, ['vid'] + label]
    Y_pred_gbdt_df = Y_pred_gbdt_df.round(3)
    Y_pred_gbdt_df.to_csv('../data/gbdt_output_tuned.csv', index=False, header=False)


def universal_cvtest():
    values = [0.52, 60, 110]
    objective(values)


def ensemble_model():
    X, y, X_test, feature, label, test_vid = get_data()

    params = get_best_params()
    y_pred_df = pd.DataFrame()
    
    print('Total Feature: %s' %(len(feature)))

    has_eval = True
    if has_eval:
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=0.2, random_state=80)
    else:
        X_train, y_train = X, y

    gbm_store = []
    rmse = [0]*5
    def eval_rmse(gbm, X_eval, y_eval, original=True):
        y_pred_eval = gbm.predict(X_eval, num_iteration=gbm.best_iteration)
        if original:
            return mean_squared_log_error(y_eval, y_pred_eval)
        else:
            return mean_squared_log_error(y_eval, np.expm1(y_pred_eval))

    is_original = [1,1,0,0,0]
    #model_0
    y_train_0 = y_train.iloc[:, 0]
    gbm_0 = lgbm(X_train, y_train_0, params)
    if has_eval:
        y_eval_0 = y_eval.iloc[:, 0]
        rmse[0] = eval_rmse(gbm_0, X_eval, y_eval_0, is_original[0])
    
    #model_1
    y_train_1 = y_train.iloc[:, 1]
    gbm_1 = lgbm(X_train, y_train_1, params)
    if has_eval:
        y_eval_1 = y_eval.iloc[:, 1]
        rmse[1] = eval_rmse(gbm_1, X_eval, y_eval_1, is_original[1])

    #model_2
    y_train_2 = np.log1p(y_train.iloc[:, 2])
    gbm_2 = lgbm(X_train, y_train_2, params)
    if has_eval:
        y_eval_2 = y_eval.iloc[:, 2]
        rmse[2] = eval_rmse(gbm_2, X_eval, y_eval_2, is_original[2])

    #model_3
    y_train_3 = np.log1p(y_train.iloc[:, 3])
    gbm_3 = lgbm(X_train, y_train_3, params)
    if has_eval:
        y_eval_3 = y_eval.iloc[:, 3]
        rmse[3] = eval_rmse(gbm_3, X_eval, y_eval_3, is_original[3])    

    #model_4
    y_train_4 = np.log1p(y_train.iloc[:, 4])
    gbm_4 = lgbm(X_train, y_train_4, params)
    if has_eval:
        y_eval_4 = y_eval.iloc[:, 4]
        rmse[4] = eval_rmse(gbm_4, X_eval, y_eval_4, is_original[4])

    print(rmse)
    score = (sum(rmse) / len(rmse))
    print('RMSE..... %s' % score)
    score = round(score, 6)

    gbm_store = [gbm_0, gbm_1, gbm_2, gbm_3, gbm_4]
    for i, gbm in enumerate(gbm_store):
        y_pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        y_pred_df[label[i]] = y_pred_test if is_original else np.expm1(y_pred_test)
        gbm.save_model('../model/gbdt_model'+str(i)+'_'+str(score)+'.txt')
        
    y_pred_df['vid'] = test_vid
    y_pred_gbdt_df = y_pred_df.loc[:, ['vid'] + label]
    # y_pred_gbdt_df = y_pred_gbdt_df.round(3)
    y_pred_gbdt_df.to_csv('../data/gbdt/gbdt_output_log1p_'+str(score)+'.csv', index=False, header=False)

if __name__ == '__main__':
    # X, y, X_test, feature, label, test_vid = get_data()
    # cvtest()
    # main()
    ensemble_model()
