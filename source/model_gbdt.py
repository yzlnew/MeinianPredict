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


def lgbm(x, y, params, has_eval, num_boost_round=1000):
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
    if not has_eval:
        lgb.cv(params, lgb_train, stratified=False, num_boost_round=1000, verbose_eval=100)
    return gbm


def get_data():
    # 读取数据
    train_df = pd.read_pickle('../data/data_train.pkl')
    test_df = pd.read_pickle('../data/data_test.pkl')

    # 获取特征列表，并填充 NaN
    num_feature = train_df.describe().columns.values.tolist()[5:]
    label = train_df.describe().columns.values.tolist()[0:5]
    cate_feature = train_df.describe(include='category').columns.values.tolist()

    most_num = train_df[num_feature+label].mean()
    most_cate ={}
    for col in cate_feature:
        most_cate[col] = train_df[col].value_counts().index[0]

    X = train_df.loc[:, num_feature].fillna(most_num)
    y = train_df.loc[:, label].fillna(most_num)
    X_test = test_df.loc[:, num_feature].fillna(most_num)
    
    X[cate_feature] = train_df.loc[:, cate_feature]
    X_test[cate_feature] = test_df.loc[:, cate_feature]
    test_vid = test_df['vid']
    return X, y, X_test, num_feature, cate_feature, label, test_vid


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
        X, y, test_size=0.2, random_state=2018)
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
        'objective': 'regression_l2',
        'metric': 'rmse',
        'sub_feature': 0.6,
        'num_leaves': 60,
        'min_data': 150,
        'min_hessian': 1,
        'verbose': -1,
        'bagging_fraction': 0.80,
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


def separate_model():
    X, y, X_test, num_feature, cate_feature, label, test_vid = get_data()

    params = get_best_params()
    y_pred_df = pd.DataFrame()
    
    print('Total Feature: %s' %((len(X.columns))))

    has_eval = 1    # 是否有验证集，无则需要 CV 验证
    if has_eval:
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=0.2, random_state=2018)
    else:
        X_train, y_train = X, y


    def eval_rmse(gbm, X_eval, y_eval, original=True):
        y_pred_eval = gbm.predict(X_eval, num_iteration=gbm.best_iteration)
        if original:
            return mean_squared_log_error(y_eval, y_pred_eval)
        else:
            return mean_squared_log_error(y_eval, np.expm1(y_pred_eval))

    is_original = [0,0,0,0,0]    # 是否是原始数据，否则进行 log1p 处理
    gbm_store = [0]*5
    rmse = [0]*5
    if_cv = 0    # 是否需要 CV
    params_store = [params]*5
    for i, params in enumerate(params_store):
        y_train_model = y_train.iloc[:, i] if is_original[i] else np.log1p(y_train.iloc[:, i])
        gbm = lgbm(X_train, y_train_model, params, has_eval)
        gbm_store[i] = gbm
        if has_eval:
            y_eval_model = y_eval.iloc[:, i]
            rmse[i] = eval_rmse(gbm, X_eval, y_eval_model, is_original[i])

    print(rmse)
    score = (sum(rmse) / len(rmse))
    print('RMSE..... %s' % score)
    score = round(score, 6)
    time_stamp = time.strftime("%Y-%m-%d_%H%M_", time.localtime()) 
    # gbm_store = [gbm_0, gbm_1, gbm_2, gbm_3, gbm_4]
    for i, gbm in enumerate(gbm_store):
        y_pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        y_pred_df[label[i]] = y_pred_test if is_original[i] else np.expm1(y_pred_test)
        gbm.save_model('../model/gbdt_model'+time_stamp+str(i)+'_'+str(score)+'.txt')
        
    y_pred_df['vid'] = test_vid
    y_pred_gbdt_df = y_pred_df.loc[:, ['vid'] + label]
    # y_pred_gbdt_df = y_pred_gbdt_df.round(3)
    y_pred_gbdt_df.to_csv('../data/gbdt/gbdt_output_log1p_'+time_stamp+str(score)+'.csv', index=False, header=False)

if __name__ == '__main__':
    # X, y, X_test, feature, label, test_vid = get_data()
    # cvtest()
    # main()
    separate_model()
