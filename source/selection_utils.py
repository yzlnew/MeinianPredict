#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
import warnings
import lightgbm as lgb

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

if __name__ == '__main__':
   print(get_low_importance('../model/gbdt_model2018-05-03_1853_4_0.029917.txt'))