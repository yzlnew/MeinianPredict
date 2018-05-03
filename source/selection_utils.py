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

if __name__ == '__main__':
   print(get_low_importance('../model/gbdt_model2018-05-03_1853_4_0.029917.txt'))