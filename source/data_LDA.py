#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import unicodedata
from sklearn import preprocessing

warnings.filterwarnings("ignore")

def LDA_feature(combine, col_name, n_topics):
    t0 = time.time()
    print('start LDA for feature %s' %col_name)
    vectorizer = CountVectorizer(min_df=1)
    train_counts = combine[0].shape[0]
    df = combine[0].append(combine[1])
    X = vectorizer.fit_transform(df.loc[:, col_name].dropna())
    lda = LatentDirichletAllocation(n_topics=n_topics, learning_offset=50., random_state=0)
    docres = lda.fit_transform(X)
    
    columns = []
    for i in range(n_topics):
        columns.append(col_name + '_' + str(i))
    new_features = pd.DataFrame(np.nan, index=range(df.shape[0]), columns=columns)
    new_features.iloc[df.loc[:, col_name].dropna().index, :] = docres
    for col in columns:
        combine[0][col] = new_features.loc[:train_counts, col]
        combine[1][col] = new_features.loc[train_counts:, col]
    print('finished. %s s used' %(time.time()-t0))


if __name__ == '__main__':
    merged_train_df = pd.read_pickle('../data/data_train_num.pkl')
    merged_test_df = pd.read_pickle('../data/data_test_num.pkl')
    combine = [merged_train_df, merged_test_df]
    LDA_feature(combine, '0113', 5)
    LDA_feature(combine, '0912', 3)  # 5效果不佳
    merged_train_df.to_pickle('../data/data_train_lda.pkl')
    merged_test_df.to_pickle('../data/data_test_lda.pkl')