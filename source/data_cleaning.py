#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

raw_data1 = pd.read_csv('../data/meinian_round1_data_part1_20180408.txt', sep='$')
raw_data2 = pd.read_csv('../data/meinian_round1_data_part2_20180408.txt', sep='$')
raw_data = pd.concat([raw_data1, raw_data2])

print(raw_data.shape)

data_compressed = raw_data.groupby(['vid', 'table_id'], as_index=False).apply(
    lambda x: ";".join(map(str, x['field_results'])))
data_compressed = pd.DataFrame(data_compressed, columns=['field_results'])

data_fmt_all = data_compressed.unstack(fill_value=None)
data_fmt_all.columns = data_fmt_all.columns.droplevel(level=0)

null_count = data_fmt_all.isnull().sum()
data_keep_50000 = data_fmt_all.drop(
    labels=null_count[null_count >= 50000].index, axis=1)

data_keep_50000.to_csv("../data/data_keep_50000.csv")
