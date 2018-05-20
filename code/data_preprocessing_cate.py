#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from sklearn import preprocessing
from utils import convert_mixed_num

warnings.filterwarnings("ignore")

merged_train_df = pd.read_pickle('../data/data_train_num.pkl')
merged_test_df = pd.read_pickle('../data/data_test_num.pkl')
combine = [merged_train_df, merged_test_df]

numerical_feature = merged_train_df.describe().columns.values
non_numerical_feature = merged_train_df.describe(include='O').columns.values[1:]

print('non numerical feature count: %s' % len(non_numerical_feature))
start = time.time()
print('dealing non numerical features...')

# 去掉前后空白
for col in non_numerical_feature:
    merged_train_df.loc[:, col] = merged_train_df.loc[:, col].str.strip()
    merged_test_df.loc[:, col] = merged_test_df.loc[:, col].str.strip()

# 二分类转换器：0,1
def converter(pat):
    def convert(data):
        if not pd.isna(data):
            if re.search(pat, data):
                return 0
            else:
                return 1
        return np.nan
    return convert

def converter_reverse(pat):
    def convert(data):
        if not pd.isna(data):
            if re.search(pat, data):
                return 1
            else:
                return 0
        return np.nan
    return convert

def convert_0421(data):
    if not pd.isna(data):
        normal = ['整齐','齐','正常','整','整齐;整齐','齐;齐','未见异常']
        if data in normal:
            return 0
        elif re.search(r'早搏',data):
            return 1
        elif re.search(r'(不齐|过|窦性)',data):
            return 2
        elif re.search(r'房颤',data):
            return 3
        elif re.search(r'齐',data):
            return 0
    return np.nan

def convert_0423(data):
    if not pd.isna(data):
        if re.search(r'(正常|未见)',data):
            return 0
        elif re.search(r'粗',data):
            return 1
        elif re.search(r'(清|鸣)',data):
            return 2
        elif re.search(r'弱',data):
            return 3
        elif re.search(r'齐',data):
            return 0
    return np.nan

def convert_0440(data):
    if not pd.isna(data):
        if data.isdigit():
            return float(data) - 86
        elif re.search(r'(无|未)',data):
            return 0
        elif re.search(r'有',data):
            return 1
    return np.nan

def convert_0429(data):
    if not pd.isna(data):
        if re.search(r'(无|未见)',data):
            return 0
        elif re.search(r'减',data):
            return 1
        elif re.search(r'(粗糙|鸣|干)',data):
            return 2
    return np.nan

def convert_0435(data):
    if not pd.isna(data):
        if re.search(r'(未见|软|正常)',data):
            return 0
        elif re.search(r'肠鸣',data):
            return 1
        elif re.search(r'不满意',data):
            return 2
        elif re.search(r'痛',data):
            return 3
    return np.nan

def convert_0436(data):
    if not pd.isna(data):
        if re.search(r'(未|无)',data):
            return 0
        elif re.search(r'青霉素',data):
            return 1
        elif re.search(r'磺胺',data):
            return 2
        elif re.search(r'(cm|CM)',data):
            return 3
        elif re.search(r'过敏',data):
            return 4
    return np.nan

def convert_0216(data):
    if not pd.isna(data):
        if re.search(r'(正常|未见)',data):
            return 0
        elif re.search(r'悬雍垂肥大',data):
            return 2
        elif re.search(r'(充血|水肿)',data):
            return 1
        elif re.search(r'过长',data):
            return 3
        elif re.search(r'切除',data):
            return 4
    return np.nan

def convert_0124(data):
    if not pd.isna(data):
        if re.search(r'\d',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0901(data):
    if not pd.isna(data):
        if re.search(r'白癜风',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0973(data):
    if not pd.isna(data):
        if re.search(r'(无|未|弃)',data):
            return 0
        elif re.search(r'已手术',data):
            return 1
        elif re.search(r'疝',data):
            return 2
    return np.nan

def convert_0974(data):
    if not pd.isna(data):
        if re.search(r'(无|弃|未)',data):
            return 0
        elif re.search(r'疹',data):
            return 1
        elif re.search(r'癣',data):
            return 2
        elif re.search(r'皮炎',data):
            return 3    
    return np.nan

def convert_100010(data):
    if not pd.isna(data):
        if data in ['-','阴性','0(-)']:
            return 0
        if data in ['+','+-'] or re.search(r'(阳性|1\+|\+1)',data):
            return 1
        if data == '++' or re.search(r'(\+-|\+2|2\+)',data):
            return 2
        if data =='+++' or re.search(r'(3\+|\+3)',data):
            return 3
        if re.search(r'-',data):
            return 0
    return np.nan

def convert_1305(data):
    if not pd.isna(data):
        if re.search(r'老年',data):
            return 3
        if re.search(r'(斑|翳)',data):
            return 2
        if re.search(r'角膜炎',data):
            return 1
        if re.search(r'(正常|未见|透明)',data):
            return 0
    return np.nan

def convert_30007(data):
    if not pd.isna(data):
        if re.search(r'(未|正)', data):
            return 0
        elif re.search(r'(Ⅰ|i)',data):
            return 1
        elif re.search(r'(Ⅱ|ii)',data):
            return 2
        elif re.search(r'(III|iii)',data):
            return 3
        elif re.search(r'Ⅳ',data):
            return 4
    return np.nan

def convert_3189(data):
    if not pd.isna(data):
        if data in ['-', '阴性'] :
            return 0
        elif data in ['0', '0.6', '1.4', '2.8', '+-']:
            return 1
        elif data in ['+', '阳性', '阳性(+)']:
            return 2
        elif data in ['++', '+++', '9.0']:
            return 3
    return np.nan

def convert_3190(data):
    if not pd.isna(data):
        if data in ['++++', '≥55(+4)', '+++', '3+', '+-', '2.8(+-)'] :
            return 0
        elif data in ['++', '2+', '+', '阳性(+)']:
            return 1
        elif data in ['-', '0(-)'] or re.search(r'0mmol/L|阴', data):
            return 2
    return np.nan

def convert_3191(data):
    if not pd.isna(data):
        if data in ['+++', '++'] :
            return 0
        elif data in ['+', '8.6(+1)', '阳性(+)']:
            return 1
        elif data in ['-', '0(-)'] or re.search(r'0mmol/L|阴', data):
            return 2
    return np.nan

def convert_3192(data):
    if not pd.isna(data):
        if data in ['+++', '4.0(+2)', '++'] :
            return 0
        elif data in ['+', '阳性(+)', '+-', '0.5(+-)']:
            return 1
        elif data in ['-', '0(-)'] or re.search(r'0mmol/L|阴', data):
            return 2
    return np.nan

def convert_3194(data):
    if not pd.isna(data):
        if data in ['+++', '++', '+-'] :
            return 0
        elif data in ['+', '阳性(+)']:
            return 1
        elif data in ['-', '阴性']:
            return 2
    return np.nan

def convert_3195(data):
    if not pd.isna(data):
        if data in ['+++'] :
            return 0
        elif data in ['++', '2+'] or re.search('\+-|\+2', data):
            return 1
        elif data in ['+'] or re.search(r'阳|\+1|1\+',data):
            return 2
        elif data in ['-', '阴性', '0(-)'] or re.search(r'0g/L',data):
            return 3
    return np.nan

def convert_3196(data):
    if not pd.isna(data):
        if data in ['正常', 'Normal', '3.4']:
            return 0
        elif re.search('\+|5.', data):
            return 1
        elif re.search('\-', data):
            return 2
    return np.nan

def convert_3197(data):
    if not pd.isna(data):
        if data == '-':
            return 0
        elif data == '+':
            return 1
        elif data == '阴性':
            return 2
        elif data == '+-':
            return 3
    return np.nan

def convert_0409_0(data):
    if not pd.isna(data):
        if re.search(r'高.*血压|血压.*高', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0409_1(data):
    if not pd.isna(data):
        if re.search(r'高.*血脂|血脂.*高', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0409_2(data):
    if not pd.isna(data):
        if re.search(r'高.*血糖|血糖.*高|糖尿病', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0409_3(data):
    if not pd.isna(data):
        if re.search(r'肥胖', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0409_4(data):
    if not pd.isna(data):
        if re.search(r'甲状腺', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0409_5(data):
    if not pd.isna(data):
        if re.search(r'脂肪', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0409_6(data):
    if not pd.isna(data):
        if re.search(r'肾', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0409_7(data):
    if not pd.isna(data):
        if re.search(r'心动过速', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0409_8(data):
    if not pd.isna(data):
        if re.search(r'心动过缓', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0409_9(data):
    if not pd.isna(data):
        if re.search(r'肝', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0434_0(data):
    if not pd.isna(data):
        if re.search(r'高.*血压|血压.*高', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0434_1(data):
    if not pd.isna(data):
        if re.search(r'高.*血脂|血脂.*高', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0434_2(data):
    if not pd.isna(data):
        if re.search(r'高.*血糖|血糖.*高|糖尿病', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_0434_3(data):
    if not pd.isna(data):
        if re.search(r'脂肪肝', data):# and re.search(r'高',data):
            return 1
        else:
            return 0
    return np.nan

def convert_4001(data):
    if not pd.isna(data):
        if re.search(r'正常|未见|良好', data):
            return 0
        elif re.search(r'轻度', data):
            return 1
        elif re.search(r'中度', data):
            return 2
        elif re.search(r'重度', data):
            return 3
    return np.nan

def convert_2228(data):
    if not pd.isna(data):
        if re.search(r'阳|\+', data):
            return 1
        elif re.search(r'阴|未|-', data):
            return 0
    return np.nan

def convert_0975(data):
    if not pd.isna(data):
        if re.search(r'无|未|nan', data):
            return 0
        elif re.search(r'脂肪瘤', data):
            return 1
    return np.nan

def convert_300005(data):
    if not pd.isna(data):
        if re.search(r'阴|未|-', data):
            return 0
        else:
            return 1
    return np.nan

def convert_woman(data):
    if not pd.isna(data):
        return '女'
    else:
        return ''

def convert_man(data):
    if not pd.isna(data):
        return '男'
    else:
        return ''

def convert_sex(data):
    if not pd.isna(data):
        if re.search(r'男', data):
            return 0
        elif re.search(r'女', data):
            return 1
    return np.nan

def convert_eyes(data):
    if not pd.isna(data):
        if re.search(r'硬化|变细|反光增强|病变|瘤|白内障|浑浊|不清|不能详辨|密度增高', data):
            return 1
        elif re.search(r'正|未|-', data):
            return 0
    return np.nan

def convert_300018(data):
    if not pd.isna(data):
        if re.search(r'阴性|^-$',data):
            return 0
        return convert_mixed_num(data)
    return np.nan

def convert_300036(data):
    if not pd.isna(data):
        if re.search(r'阴性|^-$',data):
            return 0
        if data == '+-':
            return 30
        if data == '+':
            return 40
        return convert_mixed_num(data)
    return np.nan

def convert_3203(data):
    if not pd.isna(data):
        if re.search(r'未见',data):
            return 0
        return convert_mixed_num(data)
    return np.nan
 
def convert_A702(data):
    if not pd.isna(data):
        if re.search(r'nan',data):
            return 0
        return convert_mixed_num(data)
    return np.nan    

def convert_I49012(data):
    if not pd.isna(data):
        if data == '-':
            return 0
        if re.search(r'\+1',data) or data in ['+-','+']:
            return 1.4
        if re.search(r'\+3',data) or data == '+++':
            return 5.6
        if data == '++':
            return 2.8
        return convert_mixed_num(data)
    return np.nan      

for df in combine:
    type = 'float'
    df['0124'] = df['0124'].apply(convert_0124).astype(type)
    df['0216'] = df['0216'].apply(convert_0216).astype(type)
    df['0405'] = df['0405'].apply(converter(r'(无|未)')).astype(type)
    df['0406'] = df['0406'].apply(converter(r'(未|正常)')).astype(type)
    df['0407'] = df['0407'].apply(converter(r'(未|弃)')).astype(type)
    df['0420'] = df['0420'].apply(converter(r'(未|正常)')).astype(type)
    df['0421'] = df['0421'].apply(convert_0421).astype(type)
    df['0423'] = df['0423'].apply(convert_0423).astype(type)
    df['0426'] = df['0426'].apply(converter(r'(未|正常|无)')).astype(type)
    df['0429'] = df['0429'].apply(convert_0429).astype(type)
    df['0430'] = df['0430'].apply(converter(r'(未|正常)')).astype(type)
    df['0431'] = df['0431'].apply(converter(r'(未|无)')).astype(type)
    df['0435'] = df['0435'].apply(convert_0435).astype(type)
    df['0436'] = df['0436'].apply(convert_0436).astype(type)
    df['0440'] = df['0440'].apply(convert_0423).astype(type)
    df['0707'] = df['0707'].apply(converter(r'未见')).astype(type)
    df['0901'] = df['0901'].apply(convert_0901).astype(type)
    df['0973'] = df['0973'].apply(convert_0973).astype(type)
    df['0974'] = df['0974'].apply(convert_0974).astype(type)
    df['0976'] = df['0976'].apply(converter(r'(无|弃查)')).astype(type)
    df['100010'] = df['100010'].apply(convert_100010).astype(type)
    df['1315'] = df['1315'].apply(converter(r'(未|正常)')).astype(type)
    df['1305'] = df['1305'].apply(convert_1305).astype(type)
    df['300018'] = df['300018'].apply(convert_300018).astype(type)
    df['300019'] = df['300019'].apply(convert_300018).astype(type)
    # df['300036'] = df['300036'].apply(convert_300036).astype(type)
    df['3203'] = df['3203'].apply(convert_3203).astype(type)
    df['A702'] = df['A702'].apply(convert_A702).astype(type)
    df['A704'] = df['A704'].apply(convert_A702).astype(type)
    df['I49012'] = df['I49012'].apply(convert_I49012).astype(type)

    # by zk
    df['30007'] = df['30007'].apply(convert_30007).astype(type)
    df['3189'] = df['3189'].apply(convert_3189).astype(type)
    df['3190'] = df['3190'].apply(convert_3190).astype(type)
    df['3191'] = df['3191'].apply(convert_3191).astype(type)
    df['3192'] = df['3192'].apply(convert_3192).astype(type)
    df['3194'] = df['3194'].apply(convert_3194).astype(type)
    df['3195'] = df['3195'].apply(convert_3195).astype(type)
    df['3196'] = df['3196'].apply(convert_3196).astype(type)
    df['3197'] = df['3197'].apply(convert_3197).astype(type)
    # by zk2
    df['0409_0434'] = df['0409'] + df['0434']
    df['0409_0434_0'] = df['0409_0434'].apply(converter_reverse(r'血压')).astype(type)
    df['0409_0434_1'] = df['0409_0434'].apply(converter_reverse(r'血脂')).astype(type)
    df['0409_0434_2'] = df['0409_0434'].apply(converter_reverse(r'糖')).astype(type)
    df['0409_0434_3'] = df['0409_0434'].apply(converter_reverse(r'肥胖')).astype(type)
    df['0409_0434_4'] = df['0409_0434'].apply(converter_reverse(r'甲状腺')).astype(type)
    df['0409_0434_5'] = df['0409_0434'].apply(converter_reverse(r'肾')).astype(type)
    df['0409_0434_6'] = df['0409_0434'].apply(converter_reverse(r'脂肪')).astype(type)
    # df['0409_0434_7'] = df['0409_0434'].apply(converter_reverse(r'心动过速')).astype(type)
    # df['0409_0434_8'] = df['0409_0434'].apply(converter_reverse(r'心动过缓')).astype(type)
    # df['0409_0434_9'] = df['0409_0434'].apply(converter_reverse(r'心律')).astype(type)
    df['0409_0434_10'] = df['0409_0434'].apply(converter_reverse(r'肝')).astype(type)
    # df['0409_0434_11'] = df['0409_0434'].apply(converter_reverse(r'冠心')).astype(type)
    df['0413'] = df['0413'].apply(converter_reverse(r'低盐|低脂|血糖|血压')).astype(type)
    df['4001'] = df['4001'].apply(convert_4001).astype(type)
    df['2228'] = df['2228'].apply(convert_2228).astype(type)
    df['2229'] = df['2229'].apply(convert_2228).astype(type)
    df['2231'] = df['2231'].apply(convert_2228).astype(type)
    df['2233'] = df['2233'].apply(convert_2228).astype(type)
    df['3301'] = df['3301'].apply(convert_2228).astype(type)
    #by zk3
    df['0975'] = df['0975'].apply(convert_0975).astype(type)
    df['3429'] = df['3429'].apply(convert_300005).astype(type)
    df['300005'] = df['300005'].apply(convert_300005).astype(type)
    # df['300018'] = df['300018'].apply(convert_300005).astype(type)
    # df['300019'] = df['300019'].apply(convert_300005).astype(type)
    # df['300036'] = df['300036'].apply(convert_300005).astype(type)

    # by zk4
    df['sex'] = df['0120'].apply(convert_man)
    for col_name in ['0981', '0982', '0983', '0984']:
        df['sex'] = df['sex'].str.cat(df[col_name].apply(convert_man))
    for col_name in ['0121', '0122', '0123', '0503', '0509', '0539']:
        df['sex'] = df['sex'].str.cat(df[col_name].apply(convert_woman))
    df['sex'] = df['sex'].apply(convert_sex).astype(type)
    df['eyes'] = df['1316'].str.cat(df['1314']).str.cat(df['1330'])
    df['eyes'] = df['eyes'].apply(convert_eyes)
    # by lyf
    df['3207'] = df['3207'].apply(converter(r'(未|-|阴)')).astype(type)
    df['3400'] = df['3400'].apply(converter(r'(透明)')).astype(type)
    df['3485'] = df['3485'].apply(converter(r'未|^-$|阴')).astype(type)
    df['3730'] = df['3730'].apply(converter(r'未|^0$|阴')).astype(type)

print('done!time used: %s s' %(time.time()-start))

merged_train_df.to_pickle('../data/data_train.pkl')
merged_test_df.to_pickle('../data/data_test.pkl') 