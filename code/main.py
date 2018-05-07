#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

def main():
    os.system('python data_cleaning.py')
    os.system('python data_preprocessing_num.py')
    os.system('python data_preprocessing_cate.py')
    os.system('python model_gbdt.py')

if __name__ == '__main__':
    main()