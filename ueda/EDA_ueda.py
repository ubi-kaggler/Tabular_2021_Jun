# import packages
import os
import joblib
import numpy as np
import pandas as pd
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

# setting up options
#print(pd.get_option('display.max_rows'))
pd.set_option('display.max_rows', None) #Noneにすることで全て表示
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

# import datasets
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')
submission = pd.read_csv('../data/sample_submission.csv')

#print(f'Number of rows: {train_df.shape[0]};  Number of columns: {train_df.shape[1]}; No of missing values: {sum(train_df.isna().sum())}')
#print(train_df.info())

#”基本統計量”___各変数の基本統計で、数、平均、標準偏差、最小値、第1四分位、中央値、第3四分位、最大値
#print(train_df.describe())

#テストdfでもDFの形を確認する＝trainとちゃんと類似しているかをみる

features = [feature for feature in train_df.columns if feature not in ['id', 'target']]
unique_values_train = np.zeros(2)
for feature in features:
    temp = train_df[feature].unique()
    unique_values_train = np.concatenate([unique_values_train, temp])
unique_values_train = np.unique(unique_values_train)
