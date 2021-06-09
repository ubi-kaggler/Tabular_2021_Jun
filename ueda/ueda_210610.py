import numpy as np
import pandas as pd
#import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import LabelEncoder
#from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler

# 学習データ、テストデータの読み込み
df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

#データの確認
print(df_train.info)
print(df_train.columns)
print(df_train["target"].value_counts())

#trainからtrainとtestに分ける
def split_data(df):
    train = df.drop["target"]
    test = df["target"]
    return train, test

#ランダムアンダーサンプリング
rank_6 = df_train["target"].value_counts()[0]
rank_8 = df_train["target"].value_counts()[1]
rank_9 = df_train["target"].value_counts()[2]
rank_2 = df_train["target"].value_counts()[3]
rank_3 = df_train["target"].value_counts()[4]
rank_7 = df_train["target"].value_counts()[5]
rank_1 = df_train["target"].value_counts()[6]
rank_4 = df_train["target"].value_counts()[7]
rank_5 = df_train["target"].value_counts()[8]

rus = RandomUnderSampler(
    sampling_strategy={"Class_1": rank_5, "Class_2": rank_5,"Class_3": rank_5, "Class_4": rank_5,"Class_5": rank_5, "Class_6": rank_5,"Class_7": rank_5, "Class_8": rank_5,"Class_9": rank_5}, random_state=71
)

X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

