import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# -----------------------------------
# 学習データ、テストデータの読み込み
# -----------------------------------
# 学習データ、テストデータの読み込み
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

train = train.drop(columns=['id'])
# test = test.drop(columns=['id'])

# train = train[train.horsepower != '?']
# train = train.replace('?', np.NaN)
# test = test.replace('?', np.NaN)
# train = train.dropna()

# import missingno as msno
#
# msno.matrix(df=train, figsize=(20, 14), color=(0.5, 0, 0))
# plt.show()
#
#
# def kesson_table(df):
#     null_val = df.isnull().sum()
#     percent = 100 * df.isnull().sum() / len(df)
#     kesson_table = pd.concat([null_val, percent], axis=1)
#     kesson_table_ren_columns = kesson_table.rename(
#         columns={0: '欠損数', 1: '%'})
#     return kesson_table_ren_columns


# print("訓練データの欠損情報")
# kesson = kesson_table(train)
# print(kesson)

# 学習データを特徴量と目的変数に分ける
train_x = train.drop(['target'], axis=1)
train_y = train['target']

# テストデータは特徴量のみなので、そのままでよい
test_x = test.copy()

# 学習用データと評価用データの分割
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32)

# -----------------------------------
# 特徴量作成
# -----------------------------------
# train_x = train_x.drop(columns=['car name'])
# train_x['horsepower'] = train_x['horsepower'].astype(float)
# test_x = test_x.drop(columns=['id', 'car name'])
# test_x['horsepower'] = test_x['horsepower'].astype(float)

# dtrain = xgb.DMatrix(train_x, label=train_y)
# dtest = xgb.DMatrix(test_x.values)


# それぞれのカテゴリ変数にlabel encodingを適用する
# for c in ['Sex', 'Embarked']:
#     # 学習データに基づいてどう変換するかを定める
#     le = LabelEncoder()
#     le.fit(train_x[c].fillna('NA'))

# # 学習データ、テストデータを変換する
# train_x[c] = le.transform(train_x[c].fillna('NA'))
# test_x[c] = le.transform(test_x[c].fillna('NA'))

# -----------------------------------
# モデル作成
# -----------------------------------
# import lightgbm as lgb
# from sklearn.metrics import log_loss
# from sklearn.model_selection import train_test_split
#
# tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y, test_size=0.2, shuffle=True, random_state=42,
#                                           stratify=train_y)
#
# lgb_train = lgb.Dataset(tr_x, tr_y)
# lgb_eval = lgb.Dataset(va_x, va_y)
#
# params = {'objective': 'multiclass',
#           'random_state': 10,
#           'metric': 'multi_logloss',
#           'num_class': 9}

# train
# model = lgb.train(params,
#                   lgb_train,
#                   num_boost_round=50,
#                   valid_names=['train', 'valid'],
#                   valid_sets=[lgb_train, lgb_eval])

# va_pred = model.predict(va_x)
# score = log_loss(va_y, va_pred)
# print(f'logloss: {score:.4f}')
#
# pred = model.predict(test_x)
# print(len(pred))
#
# submission = pd.DataFrame({'id': list(range(200000, 300001))})
# pred = pd.DataFrame(pred,
#                     columns=["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7", "Class_8",
#                              "Class_9"])
# submission = pd.merge(submission, pred, left_index=True, right_index=True)
# submission.to_csv('./sample_submission.csv', index=False)
# print('csv output')

# reg = xgb.XGBClassifier()
#
# # ハイパーパラメータ探索
# reg_cv = GridSearchCV(reg, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
# reg_cv.fit(train_x, train_y)

# モデルの作成および学習データを与えての学習
model = xgb.XGBClassifier()
model.fit(train_x, train_y)
print("fitted")
# テストデータの予測値を確率で出力する
pred = model.predict(test_x)

# 提出用ファイルの作成
submission = pd.DataFrame({'id': test['id'], 'target': pred})
submission.to_csv('./submission_automobile.csv', index=False)
print('csv output')
