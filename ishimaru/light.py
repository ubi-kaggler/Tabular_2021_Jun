import pandas as pd
import numpy as np

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])
print(train.info())

train_x = train.drop(['target'], axis=1)
train_y = train['target']

test_x = test.copy()


# Under Samplingの関数（X: num:ターゲット件数 label:少数派のラベル）

def under_sampling_func(X, num, label):
    # KMeansによるクラスタリング
    from sklearn.cluster import KMeans
    km = KMeans(random_state=201707)
    km.fit(X, train_y)
    X['Cluster'] = km.predict(X)

    # 群別の構成比を少数派の件数に乗じて群別の抽出件数を計算
    count_sum = X.groupby('Cluster').count().iloc[0:,0].as_matrix()
    ratio = count_sum / count_sum.sum()
    samp_num = np.round(ratio * num, 0).astype(np.int32)

    # 群別にサンプリング処理を実施
    for i in np.arange(8):
        tmp = X[X['Cluster'] == i]
        if i == 0:
            tmp1 = X.sample(samp_num[i], replace=True)
        else:
            tmp2 = X.sample(samp_num[i], replace=True)
            tmp1 = pd.concat([tmp1, tmp2])
    tmp1['Class'] = label
    return tmp1


under_sampling_func(train_x, 200000, 'Class_5')


# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# le.fit(train_y)
# train_y = le.transform(train_y)
# train_y = pd.Series(train_y)
#
# import lightgbm as lgb
# from sklearn.metrics import log_loss
#
# from sklearn.model_selection import train_test_split
# tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y, test_size=0.2, shuffle=True, random_state=42, stratify=train_y)
#
# lgb_train = lgb.Dataset(tr_x, tr_y)
# lgb_eval = lgb.Dataset(va_x, va_y)
#
# params = {'objective': 'multiclass',
#                   'random_state': 10,
#                   'metric': 'multi_logloss',
#              'num_class': 9}
#
# model = lgb.train(params,
#             lgb_train,
#             num_boost_round=50,
#             valid_names=['train', 'valid'], valid_sets=[lgb_train, lgb_eval])
#
# va_pred = model.predict(va_x)
# score = log_loss(va_y, va_pred)
# print(f'logloss: {score:.4f}')
#
# pred = model.predict(test_x)
# print(len(pred))
#
# submission = pd.DataFrame({'id': list(range(200000,300001))})
# pred = pd.DataFrame(pred,columns =["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"])
# submission = pd.merge(submission,pred,left_index=True, right_index=True)
# submission.to_csv('./sample_submission.csv', index=False)
# print('csv output')

