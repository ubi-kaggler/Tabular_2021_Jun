import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from net import Net
from torch import nn
import torch
import re


def main():

    df_train = pd.read_csv('../data/train.csv')[:10000]
    X_train  = torch.tensor(df_train.loc[:, 'feature_0':'feature_74'].values).float()
    y_train  = torch.tensor([int(re.search('[0-9]+', val).group(0)) for val in df_train['target']])
    df_test  = pd.read_csv('../data/train.csv')[-10000:]
    X_test   = torch.tensor(df_test.loc[:, 'feature_0':'feature_74'].values).float()
    y_test   = torch.tensor([int(re.search('[0-9]+', val).group(0)) for val in df_test['target']])
    input_features  = X_train.shape[1]
    hidden_features = input_features // 2
    output_features = len(set(y_train))
    net = Net(input_features, hidden_features, output_features)

    outputs = net(X_train[0:3])
    for output, label in zip(outputs, y_train):
        print(f'{output=}, {label=}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.003)  # 最適化アルゴリズム

    EPOCHS = 2000
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = net(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 99:
            print(f'epoch: {epoch:4}, loss: {loss.data}')

    with torch.no_grad():
        outputs = net(X_test)
        loss = criterion(outputs, y_test)
        print(f'{loss=}')

    print('training finished')


if __name__ == '__main__':

    main()
