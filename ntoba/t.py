import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from net import Net
from torch import nn
import torch
import re


def main():

    df = pd.read_csv('../data/train.csv')
    X_train = torch.tensor(df.loc[:, 'feature_0':'feature_74'].values).float()
    y_train = torch.tensor([int(re.search('[0-9]+', val).group(0)) for val in df['target']])
    input_features  = X_train.shape[1]
    hidden_features = input_features // 2
    output_features = len(set(y_train))
    net = Net(input_features, hidden_features, output_features)

    outputs = net(X_train[0:3])
    for output, label in zip(outputs, y_train):
        print(f'{output=}, {label=}')

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.003)  # 最適化アルゴリズム

    EPOCHS = 2000
    with torch.no_grad():
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = net(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        if epoch % 100 == 99:
            print(f'epoch: {epoch:4}, loss: {loss.data}')

    print('training finished')


if __name__ == '__main__':

    main()
