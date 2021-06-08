import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('train.csv')
for fa in range(1, 75):
    for fb in range(fa+1, 75):
        for i in range(10):
            plt.scatter(df[df['target'] == f'Class_{i+1}'][f'feature_{fa}'], df[df['target'] == f'Class_{i+1}'][f'feature_{fb}'])

        print(fa, fb)
        plt.savefig(f'{fa:02}-{fb:02}.png')
