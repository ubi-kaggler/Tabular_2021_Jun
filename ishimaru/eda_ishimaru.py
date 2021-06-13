import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib setting
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
sample_submission = pd.read_csv('../data/sample_submission.csv')

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# # 対象分布
# fig = plt.figure(figsize=(12, 8))
# gs = fig.add_gridspec(7, 4)
# ax = fig.add_subplot(gs[:-1,:])
# ax2 = fig.add_subplot(gs[-1,:])
# ax2.axis('off')
#
target_cnt = train['target'].value_counts().sort_index()
target_cum = target_cnt.cumsum()
# ax.bar(target_cnt.index, target_cnt, color=['#d4dddd' if i%2==0 else '#fafafa' for i in range(9)],
#        width=0.55,
#        edgecolor='black',
#        linewidth=0.7)
#
#
# for i in range(9):
#     ax.annotate(f'{target_cnt[i]}({target_cnt[i]/len(train)*100:.3}%)', xy=(i, target_cnt[i]+1000),
#                    va='center', ha='center',
#                )
#     ax2.barh([0], [target_cnt[i]], left=[target_cum[i] - target_cnt[i]], height=0.2,
#             edgecolor='black', linewidth=0.7, color='#d4dddd' if i%2==0 else '#fafafa'
#             )
#     ax2.annotate(i+1, xy=(target_cum[i]-target_cnt[i]/2, 0),
#                  va='center', ha='center', fontsize=10)
#
# ax.set_title('対象分布', weight='bold', fontsize=15)
# ax.grid(axis='y', linestyle='-', alpha=0.4)
#
# fig.tight_layout()
# plt.show()

target_cnt_df = pd.DataFrame(target_cnt)
target_cnt_df['ratio(%)'] = target_cnt_df/target_cnt.sum()*100
target_cnt_df.sort_values('ratio(%)', ascending=False, inplace=True)
target_cnt_df['cummulated_sum(%)'] = target_cnt_df['ratio(%)'].cumsum()

fig, ax = plt.subplots(1, 1, figsize=(15, 6))

ax.bar(range(75), 100, linewidth=0.2, edgecolor='black', alpha=0.2, color='lightgray')
#ax.bar(range(75), ((train == 0).sum() / len(train)*100)[:-1].sort_values(), linewidth=0.2, edgecolor='black', alpha=1, color='#244747')
ax.bar(range(75), ((train == 0).sum() / len(train)*100)[:-1], linewidth=0.2, edgecolor='black', alpha=1, color='#244747')


ax.set_ylim(0, 100)
ax.set_yticks(range(0, 100, 10))

ax.set_xticks(range(0, 75, 5))
ax.margins(0.01)
ax.grid(axis='y', linestyle='--', linewidth=0.2, zorder=5)
ax.set_title('Ratio of Zeros (Sorted)', loc='center', fontweight='bold')
ax.set_ylabel('ratio(%)', fontsize=12)
ax.legend()
plt.show()