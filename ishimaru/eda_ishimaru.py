import pandas as pd
from pandas_profiling import ProfileReport

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


# profile = ProfileReport(train, title="Pandas Profiling Report")
# profile.to_file("profile_report.html") # HTML ファイルを保存する