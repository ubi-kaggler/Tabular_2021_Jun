class UnderBaggingKFold(BaseCrossValidator):
    """CV に使うだけで UnderBagging できる KFold 実装

    NOTE: 少ないクラスのデータは各 Fold で重複して選択される"""

    def __init__(self, n_splits=5, shuffle=True, random_states=None,
                 test_size=0.2, whole_testing=False):
        """
        :param n_splits: Fold の分割数
        :param shuffle: 分割時にデータをシャッフルするか
        :param random_states: 各 Fold の乱数シード
        :param test_size: Under-sampling された中でテスト用データとして使う割合
        :param whole_testing: Under-sampling で選ばれなかった全てのデータをテスト用データに追加するか
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_states = random_states
        self.test_size = test_size
        self.whole_testing = whole_testing

        if random_states is not None:
            # 各 Fold の乱数シードが指定されているなら分割数をそれに合わせる
            self.n_splits = len(random_states)
        else:
            # 乱数シードが指定されていないときは分割数だけ None で埋めておく
            self.random_states = [None] * self.n_splits

        # 分割数だけ Under-sampling 用のインスタンスを作っておく
        self.samplers_ = [
            RandomUnderSampler(random_state=random_state)
            for random_state in self.random_states
        ]

    def split(self, X, y=None, groups=None):
        """データを学習用とテスト用に分割する"""
        if X.ndim < 2:
            # RandomUnderSampler#fit_resample() は X が 1d-array だと文句を言う
            X = np.vstack(X)

        for i in range(self.n_splits):
            # データを Under-sampling して均衡データにする
            sampler = self.samplers_[i]
            _, y_sampled = sampler.fit_resample(X, y)
            # 選ばれたデータのインデックスを取り出す
            sampled_indices = sampler.sample_indices_

            # 選ばれたデータを学習用とテスト用に分割する
            split_data = train_test_split(sampled_indices,
                                          shuffle=self.shuffle,
                                          test_size=self.test_size,
                                          stratify=y_sampled,
                                          random_state=self.random_states[i],
                                          )
            train_indices, test_indices = split_data

            if self.whole_testing:
                # Under-sampling で選ばれなかったデータをテスト用に追加する
                mask = np.ones(len(X), dtype=np.bool)
                mask[sampled_indices] = False
                X_indices = np.arange(len(X))
                non_sampled_indices = X_indices[mask]
                test_indices = np.concatenate([test_indices,
                                               non_sampled_indices])

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits




X = df_train.drop(["target"], axis=1)#説明変数
y = df_train["target"]#目的変数
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=0
)#テストが30% ランダムシード値を0で固定



X, y = make_classification(**args)

lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
}

lgb_train = lgb.Dataset(X, y)

# 5-Fold で乱数シードに 42 ~ 46 を指定している
folds = UnderBaggingKFold(random_states=range(42, 42 + 5))


    # 上記で作った UnderBaggingKFold を folds に指定する
result = lgb.cv(lgbm_params,
        lgb_train,
        num_boost_round=1000,
        early_stopping_rounds=10,
        seed=42,
        folds=folds,
        verbose_eval=10,
        )





