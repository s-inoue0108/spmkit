import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from typing import Literal

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
)
from sklearn.model_selection import KFold
from tqdm.auto import tqdm


@dataclass
class Result:
    score: float
    weights: np.ndarray
    feature: np.ndarray
    model: LinearRegression | list[LinearRegression]


# 回帰器
def _regressor(combo, X, y, sort_criteria):
    # 特徴量を切り出す
    idx = np.array(combo)
    X_slice = X[:, idx]

    # モデル定義
    model = LinearRegression()

    # fit
    model.fit(X_slice, y)

    # predict
    y_pred = model.predict(X_slice)

    # score
    score = (
        _get_aic(X_slice, y, y_pred)
        if sort_criteria == "aic"
        else _get_bic(X_slice, y, y_pred)
    )

    # idx, coef は元の特徴量ベクトル (size p) へ埋め込む
    weights = np.zeros(X.shape[1], dtype=float)
    weights[idx] = model.coef_
    feature = np.zeros(X.shape[1], dtype=int)
    feature[idx] = 1

    return Result(
        score=score,
        weights=weights,
        feature=feature,
        model=model,
    )


def _regressor_cv(combo, X, y, split):
    # 特徴量を切り出す
    idx = np.array(combo)
    X_slice = X[:, idx]

    # モデル定義
    model = LinearRegression()

    # internal k-fold split
    cves = []
    coefs = []
    models = []

    for train_index, test_index in split:
        # split
        X_train, X_test = X_slice[train_index], X_slice[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fit
        model.fit(X_train, y_train)

        # save model
        models.append(model)

        # predict
        y_pred = model.predict(X_test)

        # compute cve
        cve = mean_squared_error(y_test, y_pred)
        cves.append(cve)

        # coefs
        coefs.append(model.coef_)

    # fold 平均 CVE, coefs
    cve = np.mean(cves)
    coefs = np.mean(coefs)

    # idx, coef は元の特徴量ベクトル (size p) へ埋め込む
    weights = np.zeros(X.shape[1], dtype=float)
    weights[idx] = coefs
    feature = np.zeros(X.shape[1], dtype=int)
    feature[idx] = 1

    return Result(
        score=cve,
        weights=weights,
        feature=feature,
        model=models,
    )


# スコア計算関数
def _get_aic(X, y, y_pred):
    rss = np.sum((y - y_pred) ** 2)
    n = len(y)
    k = X.shape[1] + 1
    aic = n * np.log(rss / n) + 2 * k
    return aic


def _get_bic(X, y, y_pred):
    rss = np.sum((y - y_pred) ** 2)
    n = len(y)
    k = X.shape[1] + 1
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic


# exhaustive search
class ExhaustiveSearch:
    def __init__(
        self,
        n_fold: int = 5,
        random_state: int = 42,
        sort_criteria: Literal["cve", "aic", "bic"] = "cve",
    ):
        if sort_criteria not in {"cve", "aic", "bic"}:
            raise ValueError(
                f"sort_criteria must be one of 'cve', 'aic', 'bic', got {sort_criteria!r}",
            )

        self.n_fold = n_fold
        self.random_state = random_state
        self.sort_criteria = sort_criteria

        self.results_ = None
        self._best_mask = None

        # ランダムシード固定
        np.random.seed(self.random_state)

    def fit(self, X, y, n_jobs: int = -1):
        # ndarray にキャスト
        X, y = np.asarray(X), np.asarray(y)

        # 全ての特徴量の組み合わせを生成する
        combos = []
        for k in range(1, X.shape[1] + 1):
            combos.extend(combinations(range(X.shape[1]), k))

        print(f"model sorting criteria to be used: {self.sort_criteria.upper()}")

        if self.sort_criteria == "cve":
            # k-fold
            kf = KFold(
                n_splits=self.n_fold,
                random_state=self.random_state,
                shuffle=True,
            )
            split = list(kf.split(X, y))
            worker = partial(
                _regressor_cv,
                X=X,
                y=y,
                split=split,
            )
        else:
            worker = partial(
                _regressor,
                X=X,
                y=y,
                sort_criteria=self.sort_criteria,
            )

        # 並列実行
        with mp.Pool(processes=mp.cpu_count() if n_jobs == -1 else n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(worker, combos),
                    total=len(combos),
                    desc="progress",
                ),
            )

        # sort
        results.sort(key=lambda res: res.score)
        self.results_ = np.array(results)
        return self

    def get_scores(self):
        return np.array([res.score for res in self.results_])

    def get_weights(self):
        return np.array([res.weights for res in self.results_]).T

    def get_features(self):
        return np.array([res.feature for res in self.results_])

    def predict(self, X, rank: int = 0):
        # rank を指定したモデルを取り出す
        model = self.results_[rank].model

        # ndarray にキャスト
        X = np.asarray(X)

        # スライス
        X_slice = X[:, self.results_[rank].feature.astype(bool)]

        if self.sort_criteria == "cve":
            y_preds = []
            for fold in model:
                y_pred = fold.predict(X_slice)
                y_preds.append(y_pred)
            return np.mean(y_preds, axis=0)

        y_pred = model.predict(X_slice)
        return y_pred
