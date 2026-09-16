import multiprocessing as mp
from functools import partial
from itertools import product
from typing import Literal

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


# 各イテレーションで特徴量選択を行い、特徴量ごとに選択されたかどうかを返すメソッド
def _selector(iters, X, y, alphas, eps, max_iter, random_state):
    # イテレータ
    subsample_idx, alpha_idx = iters

    # サブサンプル
    X_subsample = X[subsample_idx]
    y_subsample = y[subsample_idx]

    # alpha
    alpha = alphas[alpha_idx]

    # 標準化
    scaler = StandardScaler()
    X_subsample_scaled = scaler.fit_transform(X_subsample)

    # モデル定義
    model = Lasso(
        alpha=alpha,
        random_state=random_state,
        max_iter=max_iter,
    )

    # 学習
    model.fit(X_subsample_scaled, y_subsample)

    # 選択された特徴量を 0/1 で収集
    selection = (np.abs(model.coef_) > eps).astype(int)

    return selection, alpha_idx


# Stability Selection を行うクラス
class StabilitySelection:
    """Stability Selection (N. Meinshausen and P. Bühlmann, J. R. Stat. Soc. B (2010).).

    Parameters
    ----------
    subsample_iter : int, default=100
        サブサンプリング回数
    subsample_frac : float, default=0.5
        サブサンプリング比率 (0.0 から 1.0 の値)
    n_alphas : int, default=100
        正則化係数 (alpha) の値をサンプリングする回数
    min_alpha : float, default=1e-6
        サンプリングされる alpha の最小値
    max_alpha : float, default=1.0
        サンプリングされる alpha の最大値
    eps : float, default=1e-6
        LASSO が特徴量選択を行う際の閾値
    lasso_max_iter : int, default=1000
        LASSO の最大反復回数
    random_state : int, default=42
        ランダムシード

    Attributes
    ----------
    alphas_ : ndarray of shape (n_alphas,)
        サンプリングされた alpha の配列
    selection_path_ : ndarray of shape (n_features, n_alphas)
        各特徴量における alpha ごとの選択確率プロファイル

    """

    def __init__(
        self,
        subsample_iter: int = 100,
        subsample_frac: float = 0.5,
        n_alphas: int = 100,
        min_alpha: float = 1e-6,
        max_alpha: float = 1.0,
        eps: float = 1e-6,
        lasso_max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.subsample_iter = subsample_iter  # サブサンプリング回数
        self.subsample_frac = subsample_frac  # サブサンプリング比率
        self.n_alphas = n_alphas  # LASSO の正則化係数の探索回数
        self.min_alpha = min_alpha  # LASSO の正則化係数の最小値
        self.max_alpha = max_alpha  # LASSO の正則化係数の最大値
        self.eps = eps  # スパースとみなす閾値
        self.lasso_max_iter = lasso_max_iter  # LASSO の最大反復回数
        self.random_state = random_state  # ランダムシード

        self.alphas_ = None  # alpha のプロファイル
        self.selection_path_ = None  # 選択確率

        # ランダムシード固定
        np.random.seed(self.random_state)

    # 学習を実行する関数
    def fit(self, X, y, n_jobs: int = -1):
        """モデルを学習させる

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            訓練データのデザイン行列
        y : array-like of shape (n_samples,)
            教師ラベルの配列
        n_jobs : int, default=-1
            CPU プロセス数 (-1 を指定すると全ての CPU コアを消費)

        Returns
        -------
        self
            学習済みインスタンス

        """
        # ndarray にキャスト
        X, y = np.asarray(X), np.asarray(y)

        # データサイズ
        n_sample = X.shape[0]
        n_subsample = int(n_sample * self.subsample_frac)

        # サブサンプリング用の index を生成
        # 重複を許さない
        subsample_indices = []
        seen = set()
        subsample_count = 0
        subsample_max_iter = int(self.subsample_iter * 10)

        while len(subsample_indices) < self.subsample_iter:
            if subsample_count > subsample_max_iter:
                break

            subsample_count += 1

            indice = np.random.choice(n_sample, n_subsample, replace=False)
            key = tuple(sorted(indice))

            if key not in seen:
                subsample_indices.append(indice)
                seen.add(key)

        # alpha を走査
        self.alphas_ = np.logspace(
            np.log10(self.min_alpha),
            np.log10(self.max_alpha),
            self.n_alphas,
        )
        alpha_indices = list(range(len(self.alphas_)))

        # 特徴量選択器の初期化
        worker = partial(
            _selector,
            X=X,
            y=y,
            alphas=self.alphas_,
            eps=self.eps,
            max_iter=self.lasso_max_iter,
            random_state=self.random_state,
        )

        # ループするイテレータ
        iters = product(subsample_indices, alpha_indices)

        # 並列で選択を実行
        tot_iters = len(subsample_indices) * len(alpha_indices)
        with mp.Pool(processes=mp.cpu_count() if n_jobs == -1 else n_jobs) as pool:
            selections = list(
                tqdm(
                    pool.imap_unordered(worker, iters),
                    total=tot_iters,
                    desc="progress",
                ),
            )

        # 特徴量・alpha ごとの選択回数
        selection_counts = np.zeros((X.shape[1], len(alpha_indices)), dtype=int)
        for sele in selections:
            selection_counts[:, sele[1]] += sele[0]

        # 特徴量・alpha ごとの選択確率
        self.selection_path_ = selection_counts / self.subsample_iter
        return self

    # 選択確率を取得
    def get_selection_path(self):
        """各特徴量における alpha ごとの選択確率プロファイルを取得する

        Returns
        -------
        selection_path_ : ndarray of shape (n_features, n_alphas)
            各特徴量における alpha ごとの選択確率プロファイル

        """
        return self.selection_path_

    # alphas を取得
    def get_alphas(self):
        """サンプリングされた alpha の配列を取得する

        Returns
        -------
        alphas_ : ndarray of shape (n_alphas,)
            サンプリングされた alpha の配列

        """
        return self.alphas_

    # alpha のスライス
    def _get_alpha_domain(
        self,
        lower_alpha: float | None,
        upper_alpha: float | None,
    ):
        if lower_alpha is None:
            lower_alpha = self.min_alpha
        if upper_alpha is None:
            upper_alpha = self.max_alpha

        if lower_alpha < self.min_alpha:
            return ValueError(
                "'lower_alpha' must be greater than or equal to 'min_alpha'.",
            )
        if upper_alpha > self.max_alpha:
            return ValueError(
                "'upper_alpha' must be less than or equal to 'max_alpha'.",
            )

        # alpha を指定した領域に制限
        alpha_indice = np.where(
            (self.alphas_ >= lower_alpha) & (self.alphas_ < upper_alpha),
        )[0]
        domain = self.selection_path_[:, alpha_indice]
        return domain

    def get_ave_selection_probs(
        self,
        lower_alpha: float | None = None,
        upper_alpha: float | None = None,
    ):
        """指定した alpha の領域について、特徴量ごとの平均選択率とその標準偏差を取得する

        Parameters
        ----------
        lower_alpha : float, optional
            alpha の下界
        upper_alpha : float, optional
            alpha の上界

        Returns
        -------
        ndarray of shape (n_features, 2)
            特徴量ごとの平均選択率とその標準偏差

        """
        domain = self._get_alpha_domain(
            lower_alpha=lower_alpha,
            upper_alpha=upper_alpha,
        )

        # 平均選択率とその標準偏差
        return np.vstack([domain.mean(axis=1), domain.std(axis=1)]).T

    def get_mask(
        self,
        lower_alpha: float | None = None,
        upper_alpha: float | None = None,
        selection_th: float = 0.5,
        astype: Literal["bool", "int"] = "bool",
    ):
        """指定した alpha の領域について、特徴量が選択されたかどうかを表すマスク配列を取得する

        Parameters
        ----------
        lower_alpha : float, optional
            alpha の下界
        upper_alpha : float, optional
            alpha の上界
        selection_th : float, default=0.5
            特徴量選択の閾値 (0.0 から 1.0 の値)
        astype : {"bool", "int"}, default="bool"
            マスク配列の型

        Returns
        -------
        ndarray of shape (n_features,)
            特徴量が選択されたかどうかを表すマスク配列

        """
        selection_probs = self.get_ave_selection_probs(
            lower_alpha=lower_alpha,
            upper_alpha=upper_alpha,
        )
        mask = selection_probs[:, 0] >= selection_th
        if astype == "int":
            mask = mask.astype(int)
        return mask

    def transform(
        self,
        X,
        lower_alpha: float | None = None,
        upper_alpha: float | None = None,
        selection_th: float = 0.5,
    ):
        """指定した alpha の領域について、デザイン行列をスパース表現に変換する

        Parameters
        ----------
        lower_alpha : float, optional
            alpha の下界
        upper_alpha : float, optional
            alpha の上界
        selection_th : float, default=0.5
            特徴量選択の閾値 (0.0 から 1.0 の値)

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            デザイン行列のスパース表現

        """
        X = np.asarray(X)
        mask = self.get_mask(
            lower_alpha=lower_alpha,
            upper_alpha=upper_alpha,
            selection_th=selection_th,
        )
        X[:, ~mask] = 0.0
        return X
