# Recipes

## Stability Selection

- モデル学習

```python
import matplotlib.pyplot as plt
import pandas as pd
from spmkit.models import StabilitySelection

# データ読み込み
dataset = pd.read_csv("dataset.csv")

# X, y に分割
X = dataset.drop("target", axis=1)
y = dataset["target"]

# StabilitySelection モデルを定義
# 1,000 x 100 イテレーション
model = StabilitySelection(
    subsample_iter=1000,
    n_alphas=100,
    min_alpha=1e-6,
    max_alpha=1.0,
    random_state=42,
)

# 特徴量選択を実行
model.fit(X, y)
```

- α方向の選択確率プロファイル

```python
from spmkit.plots import selection_path_plot

# 特徴量ごとの選択率を alpha に沿って表示
fig, ax = plt.subplots(figsize=(10, 6))
selection_path_plot(
    ax=ax,  # matplotlib の axis
    selection_path=model.selection_path_,  # selection_path
    alphas=model.alphas_,  # alphas
    feature_names=X.columns,  # 特徴量の名称
)
plt.show()
```

- 平均選択確率の算出と表示

```python
import matplotlib.pyplot as plt
from spmkit.plots import ave_selection_probs_bar_plot

# 平均選択率を 1e-4 から 1e-1 の範囲で取得
selection_probs = model.get_ave_selection_probs(
    lower_alpha=1e-4,
    upper_alpha=1e-1,
)

# 棒グラフをプロット
fig, ax = plt.subplots(figsize=(6, 6))
ave_selection_probs_bar_plot(
    ax=ax,  # matplotlib の axis
    selection_probs=selection_probs,  # selection_probs
    selection_th=0.8,   # 閾値
    feature_names=X.columns,  # 特徴量の名称
)
plt.show()
```

- スパース表現への変換

```python
# th = 0.8 で特徴量を絞り込む
# 不要な特徴量がゼロ埋めされたスパースなデザイン行列が返る
X_sparse = model.transform(
    X,
    lower_alpha=1e-4,
    upper_alpha=1e-1,
    selection_th=0.8,
)

# pandas dataframe にキャスト
X_sparse = pd.DataFrame(X_sparse, columns=X.columns)

# ゼロ埋めされた列を落としたデザイン行列
X_sparse_droped = X_sparse.loc[:, (X_sparse != 0).any(axis=0)]
```

-----

## Exhaustive Search

- モデル学習

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from spmkit.models import ExhaustiveSearch

# データ読み込み
dataset = pd.read_csv("dataset.csv")

# X, y に分割
X = dataset.drop("target", axis=1)
y = dataset["target"]

# train / test = 8 : 2 に分割
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# データを標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ExhaustiveSearch モデルを定義
model = ExhaustiveSearch(
    sort_criteria="aic",
    random_state=42,
)

# 学習を実行
model.fit(X_train_scaled, y_train)
```

- 重みダイアグラムの算出と表示

```python
import matplotlib.pyplot as plt
from spmkit.plots import weight_diagram

# 全てのモデルの回帰係数
weights = model.get_weights()

# 重みダイアグラム
fig, ax = plt.subplots(figsize=(8, 6))
im = weight_diagram(
    ax=ax,
    weights=weights,
    feature_names=X.columns,
    xscale="log2",
)
fig.colorbar(im, ax=ax, label="Standardized coefficient")
plt.show()
```

- 特徴量重要度の算出と表示

```python
import matplotlib.pyplot as plt
from spmkit.plots import ave_importances_bar_plot

# 特徴量重要度を計算
importances = model.get_ave_importances()

# 表示
fig, ax = plt.subplots(figsize=(6, 6))
ave_importances_bar_plot(
    ax=ax,
    importances=importances,
    feature_names=X.columns,
)
plt.show()
```

- 上位モデルによる予測と回帰係数の可視化

```python
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from spmkit.plots import coefs_waterfall_plot

# 上位 3 モデル
max_rank = 3

# 上位モデルでテストデータによる予測を行う
y_preds = [client.predict(X_test_scaled, rank=rank) for rank in range(max_rank)]

# 上位モデルで選択された特徴量を取得
features = client.get_features()

# 結果を表示する
for rank, y_pred in enumerate(y_preds):
    # R-squared
    print(f"R2 = {r2_score(y_test, y_pred):.3f}")

    # ウォーターフォールプロット
    fig, ax = plt.subplots(figsize=(6, 6))
    coefs_waterfall_plot(
        ax=ax,
        weights=weights,
        rank=rank,
        feature_names=X.columns,
    )
    plt.show()
```