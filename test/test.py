import pandas as pd

# Wine Quality データセットをダウンロードする
from sklearn.datasets import fetch_openml
from spmkit.models import StabilitySelection

# データフェッチ
dataset = fetch_openml(data_id=43257, as_frame=True)

# 不要な列を削除し、X, y に分割
X = dataset.data.drop(["quality", "id"], axis=1)
y = dataset.data["quality"]

# StabilitySelection モデルを定義
model = StabilitySelection(
    subsample_iter=1000,
    n_alphas=100,
    min_alpha=1e-6,
    max_alpha=1.0,
    random_state=42,
)

# 特徴量選択を実行
model.fit(X, y)

# alpha ごとの選択確率を算出
selection_path = pd.DataFrame(model.selection_path_, index=X.columns)

print(selection_path)
