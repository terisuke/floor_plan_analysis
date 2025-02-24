"""
madori/train_model.py

ランダムフォレストを用いたモデルの学習スクリプト。
学習後は models/ ディレクトリにモデルを保存。
"""

import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from .preprocess import load_floor_csv_for_training

def train_random_forest(data_dir="data/1F", model_path="models/floor_model.pkl"):
    # データ読み込み
    X, y, room_codes = load_floor_csv_for_training(data_dir)

    # X が可変長の場合、単純に学習できないので形をそろえる or パディングなどが必要
    # ここでは簡易に「最大次元に合わせてパディングする」例を示す
    import numpy as np
    max_len = max(len(row) for row in X)
    X_padded = []
    for row in X:
        padded = list(row) + [-1]*(max_len - len(row))
        X_padded.append(padded)
    X_padded = np.array(X_padded)

    # ランダムフォレスト回帰
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_padded, y)

    # モデル保存
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        "model": rf,
        "room_codes": room_codes,
        "max_len": max_len
    }, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_random_forest("data/1F", "models/floor_model.pkl")