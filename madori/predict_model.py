"""
madori/predict_model.py

学習済みモデルを使って間取りを予測する。
出力: 予測結果を CSVに保存し、または numpy配列として返す。
"""

import numpy as np
import joblib
import pandas as pd
from .preprocess import vector_to_madori

def predict_floor(model_path="models/floor_model.pkl", rows=7, cols=9, output_csv="predicted_floor.csv"):
    data = joblib.load(model_path)
    rf = data["model"]
    room_codes = data["room_codes"]
    max_len = data["max_len"]

    # ここでは簡単のため、ランダムなone-hot IDベクトルを用意し、
    # "RFでの推定" は ラベル値(y)を推定するだけのため本質的ではない
    # 実際にはGANやVAEなど生成モデルが必要になる可能性が高い
    # ここでは説明用にダミーの"ベクトル"を作成して通すだけとする
    # => 生成モデルの実装は別途

    # ダミー: 全マス(rows*cols)ぶん適当な部屋IDを割り当て
    total_cells = rows * cols
    # room_codes数
    n_codes = len(room_codes)

    # 例: ランダムに 0 ~ n_codes-1 のIDを割り当てる
    random_vec = np.random.randint(0, n_codes, size=total_cells)
    
    # パディングしてRFに通す(本来は正しい生成アルゴリズムが必要)
    if total_cells < max_len:
        padded = list(random_vec) + [-1]*(max_len - total_cells)
    elif total_cells > max_len:
        padded = list(random_vec[:max_len])
    else:
        padded = list(random_vec)

    X_test = np.array([padded])
    y_pred = rf.predict(X_test)  # これは単なる回帰値(例: 廊下の割合など)
    print("Predicted label (dummy regression):", y_pred)

    # random_vecをそのまま「予測された間取り」として出力(本来は生成モデルに置き換え)
    madori = vector_to_madori(random_vec, room_codes, rows=rows, cols=cols)

    # CSVに保存
    df_out = pd.DataFrame(madori)
    df_out.to_csv(output_csv, header=False, index=False)
    print(f"Predicted floor CSV saved to {output_csv}")

    return madori

if __name__ == "__main__":
    floor = predict_floor(model_path="models/floor_model.pkl",
                          rows=7, cols=9,
                          output_csv="predicted_floor.csv")
    print(floor)