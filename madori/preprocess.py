"""
madori/preprocess.py

CSVの間取りデータを読み込み、機械学習用の特徴量・ラベルを生成するモジュール。
"""

import os
import glob
import pandas as pd
import numpy as np
from .analyze_csv import CSV_TO_CONFIG_MAP

def load_floor_csv_for_training(data_dir="data/1F"):
    """
    data_dir 以下のCSVファイルをすべて走査し、
    それぞれを (特徴量X, ラベルy) の形に変換して返す。
    ここでは例として、各マスの部屋コードを one-hot 化したベクトルを特徴量とする簡易例。
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    X = []
    y = []  # ラベルが何を指すかは問題設定により変わる

    # 部屋コードを一意にリスト化
    room_codes = list(CSV_TO_CONFIG_MAP.values())  # ["L","D","K","r","t","B","c","s","e","H","co","ut"]
    code_to_idx = {rc:i for i,rc in enumerate(room_codes)}

    for filepath in csv_files:
        df = pd.read_csv(filepath, header=None).fillna(".")
        grid = df.values
        rows, cols = grid.shape

        # グリッド全体を flatten
        flattened = []
        for r in range(rows):
            for c in range(cols):
                val = str(grid[r][c]).strip().lower()
                if val in CSV_TO_CONFIG_MAP:
                    code = CSV_TO_CONFIG_MAP[val]  # 例: 'l' -> 'L'
                else:
                    code = "."
                # one-hot vector化 or 数値ID化
                if code == ".":
                    idx = -1  # 空きを -1 とする
                else:
                    idx = code_to_idx.get(code, -1)
                flattened.append(idx)
        # flattened は [マス数] の長さをもつベクトル
        # ここでは簡単に "Xとしてflattened全体" "yとして???(部屋の数など？)" の例
        # デモとして X にそのまま flattened を入れる
        X.append(flattened)

        # ラベルy: 例として「総マス数に占める 'co'(廊下) の割合」などを回帰問題にする例
        # （本来は間取りの"良し悪しスコア"などがあればそれをターゲットにしても良い）
        corridor_count = flattened.count(code_to_idx.get("co", -999))  # "co" のID
        ratio_co = corridor_count / len(flattened)
        y.append(ratio_co)

    X = np.array(X, dtype=object)  # 各サンプル毎に長さが異なる可能性があるので、object配列
    y = np.array(y)
    return X, y, room_codes

def vector_to_madori(vector, room_codes, rows=7, cols=9):
    """
    学習・推論の結果得られたベクトル -> 2次元間取りへ変換する例。
    vector: [rows*cols] の長さを想定 (部屋コードID)
    room_codes: 順序付きの部屋コード一覧 (train時に用いたもの)
    """
    madori = np.full((rows, cols), ".", dtype=object)
    idx_to_code = {i:rc for i,rc in enumerate(room_codes)}
    k = 0
    for r in range(rows):
        for c in range(cols):
            code_id = vector[k]
            k+=1
            if code_id >= 0 and code_id in idx_to_code:
                madori[r,c] = idx_to_code[code_id]
            else:
                madori[r,c] = "."
    return madori

if __name__ == "__main__":
    # テスト実行
    X, y, rcodes = load_floor_csv_for_training("data/1F")
    print("Loaded training data shape:", X.shape, y.shape)
    print("Room codes used:", rcodes)