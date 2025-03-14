"""
madori/preprocess.py

CSVの間取りデータを読み込み、機械学習用の特徴量・ラベルを生成するモジュール。
不定形のCSVを取り扱いつつ、データ増強(回転・フリップ、ノイズ付加など)を行う。
"""
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import random

# analyze_csv.py から CSV_TO_CONFIG_MAP をインポート (同じ定義を使う)
from .analyze_csv import CSV_TO_CONFIG_MAP

import torch
import torch.nn.functional as F

def load_and_augment_csvs(data_dir="data/1F", do_augmentation=True):
    """
    data_dir 以下のCSVファイルをすべて走査し、それぞれを読み込み:
      - 不定形のまま取得
      - データ増強(回転,フリップ,ノイズ付加など)を行う(オプション)
    戻り値: List[ (np.array(レイアウト2D), filename) ]
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    layouts = []
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath, header=None).fillna(".")
            grid = df.values
            rows, cols = grid.shape

            layout_2d = []
            for r in range(rows):
                row_list = []
                for c in range(cols):
                    val = str(grid[r][c]).strip()
                    # ここでも小文字化せず大文字含むままマッピング
                    if val in CSV_TO_CONFIG_MAP:
                        row_list.append(CSV_TO_CONFIG_MAP[val])
                    else:
                        row_list.append(".")
                layout_2d.append(row_list)

            layout_2d = np.array(layout_2d, dtype=object)
            layouts.append((layout_2d, filepath))

            # データ増強
            if do_augmentation:
                aug_layouts = augment_layout(layout_2d)
                for aug_lay in aug_layouts:
                    layouts.append((aug_lay, filepath+"(aug)"))

        except Exception as e:
            print(f"Warning: {filepath} 読み込み失敗: {e}")
    return layouts

def augment_layout(layout_2d):
    """
    与えられた2Dレイアウト(文字ラベル)に対して回転・フリップ・ノイズ付加などを行い、
    複数のバリエーションを返す。
    ここでは例として、回転90/180/270、左右反転/上下反転、微小ノイズ付加。
    """
    aug_list = []
    # 回転
    rot90 = np.rot90(layout_2d, k=1)
    rot180 = np.rot90(layout_2d, k=2)
    rot270 = np.rot90(layout_2d, k=3)
    aug_list.extend([rot90, rot180, rot270])

    # 左右反転
    fliplr = np.fliplr(layout_2d)
    flipud = np.flipud(layout_2d)
    aug_list.extend([fliplr, flipud])

    # 軽微ノイズ付加(ランダムに2セルだけ'.'にする)例
    noise_copy = layout_2d.copy()
    h, w = noise_copy.shape
    for _ in range(2):
        rr = random.randint(0, h-1)
        cc = random.randint(0, w-1)
        noise_copy[rr, cc] = "."
    aug_list.append(noise_copy)

    return aug_list

def layout_to_onehot(layout_2d, room_list):
    """
    layout_2d: shape=(H, W), 各セルが部屋ラベル(str)
    room_list: 全部屋ラベルの一覧(例: ["l","d","k","r","t","b","c","c1","c2","...","."] etc)
               "." を含む想定
    戻り値: (onehot_tensor: (C,H,W), index_tensor: (H,W))
    """
    label2id = {r: i for i, r in enumerate(room_list)}
    h, w = layout_2d.shape
    index_map = np.zeros((h,w), dtype=np.int64)
    for rr in range(h):
        for cc in range(w):
            lab = layout_2d[rr, cc]
            idx = label2id.get(lab, label2id["."])  # 未知の場合 "." のID
            index_map[rr,cc] = idx

    c = len(room_list)
    index_tensor = torch.tensor(index_map, dtype=torch.long)
    onehot = F.one_hot(index_tensor, num_classes=c).float()  # (H,W,C)
    onehot = onehot.permute(2,0,1)  # => (C,H,W)
    return onehot, index_tensor