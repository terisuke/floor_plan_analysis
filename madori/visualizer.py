import matplotlib.pyplot as plt
import numpy as np

def plot_madori(madori):
    """Matplotlibを使用して間取りを可視化する"""
    rows, cols = madori.shape
    fig, ax = plt.subplots()

    # 部屋ごとに色を変えるためのカラーマップ
    cmap = plt.cm.get_cmap('tab20', len(np.unique(madori)))

    # 間取りデータを数値データに変換
    numeric_madori = np.zeros((rows, cols))
    for i, room_code in enumerate(np.unique(madori)):
        numeric_madori[madori == room_code] = i

    # 数値データに基づいて間取りを描画
    ax.imshow(numeric_madori, cmap=cmap, interpolation='none')

    # グリッド線を表示
    ax.set_xticks(np.arange(cols) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False) # グリッドの補助目盛を消す

    # 部屋コードを表示
    for r in range(rows):
      for c in range(cols):
        ax.text(c, r, madori[r, c], ha='center', va='center', color='black')

    plt.show()