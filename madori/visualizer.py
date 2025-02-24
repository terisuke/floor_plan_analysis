import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

def plot_madori(madori, custom_colors=None):
    """Matplotlibを使用して間取りを可視化する
    
    Args:
        madori: numpy array of floor layout
        custom_colors: オプションで指定する色のリスト。Noneの場合はデフォルトカラーを使用
    """
    rows, cols = madori.shape
    fig, ax = plt.subplots(figsize=(8, 8))

    # カスタムカラーマップの設定
    if custom_colors is None:
        # デフォルトのカラー設定
        custom_colors = ['white', 'orange', 'skyblue', 'green', 'purple']
    cmap = colors.ListedColormap(custom_colors)
    
    # カテゴリ数を取得
    num_categories = len(np.unique(madori))
    
    # 正規化（カテゴリ値を離散的に扱う）
    norm = colors.BoundaryNorm(
        boundaries=np.arange(num_categories + 1) - 0.5,
        ncolors=num_categories
    )

    # 間取りを描画
    im = ax.imshow(madori, cmap=cmap, norm=norm)

    # グリッド線を表示
    ax.set_xticks(np.arange(cols) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 部屋コードを表示
    for r in range(rows):
        for c in range(cols):
            ax.text(c, r, madori[r, c], ha='center', va='center', color='black')

    # カラーバーを追加
    plt.colorbar(im, ax=ax, ticks=range(num_categories))
    
    plt.title("Floor Layout")
    plt.show()