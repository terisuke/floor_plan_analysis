import pandas as pd
import glob  # globモジュールをインポート
import os
from madori import core, visualizer  # madori_generator.py をインポート

def load_madori_data(filepath):
    """
    間取りデータCSVファイルを読み込み、データフレームを返す。
    空白なし、1フロア1ファイルの形式を前提とする。
    """
    try:
        df = pd.read_csv(filepath, header=None)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# CSVファイルを読み込み、データを確認
# dataディレクトリ内のすべてのCSVファイルを対象とする
data_dir = "data"  # dataディレクトリへのパス
file_paths = glob.glob(os.path.join(data_dir, "*.csv")) # dataディレクトリ内のすべての.csvファイルを取得

for filepath in file_paths:
    print(f"\n--- Processing {filepath} ---")
    df = load_madori_data(filepath)
    if df is not None:
        print("\nData (First 5 rows):")
        print(df.head())
        print("\nData Shape:", df.shape)

# 間取り生成のテスト
rows, cols = 7, 9  # 例: 7x9マス
generated_madori = core.generate_madori_rule_based(rows, cols)
print(generated_madori)

# 間取りの可視化(visualizer.pyを使う場合)
visualizer.plot_madori(generated_madori)