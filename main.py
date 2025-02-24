import pandas as pd
import glob
import os

from madori import core, visualizer
from madori.analyze_csv import analyze_1f_csv

def load_madori_data(filepath):
    try:
        df = pd.read_csv(filepath, header=None)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# CSVファイルを読み込み・表示
data_dir="data"
file_paths=glob.glob(os.path.join(data_dir,"*.csv"))
for fp in file_paths:
    print(f"\n--- Processing {fp} ---")
    df=load_madori_data(fp)
    if df is not None:
        print("\nData (First 5 rows):")
        print(df.head())
        print("\nData Shape:", df.shape)

# 1FのCSV解析
print("\n=== Analyzing 1F CSV files to propose config changes ===")
analyze_1f_csv("data/1F")

# 間取り自動生成
rows,cols=7,9
generated=core.generate_madori_rule_based(rows,cols)
print(generated)

# 可視化
visualizer.plot_madori(generated)