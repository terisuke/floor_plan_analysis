import pandas as pd
import glob
import os

# 不要: from madori import core
from madori.analyze_csv import analyze_1f_csv
from madori.visualizer import plot_madori

from madori.train_model import train_random_forest
from madori.predict_model import predict_floor

def load_madori_data(filepath):
    try:
        df = pd.read_csv(filepath, header=None)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# メイン処理
if __name__ == "__main__":
    # 1) CSVファイルを読み込み確認 (データ内容プリント)
    data_dir = "data"
    file_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    for fp in file_paths:
        print(f"\n--- Processing {fp} ---")
        df = load_madori_data(fp)
        if df is not None:
            print("\nData (First 5 rows):")
            print(df.head())
            print("\nData Shape:", df.shape)

    # 2) 1FのCSV解析(元々の分析用)
    print("\n=== Analyzing 1F CSV files to propose config changes ===")
    analyze_1f_csv("data/1F")

    # 3) モデルを学習 (ランダムフォレスト)
    #    - 必要に応じてコメントアウト/有効化
    print("\n=== Training model (RandomForest) ===")
    train_random_forest(data_dir="data/1F", model_path="models/floor_model.pkl")

    # 4) 学習済みモデルを使って間取りを予測
    #    - 予測結果をCSVに出力 & numpy配列(madori)を返す
    print("\n=== Predicting floor plan with the trained model ===")
    predicted_madori = predict_floor(model_path="models/floor_model.pkl",
                                     rows=7, cols=9,
                                     output_csv="predicted_floor.csv")

    print("\n=== Predicted Madori (Head) ===")
    print(predicted_madori[:5])  # 先頭5行だけ表示

    # 5) 可視化
    plot_madori(predicted_madori)