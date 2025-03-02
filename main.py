import pandas as pd
import glob
import os

# 不要: from madori import core
from madori.analyze_csv import analyze_1f_csv
from madori.visualizer import plot_madori

# 【旧】from madori.train_model import train_random_forest
# 代わりに cGANのtrain_ganを使うなら以下
# from madori.train_gan import train_gan

from madori.predict_model import predict_floorplans

def load_madori_data(filepath):
    try:
        df = pd.read_csv(filepath, header=None)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# メイン処理
if __name__ == "__main__":
    # 1) CSVファイルを読み込み確認
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

    # 下記は旧RandomForestの呼び出し。不要ならコメントアウト
    # print("\n=== Training model (RandomForest) ===")
    # train_random_forest(data_dir="data/1F", model_path="models/floor_model.pkl")

    # cGANで学習する場合は:
    # from madori.train_gan import train_gan
    # train_gan(data_dir="data/1F", epochs=50, batch_size=4)

    # 3) 学習済みモデルを使って間取りを予測 (GAN)
    print("\n=== Generating floor plans with the trained cGAN model ===")
    # ここではpredict_floorplans関数を試す
    # (注) predict_floorplansはdummy cond=0.2 などで動く例
    floors = predict_floorplans(num_samples=2, cond_value=0.2, gen_path="models/generator_ep100.pth")

    for i, floor_2d in enumerate(floors):
        print(f"[Generated {i}] shape={floor_2d.shape}")
        # plotできるなら可視化(要matplotlibなど)
        # ただしfloor_2dは intラベル。可視化には-> color map
        plot_madori(floor_2d.astype(str))