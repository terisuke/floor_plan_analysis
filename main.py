import pandas as pd
import re

def load_and_process_madori_data(filepath):
    """
    間取りデータCSVファイルを読み込み、前処理を行う関数。

    Args:
        filepath (str): CSVファイルのパス。

    Returns:
        tuple: 1Fと2Fのデータフレームのタプル。
               (DataFrame for 1F, DataFrame for 2F)
               エラーの場合はNoneを返す。
    """
    try:
        # CSVファイルの読み込み (区切り文字を正規表現で指定, 不要な行をスキップ)
        df = pd.read_csv(filepath, sep=',+', engine='python', skiprows=4, header=None)

        # 1Fと2Fのデータを分割
        df_1f = None
        df_2f = None
        for i in range(len(df)):
            if "2F" in str(df.iloc[i].values):
                df_1f = df.iloc[:i]
                df_2f = df.iloc[i+1:]
                break

        if df_1f is None or df_2f is None:
            print(f"Error: Could not split 1F and 2F data in {filepath}")
            return None

        # 2Fのデータフレームのインデックスとカラムを振りなおす
        df_2f = df_2f.reset_index(drop=True)  # Reset index
        df_2f = df_2f.dropna(axis=1, how='all') # Remove columns if they are all NaN
        df_1f = df_1f.dropna(axis=1, how='all') # Remove columns if they are all NaN

        # 記号説明部分を削除
        # "記号"という文字列が含まれる行を特定し、それ以降の行を削除
        for i in range(len(df_2f)):
          if "記号" in str(df_2f.iloc[i].values):
            df_2f = df_2f.iloc[:i]
            break
        
        return df_1f, df_2f


    except Exception as e:
        print(f"Error loading or processing {filepath}: {e}")
        return None

# 2つのCSVファイルを読み込み、データを確認
file_paths = ["data/U-dake建物配置 - No2112 .csv", "data/U-dake建物配置 - No2165 .csv"]

for filepath in file_paths:
    print(f"\n--- Processing {filepath} ---")
    result = load_and_process_madori_data(filepath)

    if result:
        df_1f, df_2f = result

        print("\n1F Data (First 5 rows):")
        print(df_1f.head())
        print("\n1F Data Shape:", df_1f.shape)

        print("\n2F Data (First 5 rows):")
        print(df_2f.head())
        print("\n2F Data Shape:", df_2f.shape)
    else:
        print("skipping this file")