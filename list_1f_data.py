import os
import glob
import pandas as pd
import argparse

def load_madori_data(filepath):
    """CSVファイルを読み込み、DataFrameとして返す"""
    try:
        df = pd.read_csv(filepath, header=None)
        return df
    except Exception as e:
        print(f"エラー: {filepath}の読み込みに失敗しました: {e}")
        return None

def list_1f_data(data_dir="data/1F", max_files=5, pattern=None):
    """data/1Fディレクトリ内のCSVファイルを一覧表示"""
    print(f"\n=== {data_dir}ディレクトリのCSVファイル一覧 ===")
    
    # CSVファイルのパスを取得
    if pattern:
        file_paths = glob.glob(os.path.join(data_dir, f"*{pattern}*.csv"))
        print(f"パターン '{pattern}' に一致するCSVファイルを検索しています...")
    else:
        file_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not file_paths:
        print(f"{data_dir}にCSVファイルが見つかりませんでした。")
        return
    
    # ファイルをソート
    file_paths.sort()
    
    print(f"合計{len(file_paths)}個のCSVファイルが見つかりました。\n")
    
    # ユーザーが確認するファイル数を制限
    if max_files > 0 and len(file_paths) > max_files:
        print(f"最初の{max_files}個のファイルを表示します。")
        file_paths = file_paths[:max_files]
    
    # 各CSVファイルの内容を表示
    for i, fp in enumerate(file_paths, 1):
        print(f"\n--- ファイル {i}: {os.path.basename(fp)} ---")
        df = load_madori_data(fp)
        if df is not None:
            print("\nデータ (全行):")
            print(df)
            print("\nデータの形状:", df.shape)
            print("\n間取りの生データ表示:")
            # 間取りデータをテキストとして表示
            for row in df.values:
                print(" ".join(str(cell) for cell in row))
            print("\n" + "-"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data/1Fディレクトリ内のCSVファイルを一覧表示します。')
    parser.add_argument('-d', '--dir', default="data/1F", help='データディレクトリのパス（デフォルト: data/1F）')
    parser.add_argument('-n', '--num', type=int, default=5, help='表示するファイル数（デフォルト: 5、0で全件表示）')
    parser.add_argument('-p', '--pattern', help='ファイル名のパターン（例: "2042"）')
    parser.add_argument('-a', '--all', action='store_true', help='全てのファイルを表示する（-n 0と同じ）')
    
    args = parser.parse_args()
    
    # -aオプションが指定された場合は全ファイル表示
    if args.all:
        args.num = 0
    
    list_1f_data(args.dir, args.num, args.pattern) 