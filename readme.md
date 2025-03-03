# 間取り生成AIプロジェクト

このプロジェクトは、GANを使用して家の間取りを学習し、新しい間取りを生成するStreamlitアプリケーションを提供します。

## セットアップ

1. 必要なパッケージをインストールします：

```bash
python3 -m pip install -r requirements.txt
```

## データ準備

1. `data`ディレクトリにCSVファイルを配置します。
2. ファイル名に`1F.csv`や`2F.csv`が含まれるものを自動的に振り分けます：

```bash
python3 organize_floors.py
```

このコマンドを実行すると、以下のような構造に整理されます：
```
data/
  ├── 1F/
  │   ├── house001_1F.csv
  │   └── ...
  └── 2F/
      ├── house001_2F.csv
      └── ...
```

## データ確認ツール

CSVデータの内容を確認するためのツールが用意されています：

```bash
python3 list_1f_data.py [オプション]
```

### オプション

- `-h, --help`：ヘルプメッセージを表示
- `-d DIR, --dir DIR`：データディレクトリのパス（デフォルト: data/1F）
- `-n NUM, --num NUM`：表示するファイル数（デフォルト: 5、0で全件表示）
- `-p PATTERN, --pattern PATTERN`：ファイル名のパターン（例: "2042"）
- `-a, --all`：全てのファイルを表示（-n 0と同じ）

### 使用例

1. 最初の5件のファイルを表示：
   ```bash
   python3 list_1f_data.py
   ```

2. 特定のパターンを含むファイルだけを表示：
   ```bash
   python3 list_1f_data.py -p 2042
   ```

3. すべてのファイルを表示：
   ```bash
   python3 list_1f_data.py -a
   ```

## main.pyの実行

`main.py`は間取りデータの基本的な分析と、学習済みモデルによる間取り生成を行います：

```bash
python3 main.py
```

`main.py`の主な機能：
1. CSVファイルの読み込みと基本情報の表示
2. `data/1F`ディレクトリのCSV解析とコード単位での統計情報表示
3. 学習済みcGANモデルによる間取り生成とプロット表示

詳細は以下のコードを参照してください：

```python
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

    # 3) 学習済みモデルを使って間取りを予測 (GAN)
    print("\n=== Generating floor plans with the trained cGAN model ===")
    floors = predict_floorplans(num_samples=2, cond_value=0.2, gen_path="models/generator_ep100.pth")

    for i, floor_2d in enumerate(floors):
        print(f"[Generated {i}] shape={floor_2d.shape}")
        plot_madori(floor_2d.astype(str))
```

## GANモデルの学習

1. 以下のコマンドで間取りGANの学習を開始します：

```bash
python3 -m madori.train_gan --data_dir=data/1F --epochs=300 --use_spectral_norm --use_minibatch_discrim
```

### 主要なハイパーパラメータ

- `--data_dir`: 学習データのディレクトリ（1Fまたは2F）
- `--epochs`: 学習エポック数
- `--batch_size`: バッチサイズ（デフォルト：8）
- `--noise_dim`: ノイズベクトルの次元数（デフォルト：64）
- `--lr_g`: 生成器の学習率（デフォルト：1e-4）
- `--lr_d`: 識別器の学習率（デフォルト：1e-4）
- `--save_interval`: モデルの保存間隔（エポック単位、デフォルト：10）
- `--patience`: 早期終了のための忍耐値（デフォルト：50）
- `--use_spectral_norm`: スペクトル正規化を使用（フラグ）
- `--use_minibatch_discrim`: ミニバッチ識別を使用（フラグ）

学習したモデルは`models/`ディレクトリに保存されます。

## Streamlitアプリの実行

学習したモデルを使って間取りを生成し、可視化するStreamlitアプリを起動します：

```bash
python3 -m streamlit run generate.py
```

アプリでは以下の操作が可能です：
- 生成モデルの選択
- 間取りのサイズ指定（縦横のマス数）
- 部屋の占有率の調整（0〜1）
- 生成する間取り数の設定
- 自動トリミング機能のON/OFF

アプリは通常、ブラウザで http://localhost:8501 にアクセスして使用できます。

## 代替の実行方法

モデルのテストや間取り生成のみを行いたい場合は、以下のコマンドを使用できます：

```bash
python3 main.py
```

これにより、学習済みモデルを使用していくつかのサンプル間取りが生成されます。