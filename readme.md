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