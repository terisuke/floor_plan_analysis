# predict_model.py

import numpy as np
import torch
from .train_model import VAE  # 学習時と同じモデルクラスをインポート

# 生成するサンプル数を指定
num_samples = 10

# 学習時に使用した設定を合わせる
H, W = 32, 32           # グリッドサイズ（学習データと同じ）
num_classes = 5         # カテゴリ数（学習データに応じて設定）
latent_dim = 100        # 学習時の潜在ベクトル次元

# モデルを再構築し重みをロード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(in_channels=num_classes, latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load("floorplan_vae.pth", map_location=device))
model.eval()

# ランダムな間取りを生成
generated_layouts = []
for i in range(num_samples):
    # 潜在ベクトルzを標準正規分布からサンプリング
    z = torch.randn(1, latent_dim).to(device)
    # デコーダで間取りを生成（モデルのdecode関数を利用）
    with torch.no_grad():
        logits = model.decode(z)
        # ロジットから各セルのカテゴリを選択（最大のチャネルが予測カテゴリ）
        pred_labels = logits.argmax(dim=1)  # 出力shape: (1, H, W)
    layout = pred_labels.squeeze(0).cpu().numpy()  # (H, W)のnumpy配列に変換
    # 平坦化してリストに保存
    generated_layouts.append(layout.reshape(-1))

# 生成結果をCSVに保存（各行が1つの生成間取り）
generated_layouts = np.array(generated_layouts, dtype=int)
np.savetxt("generated_floorplans.csv", generated_layouts, fmt="%d", delimiter=",")
print(f"{num_samples}件の生成間取りをgenerated_floorplans.csvに保存しました。")