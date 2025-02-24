# train_model.py

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import glob
import os
import math
from PIL import Image
import matplotlib.pyplot as plt

# --- ハイパーパラメータと設定 ---
latent_dim = 200        # 潜在ベクトルの次元数を増やして表現力を向上
batch_size = 16         # バッチサイズを小さくして学習を安定化
num_epochs = 500        # エポック数を500に増やす
learning_rate = 5e-4    # 学習率を少し小さくして細かく学習

# 追加の設定
kld_weight = 0.01      # KLD損失の重み（再構成誤差とのバランスを調整）

# GPUが使える場合はGPUを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- データ前処理: CSV読み込みとone-hotエンコーディング ---
# data/1Fディレクトリから全てのCSVファイルを読み込む
csv_files = glob.glob('data/1F/*.csv')
if not csv_files:
    raise ValueError("data/1FディレクトリにCSVファイルが見つかりません。")

# 固定グリッドサイズを設定（すべての間取りをこのサイズにリサイズ）
TARGET_H, TARGET_W = 64, 64  # 目標サイズ

# 部屋の種類を数値にマッピングする辞書を定義
ROOM_TYPES = {
    '': 0,  # 空白
    'r': 1,  # 廊下
    'b': 2,  # 浴室
    'ut': 3,  # ユーティリティ
    's': 4,  # 収納
    'co': 5,  # 玄関
    't': 6,  # トイレ
    'c': 7,  # クローゼット
    'k': 8,  # キッチン
    'd': 9,  # ダイニング
    'l': 10,  # リビング
    'H': 11,  # ホール
    'e': 12,  # エントランス
}

def process_csv(file):
    # CSVファイルを文字列として読み込む
    df = pd.read_csv(file, dtype=str)
    
    # 空白セルをNaNに変換し、その後空文字列に変換
    df = df.replace('', np.nan).fillna('')
    
    # 文字列を数値に変換
    data = np.zeros(df.shape)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iloc[i, j].strip()
            data[i, j] = ROOM_TYPES.get(value, 0)  # 未知の値は0として扱う
    
    return data

# ファイル処理部分を修正
processed_data = []
for file in csv_files:
    try:
        data = process_csv(file)
        
        # パディングを使用して正方形にする
        rows, cols = data.shape
        max_size = max(rows, cols)
        padded_data = np.zeros((max_size, max_size))
        padded_data[:rows, :cols] = data
        
        # リサイズ
        img = Image.fromarray(padded_data.astype('float32'))
        resized = img.resize((TARGET_W, TARGET_H), Image.Resampling.NEAREST)
        processed_data.append(np.array(resized))
        
        print(f"成功: {file}を処理しました")
        print(f"元のサイズ: {rows}x{cols}, パディング後: {max_size}x{max_size}")
        
    except Exception as e:
        print(f"警告: {file}の処理中にエラーが発生しました: {e}")
        continue

# 全てのデータを結合
if not processed_data:
    raise ValueError("処理可能なデータがありませんでした。")

data_array = np.stack(processed_data)
H, W = TARGET_H, TARGET_W

print(f"処理されたデータの形状: {data_array.shape}")
print(f"グリッドサイズ: {H}x{W}")

# グリッド上のユニークなカテゴリ値を取得し、カテゴリー数を決定
unique_vals = np.unique(data_array)
num_classes = len(unique_vals)
print(f"検出されたクラス数: {num_classes}")

# カテゴリー値が0,1,...連番でない場合に対応するインデックスマップを作成
class_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
# マップを適用して、全セルを0～num_classes-1のインデックスに変換
indexed_grid = np.vectorize(class_to_idx.get)(data_array)

# one-hotエンコーディング: shape (N, H, W) -> (N, H, W, C) -> (N, C, H, W)
indexed_tensor = torch.tensor(indexed_grid, dtype=torch.long)
onehot_tensor = F.one_hot(indexed_tensor, num_classes=num_classes).float()
onehot_tensor = onehot_tensor.permute(0, 3, 1, 2)  # (N, C, H, W) に変換

# 学習用データセットとデータローダーを作成
dataset = TensorDataset(onehot_tensor, indexed_tensor)  # 入力(one-hot)とターゲット(クラスインデックス)のペア
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- VAEモデルの定義 ---
class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        # Encoder: 畳み込み層で特徴を抽出し、全結合層で潜在変数の平均(mu)と対数分散(logvar)を出力
        self.enc_conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)   # 出力: 16x16 (入力32x32想定)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)            # 出力: 8x8
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)           # 出力: 4x4
        # 4x4サイズ・チャネル128の特徴マップをフラットにして潜在ベクトルへ
        self.enc_fc_mu = nn.Linear(128 * (H//8) * (W//8), latent_dim)      # mu出力
        self.enc_fc_logvar = nn.Linear(128 * (H//8) * (W//8), latent_dim)  # logvar出力

        # Decoder: 全結合層で特徴マップに復元し、転置畳み込みで元の画像サイズ(CxHxW)に生成
        self.dec_fc = nn.Linear(latent_dim, 128 * (H//8) * (W//8))
        self.dec_deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 出力: 8x8
        self.dec_deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 出力: 16x16
        self.dec_deconv3 = nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1)  # 出力: 32x32

    def encode(self, x):
        # Encoder: 入力xを畳み込みで特徴抽出しflattenしてmu, logvarを計算
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = h.contiguous().view(x.size(0), -1)
        mu = self.enc_fc_mu(h)
        logvar = self.enc_fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # 再パラメータ化: 推論時は平均をそのまま返し、学習時はN(0,1)ノイズを加えてサンプリング
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # 評価時は分散項を考慮しない（平均のみ）

    def decode(self, z):
        # Decoder: 潜在ベクトルzから元の画像次元にデコード（出力は各ピクセルのカテゴリlogit）
        h = F.relu(self.dec_fc(z))
        h = h.view(z.size(0), 128, H//8, W//8)  # 4x4x128のテンソルに変形（H//8=4, W//8=4）
        h = F.relu(self.dec_deconv1(h))
        h = F.relu(self.dec_deconv2(h))
        logits = self.dec_deconv3(h)  # 最終出力（活性化関数は適用しない：クロスエントロピーでlogitsを使用）
        return logits

    def forward(self, x):
        # VAEの順伝搬: x -> (mu, logvar) -> reparameterize -> デコードlogits
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

# モデルを初期化しデバイスに転送
model = VAE(in_channels=num_classes, latent_dim=latent_dim).to(device)

# オプティマイザの設定
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 学習ループ ---
model.train()
best_loss = float('inf')
for epoch in range(1, num_epochs+1):
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kld_loss = 0.0
    
    for batch_x, batch_labels in loader:
        batch_x = batch_x.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        
        recon_logits, mu, logvar = model(batch_x)
        recon_loss = F.cross_entropy(recon_logits, batch_labels)
        
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD / batch_x.size(0)
        
        # KLD重みを適用
        loss = recon_loss + kld_weight * KLD
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kld_loss += KLD.item()
    
    # エポックごとの損失を詳細に出力
    avg_loss = total_loss/len(loader)
    avg_recon = total_recon_loss/len(loader)
    avg_kld = total_kld_loss/len(loader)
    
    if epoch % 10 == 0:  # 10エポックごとに詳細を表示
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Recon Loss: {avg_recon:.4f}")
        print(f"  KLD Loss: {avg_kld:.4f}")
    else:  # それ以外は簡易表示
        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    # 最良モデルを保存
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "models/floorplan_vae_best.pth")

# 最終モデルも保存
torch.save(model.state_dict(), "models/floorplan_vae_final.pth")
print("学習が完了しました。")

def visualize_floor_plan(data, title=None):
    """データの可視化関数"""
    plt.figure(figsize=(10, 10))
    im = plt.imshow(data, cmap='tab20')
    if title:
        plt.title(title)
    plt.colorbar(im, ticks=range(len(ROOM_TYPES)))
    plt.clim(-0.5, len(ROOM_TYPES) - 0.5)
    plt.grid(True)
    plt.show()

# モデルの評価
model.eval()
with torch.no_grad():
    # テストデータの生成
    sample_idx = np.random.randint(len(processed_data))
    sample_data = processed_data[sample_idx]
    
    # 元のデータを表示
    visualize_floor_plan(sample_data, "Original Floor Plan")
    
    # VAEでの再構成
    sample_tensor = F.one_hot(torch.tensor(sample_data).long(), 
                             num_classes=num_classes).float()
    sample_tensor = sample_tensor.unsqueeze(0).permute(0, 3, 1, 2).to(device)
    recon_logits, _, _ = model(sample_tensor)
    
    # 再構成結果を表示
    recon_data = recon_logits[0].argmax(dim=0).cpu().numpy()
    visualize_floor_plan(recon_data, "Reconstructed Floor Plan")