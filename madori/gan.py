# madori/gan.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    条件付きGANのGenerator:
      - 入力: (noise_z, cond_vector)
      - 出力: 2Dレイアウト (C,H,W) logits
    この例ではシンプルなアップサンプリングCNNに条件ベクトルを結合する書き方。
    """
    def __init__(self, noise_dim=100, cond_dim=10, base_channels=64, out_channels=10):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim

        # 全結合で  (noise+cond) -> feature
        in_dim = noise_dim + cond_dim
        self.fc = nn.Linear(in_dim, base_channels*8*4*4)  # 4x4の特徴マップに変換する例

        # 転置畳み込みでアップサンプリング
        self.convTrans_blocks = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1),
            # 出力 (out_channels, 64, 64) を想定
            # 最終の活性化はつけず => ロジット
        )

    def forward(self, noise_z, cond):
        """
        noise_z: shape=(N, noise_dim)
        cond   : shape=(N, cond_dim)
        戻り値: shape=(N, out_channels, 64, 64) (例)
        """
        x = torch.cat([noise_z, cond], dim=1)  # (N, noise_dim+cond_dim)
        x = self.fc(x)  # (N, base_channels*8*4*4)
        x = x.view(x.size(0), -1, 4, 4)  # => (N, base_channels*8, 4,4)
        out = self.convTrans_blocks(x)   # => (N, out_channels, 64,64)
        return out

class Discriminator(nn.Module):
    """
    PatchGANベースのDiscriminator:
      - 入力: (layout + conditionをチャネル方向にconcat)
      - 出力: patchごとの real/fake 判定スカラー (N,1,H/16,W/16) など
    """
    def __init__(self, in_channels=10, cond_dim=10, base_channels=64):
        super().__init__()
        # 条件を画像チャネルにブロードキャストしてconcatするなどの工夫が必要
        # ここでは簡易的に (layoutC + 1) チャネル追加程度
        self.cond_dim = cond_dim
        # CNNブロック
        self.main = nn.Sequential(
            nn.Conv2d(in_channels + 1, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels*4, 1, kernel_size=4, stride=1, padding=0)
            # => 出力shape: (N,1, h', w')  (PatchGAN)
        )

    def forward(self, layout_logits, cond):
        """
        layout_logits: shape=(N, in_channels, H, W)
        cond        : shape=(N, cond_dim)
        戻り値: shape=(N,1,h',w') => patchごとのreal/fake判定
        """
        # condを画像サイズにブロードキャストする例（雑実装）
        # condを (N,cond_dim,1,1) にreshapeして expand
        N, _, H, W = layout_logits.size()
        cond_map = cond.view(N, self.cond_dim, 1, 1).expand(N, self.cond_dim, H, W)
        # ただし Discriminatorでは cond_dim=1想定みたいな簡易例にするのが楽
        # ここでは cond_dim=1 にしているものとする
        # layout と cond_map をチャネル方向で concat すると (N, in_channels+cond_dim, H, W)
        # しかし自前の initでは in_channels+1 => OK
        merged = torch.cat([layout_logits, cond_map], dim=1)
        out = self.main(merged)  # => (N,1,h',w')
        return out