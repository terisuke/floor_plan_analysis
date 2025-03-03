import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm  # スペクトル正規化

class MinibatchDiscrimination(nn.Module):
    """
    簡易的なミニバッチ識別レイヤー。
    サンプルごとの特徴量に相互作用を与え、多様性促進を図る。
    """
    def __init__(self, in_features, out_features, kernel_dims=5):
        super().__init__()
        # テンソル: (in_features, out_features, kernel_dims)
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))

    def forward(self, x):
        """
        x: shape=(N, in_features)
        戻り値: shape=(N, out_features)
        """
        # batch_size = N
        N = x.size(0)
        M = x.matmul(self.T.view(x.size(1), -1))  # (N, out_features*kernel_dims)
        M = M.view(N, -1, self.T.size(2))         # => (N, out_features, kernel_dims)
        out = []
        for i in range(N):
            # broadcast / L1 dist
            diff = torch.sum(torch.abs(M[i] - M), dim=2)  # shape (N, out_features)
            out.append(diff)
        out = torch.stack(out, dim=0)  # (N, N, out_features)
        # sum over batch dim (or exp(-out)? など色々)
        out = torch.sum(torch.exp(-out), dim=1)
        return out

class Generator(nn.Module):
    """
    条件付きGANのGenerator:
      - 入力: (noise_z, cond_vector)
      - 出力: 2Dレイアウト (C,H,W) logits
    """
    def __init__(self, noise_dim=100, cond_dim=10, base_channels=64, out_channels=10):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim

        # 全結合で  (noise+cond) -> feature
        in_dim = noise_dim + cond_dim
        self.fc = nn.Linear(in_dim, base_channels*8*4*4)  # 4x4の特徴マップに変換

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
        戻り値: shape=(N, out_channels, 64, 64)
        """
        x = torch.cat([noise_z, cond], dim=1)  # (N, noise_dim+cond_dim)
        x = self.fc(x)  # (N, base_channels*8*4*4)
        x = x.view(x.size(0), -1, 4, 4)  # => (N, base_channels*8, 4,4)
        out = self.convTrans_blocks(x)   # => (N, out_channels, 64,64)
        return out

class Discriminator(nn.Module):
    """
    PatchGANベースのDiscriminator + オプションでSpectralNorm, MinibatchDiscrimination, 条件埋め込み。
    """
    def __init__(
        self,
        in_channels=10,
        cond_dim=10,
        base_channels=64,
        use_spectral_norm=True,
        use_minibatch=False
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.use_minibatch = use_minibatch

        # 条件用の埋め込み(Projection Discriminatorほど本格的ではない簡易版)
        self.cond_emb = nn.Embedding(128, cond_dim)  # 128種類まで想定(適宜拡大)
        nn.init.xavier_uniform_(self.cond_emb.weight)

        # CNNブロック
        # (layoutC + 1) チャネル => in_channels + 1
        def sn(layer):
            return spectral_norm(layer) if use_spectral_norm else layer

        self.conv1 = sn(nn.Conv2d(in_channels + 1, base_channels, 4, 2, 1))
        self.conv2 = sn(nn.Conv2d(base_channels, base_channels*2, 4, 2, 1))
        self.conv3 = sn(nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1))
        self.conv4 = sn(nn.Conv2d(base_channels*4, 1, 4, 1, 0))

        self.bn2 = nn.BatchNorm2d(base_channels*2)
        self.bn3 = nn.BatchNorm2d(base_channels*4)

        # ミニバッチ識別用に次元を合わせる:
        # conv4直前 or conv3後あたりで flatten => minibatch layer => 1chに近い形に再マッピング
        if self.use_minibatch:
            # minibatch layer: 次元は適当; conv3出力 (bs, base_channels*4, h', w')
            # h'=w'=8 => flattenすると  (bs, base_channels*4*8*8)
            self.minibatch_features = base_channels*4*8*8
            self.mbd = MinibatchDiscrimination(self.minibatch_features, 50)  # 50はハイパラ

            # 最終出力にまとめる全結合
            self.fc_out = nn.Linear(50, 1)

    def forward(self, layout_logits, cond):
        """
        layout_logits: shape=(N, in_channels, H, W)
        cond        : shape=(N, cond_dim)
        戻り値: shape=(N,1,h',w') => patchごとのreal/fake判定 (or (N,1) if use_minibatch)
        """
        N, _, H, W = layout_logits.size()

        # 埋め込みを使って条件を(1, H, W)にブロードキャストする(簡易)
        # condは (N, cond_dim) 中身は連続値の場合と、IDの場合で分ける => ここでは離散ID想定に一例
        # 例: cond[:,0]をintにキャストしてEmbeddingでlookup
        # 本来は連続値なら別設計が必要
        cond_idx = cond[:,0].long().clamp(min=0, max=127)  # 0~127にクリップ
        cond_emb_vec = self.cond_emb(cond_idx)  # (N, cond_dim)
        # reshape => (N, cond_dim, 1, 1)
        cond_emb_map = cond_emb_vec.view(N, self.cond_dim, 1, 1).expand(N, self.cond_dim, H, W)

        merged = torch.cat([layout_logits, cond_emb_map], dim=1)  # (N, in_channels+cond_dim, H, W)

        x = F.leaky_relu(self.conv1(merged), 0.2, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, inplace=True)

        if self.use_minibatch:
            # minibatch discrimination
            # x => flatten
            xf = x.view(N, -1)  # (N, base_channels*4* h'* w') => (N, 4*64*8*8)など
            mb = self.mbd(xf)   # => (N, 50)
            out = self.fc_out(mb)  # => (N,1)
            return out
        else:
            # patch出力 => conv4
            out = self.conv4(x)  # (N,1, h', w')
            return out