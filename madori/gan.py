import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class MinibatchDiscrimination(nn.Module):
    """
    簡易的なミニバッチ識別レイヤー。
    サンプル間の相互作用を取り込み、多様性の確保を促す。
    """
    def __init__(self, in_features, out_features, kernel_dims=5):
        super().__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))

    def forward(self, x):
        """
        x: shape=(N, in_features)
        戻り値: shape=(N, out_features)
        """
        N = x.size(0)
        M = x.matmul(self.T.view(x.size(1), -1))
        M = M.view(N, -1, self.T.size(2))  # => (N, out_features, kernel_dims)
        out = []
        for i in range(N):
            diff = torch.sum(torch.abs(M[i] - M), dim=2)  # shape=(N, out_features)
            out.append(diff)
        out = torch.stack(out, dim=0)  # => (N, N, out_features)
        out = torch.sum(torch.exp(-out), dim=1)  # => (N, out_features)
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

        in_dim = noise_dim + cond_dim
        self.fc = nn.Linear(in_dim, base_channels*8*4*4)

        self.convTrans_blocks = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, 2, 1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels, out_channels, 4, 2, 1),
            # 出力 shape: (out_channels,64,64)
        )

    def forward(self, noise_z, cond):
        x = torch.cat([noise_z, cond], dim=1)  # shape=(N, noise_dim+cond_dim)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        out = self.convTrans_blocks(x)
        return out

class Discriminator(nn.Module):
    """
    PatchGANベースのDiscriminator + SpectralNorm, MinibatchDiscriminationのオプション
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

        # 条件用埋め込み(Projection Discriminatorほど本格的ではない簡易版)
        self.cond_emb = nn.Embedding(128, cond_dim)
        nn.init.xavier_uniform_(self.cond_emb.weight)

        def sn(layer):
            return spectral_norm(layer) if use_spectral_norm else layer

        self.conv1 = sn(nn.Conv2d(in_channels + 1, base_channels, 4, 2, 1))
        self.conv2 = sn(nn.Conv2d(base_channels, base_channels*2, 4, 2, 1))
        self.conv3 = sn(nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1))
        self.conv4 = sn(nn.Conv2d(base_channels*4, 1, 4, 1, 0))

        self.bn2 = nn.BatchNorm2d(base_channels*2)
        self.bn3 = nn.BatchNorm2d(base_channels*4)

        if self.use_minibatch:
            # conv3 出力 => flatten => minibatch => fc_out
            self.minibatch_features = base_channels*4*8*8
            self.mbd = MinibatchDiscrimination(self.minibatch_features, 50)
            self.fc_out = nn.Linear(50, 1)

    def forward(self, layout_logits, cond):
        """
        layout_logits: shape=(N, in_channels, H, W)
        cond: shape=(N, cond_dim)
        戻り値: shape=(N,1,h',w') or (N,1)
        """
        N, _, H, W = layout_logits.size()

        # condを埋め込み→(N,cond_dim,1,1)→(N,cond_dim,H,W)
        cond_idx = cond[:,0].long().clamp(min=0, max=127)
        cond_emb_vec = self.cond_emb(cond_idx)
        cond_map = cond_emb_vec.view(N, self.cond_dim, 1, 1).expand(N, self.cond_dim, H, W)

        merged = torch.cat([layout_logits, cond_map], dim=1)

        x = F.leaky_relu(self.conv1(merged), 0.2, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, inplace=True)

        if self.use_minibatch:
            xf = x.view(N, -1)
            mb = self.mbd(xf)
            out = self.fc_out(mb)  # => (N,1)
            return out
        else:
            out = self.conv4(x)  # => (N,1,h',w')
            return out