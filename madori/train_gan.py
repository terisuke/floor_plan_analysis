import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

from .gan import Generator, Discriminator
from .preprocess import load_and_augment_csvs
from .evaluate_model import evaluate_generated_layouts
import torch.nn.functional as F

# 必須部屋のリスト
REQUIRED_ROOMS = ["r", "r1", "r2", "r3", "r4", "L", "D", "K", "t", "B"]  # 例: 必須部屋

#==================================================================
# 1) 引数パース
#==================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train WGAN-GP for floor plan generation.")
    parser.add_argument("--data_dir", type=str, default="data/1F", help="Data directory with CSVs")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--noise_dim", type=int, default=64, help="Dim. of noise vector")
    parser.add_argument("--cond_dim", type=int, default=1, help="Dim. of condition vector")
    parser.add_argument("--out_channels", type=int, default=18, help="Output channels for generator")
    parser.add_argument("--lr_g", type=float, default=1e-4, help="Learning rate for generator")
    parser.add_argument("--lr_d", type=float, default=1e-5, help="Learning rate for discriminator")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval (epochs) to save checkpoint")
    # Early Stopping関連
    parser.add_argument("--patience", type=int, default=30, help="patience for early stopping")
    return parser.parse_args()

#==================================================================
# 2) Dataset
#==================================================================
class FloorPlanDataset(Dataset):
    """
    省略（従来のものと同様）。condは適当に1値...
    """
    def __init__(self, layouts, max_size=64):
        super().__init__()
        self.data = []
        self.max_size = max_size
        self.label_set = set()
        for (lay, fp) in layouts:
            self.label_set.update(lay.reshape(-1).tolist())
        self.label_list = sorted(list(self.label_set))
        self.label2id = {lab:i for i,lab in enumerate(self.label_list)}

        for (lay, _) in layouts:
            padded = self.pad_layout(lay, self.max_size)
            index = np.zeros((self.max_size,self.max_size), dtype=np.int64)
            h, w = lay.shape
            for rr in range(h):
                for cc in range(w):
                    idx = self.label2id.get(lay[rr,cc], 0)
                    index[rr,cc] = idx
            self.data.append(index)

    def pad_layout(self, layout_2d, size):
        h, w = layout_2d.shape
        pad_h = size - h
        pad_w = size - w
        new_arr = np.full((size,size), ".", dtype=object)
        new_arr[:h,:w] = layout_2d
        return new_arr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        index_map = self.data[idx]
        nonzero_count = np.count_nonzero(index_map)
        cond_val = np.array([nonzero_count/ (self.max_size*self.max_size)], dtype=np.float32)

        index_tensor = torch.tensor(index_map, dtype=torch.long)
        c = len(self.label_list)
        onehot = F.one_hot(index_tensor, num_classes=c).float().permute(2,0,1)
        return onehot, index_tensor, cond_val

#==================================================================
# 3) WGAN-GP (勾配ペナルティ) 関数
#==================================================================
def gradient_penalty(netD, real_data, fake_data, cond):
    """
    real_data, fake_data: (N, out_channels, H, W)
    cond: (N, cond_dim)
    """
    alpha = torch.rand(real_data.size(0), 1, 1, 1, device=real_data.device)
    alpha = alpha.expand_as(real_data)  # broadcast
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated = interpolated.requires_grad_(True)

    prob_interpolated = netD(interpolated, cond)  # (N,1,H',W')
    # WGAN-GPでは、出力をスカラーにする実装も多いが、unet風にサイズが大きい時はmeanなどする
    # ここでは 全画素平均などする:
    prob_mean = prob_interpolated.mean()

    grads = torch.autograd.grad(
        outputs=prob_mean,
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # shape (N, out_channels, H, W)
    grads = grads.view(grads.size(0), -1)
    grad_norm = grads.norm(2, dim=1)  # L2 norm => shape(N)

    # (|grad| - 1)^2
    gradient_pen = ((grad_norm - 1.0)**2).mean()
    return gradient_pen

#==================================================================
# 4) EarlyStopping Class (例)
#==================================================================
class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def check(self, val_loss):
        # val_lossが best_lossより [min_delta]だけ小さければ更新
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

#==================================================================
# 5) train_wgan_gp本体
#==================================================================
def train_wgan_gp(
    data_dir,
    epochs,
    batch_size,
    noise_dim,
    cond_dim,
    out_channels,
    lr_g,
    lr_d,
    save_interval,
    patience
):
    layouts = load_and_augment_csvs(data_dir=data_dir, do_augmentation=True)
    dataset = FloorPlanDataset(layouts, max_size=64)
    label_list = dataset.label_list
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = Generator(
        noise_dim=noise_dim,
        cond_dim=cond_dim,
        base_channels=64,
        out_channels=18  # 17+1に合わせる（Discriminatorが処理可能なチャンネル数）
    ).to(device)

    netD = Discriminator(
        in_channels=17,  # 実際のデータチャンネル数(18)-条件チャネル数(1)=17
        cond_dim=cond_dim,
        base_channels=64
    ).to(device)

    optG = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.0, 0.9))
    optD = optim.Adam(netD.parameters(), lr=lr_d, betas=(0.0, 0.9))

    # lr_scheduler
    schedulerG = optim.lr_scheduler.ReduceLROnPlateau(optG, mode='min', factor=0.5, patience=10)
    schedulerD = optim.lr_scheduler.ReduceLROnPlateau(optD, mode='min', factor=0.5, patience=10)

    gp_lambda = 10.0  # GPの係数

    early_stopper = EarlyStopping(patience=patience, min_delta=0.0)

    for ep in range(1, epochs+1):
        g_losses = []
        d_losses = []

        for i, (onehot, index_map, cond_val) in enumerate(loader):
            onehot = onehot.to(device)
            cond_val = cond_val.to(device)
            N = onehot.size(0)

            #=================================================
            #  a) Update Discriminator
            #=================================================
            netD.zero_grad()

            # real_score => "本物っぽさ" => WGAN なので 大きいほど本物
            real_score = netD(onehot, cond_val)
            real_score_mean = real_score.mean()

            # fake
            noise_z = torch.randn(N, noise_dim, device=device)
            fake_data = netG(noise_z, cond_val)  # shape (N,out_channels,H,W)
            fake_score = netD(fake_data.detach(), cond_val)
            fake_score_mean = fake_score.mean()

            # WGAN Loss => Discriminatorを "real_scoreを高く" かつ "fake_scoreを低く"
            # => LossD = -(E[real_score] - E[fake_score]) = E[fake_score] - E[real_score]
            d_loss_main = fake_score_mean - real_score_mean

            # gradient penalty
            gp = gradient_penalty(netD, onehot, fake_data, cond_val)
            d_loss = d_loss_main + gp_lambda * gp
            d_loss.backward()
            optD.step()
            d_losses.append(d_loss.item())

            #=================================================
            #  b) Update Generator
            #=================================================
            netG.zero_grad()
            # Re-sample noise
            noise_z2 = torch.randn(N, noise_dim, device=device)
            gen_data = netG(noise_z2, cond_val)
            gen_score = netD(gen_data, cond_val)
            gen_score_mean = gen_score.mean()
            # Generator wants gen_score to be as large as possible => -mean(gen_score)
            g_loss = -gen_score_mean
            g_loss.backward()
            optG.step()
            g_losses.append(g_loss.item())

            if i % 10 == 0:
                print(f"[Epoch {ep}/{epochs}][Batch {i}/{len(loader)}]"
                      f"  D: {d_loss.item():.4f}  G: {g_loss.item():.4f}")

        avg_g = sum(g_losses)/len(g_losses)
        avg_d = sum(d_losses)/len(d_losses)
        total_loss = avg_d + avg_g  # 例として合計を見る

        # lr_scheduler
        schedulerG.step(total_loss)
        schedulerD.step(total_loss)

        print(f"Epoch [{ep}/{epochs}]  avgD={avg_d:.4f}  avgG={avg_g:.4f}  total={total_loss:.4f}")
        if ep % save_interval == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(netG.state_dict(), f"models/generator_ep{ep}.pth")
            torch.save(netD.state_dict(), f"models/discriminator_ep{ep}.pth")

        # ---- エポック終了時に評価 ----
        # 例えば 5エポック毎などで評価
        if ep % 5 == 0:
            netG.eval()
            with torch.no_grad():
                N_eval = 8  # 評価用に8枚だけ生成
                gen_samples = []
                
                for _ in range(N_eval):
                    z = torch.randn(1, noise_dim, device=device)
                    c = torch.rand(1, cond_dim, device=device)  # 適当なcond
                    fake_logits = netG(z, c)
                    pred = fake_logits.argmax(dim=1).squeeze(0).cpu().numpy()
                    gen_samples.append(pred)
                
                # evaluate_generated_layoutsは "文字ラベル" 想定の場合があるので
                # とりあえず "." などに相当するIDだけ変換 or 文字化
                # ここでは最低限: int→str に変換して渡す簡易例
                gen_list_str = [arr.astype(str) for arr in gen_samples]
                
                eval_res = evaluate_generated_layouts(gen_list_str, REQUIRED_ROOMS)
                rooms_ok = eval_res["num_rooms_ok"]
                constraints_ok = eval_res["num_constraints_ok"]
                total = eval_res["total_samples"]

            netG.train()
            
            # 結果を表示
            print(f"[Eval @ Epoch {ep}]"
                  f"  RoomsOK={rooms_ok}/{total},  ConstrOK={constraints_ok}/{total}")

        # Early Stop 監視
        early_stopper.check(total_loss)
        if early_stopper.should_stop:
            print(f"[EarlyStopping] epoch={ep} total_loss={total_loss:.4f} -> Stop training.")
            break

    print("WGAN-GP学習完了。")
    return netG, netD, label_list

#==================================================================
# 6) main
#==================================================================
def main():
    args = parse_args()
    train_wgan_gp(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        cond_dim=args.cond_dim,
        out_channels=args.out_channels,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        save_interval=args.save_interval,
        patience=args.patience
    )

if __name__ == "__main__":
    main()